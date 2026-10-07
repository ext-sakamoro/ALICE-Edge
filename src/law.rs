//! Edge fits as [`SignalLaw`]: the transmitted model together with its
//! evidence, measured residual, valid range and provenance (feature `law`)
//!
//! [`fit_linear_fixed`] reduces a window of samples to two Q16.16 numbers.
//! [`linear_law`] keeps those two numbers exactly as Edge transmits them and
//! attaches what is needed to treat them as a claim about the data:
//!
//! | item | value |
//! |------|-------|
//! | coefficients | the Edge fit `(slope_q, intercept_q)`, converted without refitting |
//! | valid range | `[0, n − 1]`, the sample positions (evaluation outside it is refused) |
//! | evidence | `(i, samples[i])` for every sample |
//! | residual | `samples[i] − f(i)` measured over the original samples, so it includes the Q16 truncation of the fit (and any wrap of samples outside the Q16 range) |
//! | provenance | given by the caller |
//!
//! New windows are judged with [`SignalLaw::ingest`] (see [`sample_points`]).
//!
//! # Conversion
//!
//! Edge samples sit at `x = 0, 1, …, n − 1` and the receiver reconstructs
//! `y(x) = (slope_q · x + intercept_q) / 2¹⁶`. [`SignalLaw`] stores ascending
//! coefficients in `u = (x − lo) / (hi − lo) = x / (n − 1)`, so
//!
//! ```text
//! c0 = intercept_q / 2¹⁶
//! c1 = slope_q · (n − 1) / 2¹⁶
//! ```
//!
//! Both are exact in `f64` while `|slope_q| · (n − 1) < 2⁵³`, which holds for
//! every window shorter than 2²² samples. [`SignalLaw::evaluate`] then differs
//! from Edge's own reconstruction at a sample position by at most
//! `4ε(|c0| + |c1|)` (`ε = 2⁻⁵²`, three roundings), far below one Q16 step.
//!
//! The module needs `alloc` (the law owns its evidence) but not `std`; it
//! builds for bare-metal targets such as `thumbv7em-none-eabihf`.
//!
//! ```
//! use alice_edge::law::{linear_law, sample_points, IngestPolicy, Provenance, Verdict};
//!
//! // temperature × 100, rising 0.10 °C per sample
//! let samples = [2500, 2510, 2520, 2530, 2540];
//! let law = linear_law(&samples, Provenance::new("sensor 3, window 17", "fit_linear_fixed"))?;
//! assert_eq!(law.coefficients(), &[2500.0, 40.0]); // y = 2500 + 40·u, u = x / 4
//! assert_eq!(law.residual().max_abs, 0.0);
//! assert_eq!(law.evaluate(2.5)?, 2525.0); // between two samples
//! assert!(law.evaluate(5.0).is_err());    // beyond the window
//!
//! let policy = IngestPolicy { abs_tolerance: 1.0, break_factor: 4.0 };
//! let next = sample_points(&[2500, 2510, 2520, 2530, 2540]);
//! assert!(matches!(law.ingest(&next, &policy), Verdict::Supports { .. }));
//! # Ok::<(), alice_edge::law::LawError>(())
//! ```

extern crate alloc;

use alloc::vec::Vec;

pub use alice_core::law::{
    IngestPolicy, LawError, OracleCase, OracleOutcome, Provenance, ResidualStats, SignalLaw,
    SignalLawParts, ValidRange, Verdict,
};

use crate::q16_linear::{fit_linear_fixed, Q16_ONE};

/// The samples of a window as `(x, y)` points at `x = 0, 1, …, n − 1`
///
/// This is the evidence [`linear_law`] stores, and the form
/// [`SignalLaw::ingest`] takes for a new window of the same sensor.
#[must_use]
pub fn sample_points(samples: &[i32]) -> Vec<(f64, f64)> {
    samples
        .iter()
        .enumerate()
        .map(|(i, &y)| (i as f64, f64::from(y)))
        .collect()
}

/// The Edge linear fit of `samples` as a [`SignalLaw`]
///
/// The coefficients are those of [`fit_linear_fixed`] (converted as described
/// in the [module documentation](self), not refitted), the valid range is
/// `[0, n − 1]` and the residual is measured against `samples`.
///
/// Because the residual is measured against the samples and not reported by
/// the fit, it is an independent check on the fit itself: a window the fit
/// cannot represent (samples outside the Q16 range, or a window long enough
/// that the fixed-point arithmetic loses the slope) shows up as a residual
/// orders of magnitude above the Q16 step, while a fit that describes its
/// window has a residual below it. Reading [`SignalLaw::residual`] before
/// transmitting the coefficients is the cheapest way to find out which of the
/// two happened.
///
/// # Errors
///
/// [`LawError::TooFewPoints`] for fewer than two samples (a line needs two
/// points and a window of one sample has no range).
pub fn linear_law(samples: &[i32], provenance: Provenance) -> Result<SignalLaw, LawError> {
    if samples.len() < 2 {
        return Err(LawError::TooFewPoints);
    }
    let (slope_q, intercept_q) = fit_linear_fixed(samples);
    let q = f64::from(Q16_ONE);
    let span = (samples.len() - 1) as f64;
    SignalLaw::from_parts(SignalLawParts {
        coefficients: alloc::vec![f64::from(intercept_q) / q, f64::from(slope_q) * span / q],
        domain: ValidRange { lo: 0.0, hi: span },
        evidence: sample_points(samples),
        // from_parts measures the residual again over the evidence
        residual: ResidualStats {
            n: 0,
            rms: 0.0,
            max_abs: 0.0,
        },
        provenance,
        oracles: Vec::new(),
    })
}
