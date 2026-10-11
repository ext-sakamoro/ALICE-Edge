//! Robust fitting (MAD outlier filter + robust linear).
//!
//! Both entry points take the outlier factor `k` as a **plain integer
//! multiplier** of the MAD (`k = 3` means "three MADs"), not a Q16.16 value,
//! and both refuse `k <= 0` with [`RobustError::NonPositiveK`]: with `k <= 0`
//! the rule `|deviation| > k · MAD` marks every sample that is not exactly
//! at the centre as an outlier, which destroys the window instead of
//! cleaning it.

extern crate alloc;
use alloc::vec::Vec;

use crate::q16_linear::fit_linear_fixed;

/// Why a robust entry point refused its arguments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RobustError {
    /// The outlier factor `k` was zero or negative. `k` multiplies the MAD
    /// as a plain integer and has to be at least 1.
    NonPositiveK {
        /// The rejected factor.
        k: i32,
    },
}

impl core::fmt::Display for RobustError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonPositiveK { k } => write!(
                f,
                "outlier factor k must be a positive integer multiplier of the MAD, got {k}"
            ),
        }
    }
}

impl std::error::Error for RobustError {}

/// `k` has to be a positive multiplier of the MAD.
const fn check_k(k: i32) -> Result<(), RobustError> {
    if k > 0 {
        Ok(())
    } else {
        Err(RobustError::NonPositiveK { k })
    }
}

/// Replace outliers by the median using the MAD (median absolute deviation)
/// rule: a sample `x` is an outlier when `|x - median| > k * MAD`.
///
/// `k` is a plain integer multiplier (`3` = three MADs, not Q16.16).
///
/// Deviations are computed in `i64`, so the full `i32` input range is valid
/// (`i32::MIN - median` and `|i32::MIN|` do not overflow), and the threshold
/// `k * MAD` saturates instead of wrapping; a huge `k` therefore keeps every
/// sample rather than silently discarding all of them.
///
/// # Errors
///
/// [`RobustError::NonPositiveK`] when `k <= 0`, whatever the window.
pub fn filter_outliers_mad(data: &[i32], k: i32) -> Result<Vec<i32>, RobustError> {
    check_k(k)?;
    let n = data.len();
    if n < 3 {
        return Ok(data.to_vec());
    }

    // 中央値の計算
    let mut sorted = data.to_vec();
    sorted.sort_unstable();
    let median = sorted[n / 2];
    let deviation = |x: i32| (i64::from(x) - i64::from(median)).abs();

    // MAD = median(|x_i - median|)
    let mut abs_devs: Vec<i64> = sorted.iter().map(|&x| deviation(x)).collect();
    abs_devs.sort_unstable();
    let mad = abs_devs[n / 2];

    if mad == 0 {
        return Ok(data.to_vec());
    }

    let threshold = i64::from(k).saturating_mul(mad);
    Ok(data
        .iter()
        .map(|&x| if deviation(x) > threshold { median } else { x })
        .collect())
}

/// Fit linear model with outlier removal pre-processing.
///
/// Robust linear fit: MAD outlier rejection on the **residuals** of a first
/// least-squares fit, outliers imputed with that fit's prediction (so the
/// abscissa `x = i` stays intact), then a second fit.
///
/// Rule: a sample is an outlier when `|residual − median(residual)| > k · MAD(residual)`,
/// where `k` is a plain integer multiplier (`3` = three MADs, not Q16.16).
///
/// # Errors
///
/// [`RobustError::NonPositiveK`] when `k <= 0`, whatever the window.
///
/// History (2026-09-17, oracle `tests/analytic_oracle.rs`): the previous
/// wrapper ran `filter_outliers_mad` on the raw values, i.e. compared each
/// point with the **global median of a trend**: on a ramp the MAD is ≈ n/4 ·
/// slope so only gross outliers were caught at all, and a caught one was
/// replaced by the global median instead of the value the line predicts
/// there — a ramp `y = x` with one outlier at `x = 10` refit with slope
/// 0.98 (2 % off) because the "repair" itself injected `y(10) = 50`.
#[cfg(feature = "std")]
#[inline]
pub fn fit_linear_robust(data: &[i32], k: i32) -> Result<(i32, i32), RobustError> {
    use crate::q16_linear::{evaluate_linear_fixed, int_to_q16, q16_to_int};
    check_k(k)?;
    let n = data.len();
    if n < 3 {
        return Ok(fit_linear_fixed(data));
    }
    // Iterate: fit → residual MAD → impute outliers with the line's own
    // prediction → refit, until the outlier set stops changing (the first
    // fit is pulled by the outlier, so a single pass leaves ~1 % slope bias).
    let mut repaired: Vec<i32> = data.to_vec();
    let mut mask = vec![false; n];
    let mut fit = fit_linear_fixed(&repaired);
    for _ in 0..8 {
        let (slope, intercept) = fit;
        let mut res: Vec<i64> = (0..n)
            .map(|i| {
                i64::from(int_to_q16(data[i]))
                    - i64::from(evaluate_linear_fixed(slope, intercept, i as i32))
            })
            .collect();
        let mut sorted = res.clone();
        sorted.sort_unstable();
        let median = sorted[n / 2];
        for r in &mut res {
            *r -= median;
        }
        let mut abs_devs: Vec<i64> = res.iter().map(|r| r.abs()).collect();
        abs_devs.sort_unstable();
        let mad = abs_devs[n / 2];
        if mad == 0 {
            return Ok(fit);
        }
        let threshold = i64::from(k).saturating_mul(mad);
        let new_mask: Vec<bool> = res.iter().map(|r| r.abs() > threshold).collect();
        for i in 0..n {
            repaired[i] = if new_mask[i] {
                // impute with the line's own prediction at this abscissa
                q16_to_int(evaluate_linear_fixed(slope, intercept, i as i32))
            } else {
                data[i]
            };
        }
        let next = fit_linear_fixed(&repaired);
        let converged = new_mask == mask;
        mask = new_mask;
        fit = next;
        if converged {
            break;
        }
    }
    Ok(fit)
}
