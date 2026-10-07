#![cfg(feature = "std")]
//! What every entry point does with a degenerate argument, asserted by value.
//!
//! The fits run on a device, on data nobody inspected first: an empty window
//! because the sensor returned nothing, a single sample, a full-scale `i32`
//! from a disconnected ADC line, a `NaN` from a divide in the caller. Each of
//! those has exactly one correct answer, and "it does not panic" is not it —
//! a function that swallows a degenerate window and returns a plausible fit
//! is worse than one that refuses it, because the wrong coefficients are then
//! transmitted as a model of the data.
//!
//! So every case below first states which of **`Err` / early return / a
//! specific value / a panic** is the contract, and then asserts that one. The
//! cases are grouped as empty and length-one windows, the divisions by a
//! caller-supplied number, out-of-range and negative arguments, non-finite
//! arguments, the Q16.16 range, the integer overflow paths, and the two
//! constructors that refuse their argument outright.
//!
//! Two of these depend on the build profile and are written so that **both**
//! profiles are checked rather than one being skipped: Rust's arithmetic
//! operators panic on overflow with `debug_assertions` and wrap without it,
//! so `evaluate_quadratic_fixed` and `evaluate_cubic_fixed` outside the Q16
//! range panic under `cargo test` and return a wrapped value under `cargo
//! test --release`. The wrapped value is the dangerous one (it is a plausible
//! number), so it is pinned explicitly. Slice indexing and division by zero,
//! by contrast, are checked in every profile.
//!
//! Panic messages reach the captured per-test output and are only printed
//! when a case fails, so no panic hook is installed here.

use std::panic::{catch_unwind, AssertUnwindSafe};

use alice_edge::adaptive_polyfit::{
    evaluate_cubic_fixed, evaluate_quadratic_fixed, fit_cubic_fixed, fit_quadratic_fixed,
    should_use_linear,
};
use alice_edge::constant_fit::{compute_residual_error, fit_constant_fixed};
use alice_edge::piecewise::fit_piecewise_linear;
use alice_edge::q16_linear::{evaluate_linear_fixed, fit_linear_fixed, int_to_q16, q16_to_int};
use alice_edge::ring_buffer::RingBuffer;
use alice_edge::robust::{filter_outliers_mad, fit_linear_robust};
use alice_edge::sensor_fusion::{
    FusedSensor, FusionConfig, KalmanFilter1D, KalmanFilter2D, SensorInput,
};
use alice_edge::simd_fit::fit_linear_simd;

/// What a call did: the value it returned, or the message of the panic it
/// raised. Both are outcomes to be asserted, never collapsed into "fine".
#[derive(Debug)]
enum Outcome<T> {
    Value(T),
    Panic(String),
}

fn run<T>(f: impl FnOnce() -> T) -> Outcome<T> {
    match catch_unwind(AssertUnwindSafe(f)) {
        Ok(v) => Outcome::Value(v),
        Err(e) => {
            let msg = e
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| e.downcast_ref::<&str>().copied())
                .unwrap_or("<panic payload is not a string>");
            Outcome::Panic(msg.to_owned())
        }
    }
}

#[track_caller]
fn value<T>(what: &str, o: Outcome<T>) -> T {
    match o {
        Outcome::Value(v) => v,
        Outcome::Panic(m) => panic!("{what} was expected to return a value, it panicked: {m}"),
    }
}

#[track_caller]
fn panic_containing<T: core::fmt::Debug>(what: &str, needle: &str, o: Outcome<T>) {
    match o {
        Outcome::Value(v) => {
            panic!("{what} was expected to panic with `{needle}`, it returned {v:?}")
        }
        Outcome::Panic(m) => assert!(
            m.contains(needle),
            "{what} panicked with `{m}`, which does not contain `{needle}`"
        ),
    }
}

// ---------------------------------------------------------------------------
// Empty and length-one windows
// ---------------------------------------------------------------------------

/// Contract: **a specific value, not a panic and not an `Err`.** A window can
/// legitimately be empty (the sensor produced nothing in the interval), and
/// the zero fit `y = 0` is the only model with no evidence behind it. Every
/// fit therefore returns all-zero coefficients, and the derived helpers
/// return the empty result rather than indexing into nothing.
#[test]
fn an_empty_window_gives_the_zero_fit_from_every_entry_point() {
    assert_eq!(
        value("fit_linear_fixed", run(|| fit_linear_fixed(&[]))),
        (0, 0)
    );
    assert_eq!(
        value("fit_linear_simd", run(|| fit_linear_simd(&[]))),
        (0, 0)
    );
    assert_eq!(
        value("fit_constant_fixed", run(|| fit_constant_fixed(&[]))),
        0
    );
    assert_eq!(
        value("fit_quadratic_fixed", run(|| fit_quadratic_fixed(&[]))),
        (0, 0, 0)
    );
    assert_eq!(
        value("fit_cubic_fixed", run(|| fit_cubic_fixed(&[]))),
        (0, 0, 0, 0)
    );
    assert_eq!(
        value(
            "compute_residual_error",
            run(|| compute_residual_error(&[], 0, 0))
        ),
        0
    );
    // an empty window is not evidence that a line beats a constant
    assert!(!value("should_use_linear", run(|| should_use_linear(&[]))));
    assert!(value("filter_outliers_mad", run(|| filter_outliers_mad(&[], 3))).is_empty());
    assert_eq!(
        value("fit_linear_robust", run(|| fit_linear_robust(&[], 3))),
        (0, 0)
    );
    assert!(value(
        "fit_piecewise_linear",
        run(|| fit_piecewise_linear(&[], 0, 4))
    )
    .is_empty());
}

/// Contract: **a specific value.** One sample is a constant, not a line: the
/// slope is zero and the intercept is that sample in Q16.16. The piecewise
/// fit returns one segment covering it rather than refusing to split.
#[test]
fn a_single_sample_window_gives_a_constant_fit() {
    assert_eq!(fit_linear_fixed(&[42]), (0, int_to_q16(42)));
    assert_eq!(fit_linear_simd(&[42]), (0, int_to_q16(42)));
    assert_eq!(fit_constant_fixed(&[42]), int_to_q16(42));
    assert_eq!(fit_quadratic_fixed(&[42]), (0, 0, int_to_q16(42)));
    assert_eq!(fit_cubic_fixed(&[42]), (0, 0, 0, int_to_q16(42)));
    assert!(!should_use_linear(&[42]));

    let segments = fit_piecewise_linear(&[42], 0, 4);
    assert_eq!(segments.len(), 1);
    assert_eq!((segments[0].start, segments[0].end), (0, 1));
    assert_eq!(
        (segments[0].slope, segments[0].intercept),
        (0, int_to_q16(42))
    );
}

// ---------------------------------------------------------------------------
// The divisions by a caller-supplied number
// ---------------------------------------------------------------------------

/// Contract: **early return, with the guarded value asserted.** Three
/// divisions take a number derived from the argument: the window length (the
/// mean), the normal-equation determinant (the slope) and the median absolute
/// deviation (the outlier threshold). Each guards its zero, and this case
/// pins the guarded answer rather than only the absence of a panic — the
/// division by zero would be caught in every profile anyway, so "it did not
/// panic" would pass even with the guards removed from a path that cannot
/// reach zero.
#[test]
fn every_division_by_a_derived_number_guards_its_zero() {
    // length zero: the mean is not computed at all
    assert_eq!(fit_constant_fixed(&[]), 0);

    // the determinant n²(n² − 1) / 12 is zero only for n < 2, which the
    // length-one branch takes first; n = 2 is the smallest window that
    // reaches the division, and the fit through two points is exact
    assert_eq!(fit_linear_fixed(&[0, 0]), (0, 0));
    assert_eq!(
        fit_linear_fixed(&[100, 200]),
        (int_to_q16(100), int_to_q16(100))
    );

    // MAD zero (every sample identical): the window is returned untouched
    // instead of dividing by it
    let flat = [7i32; 9];
    assert_eq!(filter_outliers_mad(&flat, 3), flat.to_vec());
    assert_eq!(fit_linear_robust(&flat, 3), (0, int_to_q16(7)));

    // a minimum segment length below two cannot terminate the recursion, so
    // it is normalised to two rather than dividing the window forever
    let ramp: Vec<i32> = (0..16).collect();
    let with_zero = fit_piecewise_linear(&ramp, 0, 0);
    let with_two = fit_piecewise_linear(&ramp, 0, 2);
    assert_eq!(with_zero, with_two);
    assert!(with_zero.iter().all(|s| s.end - s.start >= 2));
    assert_eq!(with_zero.last().expect("16 samples give segments").end, 16);
}

// ---------------------------------------------------------------------------
// Out-of-range and negative arguments
// ---------------------------------------------------------------------------

/// Contract: **a specific value, documented as lossy.** The MAD rule keeps a
/// sample when `|x − median| <= k · MAD`, so `k <= 0` keeps nothing: every
/// sample is replaced by the median. That is the arithmetic working as
/// written, and it is also a way to destroy a window with one bad argument,
/// so it is pinned by value here and the saturating upper end is pinned next
/// to it (a huge `k` keeps everything instead of wrapping negative).
#[test]
fn a_non_positive_mad_factor_collapses_the_window_onto_its_median() {
    let w = [1i32, 2, 3, 4, 100];
    let median = 3i32;

    for k in [0i32, -1, -5, i32::MIN] {
        assert_eq!(
            filter_outliers_mad(&w, k),
            vec![median; w.len()],
            "k = {k} should replace every sample by the median"
        );
    }
    // the other end: the threshold saturates, so nothing is rejected
    for k in [i32::MAX, 1 << 30] {
        assert_eq!(filter_outliers_mad(&w, k), w.to_vec(), "k = {k}");
    }
    // between the two ends the outlier, and only the outlier, is replaced
    assert_eq!(filter_outliers_mad(&w, 2), vec![1, 2, 3, 4, median]);
}

/// Contract: **a specific value.** `fit_piecewise_linear` takes two
/// caller-supplied bounds and neither is validated. A negative error budget
/// cannot be met by any segment, so the window is split down to the minimum
/// segment length; a minimum longer than the window produces one segment.
/// Both are total, and both are pinned so that a change of either bound's
/// meaning is a failure here.
#[test]
fn piecewise_bounds_outside_their_intended_range_stay_total() {
    let ramp: Vec<i32> = (0..16).collect();

    let minimal = fit_piecewise_linear(&ramp, -1, 2);
    assert_eq!(minimal.len(), 8);
    assert!(minimal.iter().all(|s| s.end - s.start == 2));
    assert_eq!(minimal[0].start, 0);
    assert_eq!(minimal[7].end, 16);

    for min_len in [16usize, 17, usize::MAX] {
        let whole = fit_piecewise_linear(&ramp, 0, min_len);
        assert_eq!(whole.len(), 1, "min_len = {min_len}");
        assert_eq!((whole[0].start, whole[0].end), (0, 16));
        assert_eq!(whole[0].slope, int_to_q16(1));
    }
}

/// Contract: **`Err`, not a clamped value.** A law is a claim about the range
/// its evidence covers; asked outside that range, including for a non-finite
/// `x`, it refuses rather than extrapolating. A window shorter than two
/// samples has no range at all and is refused at construction.
#[cfg(feature = "law")]
#[test]
fn a_law_refuses_arguments_outside_the_range_its_evidence_covers() {
    use alice_edge::law::{linear_law, sample_points, IngestPolicy, LawError, Provenance, Verdict};

    let prov = || Provenance::new("panic contract", "fit_linear_fixed");

    assert!(matches!(
        linear_law(&[], prov()),
        Err(LawError::TooFewPoints)
    ));
    assert!(matches!(
        linear_law(&[42], prov()),
        Err(LawError::TooFewPoints)
    ));
    assert!(sample_points(&[]).is_empty());

    let law = linear_law(&[2500, 2510, 2520], prov()).expect("three samples are enough");
    // inside: exact at the sample positions
    assert_eq!(law.evaluate(0.0).expect("x = 0 is in range"), 2500.0);
    assert_eq!(law.evaluate(2.0).expect("x = 2 is in range"), 2520.0);
    // outside, on both sides, and non-finite: refused, never clamped
    for x in [-1.0f64, -0.000_001, 2.000_001, 3.0, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(law.evaluate(x), Err(LawError::OutOfRange)),
            "evaluate({x}) should be refused"
        );
    }

    let policy = IngestPolicy {
        abs_tolerance: 1.0,
        break_factor: 4.0,
    };
    // no evidence is its own verdict, not an empty "supports"
    assert!(matches!(law.ingest(&[], &policy), Verdict::NoEvidence));
    // a non-finite or out-of-range point is counted, not judged
    assert!(matches!(
        law.ingest(&[(f64::NAN, 1.0)], &policy),
        Verdict::OutOfRange { outside: 1 }
    ));
    assert!(matches!(
        law.ingest(&[(99.0, 1.0)], &policy),
        Verdict::OutOfRange { outside: 1 }
    ));
}

// ---------------------------------------------------------------------------
// Non-finite arguments
// ---------------------------------------------------------------------------

/// Contract: **a specific value, and that value is `NaN`.** The filters take
/// `f32` through public fields with no validation anywhere — not in the
/// constructor, not in `predict`, not in `update` — so the domain belongs to
/// the caller. What this case fixes is that a non-finite measurement is *not*
/// silently absorbed: it reaches the estimate, where a caller checking
/// `is_finite()` can see it, rather than being dropped or clamped into a
/// plausible number.
#[test]
fn a_non_finite_measurement_reaches_the_estimate_rather_than_being_absorbed() {
    let mut k = KalmanFilter1D::new(0.0, 1.0, 0.01, 0.1);
    assert!(k.filter(f32::NAN).is_nan());
    assert!(k.estimate().is_nan());
    // the covariance keeps shrinking normally: the gain did not see the NaN
    assert!(k.error().is_finite());

    let mut k = KalmanFilter1D::new(0.0, 1.0, 0.01, 0.1);
    assert_eq!(k.filter(f32::INFINITY), f32::INFINITY);

    let mut k2 = KalmanFilter2D::new(0.05, 0.01, 0.5);
    let (p, v) = k2.filter(f32::NAN);
    assert!(p.is_nan() && v.is_nan());

    // all-zero noise makes the gain 0 / 0: NaN, not a division-by-zero panic
    // (float division by zero is defined, unlike the integer one)
    let mut degenerate = KalmanFilter1D::new(0.0, 0.0, 0.0, 0.0);
    degenerate.update(1.0);
    assert!(degenerate.estimate().is_nan());
    assert!(degenerate.error().is_nan());
}

/// Contract: **early return with the estimate unchanged.** The fusion weights
/// a sensor by `1 / noise`, so a non-positive or non-finite noise produces a
/// non-positive or non-finite total weight, and the update is skipped: the
/// previous estimate is returned untouched. An infinite *value* is a
/// different case — it is a measurement, and the outlier test rejects it and
/// counts it.
#[test]
fn the_fusion_skips_its_update_when_no_input_carries_a_usable_weight() {
    let one = |value: f32, noise: f32| {
        let mut f = FusedSensor::new(FusionConfig::default());
        let out = f.fuse(&[SensorInput { value, noise }]);
        (out, f.estimate(), f.fusion_count(), f.rejected_count())
    };

    // no inputs at all: the estimate is returned and nothing is counted
    let mut empty = FusedSensor::new(FusionConfig::default());
    assert_eq!(empty.fuse(&[]), 0.0);
    assert_eq!((empty.fusion_count(), empty.rejected_count()), (0, 0));

    // noise = 0 gives an infinite weight, and infinity / infinity is NaN
    let (out, est, fused, rejected) = one(1.0, 0.0);
    assert!(out.is_nan() && est.is_nan());
    assert_eq!((fused, rejected), (1, 0));

    // NaN noise: not rejected (NaN compares false), but carries no weight
    let (out, est, fused, rejected) = one(1.0, f32::NAN);
    assert_eq!((out, est), (0.0, 0.0));
    assert_eq!((fused, rejected), (1, 0));

    // negative noise: a negative total weight is not usable either
    let (out, est, fused, rejected) = one(5.0, -1.0);
    assert_eq!((out, est), (0.0, 0.0));
    assert_eq!((fused, rejected), (1, 0));

    // an infinite measurement with usable noise is rejected as an outlier
    let (out, est, fused, rejected) = one(f32::INFINITY, 1.0);
    assert_eq!((out, est), (0.0, 0.0));
    assert_eq!((fused, rejected), (1, 1));
}

// ---------------------------------------------------------------------------
// The Q16.16 range
// ---------------------------------------------------------------------------

/// Contract: **a specific wrapped value, by design.** Q16.16 covers
/// `|y| < 2¹⁵`; `int_to_q16` is a shift, which in Rust wraps the value in
/// every profile (only the shift *amount* is checked), so samples outside the
/// range wrap silently. That is the documented contract, and the `law`
/// feature exists so that the wrap shows up as a residual. The exact wrapped
/// values are pinned here because a change from wrapping to saturating would
/// otherwise be invisible.
#[test]
fn samples_outside_the_q16_range_wrap_rather_than_saturate() {
    assert_eq!(int_to_q16(32_767), 32_767 << 16);
    assert_eq!(int_to_q16(32_768), i32::MIN);
    assert_eq!(int_to_q16(-32_768), -32_768 << 16);
    assert_eq!(int_to_q16(-32_769), 2_147_418_112);
    assert_eq!(int_to_q16(100_000), -2_036_334_592);
    assert_eq!(int_to_q16(i32::MAX), -65_536);
    assert_eq!(q16_to_int(int_to_q16(i32::MAX)), -1);

    // the fit of a full-scale constant window wraps the same way, and the
    // slope stays exactly zero
    assert_eq!(fit_linear_fixed(&[i32::MAX; 4]), (0, -65_536));
    assert_eq!(fit_linear_fixed(&[i32::MIN; 4]), (0, 0));
    assert_eq!(fit_linear_simd(&[i32::MAX; 16]), (0, -65_536));
    assert_eq!(fit_constant_fixed(&[i32::MAX; 8]), -65_536);

    // the linear evaluator is written with wrapping operators, so it is
    // total in both profiles
    assert_eq!(
        evaluate_linear_fixed(i32::MAX, i32::MAX, i32::MAX),
        i32::MIN
    );
    assert_eq!(
        evaluate_linear_fixed(i32::MIN, i32::MIN, i32::MIN),
        i32::MIN
    );
}

// ---------------------------------------------------------------------------
// The integer overflow paths
// ---------------------------------------------------------------------------

/// Contract: **a panic with `debug_assertions`, a specific wrapped value
/// without it.** Unlike `evaluate_linear_fixed`, the quadratic and cubic
/// evaluators use plain `*` and `+`, so a reconstruction outside the Q16
/// range overflows. Rust panics on that in a debug build and wraps in a
/// release build, and the release answer is the dangerous one because it is a
/// plausible number, so this case asserts the panic *and* the wrapped value
/// rather than running in one profile only.
#[test]
fn the_polynomial_evaluators_overflow_in_debug_and_wrap_in_release() {
    // `(a·x² + b·x) as i32 + c` overflows in the final i32 addition
    let add_overflow_q = || evaluate_quadratic_fixed(1 << 16, 0, i32::MAX, 1);
    let add_overflow_c = || evaluate_cubic_fixed(1 << 16, 0, 0, i32::MAX, 1);
    // `(a as i64) * x * x` overflows in i64 before the cast
    let mul_overflow_q = || evaluate_quadratic_fixed(i32::MAX, 0, 0, i32::MAX);
    let mul_overflow_c = || evaluate_cubic_fixed(i32::MAX, 0, 0, 0, 1 << 21);

    if cfg!(debug_assertions) {
        panic_containing(
            "quadratic i32 add",
            "attempt to add with overflow",
            run(add_overflow_q),
        );
        panic_containing(
            "cubic i32 add",
            "attempt to add with overflow",
            run(add_overflow_c),
        );
        panic_containing(
            "quadratic i64 multiply",
            "attempt to multiply with overflow",
            run(mul_overflow_q),
        );
        panic_containing(
            "cubic i64 multiply",
            "attempt to multiply with overflow",
            run(mul_overflow_c),
        );
    } else {
        assert_eq!(
            value("quadratic i32 add", run(add_overflow_q)),
            -2_147_418_113
        );
        assert_eq!(value("cubic i32 add", run(add_overflow_c)), -2_147_418_113);
        assert_eq!(
            value("quadratic i64 multiply", run(mul_overflow_q)),
            i32::MAX
        );
        assert_eq!(value("cubic i64 multiply", run(mul_overflow_c)), 0);
    }

    // inside the Q16 range both evaluators are exact in either profile:
    // y = x² from the window [0, 1, 4, 9, 16, 25, 36, 49]
    let (a, b, c) = fit_quadratic_fixed(&[0, 1, 4, 9, 16, 25, 36, 49]);
    for x in 0i32..8 {
        assert_eq!(q16_to_int(evaluate_quadratic_fixed(a, b, c, x)), x * x);
    }
    let (ca, cb, cc, cd) = fit_cubic_fixed(&[0, 1, 8, 27, 64, 125, 216, 343]);
    for x in 0i32..8 {
        assert_eq!(
            q16_to_int(evaluate_cubic_fixed(ca, cb, cc, cd, x)),
            x * x * x
        );
    }
}

/// Contract: **degrade to the lower-degree fit, never panic.** The quadratic
/// and cubic normal equations are solved with checked `i128` arithmetic, so a
/// window long enough to overflow the moment determinant falls back instead
/// of returning wrapped coefficients. A 5000-sample ramp is past the cubic's
/// exact capacity, and the answer it falls back to is the exact line through
/// that ramp.
#[test]
fn an_overlong_window_degrades_to_the_lower_degree_fit() {
    let ramp: Vec<i32> = (0..5_000).map(|i| i * 3).collect();

    let (a, b, c, d) = value("fit_cubic_fixed", run(|| fit_cubic_fixed(&ramp)));
    assert_eq!((a, b), (0, 0), "the cubic and quadratic terms are dropped");
    assert_eq!((c, d), (int_to_q16(3), 0));

    let (qa, qb, qc) = value("fit_quadratic_fixed", run(|| fit_quadratic_fixed(&ramp)));
    assert_eq!(qa, 0, "the quadratic term is dropped");
    assert_eq!((qb, qc), (int_to_q16(3), 0));
}

/// Contract: **exact, for every window up to the measured capacity.** The
/// linear fit shifts the numerator up by 16 before dividing, so the range in
/// which it is exact is bounded by `i64`, and the bound depends on the window
/// length and the slope together rather than on either alone. The three
/// capacities below were found by bisection on exact ramps `y = step · x`
/// (the largest window whose fit is still the exact slope): 6410 for
/// `step = 1`, 4870 for `step = 3`, 2027 for `step = 100`. Each assertion is
/// the exact answer, so no wrapped value is written down here.
///
/// The companion case
/// `the_linear_fit_should_stay_exact_past_its_current_capacity` is ignored
/// and records what happens one sample past each of these.
#[test]
fn the_linear_fit_is_exact_up_to_its_measured_capacity() {
    // (slope of the ramp, largest window that is still exact — bisected)
    for (step, capacity) in [(1i32, 6_410usize), (3, 4_870), (100, 2_027)] {
        let ramp: Vec<i32> = (0..capacity).map(|i| i as i32 * step).collect();
        assert_eq!(
            fit_linear_fixed(&ramp),
            (int_to_q16(step), 0),
            "step {step}, window {capacity}"
        );
        assert_eq!(
            fit_linear_simd(&ramp),
            (int_to_q16(step), 0),
            "the SIMD kernel must agree (step {step}, window {capacity})"
        );
    }

    // a constant window never loses the slope, however long it is: the
    // numerator is exactly zero, so there is nothing for the shift to lose
    assert_eq!(
        fit_linear_fixed(&vec![32_767; 20_000]),
        (0, int_to_q16(32_767))
    );
}

/// Known defect, recorded rather than asserted: one sample past the capacity
/// of the case above, `fit_linear_fixed` returns a slope with the wrong sign
/// and magnitude instead of degrading or refusing.
///
/// The mechanism is that `slope_num << Q16_SHIFT` is a multiplication by
/// 2¹⁶ written as a shift, and **a shift does not check the value for
/// overflow** — only the shift amount is checked — so the numerator wraps
/// with no diagnostic in either build profile. `fit_quadratic_fixed` and
/// `fit_cubic_fixed` solve the same normal equations with checked `i128` and
/// are exact here, so the defect is specific to the first-degree path;
/// `fit_linear_simd` shares the same expression and the same limit.
///
/// With the `law` feature the break is visible in the residual (measured:
/// RMS 1.6e-13 at the capacity, 8.4e3 one sample past it), but the raw fit
/// reports nothing.
#[test]
#[ignore = "known defect: the linear fit multiplies by 2^16 with a shift, which does not check the value for overflow, so a long window wraps"]
fn the_linear_fit_should_stay_exact_past_its_current_capacity() {
    for (step, capacity) in [(1i32, 6_410usize), (3, 4_870), (100, 2_027)] {
        let ramp: Vec<i32> = (0..=capacity).map(|i| i as i32 * step).collect();
        assert_eq!(
            fit_linear_fixed(&ramp),
            (int_to_q16(step), 0),
            "step {step}, window {}",
            capacity + 1
        );
    }
}

/// Contract: **bit-identical, across and past the capacity boundary.** The
/// scalar fit and the SSE2 / NEON kernels are three implementations of one
/// formula, and only one of them runs on a given target, so a divergence
/// between them is a divergence between devices. They agree today both where
/// the fit is exact and where it wraps, and this case fixes that: a change to
/// one of the three that is not made to the others fails here.
#[test]
fn the_simd_kernels_agree_with_the_scalar_fit_bit_for_bit() {
    // lengths around the kernel threshold (8) and its remainder handling
    for n in 0usize..40 {
        let w: Vec<i32> = (0..n).map(|i| 13 * i as i32 - 500).collect();
        assert_eq!(fit_linear_simd(&w), fit_linear_fixed(&w), "window {n}");
    }
    // across the capacity boundary of the `step = 3` ramp (4870), on both
    // sides, so the two implementations stay together through the wrap
    for n in [4_000usize, 4_869, 4_870, 4_871, 4_872, 5_000, 8_192] {
        let w: Vec<i32> = (0..n).map(|i| i as i32 * 3).collect();
        assert_eq!(fit_linear_simd(&w), fit_linear_fixed(&w), "window {n}");
    }
    // full-scale samples, where the Q16 conversion itself wraps
    for n in [1usize, 7, 8, 9, 64, 1_000] {
        let hi: Vec<i32> = vec![i32::MAX; n];
        let lo: Vec<i32> = vec![i32::MIN; n];
        assert_eq!(
            fit_linear_simd(&hi),
            fit_linear_fixed(&hi),
            "i32::MAX x {n}"
        );
        assert_eq!(
            fit_linear_simd(&lo),
            fit_linear_fixed(&lo),
            "i32::MIN x {n}"
        );
    }
}

/// Contract: **a panic, in every profile.** `RingBuffer<T, 0>` has no slot to
/// write into; `push` computes its index modulo the capacity, and an integer
/// division by zero is checked in release builds too. Reading from it is
/// still total (`None`), so the panic is specific to `push`.
#[test]
#[should_panic(expected = "divisor of zero")]
fn pushing_into_a_zero_capacity_ring_buffer_panics() {
    let mut rb = RingBuffer::<i32, 0>::new();
    rb.push(1);
}

/// Contract: **`None`, not a panic.** Every read of a ring buffer is total,
/// including a zero-capacity one and an index past the live elements.
#[test]
fn every_read_of_a_ring_buffer_is_total() {
    let empty = RingBuffer::<i32, 0>::new();
    assert_eq!(empty.len(), 0);
    assert!(empty.is_empty());
    assert_eq!(empty.first(), None);
    assert_eq!(empty.last(), None);
    assert_eq!(empty.get(0), None);
    assert_eq!(empty.iter().count(), 0);

    let mut rb = RingBuffer::<i32, 4>::new();
    assert_eq!(rb.first(), None);
    rb.push(1);
    assert_eq!(rb.get(1), None);
    assert_eq!(rb.get(usize::MAX), None);
    for i in 2..10 {
        rb.push(i);
    }
    // full and wrapped: the window is the last four pushes, in order
    assert_eq!(rb.iter().copied().collect::<Vec<i32>>(), vec![6, 7, 8, 9]);
    assert_eq!(rb.get(4), None);
}

// ---------------------------------------------------------------------------
// Constructors that refuse their argument
// ---------------------------------------------------------------------------

/// Contract: **a panic naming the bound that was broken.** The classifier has
/// a closed domain for `num_classes`, enforced at construction on both sides.
/// It is enforced there, and not where the value is first used, because an
/// accepted zero fails inside `classify` with a generic bounds-check message,
/// several calls away from the argument that caused it. The two bounds are
/// separate assertions so that the message identifies which end was broken,
/// and each `should_panic` names the message it expects rather than accepting
/// any panic.
#[cfg(feature = "ml")]
#[test]
#[should_panic(expected = "exceeds MAX_CLASSES")]
fn more_classes_than_the_logits_buffer_holds_is_refused_at_construction() {
    let _ = alice_edge::object_classifier::TernaryClassifier::new(
        alice_edge::object_classifier::TernaryClassifier::MAX_CLASSES + 1,
    );
}

/// Contract: **a panic naming the lower bound.** See the case above.
#[cfg(feature = "ml")]
#[test]
#[should_panic(expected = "num_classes must be at least 1")]
fn zero_classes_is_refused_at_construction() {
    let _ = alice_edge::object_classifier::TernaryClassifier::new(0);
}

/// Contract: **a panic naming the lower bound, from `from_weights` too.** The
/// check is on every construction path, not only on the one that generates
/// its own weights.
#[cfg(feature = "ml")]
#[test]
#[should_panic(expected = "num_classes must be at least 1")]
fn zero_classes_is_refused_by_from_weights_as_well() {
    use alice_edge::object_classifier::{TernaryClassifier, FEATURE_DIM, HIDDEN_DIM};
    let w1 = vec![0i8; HIDDEN_DIM * FEATURE_DIM];
    let w2 = vec![0i8; HIDDEN_DIM * HIDDEN_DIM];
    let _ = TernaryClassifier::from_weights(&w1, &w2, &[], 0);
}

/// Contract: **a panic naming the upper bound, from `from_weights` too.**
#[cfg(feature = "ml")]
#[test]
#[should_panic(expected = "exceeds MAX_CLASSES")]
fn too_many_classes_is_refused_by_from_weights_as_well() {
    use alice_edge::object_classifier::{TernaryClassifier, FEATURE_DIM, HIDDEN_DIM};
    let n = TernaryClassifier::MAX_CLASSES + 1;
    let w1 = vec![0i8; HIDDEN_DIM * FEATURE_DIM];
    let w2 = vec![0i8; HIDDEN_DIM * HIDDEN_DIM];
    let w3 = vec![0i8; n * HIDDEN_DIM];
    let _ = TernaryClassifier::from_weights(&w1, &w2, &w3, n);
}

/// Contract: **a panic naming the layer and both lengths.** A weight vector
/// of the wrong length is a loaded model that does not match the
/// architecture. The kernel constructor one level down also rejects it, but
/// with a bare `assertion left == right failed` that names neither the layer
/// nor the expected length, so the check is made here where both are known.
#[cfg(feature = "ml")]
#[test]
#[should_panic(expected = "w3 has 0 values, expected num_classes * HIDDEN_DIM = 128")]
fn a_weight_vector_of_the_wrong_length_is_refused_with_the_layer_named() {
    use alice_edge::object_classifier::{TernaryClassifier, FEATURE_DIM, HIDDEN_DIM};
    let w1 = vec![0i8; HIDDEN_DIM * FEATURE_DIM];
    let w2 = vec![0i8; HIDDEN_DIM * HIDDEN_DIM];
    let _ = TernaryClassifier::from_weights(&w1, &w2, &[], 4);
}

/// Contract: **a value, across the whole closed domain.** The bounds above
/// would also be satisfied by a constructor that refused everything, so the
/// usable ends and the middle are asserted here: every `num_classes` from 1
/// to `MAX_CLASSES` constructs, classifies, and reports a finite confidence,
/// through both construction paths.
#[cfg(feature = "ml")]
#[test]
fn the_whole_closed_domain_of_num_classes_constructs_and_classifies() {
    use alice_edge::object_classifier::{SdfFeatures, TernaryClassifier, FEATURE_DIM, HIDDEN_DIM};

    let features = SdfFeatures::from_primitive(0, &[1.0, 0.5, 0.25], [1.0, 2.0, 0.5], 128);
    let w1 = vec![1i8; HIDDEN_DIM * FEATURE_DIM];
    let w2 = vec![-1i8; HIDDEN_DIM * HIDDEN_DIM];

    for n in 1..=TernaryClassifier::MAX_CLASSES {
        let generated = value("TernaryClassifier::new", run(|| TernaryClassifier::new(n)));
        let (_, confidence) = value("classify", run(|| generated.classify(&features)));
        assert!(
            confidence.is_finite(),
            "num_classes = {n} should classify to a finite confidence"
        );

        let w3 = vec![0i8; n * HIDDEN_DIM];
        let loaded = value(
            "TernaryClassifier::from_weights",
            run(|| TernaryClassifier::from_weights(&w1, &w2, &w3, n)),
        );
        let (_, confidence) = value("classify", run(|| loaded.classify(&features)));
        assert!(
            confidence.is_finite(),
            "num_classes = {n} from loaded weights should classify to a finite confidence"
        );
    }
}

// ---------------------------------------------------------------------------
// No invariant is enforced on any mutation path
// ---------------------------------------------------------------------------

/// Contract: **the caller owns the domain, and this is asserted rather than
/// assumed.** `FusionConfig`, `KalmanFilter1D` and `SensorInput` expose their
/// numbers as public fields with no constructor check, no setter and no
/// `clamp`, so there is no invariant for a later mutation to escape. This
/// case states that explicitly: a value written straight into a field after
/// construction is used as written. If a guard is ever added to the
/// constructor, it has to be added to the field access as well, and this case
/// is where that shows up.
#[test]
fn the_filter_state_is_public_and_unguarded_on_every_path() {
    // a value the constructor would plausibly reject, written directly
    let mut k = KalmanFilter1D::new(0.0, 1.0, 0.01, 0.1);
    k.q = -1.0;
    k.predict();
    assert_eq!(
        k.error(),
        0.0,
        "a negative process noise is used as written"
    );

    let mut k = KalmanFilter1D::new(0.0, 1.0, 0.01, 0.1);
    k.r = f32::NAN;
    assert!(k.filter(1.0).is_nan());

    // the same for the fusion configuration
    let mut cfg = FusionConfig::default();
    assert_eq!(
        (
            cfg.process_noise,
            cfg.default_measurement_noise,
            cfg.outlier_threshold
        ),
        (0.01, 1.0, 3.0)
    );
    cfg.outlier_threshold = -1.0;
    let mut f = FusedSensor::new(cfg);
    // every input is now further than −1 standard deviations away, so all are
    // rejected and the estimate never moves
    assert_eq!(
        f.fuse(&[SensorInput {
            value: 1.0,
            noise: 1.0
        }]),
        0.0
    );
    assert_eq!(f.rejected_count(), 1);
}
