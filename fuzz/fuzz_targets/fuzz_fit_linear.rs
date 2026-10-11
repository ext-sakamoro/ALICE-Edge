//! Linear fit must never panic on any input, and three contracts hold.
//!
//! 1. Scalar / SIMD parity: `fit_linear_simd` is a re-implementation of
//!    `fit_linear_fixed`, so the two return the same `(slope, intercept)`.
//!
//! 2. `fit_linear_robust(data, k)` filters on the residuals of a fit, not on
//!    the raw values, and fills an outlier with the line's own prediction.
//!    The rule and its arithmetic are written out in
//!    `tests/support/robust_reference.rs` (included below). `k` is a plain
//!    integer multiplier of the MAD; `k <= 0` is refused with
//!    `RobustError::NonPositiveK { k }` for every `data`. For every `data`
//!    and every `k > 0`:
//!    - fewer than 3 samples, a first-round residual MAD of 0, or no
//!      first-round residual with `|r| > k · MAD` (strict `>`, `k · MAD`
//!      saturated in `i64`) → the result is `fit_linear_fixed(data)`
//!    - the result equals the fit the documented rule produces
//!    - the result is `fit_linear_fixed(filled)` for the rule's last filled
//!      series, which has the input's length and whose element `i` is
//!      bit-exact either `data[i]` or the line prediction
//!      `q16_to_int(evaluate_linear_fixed(slope, intercept, i as i32))`
//!    The function returns only the fit, so the filled series is recomputed
//!    here from the rule. The inputs are integers: there is no NaN case.
//!
//! 3. `filter_outliers_mad(data, k)` has its own contract on the raw values:
//!    `k <= 0` is refused with the same error; otherwise same length; fewer than 3 samples or a MAD of 0 pass through; otherwise
//!    sample `x` becomes the median exactly when `|x − median| > k · MAD`
//!    (in `i64`, product saturated), and stays `x` otherwise. It is not the
//!    pre-pass of `fit_linear_robust` any more, so the two are checked
//!    separately.
#![no_main]

use alice_edge::{filter_outliers_mad, fit_linear_simd, RobustError};
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

include!("../../tests/support/robust_reference.rs");

#[derive(Arbitrary, Debug)]
struct Input {
    data: Vec<i32>,
    mad_k: i32,
}

fuzz_target!(|input: Input| {
    let data = &input.data;
    let k = input.mad_k;

    let scalar = fit_linear_fixed(data);
    let simd = fit_linear_simd(data);
    assert_eq!(
        scalar,
        simd,
        "scalar / SIMD fit diverge on {} samples",
        data.len()
    );

    check_robust_contract(data, k);

    if k <= 0 {
        assert_eq!(
            filter_outliers_mad(data, k),
            Err(RobustError::NonPositiveK { k }),
            "filter_outliers_mad must refuse k = {k}"
        );
        return;
    }
    let filtered = filter_outliers_mad(data, k).expect("k > 0 is accepted");
    assert_eq!(filtered.len(), data.len());
    if data.len() < 3 {
        assert_eq!(&filtered, data, "fewer than 3 samples must pass through");
        return;
    }
    let mut sorted = data.clone();
    sorted.sort_unstable();
    let median = sorted[data.len() / 2];
    let dev = |x: i32| (i64::from(x) - i64::from(median)).abs();
    let mut devs: Vec<i64> = data.iter().map(|&x| dev(x)).collect();
    devs.sort_unstable();
    let mad = devs[data.len() / 2];
    let threshold = i64::from(k).saturating_mul(mad);
    for (i, (&before, &after)) in data.iter().zip(&filtered).enumerate() {
        let expected = if mad != 0 && dev(before) > threshold {
            median
        } else {
            before
        };
        assert_eq!(
            after, expected,
            "filter_outliers_mad element {i} (k = {k}, MAD = {mad})"
        );
    }
});
