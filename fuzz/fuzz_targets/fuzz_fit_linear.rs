//! Linear fit must never panic on any input, and three contracts hold.
//!
//! 1. Scalar / SIMD parity: `fit_linear_simd` is a re-implementation of
//!    `fit_linear_fixed`, so the two return the same `(slope, intercept)`.
//!
//! 2. `fit_linear_robust(data, k)` flags outliers on the residuals of a
//!    line and refits on the kept samples. The rule and its arithmetic are
//!    written out in `tests/support/robust_reference.rs` (included below).
//!    `k` is a plain integer multiplier of the MAD; `k <= 0` is refused with
//!    `RobustError::NonPositiveK { k }` for every `data`. For every `data`
//!    and every `k > 0`:
//!    - fewer than 3 samples, or an empty final outlier set → the result is
//!      `fit_linear_fixed(data)`, bit for bit
//!    - the result equals the fit the documented rule produces (repeated-
//!      median start, `|r| > k · max(MAD, Q16_ONE)` with the product
//!      saturated in `i64`, least-squares refit until the set is stable)
//!    - the result is the least-squares fit of the kept indices, bit for
//!      bit, and more than half of the samples are kept
//!    The inputs are integers: there is no NaN case.
//!
//! 3. `filter_outliers_mad(data, k)` has its own contract on the raw values:
//!    `k <= 0` is refused with the same error; otherwise same length; fewer
//!    than 3 samples pass through; otherwise sample `x` becomes the median
//!    exactly when `|x − median| > k · max(MAD, 1)` (in `i64`, product
//!    saturated), and stays `x` otherwise. It is not the pre-pass of
//!    `fit_linear_robust`, so the two are checked separately.
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
    let threshold = i64::from(k).saturating_mul(mad.max(1));
    for (i, (&before, &after)) in data.iter().zip(&filtered).enumerate() {
        let expected = if dev(before) > threshold {
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
