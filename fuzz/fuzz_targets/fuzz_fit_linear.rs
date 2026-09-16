//! Linear fit must never panic on any input and the scalar / SIMD paths must
//! agree: `fit_linear_simd` is a pure re-implementation of
//! `fit_linear_fixed`, so parity is a hard invariant.
//!
//! `filter_outliers_mad` (the pre-pass of `fit_linear_robust`) is checked
//! against its contract: same length, every sample is either kept or replaced
//! by the median, and nothing is replaced when `k * MAD` covers the largest
//! deviation.
#![no_main]

use alice_edge::{filter_outliers_mad, fit_linear_fixed, fit_linear_robust, fit_linear_simd};
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    data: Vec<i32>,
    mad_k: i32,
}

fuzz_target!(|input: Input| {
    let data = &input.data;
    let scalar = fit_linear_fixed(data);
    let simd = fit_linear_simd(data);
    assert_eq!(scalar, simd, "scalar / SIMD fit diverge on {} samples", data.len());

    let filtered = filter_outliers_mad(data, input.mad_k);
    assert_eq!(filtered.len(), data.len());
    assert_eq!(fit_linear_robust(data, input.mad_k), fit_linear_fixed(&filtered));

    if data.len() < 3 {
        assert_eq!(&filtered, data, "fewer than 3 samples must pass through");
        return;
    }

    // Reference median / MAD in i64 (the filter's own contract).
    let mut sorted = data.clone();
    sorted.sort_unstable();
    let median = sorted[data.len() / 2];
    let dev = |x: i32| (i64::from(x) - i64::from(median)).abs();
    let mut devs: Vec<i64> = sorted.iter().map(|&x| dev(x)).collect();
    devs.sort_unstable();
    let mad = devs[data.len() / 2];
    let max_dev = *devs.last().unwrap();

    for (&before, &after) in data.iter().zip(&filtered) {
        assert!(
            after == before || after == median,
            "filter produced a value that is neither the sample nor the median"
        );
    }
    if mad == 0 || i64::from(input.mad_k).saturating_mul(mad) >= max_dev {
        assert_eq!(&filtered, data, "no deviation exceeds k * MAD, nothing may be replaced");
    }
});
