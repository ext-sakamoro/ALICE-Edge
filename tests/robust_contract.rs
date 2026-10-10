//! Contract of `fit_linear_robust`: MAD filtering on the residuals of a fit,
//! outliers filled with the line's own prediction, refit until the outlier
//! set stops changing.
//!
//! The rule and its arithmetic are written out in
//! `tests/support/robust_reference.rs`; the fuzz target
//! `fuzz/fuzz_targets/fuzz_fit_linear.rs` includes the same file, so both
//! check the same clauses:
//! - fewer than 3 samples, a first-round residual MAD of 0, or no
//!   first-round residual with `|r| > k · MAD` → `fit_linear_fixed(data)`
//! - the result equals the fit the documented rule produces, for any `k`
//!   (including `k <= 0`)
//! - the result is `fit_linear_fixed(filled)`, where the filled series has
//!   the input's length and element `i` is bit-exact either `data[i]` or the
//!   line prediction at `i`
#![cfg(feature = "std")]

use alice_edge::robust::{filter_outliers_mad, fit_linear_robust};

include!("support/robust_reference.rs");

const KS: [i32; 11] = [i32::MIN, -5, -1, 0, 1, 2, 3, 4, 10, 1 << 20, i32::MAX];

/// xorshift32: a fixed, platform-independent stream of windows.
struct Rng(u32);
impl Rng {
    fn next(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }
    fn range(&mut self, lo: i32, hi: i32) -> i32 {
        let span = (i64::from(hi) - i64::from(lo) + 1) as u64;
        (i64::from(lo) + (u64::from(self.next()) % span) as i64) as i32
    }
}

/// The input the time-boxed fuzz run found on 2026-10-10
/// (fuzz/regressions/fuzz_fit_linear/crash-284d0a5c…): `[4, -1, 0]`, k = 0.
///
/// oracle (by hand): the least-squares line through (0, 4), (1, −1), (2, 0)
/// has slope −2 and intercept 3. Its residuals 1, −2, 1 centre on the median
/// 1 to 0, −3, 0, whose MAD is 0, so the rule returns that plain fit.
#[test]
fn regression_mad_zero_residuals_return_the_plain_fit() {
    let data = [4, -1, 0];
    let expected = (int_to_q16(-2), int_to_q16(3));
    assert_eq!(expected, (-131_072, 196_608));
    assert_eq!(fit_linear_robust(&data, 0), expected);
    check_robust_contract(&data, 0);

    // The fit is not the plain fit of the raw-value MAD filter: that filter
    // (median 0, MAD 1, threshold 0) maps the window to [0, 0, 0]. Comparing
    // the two was the stale assertion this input exposed.
    assert_eq!(filter_outliers_mad(&data, 0), vec![0, 0, 0]);
    assert_eq!(fit_linear_fixed(&filter_outliers_mad(&data, 0)), (0, 0));
}

/// Small windows (ties between a residual and `k · MAD` are common at this
/// size) against every `k` in `KS`.
#[test]
fn small_windows_follow_the_rule_for_every_k() {
    let mut rng = Rng(0x9e37_79b9);
    let mut filled = 0usize;
    for _ in 0..3000 {
        let n = rng.range(0, 9) as usize;
        let data: Vec<i32> = (0..n).map(|_| rng.range(-8, 8)).collect();
        for k in KS {
            check_robust_contract(&data, k);
            if robust_reference(&data, k).filled.is_some() {
                filled += 1;
            }
        }
    }
    // the sweep must reach the filling path, not only the early returns
    assert!(filled > 1000, "only {filled} cases reached a fill");
}

/// Ramps with injected outliers, the case the robust fit exists for.
#[test]
fn ramps_with_outliers_follow_the_rule_for_every_k() {
    let mut rng = Rng(0x1234_5678);
    for _ in 0..500 {
        let n = rng.range(3, 40) as usize;
        let slope = rng.range(-50, 50);
        let offset = rng.range(-1000, 1000);
        let mut data: Vec<i32> = (0..n as i32)
            .map(|x| offset + slope * x + rng.range(-2, 2))
            .collect();
        for _ in 0..rng.range(1, 3) {
            let i = rng.range(0, n as i32 - 1) as usize;
            data[i] = rng.range(-30_000, 30_000);
        }
        for k in KS {
            check_robust_contract(&data, k);
        }
    }
}

/// Values outside the Q16.16 range (the `int_to_q16` shift wraps) and
/// full-range extremes.
#[test]
fn wide_values_follow_the_rule_for_every_k() {
    let mut rng = Rng(0xdead_beef);
    for _ in 0..500 {
        let n = rng.range(3, 12) as usize;
        let data: Vec<i32> = (0..n).map(|_| rng.next() as i32).collect();
        for k in KS {
            check_robust_contract(&data, k);
        }
    }
    for data in [
        vec![i32::MIN, i32::MAX, 0],
        vec![i32::MAX; 5],
        vec![i32::MIN, i32::MIN, i32::MAX, 0, 1],
    ] {
        for k in KS {
            check_robust_contract(&data, k);
        }
    }
}
