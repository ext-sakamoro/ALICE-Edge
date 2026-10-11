//! Contract of `fit_linear_robust`: a repeated-median start, MAD flagging on
//! the residuals with a scale floor of one count, least-squares refit on the
//! kept samples until the outlier set stops changing.
//!
//! The rule and its arithmetic are written out in
//! `tests/support/robust_reference.rs`; the fuzz target
//! `fuzz/fuzz_targets/fuzz_fit_linear.rs` includes the same file, so both
//! check the same clauses:
//! - `k <= 0` is refused with `RobustError::NonPositiveK { k }` (for both
//!   `fit_linear_robust` and `filter_outliers_mad`)
//! - fewer than 3 samples, or an empty final outlier set →
//!   `fit_linear_fixed(data)`, bit for bit
//! - the result equals the fit the documented rule produces, for every
//!   `k > 0`
//! - the result is the least-squares fit of the kept indices, bit for bit,
//!   and more than half of the samples are kept
#![cfg(feature = "std")]

use alice_edge::robust::{filter_outliers_mad, fit_linear_robust, RobustError, ROBUST_MAX_N};

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
/// (fuzz/regressions/fuzz_fit_linear/crash-284d0a5c…): `[4, -1, 0]`, found
/// with k = 0, which is now refused.
///
/// oracle (by hand): the repeated-median rows are −3.5 (slopes −5 and −2),
/// −2 (−5 and 1) and −0.5 (−2 and 1), so the slope is −2; the intercept
/// candidates are 4, 1 and 4, so the start is `4 − 2x`. Its residuals are 0,
/// −3 and 0 counts: median 0, MAD 0, scale floor one count. With k = 1 the
/// middle sample (3 counts > 1) is flagged and the least-squares line of the
/// other two is `(−2, 4)`; with k = 3 it is not above 3 counts, nothing is
/// flagged and the plain fit `(−2, 3)` comes back (whose centred residuals
/// are 0, −3, 0 counts again). Before the floor this window took the MAD-0
/// early return for every k.
#[test]
fn regression_mad_zero_residuals_use_the_scale_floor() {
    let data = [4, -1, 0];
    let plain = (int_to_q16(-2), int_to_q16(3));
    assert_eq!(fit_linear_fixed(&data), plain);
    assert_eq!(
        fit_linear_robust(&data, 1),
        Ok((int_to_q16(-2), int_to_q16(4)))
    );
    for k in [1, 3, i32::MAX] {
        if k > 1 {
            assert_eq!(fit_linear_robust(&data, k), Ok(plain), "k = {k}");
        }
        check_robust_contract(&data, k);
    }
    assert_eq!(
        fit_linear_robust(&data, 0),
        Err(RobustError::NonPositiveK { k: 0 })
    );
    assert_eq!(
        filter_outliers_mad(&data, 0),
        Err(RobustError::NonPositiveK { k: 0 })
    );
}

/// `k <= 0` is refused by both entry points, for every window shape
/// (empty, short, flat, a ramp with an outlier).
#[test]
fn non_positive_k_is_refused_by_both_functions() {
    let windows: [&[i32]; 5] = [&[], &[1, 2], &[7; 9], &[1, 2, 3, 4, 100], &[4, -1, 0]];
    for data in windows {
        for k in [0, -1, -5, i32::MIN] {
            let err = RobustError::NonPositiveK { k };
            assert_eq!(fit_linear_robust(data, k), Err(err), "{data:?}, k = {k}");
            assert_eq!(filter_outliers_mad(data, k), Err(err), "{data:?}, k = {k}");
        }
    }
    assert_eq!(
        RobustError::NonPositiveK { k: -1 }.to_string(),
        "outlier factor k must be a positive integer multiplier of the MAD, got -1"
    );
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
            if k > 0
                && robust_reference(&data, k)
                    .outliers
                    .is_some_and(|o| o.contains(&true))
            {
                filled += 1;
            }
        }
    }
    // the sweep must reach the refit of a strict subset, not only the plain fit
    assert!(filled > 1000, "only {filled} cases flagged an outlier");
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

/// `k · MAD` past `i64::MAX` must saturate, not wrap.
///
/// oracle (by hand, n = 4, x = 0..3, so Σx = 6 and the denominator
/// n²(n²−1)/12 = 20): Σy = 65 535, Σxy = 65 397, slope numerator
/// 4·65 397 − 6·65 535 = −131 622, slope = (−131 622 · 2¹⁶) / 20 truncated
/// = −431 298 969, intercept = (65 535 · 2¹⁶ + 431 298 969 · 6) / 4 truncated
/// = 1 720 673 893. Every sample but the third lies outside Q16.16, so
/// `int_to_q16` wraps and the centred residuals are
/// [−5 150 749 490, −4 713 552 281, 0, 436 345 241] with MAD 4 713 552 281.
/// `i32::MAX · MAD` = 10 122 276 442 727 048 807 > `i64::MAX`: saturated it
/// is `i64::MAX`, no residual exceeds it, and the result is the plain fit.
/// Wrapped it would be −8 324 467 630 982 502 809, every sample an outlier,
/// and the refit of the line predictions [26 255, 19 674, 13 093, 6 512]
/// would give (−431 292 416, 1 720 647 680) instead.
#[test]
fn huge_k_times_mad_saturates_and_keeps_every_sample() {
    let data = [32_778, 32_868, -32_862, 32_751];
    let k = i32::MAX;
    let expected = (-431_298_969, 1_720_673_893);

    let plain = fit_linear_fixed(&data);
    assert_eq!(plain, expected);
    let (res, mad) = centred_residuals(&data, plain);
    assert_eq!(res, vec![-5_150_749_490, -4_713_552_281, 0, 436_345_241]);
    assert_eq!(mad, 4_713_552_281);
    assert!(i128::from(k) * i128::from(mad) > i128::from(i64::MAX));
    assert!(res.iter().all(|&r| !is_outlier(r, k, mad)));

    assert_eq!(robust_reference(&data, k).fit, expected);
    assert_eq!(fit_linear_robust(&data, k), Ok(expected));
    check_robust_contract(&data, k);
}

/// The MAD rule is strict: a sample exactly at `k · MAD` from the median is
/// kept.
///
/// oracle (by hand): [0, 1, 2, 3, 5] has median 2 and absolute deviations
/// [2, 1, 0, 1, 3], whose median (MAD) is 1. With k = 3 the sample 5 sits
/// exactly at 3 · 1 and is kept, so the window is unchanged. With k = 2 the
/// sample 0 sits exactly at 2 · 1 and is kept, while 5 (deviation 3 > 2) is
/// replaced by the median.
#[test]
fn mad_filter_keeps_a_sample_exactly_at_the_threshold() {
    let data = [0, 1, 2, 3, 5];
    assert_eq!(filter_outliers_mad(&data, 3), Ok(vec![0, 1, 2, 3, 5]));
    assert_eq!(filter_outliers_mad(&data, 2), Ok(vec![0, 1, 2, 3, 2]));
}

// ---------------------------------------------------------------------------
// Closed-form oracles for the residual rule, derived by hand
// ---------------------------------------------------------------------------
//
// Each window below is an exact line (integer samples on `y = a + b·x`) with
// one corrupted sample. The repeated-median start passes through the line
// (every row but the corrupted one has a majority of exact pair slopes), so
// the only non-zero residual is the corrupted sample's, the residual MAD is
// 0, the scale floor of one count (`Q16_ONE`) applies and the sample is
// flagged; the least-squares refit of the kept samples is the exact line.

/// `[base; n]` with `base + spike` at index `at`.
fn flat_with(n: usize, base: i32, at: usize, spike: i32) -> Vec<i32> {
    let mut d = vec![base; n];
    d[at] += spike;
    d
}

/// The ramp `y = 10·i` with `delta` added at index `at`.
fn ramp_with(n: usize, at: usize, delta: i32) -> Vec<i32> {
    let mut d: Vec<i32> = (0..n as i32).map(|i| 10 * i).collect();
    d[at] += delta;
    d
}

/// oracle (by hand): flat 100 with 5000 at the centre. Every row of the
/// repeated median but the spike's has slopes 0 to all flat samples, so its
/// median is 0 and so is the median of the rows; the intercept is the
/// median of `y_i << 16`, 100. Residuals are 0 except 4900 at the spike,
/// MAD 0, scale floor 1 count, threshold 3 counts: the spike is flagged and
/// the refit of the flat samples is `(0, 100)`. Before the floor the MAD-0
/// window returned the least-squares fit with the spike in it.
#[test]
fn flat_window_with_a_centre_spike_gives_the_flat_line() {
    for n in [5usize, 9, 33] {
        let data = flat_with(n, 100, n / 2, 4900);
        assert_eq!(data[n / 2], 5000);
        assert_eq!(
            fit_linear_robust(&data, 3),
            Ok((0, int_to_q16(100))),
            "n = {n}"
        );
        check_robust_contract(&data, 3);
    }
}

/// oracle (by hand): a spike at either end of a flat window or of the ramp
/// `y = 10·i`. Every row but the spike's has its exact pair slopes in the
/// majority (`n − 2` of `n − 1`), so its median is the line's slope, and the
/// spike row is one of `n`; the start is the exact line, the spike is the
/// only non-zero residual and is flagged for `k = 3` and `k = 10` (its
/// residual, 5000 counts, is far above `k` counts).
#[test]
fn a_spike_at_either_end_is_flagged_and_the_line_is_exact() {
    for n in [4usize, 5, 8, 16] {
        for at in [0, n - 1] {
            for k in [3, 10] {
                let flat = flat_with(n, 100, at, 5000);
                assert_eq!(
                    fit_linear_robust(&flat, k),
                    Ok((0, int_to_q16(100))),
                    "flat, n = {n}, spike at {at}, k = {k}"
                );
                check_robust_contract(&flat, k);
                let ramp = ramp_with(n, at, 5000);
                assert_eq!(
                    fit_linear_robust(&ramp, k),
                    Ok((int_to_q16(10), 0)),
                    "ramp, n = {n}, spike at {at}, k = {k}"
                );
                check_robust_contract(&ramp, k);
            }
        }
    }
}

/// oracle (by hand): the ramp `y = 10·i`, n = 11, with ±1000 at the centre
/// and at either end. Ten of the eleven rows of the repeated median have
/// nine exact pair slopes 10 out of ten, so the slope is 10 and the
/// intercept the median of `(y_i << 16) − 10·i·2¹⁶` = 0; the corrupted
/// sample is the only non-zero residual and the refit is exact:
/// `(655360, 0)`. Before, the odd window with a centre spike stopped at the
/// MAD-0 early return with an intercept of 86.4.
#[test]
fn exact_ramp_with_one_corrupted_sample_is_recovered_exactly() {
    let n = 11;
    for at in [0, n / 2, n - 1] {
        for delta in [1000, -1000] {
            let data = ramp_with(n, at, delta);
            assert_eq!(
                fit_linear_robust(&data, 3),
                Ok((655_360, 0)),
                "at {at}, delta {delta}"
            );
            check_robust_contract(&data, 3);
        }
    }
}

/// oracle (by hand): the scale floor is exactly one count. Flat 100, n = 9,
/// with `100 + d` at the centre, k = 3. The start is `(0, 100)`, the centre
/// residual is `d` counts and the MAD is 0, so the threshold is 3 counts.
/// `d = 3` is not above it: nothing is flagged and the result is the plain
/// fit (whose centred residuals are again 0 and 3 counts, so the outlier
/// set stays empty). `d = 4` is flagged and the refit is `(0, 100)`.
#[test]
fn the_scale_floor_is_one_count() {
    let kept = flat_with(9, 100, 4, 3);
    assert_eq!(fit_linear_robust(&kept, 3), Ok(fit_linear_fixed(&kept)));
    // slope 0 by symmetry, intercept 903·2¹⁶ / 9 truncated
    assert_eq!(fit_linear_fixed(&kept), (0, 6_575_445));
    check_robust_contract(&kept, 3);

    let flagged = flat_with(9, 100, 4, 4);
    assert_eq!(fit_linear_robust(&flagged, 3), Ok((0, int_to_q16(100))));
    check_robust_contract(&flagged, 3);
}

/// oracle (by hand): one-count noise is not an outlier. In
/// `[100, 100, 101, 100, 100, 100, 101, 100, 100]` the repeated-median start
/// is `(0, 100)` (seven of nine rows have median slope 0), the residuals
/// are 0 and one count, the MAD is 0 and the floor makes the threshold 3
/// counts: nothing is flagged and the result is the plain fit. Without the
/// floor (threshold `k · 0`) both 101 samples would be dropped.
#[test]
fn one_count_noise_on_a_flat_window_keeps_the_plain_fit() {
    let data = [100, 100, 101, 100, 100, 100, 101, 100, 100];
    let plain = fit_linear_fixed(&data);
    assert_eq!(plain, (0, 6_568_163));
    assert_eq!(fit_linear_robust(&data, 3), Ok(plain));
    check_robust_contract(&data, 3);
}

/// oracle (by hand): the filter's scale floor is one count. `[100; 9]` has
/// median 100 and MAD 0, so with k = 3 the threshold is 3: a centre of 5000
/// or 104 is replaced by 100, a centre of 103 is kept, and the one-count
/// alternation `[10, 11, 10, 11, 10]` (median 10, MAD 0) is unchanged.
#[test]
fn filter_scale_floor_is_one_count() {
    assert_eq!(
        filter_outliers_mad(&flat_with(9, 100, 4, 4900), 3),
        Ok(vec![100; 9])
    );
    assert_eq!(
        filter_outliers_mad(&flat_with(9, 100, 4, 4), 3),
        Ok(vec![100; 9])
    );
    let kept = flat_with(9, 100, 4, 3);
    assert_eq!(filter_outliers_mad(&kept, 3), Ok(kept.clone()));
    assert_eq!(
        filter_outliers_mad(&[10, 11, 10, 11, 10], 3),
        Ok(vec![10, 11, 10, 11, 10])
    );
}

/// oracle (by hand): the broken line of the golden windows, `3x` for
/// `x < 40` and `200 − 2x` from 40 on (the sample at 40, 120, is on both
/// lines, so the first segment holds 41 samples and the second 39 besides
/// it). The rows 0..=40 of the repeated median have their 40 slopes to the
/// first segment exactly 3 and their other slopes below 3, so their median
/// (index 39 of 79) is 3; the 39 rows past 40 have medians above −2 and
/// below 3. The 80 row medians sort as 39 values below 3 and 41 threes, so
/// both middle values are 3 and the slope is 3; the intercept candidates are
/// 41 zeros and 39 negatives, so it is 0. Every sample past 40 is then at
/// least 5 counts off: flagged for k = 1 and 3, and the refit of the first
/// segment is exactly `(3, 0)`. With k = 10 the threshold is 10 counts, so
/// the junction samples 41 and 42 (5 and 10 counts off the first line) are
/// kept, each refit tilts towards the second segment and admits more of it,
/// and after four refits every sample is kept: the plain fit (measured:
/// 43 → 45 → 52 → 80 kept samples). With `k = i32::MAX` nothing is flagged
/// either.
#[test]
fn broken_line_takes_the_segment_the_repeated_median_selects() {
    let data: Vec<i32> = (0..80)
        .map(|x: i32| if x < 40 { 3 * x } else { 120 - 2 * (x - 40) })
        .collect();
    for k in [1, 3] {
        assert_eq!(
            fit_linear_robust(&data, k),
            Ok((int_to_q16(3), 0)),
            "k = {k}"
        );
        check_robust_contract(&data, k);
    }
    for k in [10, i32::MAX] {
        assert_eq!(
            fit_linear_robust(&data, k),
            Ok(fit_linear_fixed(&data)),
            "k = {k}"
        );
        check_robust_contract(&data, k);
    }
}

/// Three samples cannot identify a line and an outlier: any two of them
/// define a line on which they lie exactly, so the third always looks like
/// the outlier. The rule claims no filtering here. What it does is pinned:
/// with the spike at the end, the repeated-median start is the line
/// `100 + 2450·x` through the two outer samples (row medians 1225, 2450,
/// 3675; intercept candidates 100, −2350, 100), the middle sample — a good
/// one — is the only non-zero residual and is flagged, and the result is
/// the line through the spike. With the spike in the middle the start is
/// flat, the spike is flagged and the result is `(0, 100)`.
#[test]
fn three_samples_are_not_identifiable_and_the_result_is_pinned() {
    let end = [100, 100, 5000];
    assert_eq!(
        fit_linear_robust(&end, 3),
        Ok((int_to_q16(2450), int_to_q16(100)))
    );
    check_robust_contract(&end, 3);
    let centre = [100, 5000, 100];
    assert_eq!(fit_linear_robust(&centre, 3), Ok((0, int_to_q16(100))));
    check_robust_contract(&centre, 3);
}

/// The window length is bounded: `ROBUST_MAX_N` samples are accepted by
/// both entry points, one more is refused with `WindowTooLong` (after the
/// `k` check, so a bad `k` is reported first), whatever the samples.
#[test]
fn window_length_is_bounded_by_robust_max_n() {
    assert_eq!(ROBUST_MAX_N, 1024);
    let at_max: Vec<i32> = (0..ROBUST_MAX_N as i32).map(|i| 3 * i - 7).collect();
    assert_eq!(
        fit_linear_robust(&at_max, 3),
        Ok((int_to_q16(3), int_to_q16(-7)))
    );
    assert_eq!(filter_outliers_mad(&at_max, 3), Ok(at_max.clone()));

    let mut too_long = at_max.clone();
    too_long.push(0);
    let err = RobustError::WindowTooLong {
        n: ROBUST_MAX_N + 1,
        max: ROBUST_MAX_N,
    };
    for k in [1, 3, i32::MAX] {
        assert_eq!(fit_linear_robust(&too_long, k), Err(err), "k = {k}");
        assert_eq!(filter_outliers_mad(&too_long, k), Err(err), "k = {k}");
    }
    assert_eq!(
        fit_linear_robust(&too_long, 0),
        Err(RobustError::NonPositiveK { k: 0 })
    );
    assert_eq!(
        err.to_string(),
        "window of 1025 samples is longer than the robust bound of 1024"
    );
}
