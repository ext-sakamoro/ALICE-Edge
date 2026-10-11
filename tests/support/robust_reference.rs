// Reference model of `alice_edge::fit_linear_robust`, written from the rule
// in its documentation and nothing else. Shared through `include!` by
// tests/robust_contract.rs and fuzz/fuzz_targets/fuzz_fit_linear.rs, so the
// unit tests and the fuzz target check the same contract.
//
// The rule, with the arithmetic the crate uses:
//
// - `k` is a plain integer multiplier of the MAD (`3` = three MADs, not
//   Q16.16); `k <= 0` is refused with `RobustError::NonPositiveK { k }`
//   before the window is looked at, so the rule below is for `k > 0` only
// - a window longer than `alice_edge::ROBUST_MAX_N` (1024) is then refused
//   with `RobustError::WindowTooLong { n, max: ROBUST_MAX_N }`
// - windows of fewer than 3 samples: the plain fit `fit_linear_fixed(data)`
// - start: the repeated-median line. Pair slope
//   `((y_j - y_i) << 16) / (j - i)` in `i64` (truncated toward zero); row
//   median over `j != i`, slope = median of the row medians, intercept =
//   median of `(y_i << 16) - slope * i`; every median here of an even count
//   is the floor of the mean of the two middle values; both cast `as i32`
// - then repeat at most `ROBUST_MAX_ROUNDS` times:
//   - residual `r_i = i64(int_to_q16(data[i])) - i64(evaluate_linear_fixed(
//     slope, intercept, i as i32))` (Q16.16, in `i64`)
//   - centre: `r_i -= median`, where `median` is `sorted(r)[n / 2]`
//   - `MAD = sorted(|r_i|)[n / 2]`, scale `max(MAD, 1 << 16)` (one count)
//   - sample `i` is an outlier when `|r_i| > i64(k).saturating_mul(scale)`
//     (strict `>`)
//   - if the outlier set equals the set the current line was fitted
//     without, stop; otherwise the line becomes the least-squares fit of the
//     kept samples (`i128`; with none flagged, `fit_linear_fixed(data)`)
//
// All values are integers, so there is no NaN to exclude.

use alice_edge::{evaluate_linear_fixed, fit_linear_fixed, int_to_q16};

/// Refits before the loop gives up.
pub const ROBUST_MAX_ROUNDS: usize = 8;

/// Median of a sorted copy; an even count takes the floor of the mean of
/// the two middle values (computed in `i128`, so it cannot overflow).
pub fn median_floor_mean(v: &[i64]) -> i64 {
    let mut s = v.to_vec();
    s.sort_unstable();
    let m = s.len();
    if m % 2 == 1 {
        s[m / 2]
    } else {
        (i128::from(s[m / 2 - 1]) + i128::from(s[m / 2])).div_euclid(2) as i64
    }
}

/// The repeated-median start, written with sorts.
pub fn repeated_median(data: &[i32]) -> (i32, i32) {
    let n = data.len();
    let rows: Vec<i64> = (0..n)
        .map(|i| {
            let cells: Vec<i64> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    ((i64::from(data[j]) - i64::from(data[i])) << 16) / (j as i64 - i as i64)
                })
                .collect();
            median_floor_mean(&cells)
        })
        .collect();
    let slope = median_floor_mean(&rows) as i32;
    let cands: Vec<i64> = (0..n)
        .map(|i| (i64::from(data[i]) << 16) - i64::from(slope) * i as i64)
        .collect();
    (slope, median_floor_mean(&cands) as i32)
}

/// Least squares of the kept samples (`keep[i]`), from the normal equations
/// in `i128`; `None` with fewer than 2 kept.
pub fn least_squares_kept(data: &[i32], keep: &[bool]) -> Option<(i32, i32)> {
    let pts: Vec<(i128, i128)> = (0..data.len())
        .filter(|&i| keep[i])
        .map(|i| (i as i128, i128::from(data[i])))
        .collect();
    let m = pts.len() as i128;
    if m < 2 {
        return None;
    }
    let sx: i128 = pts.iter().map(|p| p.0).sum();
    let sy: i128 = pts.iter().map(|p| p.1).sum();
    let sxx: i128 = pts.iter().map(|p| p.0 * p.0).sum();
    let sxy: i128 = pts.iter().map(|p| p.0 * p.1).sum();
    let slope = ((m * sxy - sx * sy) << 16) / (m * sxx - sx * sx);
    let intercept = ((sy << 16) - slope * sx) / m;
    Some((slope as i32, intercept as i32))
}

/// Median-centred Q16.16 residuals of `data` against `fit`, and their MAD.
pub fn centred_residuals(data: &[i32], fit: (i32, i32)) -> (Vec<i64>, i64) {
    let n = data.len();
    let mut res: Vec<i64> = (0..n)
        .map(|i| {
            i64::from(int_to_q16(data[i]))
                - i64::from(evaluate_linear_fixed(fit.0, fit.1, i as i32))
        })
        .collect();
    let mut sorted = res.clone();
    sorted.sort_unstable();
    let median = sorted[n / 2];
    for r in &mut res {
        *r -= median;
    }
    let mut abs: Vec<i64> = res.iter().map(|r| r.abs()).collect();
    abs.sort_unstable();
    (res, abs[n / 2])
}

/// `|r| > k · max(MAD, one count)`, with the product saturated in `i64`.
pub fn is_outlier(r: i64, k: i32, mad: i64) -> bool {
    r.abs() > i64::from(k).saturating_mul(mad.max(1 << 16))
}

/// How the reference loop ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobustExit {
    /// fewer than 3 samples
    TooShort,
    /// the outlier set did not change
    Converged,
    /// `ROBUST_MAX_ROUNDS` refits without convergence
    RoundLimit,
}

/// The fit the rule produces and the outlier set it was fitted without.
#[derive(Debug, Clone)]
pub struct RobustTrace {
    pub fit: (i32, i32),
    pub start: (i32, i32),
    /// `None` for `TooShort`
    pub outliers: Option<Vec<bool>>,
    pub exit: RobustExit,
}

/// The rule for `k > 0`; a non-positive `k` has no rule, it is refused.
pub fn robust_reference(data: &[i32], k: i32) -> RobustTrace {
    assert!(k > 0, "the rule is defined for k > 0 only, got {k}");
    let n = data.len();
    if n < 3 {
        let fit = fit_linear_fixed(data);
        return RobustTrace {
            fit,
            start: fit,
            outliers: None,
            exit: RobustExit::TooShort,
        };
    }
    let start = repeated_median(data);
    let mut fit = start;
    let mut fitted_without: Option<Vec<bool>> = None;
    for _ in 0..ROBUST_MAX_ROUNDS {
        let (res, mad) = centred_residuals(data, fit);
        let mask: Vec<bool> = res.iter().map(|&r| is_outlier(r, k, mad)).collect();
        if fitted_without.as_ref() == Some(&mask) {
            return RobustTrace {
                fit,
                start,
                outliers: fitted_without,
                exit: RobustExit::Converged,
            };
        }
        let keep: Vec<bool> = mask.iter().map(|&o| !o).collect();
        fit = if mask.iter().any(|&o| o) {
            least_squares_kept(data, &keep).expect("the MAD bound keeps n / 2 + 1 samples")
        } else {
            fit_linear_fixed(data)
        };
        fitted_without = Some(mask);
    }
    RobustTrace {
        fit,
        start,
        outliers: fitted_without,
        exit: RobustExit::RoundLimit,
    }
}

/// Checks `fit_linear_robust(data, k)` against the contract and panics with
/// the failing clause. Used by the unit tests and the fuzz target.
pub fn check_robust_contract(data: &[i32], k: i32) {
    // 0. k <= 0 is refused, whatever the window
    if k <= 0 {
        assert_eq!(
            alice_edge::fit_linear_robust(data, k),
            Err(alice_edge::RobustError::NonPositiveK { k }),
            "k = {k} must be refused"
        );
        return;
    }
    // 0b. a window past the bound is refused, after the k check
    if data.len() > alice_edge::ROBUST_MAX_N {
        assert_eq!(
            alice_edge::fit_linear_robust(data, k),
            Err(alice_edge::RobustError::WindowTooLong {
                n: data.len(),
                max: alice_edge::ROBUST_MAX_N
            }),
            "n = {} must be refused",
            data.len()
        );
        return;
    }
    let got = alice_edge::fit_linear_robust(data, k).expect("k > 0 is accepted");
    let plain = fit_linear_fixed(data);
    let trace = robust_reference(data, k);

    // 1. fewer than 3 samples, or no outlier in the final set → the plain
    //    fit, bit for bit
    match &trace.outliers {
        None => assert_eq!(got, plain, "fewer than 3 samples must give the plain fit"),
        Some(out) if !out.contains(&true) => {
            assert_eq!(got, plain, "no outlier flagged (k = {k}): the plain fit")
        }
        Some(_) => {}
    }

    // 2. the fit is the one the documented rule produces
    assert_eq!(
        got, trace.fit,
        "fit differs from the documented rule (k = {k}, exit {:?}, start {:?})",
        trace.exit, trace.start
    );

    // 3. the fit is the least-squares fit of the kept indices, bit for bit,
    //    and at least n / 2 + 1 samples are kept
    if let Some(out) = &trace.outliers {
        let keep: Vec<bool> = out.iter().map(|&o| !o).collect();
        let kept = keep.iter().filter(|&&x| x).count();
        assert!(kept > data.len() / 2, "only {kept} of {} kept", data.len());
        if kept < data.len() {
            assert_eq!(
                Some(got),
                least_squares_kept(data, &keep),
                "fit is not the least-squares fit of the kept samples"
            );
        }
    }
}
