//! Robust fitting (MAD outlier filter + robust linear).
//!
//! Both entry points take the outlier factor `k` as a **plain integer
//! multiplier** of the MAD (`k = 3` means "three MADs"), not a Q16.16 value,
//! and both refuse `k <= 0` with [`RobustError::NonPositiveK`]: with `k <= 0`
//! the rule `|deviation| > k · MAD` marks every sample that is not exactly
//! at the centre as an outlier, which destroys the window instead of
//! cleaning it.
//!
//! Both also refuse a window longer than [`ROBUST_MAX_N`] samples with
//! [`RobustError::WindowTooLong`]: the fit's start is quadratic in the
//! window length, and the bound fixes its worst-case time and memory.

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
    /// The window is longer than [`ROBUST_MAX_N`] samples.
    WindowTooLong {
        /// The window length.
        n: usize,
        /// [`ROBUST_MAX_N`].
        max: usize,
    },
}

impl core::fmt::Display for RobustError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::NonPositiveK { k } => write!(
                f,
                "outlier factor k must be a positive integer multiplier of the MAD, got {k}"
            ),
            Self::WindowTooLong { n, max } => write!(
                f,
                "window of {n} samples is longer than the robust bound of {max}"
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

/// Longest window either robust entry point accepts.
///
/// The bound is on [`fit_linear_robust`], whose repeated-median start makes
/// `n (n − 1)` pair-slope divisions: at `n = 1024` that is 1 047 552 `i64`
/// divisions, 1 026 median selections (1 024 of 1 023 values and 2 of
/// 1 024) and at most [`ROBUST_MAX_ROUNDS`] rounds of 2 selections and 2
/// passes over the window. Measured on an arm64 desktop core (release, 41
/// calls per input, five input shapes from flat to full-range random): 1.5 to
/// 1.9 ms per call, the slowest call 1.92 ms. The heap the fit requests is
/// exactly `18 · n` bytes, 18 432 bytes at `n = 1024`: two `i64` buffers of
/// capacity `n` and two `bool` masks of length `n`.
/// [`filter_outliers_mad`] is `O(n log n)` with two `Vec`s of `n` values;
/// it takes the same bound so that the two entry points accept the same
/// windows.
pub const ROBUST_MAX_N: usize = 1024;

/// The window has to be at most [`ROBUST_MAX_N`] samples.
const fn check_len(n: usize) -> Result<(), RobustError> {
    if n <= ROBUST_MAX_N {
        Ok(())
    } else {
        Err(RobustError::WindowTooLong {
            n,
            max: ROBUST_MAX_N,
        })
    }
}

/// Replace outliers by the median using the MAD (median absolute deviation)
/// rule: a sample `x` is an outlier when `|x − median| > k · max(MAD, 1)`.
///
/// `k` is a plain integer multiplier (`3` = three MADs, not Q16.16).
///
/// The scale has a floor of one count: the samples are integers, so one
/// count is the smallest spread a window can show, and a window whose MAD is
/// 0 (more than half of it identical) still has a finite threshold of `k`
/// counts. Without the floor the threshold of such a window is 0 and every
/// sample off the median would be replaced; with it, a single spike on a
/// flat window is replaced and one-count noise is kept.
///
/// The median is the upper median `sorted[n / 2]`. Deviations are computed
/// in `i64`, so the full `i32` input range is valid (`i32::MIN - median` and
/// `|i32::MIN|` do not overflow), and the threshold `k · max(MAD, 1)`
/// saturates instead of wrapping; a huge `k` therefore keeps every sample
/// rather than silently discarding all of them.
///
/// # Errors
///
/// [`RobustError::NonPositiveK`] when `k <= 0`, whatever the window;
/// otherwise [`RobustError::WindowTooLong`] when the window is longer than
/// [`ROBUST_MAX_N`].
pub fn filter_outliers_mad(data: &[i32], k: i32) -> Result<Vec<i32>, RobustError> {
    check_k(k)?;
    let n = data.len();
    check_len(n)?;
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

    let threshold = i64::from(k).saturating_mul(mad.max(1));
    Ok(data
        .iter()
        .map(|&x| if deviation(x) > threshold { median } else { x })
        .collect())
}

/// Rounds of "flag → refit" before [`fit_linear_robust`] stops.
pub const ROBUST_MAX_ROUNDS: usize = 8;

/// Floor of `((a + b) / 2)` without overflow.
#[inline]
const fn floor_mean(a: i64, b: i64) -> i64 {
    (a >> 1) + (b >> 1) + (a & b & 1)
}

/// Median of `v` (reordered in place); an even count takes the floor of the
/// mean of the two middle values. `v` must not be empty.
fn median_floor(v: &mut [i64]) -> i64 {
    let m = v.len();
    let (left, &mut hi, _) = v.select_nth_unstable(m / 2);
    if m % 2 == 1 {
        return hi;
    }
    // the lower middle value is the largest of the left part
    let lo = left.iter().copied().max().unwrap_or(hi);
    floor_mean(lo, hi)
}

/// Upper median `sorted(v)[n / 2]` of `v` (reordered in place).
fn upper_median(v: &mut [i64]) -> i64 {
    let m = v.len();
    *v.select_nth_unstable(m / 2).1
}

/// Siegel's repeated-median line of `data` at `x = i`, `n >= 2`.
///
/// slope = median over `i` of (median over `j != i` of
/// `((y_j − y_i) << 16) / (j − i)`), each pair slope in `i64` truncated
/// toward zero, the medians [`median_floor`]; intercept = [`median_floor`]
/// of `(y_i << 16) − slope · i`. Both are cast to `i32` with the Q16.16
/// wrap. `rows` and `cells` are scratch buffers of capacity `n`.
fn repeated_median_start(data: &[i32], rows: &mut Vec<i64>, cells: &mut Vec<i64>) -> (i32, i32) {
    let n = data.len();
    rows.clear();
    for i in 0..n {
        let yi = i64::from(data[i]);
        cells.clear();
        for (j, &yj) in data.iter().enumerate() {
            if j != i {
                // |y_j − y_i| < 2³², so the shifted difference fits in i64
                let num = (i64::from(yj) - yi) << 16;
                cells.push(num / (j as i64 - i as i64));
            }
        }
        rows.push(median_floor(cells));
    }
    let slope = median_floor(rows) as i32;
    rows.clear();
    for (i, &y) in data.iter().enumerate() {
        rows.push((i64::from(y) << 16) - i64::from(slope) * i as i64);
    }
    (slope, median_floor(rows) as i32)
}

/// Least-squares line of the samples whose `outlier` flag is false, at
/// `x = i`, in `i128`. With no flag set it is `fit_linear_fixed(data)`
/// itself (bit for bit, including that function's wrap on windows it cannot
/// represent).
///
/// The caller keeps at least `n / 2 + 1 >= 2` samples (see
/// [`fit_linear_robust`]), so the determinant `m·Σx² − (Σx)²` of distinct
/// abscissae is positive.
fn fit_kept(data: &[i32], outlier: &[bool]) -> (i32, i32) {
    if !outlier.contains(&true) {
        return fit_linear_fixed(data);
    }
    let (mut m, mut sx, mut sy, mut sxx, mut sxy) = (0i128, 0i128, 0i128, 0i128, 0i128);
    for (i, (&y, &out)) in data.iter().zip(outlier).enumerate() {
        if !out {
            let (x, y) = (i as i128, i128::from(y));
            m += 1;
            sx += x;
            sy += y;
            sxx += x * x;
            sxy += x * y;
        }
    }
    let det = m * sxx - sx * sx;
    let slope = ((m * sxy - sx * sy) << 16) / det;
    let intercept = ((sy << 16) - slope * sx) / m;
    (slope as i32, intercept as i32)
}

/// Robust linear fit: MAD outlier rejection on the **residuals** of a line,
/// least-squares refit on the samples that are kept.
///
/// Rule, for `k >= 1` and a window of `n >= 3` samples at `x = i`:
///
/// 1. **Start**: Siegel's repeated-median line. slope = median over `i` of
///    (median over `j != i` of the pair slope `((y_j − y_i) << 16) / (j − i)`,
///    `i64`, truncated toward zero); intercept = median of
///    `(y_i << 16) − slope · i`. An even count takes the floor of the mean
///    of its two middle values, the tie-break that decides which line wins
///    when two candidates hold the same number of samples. The repeated
///    median is not pulled by fewer than half of the samples, so a spike at
///    the end of a window, where it has the most leverage on least squares,
///    does not move the start.
/// 2. **Flag**: Q16.16 residuals `r_i = (y_i << 16) − line(i)` in `i64`,
///    centred on their upper median `sorted(r)[n / 2]`; sample `i` is an
///    outlier when `|r_i − median(r)| > k · max(MAD(r), Q16_ONE)` (strict
///    `>`, the product saturated in `i64`). The scale floor is one count
///    (`Q16_ONE`): the samples are integers, so one count is the smallest
///    spread a window can show. A window whose residual MAD is 0 — an exact
///    line with one corrupted sample — still has a threshold of `k` counts,
///    so the corrupted sample is flagged and one-count noise is kept.
/// 3. **Refit**: least squares on the kept samples only, in `i128` (when
///    nothing is flagged it is `fit_linear_fixed(data)`, bit for bit). This
///    is the exact fixed point of "fill each outlier with the line's
///    prediction and refit": samples on the line do not move a
///    least-squares line through them.
/// 4. Repeat 2–3 from the refit until the flagged set equals the set the
///    current line was fitted without (the start has no set, so at least one
///    refit is made), at most [`ROBUST_MAX_ROUNDS`] refits; the result is the
///    last refit. The threshold is at least the MAD, so at least `n / 2 + 1`
///    samples are always kept.
///
/// `k` is a plain integer multiplier (`3` = three MADs, not Q16.16). Fewer
/// than 3 samples give `fit_linear_fixed(data)`.
///
/// Three samples cannot identify a line and an outlier: any two of them lie
/// exactly on a line, so the third always looks like the outlier, and the
/// rule claims no filtering for `n = 3` (with a spike at the end the result
/// is the line through the spike and the far sample).
///
/// Cost: the start computes `n (n − 1)` pair slopes (`i64` divisions) and
/// `n + 2` median selections; each round two selections and two passes. The
/// window is bounded by [`ROBUST_MAX_N`], which documents the worst-case
/// time and the exact heap (`18 · n` bytes).
///
/// # Errors
///
/// [`RobustError::NonPositiveK`] when `k <= 0`, whatever the window;
/// otherwise [`RobustError::WindowTooLong`] when the window is longer than
/// [`ROBUST_MAX_N`].
///
/// History: 2026-09-17 the filter moved from the raw values to the residuals
/// of a fit (a ramp with one outlier refit 2 % off); 0.2.0 the start moved
/// from least squares to the repeated median, the refit from "fill outliers
/// with the prediction" to least squares on the kept samples, and the MAD
/// got its floor of one count in place of an early return at MAD 0.
#[cfg(feature = "std")]
pub fn fit_linear_robust(data: &[i32], k: i32) -> Result<(i32, i32), RobustError> {
    use crate::q16_linear::{evaluate_linear_fixed, int_to_q16, Q16_ONE};
    check_k(k)?;
    let n = data.len();
    check_len(n)?;
    if n < 3 {
        return Ok(fit_linear_fixed(data));
    }
    let mut res: Vec<i64> = Vec::with_capacity(n);
    let mut scratch: Vec<i64> = Vec::with_capacity(n);
    let mut fit = repeated_median_start(data, &mut res, &mut scratch);
    let mut fitted_without: Option<Vec<bool>> = None;
    let mut flagged = vec![false; n];
    for _ in 0..ROBUST_MAX_ROUNDS {
        let (slope, intercept) = fit;
        res.clear();
        res.extend(data.iter().enumerate().map(|(i, &y)| {
            i64::from(int_to_q16(y)) - i64::from(evaluate_linear_fixed(slope, intercept, i as i32))
        }));
        scratch.clear();
        scratch.extend_from_slice(&res);
        let centre = upper_median(&mut scratch);
        scratch.clear();
        scratch.extend(res.iter().map(|r| (r - centre).abs()));
        let mad = upper_median(&mut scratch);
        let threshold = i64::from(k).saturating_mul(mad.max(i64::from(Q16_ONE)));
        for (f, r) in flagged.iter_mut().zip(&res) {
            *f = (r - centre).abs() > threshold;
        }
        if fitted_without.as_deref() == Some(flagged.as_slice()) {
            break;
        }
        fit = fit_kept(data, &flagged);
        match &mut fitted_without {
            Some(prev) => prev.copy_from_slice(&flagged),
            None => fitted_without = Some(flagged.clone()),
        }
    }
    Ok(fit)
}
