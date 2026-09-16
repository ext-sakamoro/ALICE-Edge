//! Robust fitting (MAD outlier filter + robust linear).

extern crate alloc;
use alloc::vec::Vec;

use crate::q16_linear::fit_linear_fixed;

/// Replace outliers by the median using the MAD (median absolute deviation)
/// rule: a sample `x` is an outlier when `|x - median| > k * MAD`.
///
/// Deviations are computed in `i64`, so the full `i32` input range is valid
/// (`i32::MIN - median` and `|i32::MIN|` do not overflow), and the threshold
/// `k * MAD` saturates instead of wrapping; a huge `k` therefore keeps every
/// sample rather than silently discarding all of them.
pub fn filter_outliers_mad(data: &[i32], k: i32) -> Vec<i32> {
    let n = data.len();
    if n < 3 {
        return data.to_vec();
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
        return data.to_vec();
    }

    let threshold = i64::from(k).saturating_mul(mad);
    data.iter()
        .map(|&x| if deviation(x) > threshold { median } else { x })
        .collect()
}

/// Fit linear model with outlier removal pre-processing.
///
/// Applies MAD-based outlier filter, then fits a linear model.
/// Convenience wrapper combining `filter_outliers_mad` + `fit_linear_fixed`.
#[cfg(feature = "std")]
#[inline]
#[must_use]
pub fn fit_linear_robust(data: &[i32], mad_k: i32) -> (i32, i32) {
    let filtered = filter_outliers_mad(data, mad_k);
    fit_linear_fixed(&filtered)
}
