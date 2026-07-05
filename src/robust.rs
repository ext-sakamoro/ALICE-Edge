//! Robust fitting (MAD outlier filter + robust linear).

extern crate alloc;
use alloc::vec::Vec;

use crate::q16_linear::fit_linear_fixed;

pub fn filter_outliers_mad(data: &[i32], k: i32) -> Vec<i32> {
    let n = data.len();
    if n < 3 {
        return data.to_vec();
    }

    // 中央値の計算
    let mut sorted = data.to_vec();
    sorted.sort_unstable();
    let median = sorted[n / 2];

    // MAD = median(|x_i - median|)
    let mut abs_devs: Vec<i32> = sorted.iter().map(|&x| (x - median).abs()).collect();
    abs_devs.sort_unstable();
    let mad = abs_devs[n / 2];

    if mad == 0 {
        return data.to_vec();
    }

    let threshold = (k as i64 * mad as i64) as i32;
    data.iter()
        .map(|&x| {
            if (x - median).abs() > threshold {
                median
            } else {
                x
            }
        })
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
