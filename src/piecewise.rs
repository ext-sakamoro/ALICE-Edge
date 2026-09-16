//! Piecewise linear fitting (`PiecewiseSegment` / `fit_piecewise_linear`).

// These imports are only consumed by `fit_piecewise_linear`, which itself is
// gated on `feature = "std"` (it needs `Vec`). Match the gate so the default
// no-std build does not warn on them, while the CI std build still resolves
// the symbols.
#[cfg(feature = "std")]
use crate::constant_fit::compute_residual_error;
#[cfg(feature = "std")]
use crate::q16_linear::{evaluate_linear_fixed, fit_linear_fixed, int_to_q16};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PiecewiseSegment {
    /// セグメント開始インデックス
    pub start: usize,
    /// セグメント終了インデックス（排他）
    pub end: usize,
    /// Q16.16 傾き
    pub slope: i32,
    /// Q16.16 切片
    pub intercept: i32,
}

/// Fit piecewise linear model with automatic breakpoint detection.
///
/// Splits data into segments where each segment has residual error below
/// `max_error_q32` (Q32.32 threshold). Greedy top-down splitting.
///
/// Returns segments in order. Each segment independently fits a linear model.
///
/// # Arguments
///
/// * `data` - Raw sensor readings
/// * `max_error_q32` - Maximum residual error per segment (Q32.32)
/// * `min_segment_len` - Minimum segment length (default: 4)
#[cfg(feature = "std")]
#[must_use]
pub fn fit_piecewise_linear(
    data: &[i32],
    max_error_q32: i64,
    min_segment_len: usize,
) -> Vec<PiecewiseSegment> {
    fn split_recursive(
        data: &[i32],
        offset: usize,
        max_err: i64,
        min_len: usize,
        out: &mut Vec<PiecewiseSegment>,
    ) {
        let (slope, intercept) = fit_linear_fixed(data);
        let error = compute_residual_error(data, slope, intercept);

        if error <= max_err || data.len() <= min_len {
            out.push(PiecewiseSegment {
                start: offset,
                end: offset + data.len(),
                slope,
                intercept,
            });
            return;
        }

        // 最大残差地点で分割
        let mut max_residual = 0i64;
        let mut split_at = data.len() / 2;
        for (i, &val) in data.iter().enumerate() {
            let predicted = evaluate_linear_fixed(slope, intercept, i as i32);
            let actual = int_to_q16(val);
            let diff = (predicted as i64 - actual as i64).abs();
            if diff > max_residual {
                max_residual = diff;
                split_at = i;
            }
        }

        // 分割点が端に寄りすぎないよう調整
        if split_at < min_len {
            split_at = min_len;
        }
        if split_at > data.len() - min_len {
            split_at = data.len() - min_len;
        }

        split_recursive(&data[..split_at], offset, max_err, min_len, out);
        split_recursive(&data[split_at..], offset + split_at, max_err, min_len, out);
    }

    let mut segments = Vec::new();
    let min_len = if min_segment_len < 2 {
        2
    } else {
        min_segment_len
    };

    if data.len() < min_len {
        if !data.is_empty() {
            let (slope, intercept) = fit_linear_fixed(data);
            segments.push(PiecewiseSegment {
                start: 0,
                end: data.len(),
                slope,
                intercept,
            });
        }
        return segments;
    }

    split_recursive(data, 0, max_error_q32, min_len, &mut segments);
    segments
}
