//! Constant fitting + residual error (`fit_constant_fixed` / `compute_residual_error`).

use crate::q16_linear::{evaluate_linear_fixed, int_to_q16, Q16_SHIFT};

pub const fn fit_constant_fixed(data: &[i32]) -> i32 {
    let n = data.len();
    if n == 0 {
        return 0;
    }

    let ptr = data.as_ptr();
    let mut sum: i64 = 0;

    // Loop unrolling (4x)
    let mut i = 0;
    while i + 4 <= n {
        // SAFETY: i+3 < n はループ条件で保証。ptr は data.as_ptr() で有効。
        unsafe {
            let v0 = *ptr.add(i) as i64;
            let v1 = *ptr.add(i + 1) as i64;
            let v2 = *ptr.add(i + 2) as i64;
            let v3 = *ptr.add(i + 3) as i64;
            sum = sum.wrapping_add(v0 + v1 + v2 + v3);
        }
        i += 4;
    }

    // Remainder
    while i < n {
        // SAFETY: i < n はループ条件で保証。
        unsafe {
            sum = sum.wrapping_add(*ptr.add(i) as i64);
        }
        i += 1;
    }

    let mean = (sum << Q16_SHIFT) / n as i64;
    mean as i32
}

/// Compute residual error (sum of squared differences) - Optimized
///
/// Each prediction-vs-actual difference is right-shifted by 8 bits (`>> 8`)
/// before squaring to prevent i64 overflow when accumulating over large
/// datasets. The returned value is thus scaled by 2^{-16} relative to the
/// true Q16.16 squared error. Use this only for relative comparisons
/// (e.g., `should_use_linear`), not for absolute error reporting.
#[inline(always)]
#[must_use]
pub const fn compute_residual_error(data: &[i32], slope: i32, intercept: i32) -> i64 {
    let n = data.len();
    let ptr = data.as_ptr();
    let mut error: i64 = 0;

    let mut i = 0;
    while i < n {
        // SAFETY: i < n はループ条件で保証。ptr は data.as_ptr() で有効。
        unsafe {
            let y = *ptr.add(i);
            let predicted = evaluate_linear_fixed(slope, intercept, i as i32);
            let actual = int_to_q16(y);
            let diff = (predicted as i64 - actual as i64) >> 8; // Scale down to prevent overflow
            error = error.wrapping_add(diff.wrapping_mul(diff));
        }
        i += 1;
    }

    error
}
