#![no_std]
#![deny(warnings)]

//! ALICE-Edge: Embedded Model Generator (Ultimate Optimization)
//!
//! "Don't send data. Send the law."
//!
//! Ultra-lightweight procedural compression for embedded systems.
//! Runs on Cortex-M, ESP32, AVR, RISC-V - anywhere with 1KB of stack.
//!
//! # Optimization Level: LUDICROUS
//!
//! - **Algorithm**: Closed-form formulas for Sum(x) and Sum(x²) - O(1)
//! - **Loop**: Calculates ONLY Sum(y) and Sum(xy). Other sums eliminated.
//! - **Factorization**: Reduces multiplications via algebraic transformation.
//! - **Division**: Minimized to essential operations only.
//! - **Safety**: Unsafe pointer arithmetic for zero bounds checks.
//!
//! # Philosophy
//!
//! Raw sensor data never leaves the device. Instead, we fit a mathematical
//! model on-device and transmit only the coefficients. This provides:
//!
//! - **Compression**: 1000 samples → 8 bytes (slope + intercept)
//! - **Privacy**: Raw data is discarded immediately
//! - **Bandwidth**: 100-1000x reduction in transmission
//!
//! # Example
//!
//! ```ignore
//! use alice_edge::{fit_linear_fixed, evaluate_linear_fixed};
//!
//! // Sensor readings (e.g., temperature * 100)
//! let samples = [2500, 2510, 2520, 2530, 2540]; // 25.00°C rising
//!
//! // Fit model on-device
//! let (slope, intercept) = fit_linear_fixed(&samples);
//!
//! // Transmit only 8 bytes instead of 20 bytes!
//! transmit_coefficients(slope, intercept);
//!
//! // On receiver: reconstruct any point
//! let temp_at_3 = evaluate_linear_fixed(slope, intercept, 3);
//! ```
//!
//! # Related Projects
//!
//! - [ALICE-Zip](https://github.com/ext-sakamoro/ALICE-Zip) - Full compression engine
//! - [ALICE-DB](https://github.com/ext-sakamoro/ALICE-DB) - Model-based database
//! - [ALICE-Streaming-Protocol](https://github.com/ext-sakamoro/ALICE-Streaming-Protocol) - Video streaming

/// Q16.16 fixed-point format constant
pub const Q16_SHIFT: i32 = 16;
pub const Q16_ONE: i32 = 1 << Q16_SHIFT;

/// Fit a linear model using least squares (Ultimate Optimization)
///
/// Returns `(slope, intercept)` in Q16.16 format.
///
/// # Algorithm
///
/// Uses the normal equations for least squares:
/// - slope = (n*Σxy - Σx*Σy) / (n*Σx² - (Σx)²)
/// - intercept = (Σy - slope*Σx) / n
///
/// # Ultimate Optimizations
///
/// 1. **O(1) x-sums**: Sum(x) = n(n-1)/2, Sum(x²) = n(n-1)(2n-1)/6
/// 2. **Factorization**: x0*y0 + x1*y1 + ... → x0*(Σy) + weighted_sum
/// 3. **Loop only computes**: Sum(y) and Sum(xy) - everything else is O(1)
/// 4. **Zero bounds checks**: Unsafe pointer arithmetic
///
/// # Safety
///
/// This function uses `unsafe` to skip bounds checks.
/// The caller guarantees `data` is valid (which is always true for a slice).
///
/// # Arguments
///
/// * `data` - Slice of sensor readings (raw integers)
///
/// # Returns
///
/// * `(slope, intercept)` - Q16.16 fixed-point coefficients
///
/// # Example
///
/// ```
/// use alice_edge::fit_linear_fixed;
///
/// let data = [100, 200, 300, 400, 500];
/// let (slope, intercept) = fit_linear_fixed(&data);
///
/// // slope ≈ 100.0 in Q16.16 = 6553600
/// // intercept ≈ 100.0 in Q16.16 = 6553600
/// ```
#[inline(always)]
pub fn fit_linear_fixed(data: &[i32]) -> (i32, i32) {
    let n = data.len();

    // Early exit for small data
    if n < 2 {
        if n == 1 {
            // SAFETY: n==1 confirmed
            let val = unsafe { *data.get_unchecked(0) };
            return (0, val << Q16_SHIFT);
        }
        return (0, 0);
    }

    let n64 = n as i64;

    // ========================================
    // 1. O(1) Calculation of Sum(x)
    // ========================================
    // Sum(x) = 0 + 1 + 2 + ... + (n-1) = n(n-1)/2
    let sum_x = (n64 * (n64 - 1)) >> 1;

    // NOTE: sum_xx is NO LONGER NEEDED!
    // Denominator is calculated directly using identity: D = n²(n²-1)/12

    // ========================================
    // 2. O(N) Loop: Only Sum(y) and Sum(xy)
    // ========================================
    let mut sum_y: i64 = 0;
    let mut sum_xy: i64 = 0;

    let ptr = data.as_ptr();
    let mut i = 0;

    // Unrolled loop (4x) with factorized sum_xy
    while i + 4 <= n {
        unsafe {
            // Load 4 values
            let y0 = *ptr.add(i) as i64;
            let y1 = *ptr.add(i + 1) as i64;
            let y2 = *ptr.add(i + 2) as i64;
            let y3 = *ptr.add(i + 3) as i64;

            let x0 = i as i64;

            // Sum(y) accumulation
            let local_sum_y = y0 + y1 + y2 + y3;
            sum_y = sum_y.wrapping_add(local_sum_y);

            // Factorized Sum(xy):
            // x0*y0 + (x0+1)*y1 + (x0+2)*y2 + (x0+3)*y3
            // = x0*(y0+y1+y2+y3) + (0*y0 + 1*y1 + 2*y2 + 3*y3)
            // = x0 * local_sum_y + weighted_y
            // This reduces 4 multiplications to 1!
            let weighted_y = y1 + (y2 << 1) + y3 * 3; // y1 + 2*y2 + 3*y3
            sum_xy = sum_xy.wrapping_add(x0 * local_sum_y + weighted_y);
        }
        i += 4;
    }

    // Remainder loop
    while i < n {
        unsafe {
            let y = *ptr.add(i) as i64;
            let x = i as i64;
            sum_y = sum_y.wrapping_add(y);
            sum_xy = sum_xy.wrapping_add(x * y);
        }
        i += 1;
    }

    // ========================================
    // 3. Solve Normal Equations
    // ========================================
    // Denominator = n * Sum(x²) - Sum(x)²
    //
    // THE DIVINE OPTIMIZATION:
    // D = n²(n²-1)/12  (mathematical identity)
    // This eliminates sum_xx entirely!
    //
    // Note: n⁴ overflow at n > 55,000, safe for embedded buffers (< 1024)
    let n_sq = n64 * n64;
    let denominator = (n_sq * (n_sq - 1)) / 12;

    if denominator == 0 {
        return (0, (sum_y / n64) as i32);
    }

    // Slope (Q16.16)
    // slope = (n * Sum(xy) - Sum(x) * Sum(y)) / denominator
    let slope_num = (n64 * sum_xy).wrapping_sub(sum_x * sum_y);
    let slope = (slope_num << Q16_SHIFT) / denominator;

    // Intercept (Q16.16)
    // intercept = (Sum(y) - slope * Sum(x)) / n
    let sum_y_fixed = sum_y << Q16_SHIFT;
    let slope_term = slope.wrapping_mul(sum_x);
    let intercept = (sum_y_fixed.wrapping_sub(slope_term)) / n64;

    (slope as i32, intercept as i32)
}

/// Evaluate linear model at a given point (Unsafe & Optimized)
///
/// Computes `y = slope * x + intercept` in Q16.16 arithmetic.
///
/// # Performance
///
/// Single multiply-add operation, optimized for ARM MLA instruction.
///
/// # Arguments
///
/// * `slope` - Q16.16 slope coefficient
/// * `intercept` - Q16.16 intercept coefficient
/// * `x` - Integer x position
///
/// # Returns
///
/// * Q16.16 fixed-point result
#[inline(always)]
pub fn evaluate_linear_fixed(slope: i32, intercept: i32, x: i32) -> i32 {
    // Single instruction multiply-add if target supports it (e.g. MLA on ARM)
    let mx = (slope as i64).wrapping_mul(x as i64);
    (mx as i32).wrapping_add(intercept)
}

/// Convert Q16.16 fixed-point to integer (truncate)
#[inline(always)]
pub const fn q16_to_int(q: i32) -> i32 {
    q >> Q16_SHIFT
}

/// Convert integer to Q16.16 fixed-point
#[inline(always)]
pub const fn int_to_q16(i: i32) -> i32 {
    i << Q16_SHIFT
}

/// Convert Q16.16 to float (for debugging, requires std)
#[cfg(feature = "std")]
#[inline(always)]
pub fn q16_to_f32(q: i32) -> f32 {
    q as f32 / Q16_ONE as f32
}

/// Fit a constant model (just the mean) - Optimized
///
/// Returns the mean value in Q16.16 format.
/// Use this when data has no trend.
#[inline(always)]
pub fn fit_constant_fixed(data: &[i32]) -> i32 {
    let n = data.len();
    if n == 0 {
        return 0;
    }

    let ptr = data.as_ptr();
    let mut sum: i64 = 0;

    // Loop unrolling (4x)
    let mut i = 0;
    while i + 4 <= n {
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
/// Returns error in Q32.32 format (very large numbers possible).
/// Useful for deciding whether linear model is good enough.
#[inline(always)]
pub fn compute_residual_error(data: &[i32], slope: i32, intercept: i32) -> i64 {
    let n = data.len();
    let ptr = data.as_ptr();
    let mut error: i64 = 0;

    let mut i = 0;
    while i < n {
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

/// Model selection: Choose between constant and linear - Optimized
///
/// Returns `true` if linear model is significantly better than constant.
/// Uses a simple threshold on relative error improvement.
#[inline(always)]
pub fn should_use_linear(data: &[i32]) -> bool {
    if data.len() < 3 {
        return false;
    }

    let constant = fit_constant_fixed(data);
    let (slope, intercept) = fit_linear_fixed(data);

    let error_constant = compute_residual_error(data, 0, constant);
    let error_linear = compute_residual_error(data, slope, intercept);

    // Use linear if it reduces error by at least 50%
    error_linear < error_constant / 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_linear_perfect() {
        // y = 100x + 50
        let data: [i32; 5] = [50, 150, 250, 350, 450];
        let (slope, intercept) = fit_linear_fixed(&data);

        // slope should be ~100 in Q16.16 = 100 * 65536 = 6553600
        assert!((slope - 6553600).abs() < 100);
        // intercept should be ~50 in Q16.16 = 50 * 65536 = 3276800
        assert!((intercept - 3276800).abs() < 100);
    }

    #[test]
    fn test_fit_constant() {
        let data: [i32; 4] = [100, 100, 100, 100];
        let (slope, intercept) = fit_linear_fixed(&data);

        // slope should be ~0
        assert!(slope.abs() < 100);
        // intercept should be ~100 in Q16.16
        assert!((intercept - 6553600).abs() < 100);
    }

    #[test]
    fn test_evaluate() {
        let slope = 6553600; // 100.0 in Q16.16
        let intercept = 3276800; // 50.0 in Q16.16

        let y = evaluate_linear_fixed(slope, intercept, 3);
        // y = 100 * 3 + 50 = 350 in Q16.16 = 22937600
        assert!((y - 22937600).abs() < 1000);
    }

    #[test]
    fn test_q16_conversion() {
        assert_eq!(int_to_q16(100), 6553600);
        assert_eq!(q16_to_int(6553600), 100);
    }

    #[test]
    fn test_single_point() {
        let data: [i32; 1] = [42];
        let (slope, intercept) = fit_linear_fixed(&data);

        assert_eq!(slope, 0);
        assert_eq!(q16_to_int(intercept), 42);
    }

    #[test]
    fn test_empty() {
        let data: [i32; 0] = [];
        let (slope, intercept) = fit_linear_fixed(&data);

        assert_eq!(slope, 0);
        assert_eq!(intercept, 0);
    }

    #[test]
    fn test_loop_unrolling_boundary() {
        // Test with exactly 4 elements (perfect unroll)
        let data4: [i32; 4] = [10, 20, 30, 40];
        let (slope4, _) = fit_linear_fixed(&data4);
        assert!((slope4 - 655360).abs() < 100); // slope = 10

        // Test with 5 elements (4 + 1 remainder)
        let data5: [i32; 5] = [10, 20, 30, 40, 50];
        let (slope5, _) = fit_linear_fixed(&data5);
        assert!((slope5 - 655360).abs() < 100); // slope = 10

        // Test with 7 elements (4 + 3 remainder)
        let data7: [i32; 7] = [10, 20, 30, 40, 50, 60, 70];
        let (slope7, _) = fit_linear_fixed(&data7);
        assert!((slope7 - 655360).abs() < 100); // slope = 10
    }
}
