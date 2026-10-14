//! Q16 fixed-point utilities + linear fitting (`fit_linear_fixed` / `evaluate_linear_fixed` / conversions).

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
/// 2. **Factorization**: x0*y0 + x1*y1 + ... → x0*(Σy) + `weighted_sum`
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
/// // slope ≈ 100.0 in Q16.16 = 6_553_600
/// // intercept ≈ 100.0 in Q16.16 = 6_553_600
/// ```
#[inline(always)]
#[must_use]
pub fn fit_linear_fixed(data: &[i32]) -> (i32, i32) {
    let n = data.len();

    // Early exit for small data
    if n < 2 {
        if n == 1 {
            // SAFETY: n==1 confirmed
            let val = unsafe { *data.get_unchecked(0) };
            // Use wrapping_shl: values outside Q16.16 representable range
            // (abs > 32767) are expected to wrap, matching the Q16.16 contract.
            return (0, val.wrapping_shl(Q16_SHIFT as u32));
        }
        return (0, 0);
    }

    // Σx and the denominator are closed forms of n (see `solve_linear_normal_q16`);
    // the loop only accumulates Σy and Σxy
    // ========================================
    // 2. O(N) Loop: Only Sum(y) and Sum(xy)
    // ========================================
    let mut sum_y: i64 = 0;
    let mut sum_xy: i64 = 0;

    let ptr = data.as_ptr();
    let mut i = 0;

    // Unrolled loop (4x) with factorized sum_xy
    while i + 4 <= n {
        // SAFETY: i+3 < n はループ条件で保証。ptr は data.as_ptr() で有効。
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
        // SAFETY: i < n はループ条件で保証。ptr は data.as_ptr() で有効。
        unsafe {
            let y = *ptr.add(i) as i64;
            let x = i as i64;
            sum_y = sum_y.wrapping_add(y);
            sum_xy = sum_xy.wrapping_add(x * y);
        }
        i += 1;
    }

    solve_linear_normal_q16(n, sum_y, sum_xy)
}

/// Solve the least squares normal equations on `x = 0..n−1` from `Σy` and `Σxy`
/// and return `(slope, intercept)` in Q16.16 (shared by [`fit_linear_fixed`] and
/// the SIMD kernels in `simd_fit`, so every path rounds the same way).
///
/// - slope = ((n·Σxy − Σx·Σy) << 16) / D, with Σx = n(n−1)/2 and
///   D = n·Σx² − (Σx)² = n²(n²−1)/12
/// - intercept = ((Σy << 16) − slope·Σx) / n
///
/// The divisions truncate toward zero. Everything is computed in `i128`: in
/// `i64`, n·Σxy and its Q16 shift overflow from about n = 10⁵ (an exact line
/// with slope 3 already does), and D itself from n ≈ 1.02·10⁵, which made the
/// fit return a wrapped line (and panic on the overflow in debug builds).
/// `i128` holds every intermediate for any `n` whose `Σxy` fits the `i64`
/// accumulator (|y| < 2¹⁵, the Q16.16 contract, keeps it exact up to
/// n ≈ 2.4·10⁷). The result keeps the low 32 bits (`as i32`), as before.
///
/// `n` must be at least 2 (D = 0 otherwise).
#[inline]
#[must_use]
pub(crate) fn solve_linear_normal_q16(n: usize, sum_y: i64, sum_xy: i64) -> (i32, i32) {
    let n = n as i128;
    let sum_y = i128::from(sum_y);
    let sum_xy = i128::from(sum_xy);
    let sum_x = (n * (n - 1)) >> 1;
    let n_sq = n * n;
    let denominator = (n_sq * (n_sq - 1)) / 12;
    debug_assert!(denominator > 0, "solve_linear_normal_q16 needs n >= 2");

    let slope = ((n * sum_xy - sum_x * sum_y) << Q16_SHIFT) / denominator;
    let intercept = ((sum_y << Q16_SHIFT) - slope * sum_x) / n;

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
#[must_use]
pub const fn evaluate_linear_fixed(slope: i32, intercept: i32, x: i32) -> i32 {
    // Single instruction multiply-add if target supports it (e.g. MLA on ARM)
    let mx = (slope as i64).wrapping_mul(x as i64);
    (mx as i32).wrapping_add(intercept)
}

/// Convert Q16.16 fixed-point to integer (truncate)
#[inline(always)]
#[must_use]
pub const fn q16_to_int(q: i32) -> i32 {
    q >> Q16_SHIFT
}

/// Convert integer to Q16.16 fixed-point
#[inline(always)]
#[must_use]
pub const fn int_to_q16(i: i32) -> i32 {
    i << Q16_SHIFT
}

/// Convert Q16.16 to float (plain `f32` arithmetic, available in `no_std`)
#[inline(always)]
#[must_use]
pub fn q16_to_f32(q: i32) -> f32 {
    const INV_Q16_ONE: f32 = 1.0 / (1i32 << 16) as f32;
    q as f32 * INV_Q16_ONE
}
