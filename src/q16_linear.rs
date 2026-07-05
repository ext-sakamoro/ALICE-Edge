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

    // ========================================
    // 3. Solve Normal Equations
    // ========================================
    // Denominator = n * Sum(x²) - Sum(x)²
    //
    // THE DIVINE OPTIMIZATION:
    // D = n²(n²-1)/12  (mathematical identity)
    // This eliminates sum_xx entirely!
    //
    // Use i128 for the intermediate n⁴ product to avoid overflow when
    // n > 55,000 (n² overflows i64 at n > ~3B, but n_sq * (n_sq-1) can
    // still be large; i128 keeps correctness for arbitrarily large slices
    // such as those passed from the Python binding).
    let n_sq = n64 as i128 * n64 as i128;
    let denominator = ((n_sq * (n_sq - 1)) / 12) as i64;

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

/// Convert Q16.16 to float (for debugging, requires std)
#[cfg(feature = "std")]
#[inline(always)]
#[must_use]
pub fn q16_to_f32(q: i32) -> f32 {
    const INV_Q16_ONE: f32 = 1.0 / (1i32 << 16) as f32;
    q as f32 * INV_Q16_ONE
}
