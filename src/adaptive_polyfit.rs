//! Adaptive polynomial fitting (`should_use_linear` / `fit_quadratic_fixed` / `fit_cubic_fixed` + evaluators).

use crate::constant_fit::{compute_residual_error, fit_constant_fixed};
use crate::q16_linear::{fit_linear_fixed, Q16_SHIFT};

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

// ── E1: 多項式フィット (2次/3次) ──────────────────────────────────────

/// Fit a quadratic model y = ax² + bx + c using least squares (Q16.16).
///
/// Returns `(a, b, c)` in Q16.16 fixed-point format.
/// Useful for curved sensor data (e.g., temperature diurnal cycles).
///
/// # Algorithm
///
/// Solves the 3x3 normal equations via Cramer's rule.
/// All x-sums are O(1) using closed-form identities:
/// - Σx   = n(n-1)/2
/// - Σx²  = n(n-1)(2n-1)/6
/// - Σx³  = [n(n-1)/2]²
/// - Σx⁴  = n(n-1)(2n-1)(3n²-3n-1)/30
#[inline]
#[must_use]
#[allow(clippy::suspicious_operation_groupings)]
pub fn fit_quadratic_fixed(data: &[i32]) -> (i32, i32, i32) {
    let n = data.len();
    if n < 3 {
        if n == 2 {
            // 2点 → 線形退化
            let (s, i) = fit_linear_fixed(data);
            return (0, s, i);
        }
        if n == 1 {
            // SAFETY: n==1 confirmed above
            let val = unsafe { *data.get_unchecked(0) };
            return (0, 0, val.wrapping_shl(Q16_SHIFT as u32));
        }
        return (0, 0, 0);
    }

    let n64 = n as i64;
    let n128 = n as i128;

    // O(1) x-sum 計算
    let sx = (n64 * (n64 - 1)) / 2;
    let sx2 = (n64 * (n64 - 1) * (2 * n64 - 1)) / 6;
    let sx3 = ((n128 * (n128 - 1)) / 2) * ((n128 * (n128 - 1)) / 2);
    let sx3_i64 = sx3 as i64;
    let sx4 = (n128 * (n128 - 1) * (2 * n128 - 1) * (3 * n128 * n128 - 3 * n128 - 1)) / 30;
    let sx4_i64 = sx4 as i64;

    // O(N) ループ: Σy, Σxy, Σx²y
    let mut sy: i64 = 0;
    let mut sxy: i64 = 0;
    let mut sx2y: i64 = 0;

    let ptr = data.as_ptr();
    for i in 0..n {
        // SAFETY: i < n はループ条件で保証。ptr は data.as_ptr() で有効。
        let y = unsafe { *ptr.add(i) } as i64;
        let x = i as i64;
        sy += y;
        sxy += x * y;
        sx2y += x * x * y;
    }

    // 3x3 正規方程式: Cramer の公式
    // | n    sx   sx2  | | c |   | sy   |
    // | sx   sx2  sx3  | | b | = | sxy  |
    // | sx2  sx3  sx4  | | a |   | sx2y |
    let det = {
        n64 as i128 * (sx2 as i128 * sx4_i64 as i128 - sx3_i64 as i128 * sx3_i64 as i128)
            - sx as i128 * (sx as i128 * sx4_i64 as i128 - sx3_i64 as i128 * sx2 as i128)
            + sx2 as i128 * (sx as i128 * sx3_i64 as i128 - sx2 as i128 * sx2 as i128)
    };

    if det == 0 {
        let (s, i) = fit_linear_fixed(data);
        return (0, s, i);
    }

    // a (x²係数)
    let det_a = {
        n64 as i128 * (sx2 as i128 * sx2y as i128 - sx3_i64 as i128 * sxy as i128)
            - sx as i128 * (sx as i128 * sx2y as i128 - sx3_i64 as i128 * sy as i128)
            + sx2 as i128 * (sx as i128 * sxy as i128 - sx2 as i128 * sy as i128)
    };
    let a = ((det_a << Q16_SHIFT as i128) / det) as i32;

    // b (x係数)
    let det_b = {
        n64 as i128 * (sxy as i128 * sx4_i64 as i128 - sx2y as i128 * sx3_i64 as i128)
            - sy as i128 * (sx as i128 * sx4_i64 as i128 - sx2 as i128 * sx3_i64 as i128)
            + sx2 as i128 * (sx as i128 * sx2y as i128 - sx2 as i128 * sxy as i128)
    };
    let b = ((det_b << Q16_SHIFT as i128) / det) as i32;

    // c (定数)
    let det_c = {
        sy as i128 * (sx2 as i128 * sx4_i64 as i128 - sx3_i64 as i128 * sx3_i64 as i128)
            - sxy as i128 * (sx as i128 * sx4_i64 as i128 - sx3_i64 as i128 * sx2 as i128)
            + sx2y as i128 * (sx as i128 * sx3_i64 as i128 - sx2 as i128 * sx2 as i128)
    };
    let c = ((det_c << Q16_SHIFT as i128) / det) as i32;

    (a, b, c)
}

/// Evaluate quadratic model at a given point: y = a*x² + b*x + c
#[inline(always)]
#[must_use]
pub const fn evaluate_quadratic_fixed(a: i32, b: i32, c: i32, x: i32) -> i32 {
    let x64 = x as i64;
    // a,b,c はすべて Q16.16。x は整数。Q16.16 * int = Q16.16 (シフト不要)
    let ax2 = (a as i64) * x64 * x64;
    let bx = (b as i64) * x64;
    (ax2 + bx) as i32 + c
}

/// Fit a cubic model y = ax³ + bx² + cx + d using least squares (Q16.16).
///
/// Returns `(a, b, c, d)`. For very high-order trends.
/// Falls back to quadratic if n < 4.
#[inline]
#[must_use]
#[allow(clippy::needless_range_loop)]
pub fn fit_cubic_fixed(data: &[i32]) -> (i32, i32, i32, i32) {
    let n = data.len();
    if n < 4 {
        let (a, b, c) = fit_quadratic_fixed(data);
        return (0, a, b, c);
    }

    let n64 = n as i64;

    // O(N) ループ: Σy, Σxy, Σx²y, Σx³y, Σx^k (k=1..6)
    let mut sy: i128 = 0;
    let mut sxy: i128 = 0;
    let mut sx2y: i128 = 0;
    let mut sx3y: i128 = 0;
    let mut sxk = [0i128; 7]; // sxk[k] = Σx^k
    sxk[0] = n64 as i128;

    let ptr = data.as_ptr();
    for i in 0..n {
        // SAFETY: i < n はループ条件で保証。ptr は data.as_ptr() で有効。
        let y = unsafe { *ptr.add(i) } as i128;
        let x = i as i128;
        let x2 = x * x;
        let x3 = x2 * x;
        sy += y;
        sxy += x * y;
        sx2y += x2 * y;
        sx3y += x3 * y;
        // x^1..x^6 の蓄積
        let mut xk = x;
        for slot in &mut sxk[1..] {
            *slot += xk;
            xk *= x;
        }
    }

    // 4x4 正規方程式を前進消去で解く（ガウス消去法、i128精度）
    // M * [d, c, b, a]^T = rhs
    // M[i][j] = Σx^(i+j),  rhs[i] = Σx^i * y
    let mut m = [[0i128; 5]; 4]; // 拡大行列 [4][5]
    for (i, row) in m.iter_mut().enumerate() {
        row[..4].copy_from_slice(&sxk[i..i + 4]);
    }
    m[0][4] = sy;
    m[1][4] = sxy;
    m[2][4] = sx2y;
    m[3][4] = sx3y;

    // 前進消去
    for col in 0..4 {
        // ピボット選択
        let mut max_row = col;
        let mut max_val = m[col][col].abs();
        for (row_idx, row_data) in m.iter().enumerate().skip(col + 1) {
            if row_data[col].abs() > max_val {
                max_val = row_data[col].abs();
                max_row = row_idx;
            }
        }
        if max_val == 0 {
            // 退化 → 2次に降格
            let (a2, b2, c2) = fit_quadratic_fixed(data);
            return (0, a2, b2, c2);
        }
        m.swap(col, max_row);

        let pivot = m[col][col];
        for row in (col + 1)..4 {
            let factor = m[row][col];
            if factor == 0 {
                continue;
            }
            for j in col..5 {
                m[row][j] = m[row][j] * pivot - m[col][j] * factor;
            }
        }
    }

    // 後退代入 (Q16.16 シフト付き)
    let mut sol = [0i128; 4]; // d, c, b, a
    for i in (0..4).rev() {
        let mut sum = m[i][4] << Q16_SHIFT;
        for j in (i + 1)..4 {
            sum -= m[i][j] * sol[j];
        }
        if m[i][i] != 0 {
            sol[i] = sum / m[i][i];
        }
    }

    (sol[3] as i32, sol[2] as i32, sol[1] as i32, sol[0] as i32)
}

/// Evaluate cubic model: y = a*x³ + b*x² + c*x + d
#[inline(always)]
#[must_use]
pub const fn evaluate_cubic_fixed(a: i32, b: i32, c: i32, d: i32, x: i32) -> i32 {
    let x64 = x as i64;
    let x2 = x64 * x64;
    let x3 = x2 * x64;
    // a,b,c,d はすべて Q16.16。x は整数。Q16.16 * int^k = Q16.16 (シフト不要)
    let ax3 = (a as i64) * x3;
    let bx2 = (b as i64) * x2;
    let cx = (c as i64) * x64;
    (ax3 + bx2 + cx) as i32 + d
}
