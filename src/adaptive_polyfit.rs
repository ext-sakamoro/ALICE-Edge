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
/// Solves the 3x3 normal equations via Cramer's rule in `i128`.
/// All x-sums are O(1) using closed-form identities:
/// - Σx   = n(n-1)/2
/// - Σx²  = n(n-1)(2n-1)/6
/// - Σx³  = [n(n-1)/2]²
/// - Σx⁴  = n(n-1)(2n-1)(3n²-3n-1)/30
///
/// # Overflow
///
/// Every product is checked. If an intermediate value does not fit in
/// `i128` (measured: windows beyond 4096 samples, independent of `|y|`), the
/// fit degrades to the linear model `(0, slope, intercept)` instead of
/// panicking or returning garbage. The same happens for a singular system.
#[inline]
#[must_use]
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

    match solve_quadratic(data) {
        Some(coeffs) => coeffs,
        None => {
            let (s, i) = fit_linear_fixed(data);
            (0, s, i)
        }
    }
}

/// Cramer's rule for the quadratic normal equations; `None` on `i128`
/// overflow or a singular matrix.
fn solve_quadratic(data: &[i32]) -> Option<(i32, i32, i32)> {
    let n = data.len() as i128;

    // O(1) x-sum 計算 (closed forms are exact integers)
    let sx = n.checked_mul(n - 1)? / 2;
    let sx2 = n.checked_mul(n - 1)?.checked_mul(2 * n - 1)? / 6;
    let sx3 = sx.checked_mul(sx)?;
    let sx4 = n
        .checked_mul(n - 1)?
        .checked_mul(2 * n - 1)?
        .checked_mul((3 * n).checked_mul(n)? - 3 * n - 1)?
        / 30;

    // O(N) ループ: Σy, Σxy, Σx²y
    let mut sy: i128 = 0;
    let mut sxy: i128 = 0;
    let mut sx2y: i128 = 0;
    for (i, &sample) in data.iter().enumerate() {
        let y = i128::from(sample);
        let x = i as i128;
        sy = sy.checked_add(y)?;
        sxy = sxy.checked_add(x.checked_mul(y)?)?;
        sx2y = sx2y.checked_add(x.checked_mul(x)?.checked_mul(y)?)?;
    }

    // 3x3 正規方程式: Cramer の公式
    // | n    sx   sx2  | | c |   | sy   |
    // | sx   sx2  sx3  | | b | = | sxy  |
    // | sx2  sx3  sx4  | | a |   | sx2y |
    let m = [[n, sx, sx2], [sx, sx2, sx3], [sx2, sx3, sx4]];
    let det = det3(&m)?;
    if det == 0 {
        return None;
    }

    let det_a = det3(&[[n, sx, sy], [sx, sx2, sxy], [sx2, sx3, sx2y]])?;
    let det_b = det3(&[[n, sy, sx2], [sx, sxy, sx3], [sx2, sx2y, sx4]])?;
    let det_c = det3(&[[sy, sx, sx2], [sxy, sx2, sx3], [sx2y, sx3, sx4]])?;

    let q16 = |d: i128| d.checked_shl(Q16_SHIFT as u32).map(|v| (v / det) as i32);
    Some((q16(det_a)?, q16(det_b)?, q16(det_c)?))
}

/// Determinant of a 3×3 `i128` matrix; `None` on overflow.
fn det3(m: &[[i128; 3]; 3]) -> Option<i128> {
    let minor =
        |a: i128, b: i128, c: i128, d: i128| a.checked_mul(d)?.checked_sub(b.checked_mul(c)?);
    let t0 = m[0][0].checked_mul(minor(m[1][1], m[1][2], m[2][1], m[2][2])?)?;
    let t1 = m[0][1].checked_mul(minor(m[1][0], m[1][2], m[2][0], m[2][2])?)?;
    let t2 = m[0][2].checked_mul(minor(m[1][0], m[1][1], m[2][0], m[2][1])?)?;
    t0.checked_sub(t1)?.checked_add(t2)
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
///
/// # Algorithm
///
/// The 4×4 normal equations are solved with Bareiss (fraction-free Gaussian
/// elimination with exact division) in `i128`: every intermediate entry is a
/// minor of the original matrix, so growth is bounded by determinants rather
/// than doubling in magnitude at each elimination step.
///
/// # Overflow
///
/// Every product is checked. If an intermediate value still does not fit in
/// `i128`, or the system is singular, the fit degrades to the quadratic model
/// `(0, a, b, c)` instead of panicking or returning garbage.
///
/// Exact-integer capacity (measured): windows of up to 64 samples (32 at the
/// full `i32` range) get a cubic; longer windows degrade, because the 4×4
/// moment determinant grows like `n^16`. Longer cubic windows need the
/// centred-abscissa formulation (odd moments vanish, the system splits into
/// two 2×2 blocks), tracked as a follow-up.
#[inline]
#[must_use]
pub fn fit_cubic_fixed(data: &[i32]) -> (i32, i32, i32, i32) {
    let n = data.len();
    if n < 4 {
        let (a, b, c) = fit_quadratic_fixed(data);
        return (0, a, b, c);
    }

    match solve_cubic(data) {
        Some(coeffs) => coeffs,
        None => {
            let (a, b, c) = fit_quadratic_fixed(data);
            (0, a, b, c)
        }
    }
}

/// Bareiss elimination for the cubic normal equations; `None` on `i128`
/// overflow or a singular matrix.
fn solve_cubic(data: &[i32]) -> Option<(i32, i32, i32, i32)> {
    // O(N) ループ: Σy, Σxy, Σx²y, Σx³y, Σx^k (k=0..6)
    let mut sy: i128 = 0;
    let mut sxy: i128 = 0;
    let mut sx2y: i128 = 0;
    let mut sx3y: i128 = 0;
    let mut sxk = [0i128; 7]; // sxk[k] = Σx^k
    sxk[0] = data.len() as i128;

    for (i, &sample) in data.iter().enumerate() {
        let y = i128::from(sample);
        let x = i as i128;
        let x2 = x.checked_mul(x)?;
        let x3 = x2.checked_mul(x)?;
        sy = sy.checked_add(y)?;
        sxy = sxy.checked_add(x.checked_mul(y)?)?;
        sx2y = sx2y.checked_add(x2.checked_mul(y)?)?;
        sx3y = sx3y.checked_add(x3.checked_mul(y)?)?;
        // x^1..x^6 の蓄積
        let mut xk = x;
        for slot in &mut sxk[1..] {
            *slot = slot.checked_add(xk)?;
            xk = xk.checked_mul(x)?;
        }
    }

    // 4x4 正規方程式 (拡大行列 [4][5])
    // M * [d, c, b, a]^T = rhs,  M[i][j] = Σx^(i+j),  rhs[i] = Σx^i * y
    let mut m = [[0i128; 5]; 4];
    for (i, row) in m.iter_mut().enumerate() {
        row[..4].copy_from_slice(&sxk[i..i + 4]);
    }
    m[0][4] = sy;
    m[1][4] = sxy;
    m[2][4] = sx2y;
    m[3][4] = sx3y;

    // Bareiss 前進消去: m[i][j] ← (m[i][j]·pivot − m[i][k]·m[k][j]) / prev
    // (prev = 1 つ前の pivot、除算は常に割り切れる)
    let mut prev: i128 = 1;
    for k in 0..4 {
        // 部分 pivot 選択 (行 k.. の中で |m[r][k]| 最大)
        let mut max_row = k;
        let mut max_val = m[k][k].abs();
        for (r, row) in m.iter().enumerate().skip(k + 1) {
            if row[k].abs() > max_val {
                max_val = row[k].abs();
                max_row = r;
            }
        }
        if max_val == 0 {
            return None; // 退化
        }
        m.swap(k, max_row);

        let pivot_row = m[k];
        let pivot = pivot_row[k];
        for row in m.iter_mut().skip(k + 1) {
            let factor = row[k];
            for (cell, &pivot_cell) in row[k..].iter_mut().zip(&pivot_row[k..]) {
                let lhs = cell.checked_mul(pivot)?;
                let rhs = factor.checked_mul(pivot_cell)?;
                *cell = lhs.checked_sub(rhs)? / prev;
            }
        }
        prev = pivot;
    }

    // 後退代入 (Q16.16 シフト付き)
    let mut sol = [0i128; 4]; // d, c, b, a
    for i in (0..4).rev() {
        let mut sum = m[i][4].checked_shl(Q16_SHIFT as u32)?;
        for j in (i + 1)..4 {
            sum = sum.checked_sub(m[i][j].checked_mul(sol[j])?)?;
        }
        sol[i] = sum / m[i][i];
    }

    Some((sol[3] as i32, sol[2] as i32, sol[1] as i32, sol[0] as i32))
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
