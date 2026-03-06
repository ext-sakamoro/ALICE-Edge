// SPDX-License-Identifier: MIT
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::module_name_repetitions,
    clippy::inline_always,
    clippy::too_many_lines
)]
#![cfg_attr(not(feature = "std"), no_std)]
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
//! ```
//! use alice_edge::{fit_linear_fixed, evaluate_linear_fixed, int_to_q16};
//!
//! // Sensor readings (e.g., temperature * 100)
//! let samples = [2500, 2510, 2520, 2530, 2540]; // 25.00°C rising
//!
//! // Fit model on-device
//! let (slope, intercept) = fit_linear_fixed(&samples);
//!
//! // Transmit only 8 bytes instead of 20 bytes!
//! // transmit_coefficients(slope, intercept);
//!
//! // On receiver: reconstruct any point
//! let temp_at_3 = evaluate_linear_fixed(slope, intercept, int_to_q16(3));
//! assert!(temp_at_3 != 0); // 非ゼロの予測値
//! ```
//!
//! # Related Projects
//!
//! - [ALICE-Zip](https://github.com/ext-sakamoro/ALICE-Zip) - Full compression engine
//! - [ALICE-DB](https://github.com/ext-sakamoro/ALICE-DB) - Model-based database
//! - [ALICE-Streaming-Protocol](https://github.com/ext-sakamoro/ALICE-Streaming-Protocol) - Video streaming

#[cfg(feature = "asp")]
pub mod asp_bridge;
#[cfg(feature = "codec")]
pub mod codec_bridge;
#[cfg(feature = "dashboard")]
pub mod dashboard;
#[cfg(feature = "db")]
pub mod db_bridge;
#[cfg(feature = "depth-camera")]
pub mod depth_capture;
#[cfg(feature = "edge-pipeline")]
pub mod edge_pipeline;
#[cfg(feature = "ffi")]
pub mod ffi;
#[cfg(feature = "ml")]
pub mod ml_bridge;
#[cfg(feature = "mqtt")]
pub mod mqtt_bridge;
#[cfg(feature = "ml")]
pub mod object_classifier;
#[cfg(feature = "std")]
pub mod ota;
#[cfg(feature = "pyo3")]
pub mod python;
pub mod ring_buffer;
#[cfg(feature = "sdf")]
pub mod sdf_compress;
pub mod sensor_fusion;
#[cfg(feature = "sensors")]
pub mod sensors;
#[cfg(feature = "std")]
pub mod telemetry;
#[cfg(feature = "std")]
pub mod watchdog;
#[cfg(feature = "zip")]
pub mod zip_bridge;

// ── D5: 共通推論トレイト ────────────────────────────────────────────────

/// Edge 推論エンジンの共通トレイト
///
/// `EdgeClassifier`（`ml_bridge`）と `TernaryClassifier`（`object_classifier`）を
/// 統一的に扱うための抽象インタフェース。
#[cfg(feature = "ml")]
pub trait EdgeInference {
    /// 推論結果の型
    type Output;
    /// f32 特徴量ベクトルからの推論（クラスIDと信頼度を返す）
    fn infer(&mut self, features: &[f32]) -> (usize, f32);
    /// モデルの重みメモリ使用量（バイト）
    fn model_size_bytes(&self) -> usize;
}

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

/// Fit a constant model (just the mean) - Optimized
///
/// Returns the mean value in Q16.16 format.
/// Use this when data has no trend.
#[inline(always)]
#[must_use]
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

/// Model selection: Choose between constant and linear - Optimized
///
/// Returns `true` if linear model is significantly better than constant.
/// Uses a simple threshold on relative error improvement.
#[inline(always)]
#[must_use]
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

// ── E2: 区間分割フィット (piecewise linear) ───────────────────────────

/// 区間分割線形回帰の1セグメント結果
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

// ── E3: 外れ値除去 (MAD ベース) ───────────────────────────────────────

/// Filter outliers using Median Absolute Deviation (MAD).
///
/// Returns filtered data with outliers replaced by the median.
/// Threshold: points > `k` * MAD from median are considered outliers.
///
/// # Arguments
///
/// * `data` - Raw sensor readings
/// * `k` - Sensitivity multiplier (recommended: 3 for mild, 2 for aggressive)
///
/// # Returns
///
/// * Filtered data (same length, outliers replaced by median)
#[cfg(feature = "std")]
#[must_use]
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

// ── E5: SIMD 高速化 (NEON/SSE2) ───────────────────────────────────────

/// SIMD-accelerated linear fit using platform intrinsics.
///
/// Falls back to `fit_linear_fixed` on unsupported platforms.
/// On `x86_64` with SSE2 or aarch64 with NEON, uses 4-wide SIMD accumulation.
#[inline]
#[must_use]
pub fn fit_linear_simd(data: &[i32]) -> (i32, i32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            // SAFETY: SSE2 サポートを上記で確認済み。data は有効なスライス。
            return unsafe { fit_linear_sse2(data) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: aarch64 は NEON を常にサポート。data は有効なスライス。
        unsafe { fit_linear_neon(data) }
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        fit_linear_fixed(data)
    }
}

/// # Safety
/// 呼び出し元で SSE2 サポートを確認済みであること。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn fit_linear_sse2(data: &[i32]) -> (i32, i32) {
    use core::arch::x86_64::*;

    let n = data.len();
    if n < 8 {
        return fit_linear_fixed(data);
    }

    let n64 = n as i64;

    // Σx = n(n-1)/2, denominator = n²(n²-1)/12
    let sum_x = (n64 * (n64 - 1)) >> 1;
    let n_sq = n64 as i128 * n64 as i128;
    let denominator = ((n_sq * (n_sq - 1)) / 12) as i64;

    if denominator == 0 {
        return fit_linear_fixed(data);
    }

    // SIMD: 4-wide i32 → i64 accumulation
    let mut sum_y_lo = _mm_setzero_si128();
    let mut sum_y_hi = _mm_setzero_si128();
    let mut sum_xy_lo = _mm_setzero_si128();
    let mut sum_xy_hi = _mm_setzero_si128();

    let ptr = data.as_ptr();
    let mut i = 0usize;

    // x インデックスベクトル: [0,1,2,3] から開始し +4 ずつ増加
    let mut x_vec = _mm_set_epi32(3, 2, 1, 0);
    let four = _mm_set1_epi32(4);

    while i + 4 <= n {
        let y_vec = _mm_loadu_si128(ptr.add(i) as *const __m128i);

        // i32 → i64 拡張 (下位2要素、上位2要素)
        let y_lo = _mm_cvtepi32_epi64(y_vec);
        let y_hi = _mm_cvtepi32_epi64(_mm_shuffle_epi32(y_vec, 0b_01_00_11_10));

        let x_lo = _mm_cvtepi32_epi64(x_vec);
        let x_hi = _mm_cvtepi32_epi64(_mm_shuffle_epi32(x_vec, 0b_01_00_11_10));

        // Σy 蓄積
        sum_y_lo = _mm_add_epi64(sum_y_lo, y_lo);
        sum_y_hi = _mm_add_epi64(sum_y_hi, y_hi);

        // Σxy 蓄積 (SSE2 には _mm_mullo_epi64 がないため、手動で乗算)
        // x_lo * y_lo の各要素を個別に処理
        let xy_lo = sse2_mul_epi64(x_lo, y_lo);
        let xy_hi = sse2_mul_epi64(x_hi, y_hi);
        sum_xy_lo = _mm_add_epi64(sum_xy_lo, xy_lo);
        sum_xy_hi = _mm_add_epi64(sum_xy_hi, xy_hi);

        x_vec = _mm_add_epi32(x_vec, four);
        i += 4;
    }

    // 水平加算
    let sum_y_vec = _mm_add_epi64(sum_y_lo, sum_y_hi);
    let sum_xy_vec = _mm_add_epi64(sum_xy_lo, sum_xy_hi);

    let mut sum_y_arr = [0i64; 2];
    let mut sum_xy_arr = [0i64; 2];
    _mm_storeu_si128(sum_y_arr.as_mut_ptr() as *mut __m128i, sum_y_vec);
    _mm_storeu_si128(sum_xy_arr.as_mut_ptr() as *mut __m128i, sum_xy_vec);

    let mut sum_y = sum_y_arr[0] + sum_y_arr[1];
    let mut sum_xy = sum_xy_arr[0] + sum_xy_arr[1];

    // 端数処理
    while i < n {
        let y = *ptr.add(i) as i64;
        let x = i as i64;
        sum_y += y;
        sum_xy += x * y;
        i += 1;
    }

    let slope_num = (n64 * sum_xy).wrapping_sub(sum_x * sum_y);
    let slope = (slope_num << Q16_SHIFT) / denominator;
    let sum_y_fixed = sum_y << Q16_SHIFT;
    let slope_term = slope.wrapping_mul(sum_x);
    let intercept = (sum_y_fixed.wrapping_sub(slope_term)) / n64;

    (slope as i32, intercept as i32)
}

/// SSE2 の符号付き i64 乗算ヘルパー（両オペランドが i32 範囲の場合）
///
/// `_mm_mul_epu32` で下位 32bit を unsigned 乗算し、負数の場合は
/// 符号補正を行う。store/load ラウンドトリップを回避。
///
/// 数学的導出:
/// p = a_lo * b_lo (unsigned), a_lo = a mod 2^32, b_lo = b mod 2^32
/// 真の積 = p - (a<0 ? b<<32 : 0) - (b<0 ? a<<32 : 0) (mod 2^64)
/// # Safety
/// SSE2 有効なコンテキストから呼び出すこと。a, b は sign-extended i64。
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
#[inline(always)]
unsafe fn sse2_mul_epi64(
    a: core::arch::x86_64::__m128i,
    b: core::arch::x86_64::__m128i,
) -> core::arch::x86_64::__m128i {
    use core::arch::x86_64::*;
    // unsigned 乗算: 各 64bit レーンの下位 32bit → 64bit 積
    let p = _mm_mul_epu32(a, b);
    // 符号マスク (sign-extended i64 なので srai_epi32 で全 64bit が 0 or -1)
    let a_neg = _mm_srai_epi32(a, 31);
    let b_neg = _mm_srai_epi32(b, 31);
    // 符号補正: a<0 なら b<<32 を引く、b<0 なら a<<32 を引く
    let corr_a = _mm_and_si128(a_neg, _mm_slli_epi64(b, 32));
    let corr_b = _mm_and_si128(b_neg, _mm_slli_epi64(a, 32));
    _mm_sub_epi64(_mm_sub_epi64(p, corr_a), corr_b)
}

/// # Safety
/// aarch64 ターゲットから呼び出すこと（NEON は常時利用可能）。
#[cfg(target_arch = "aarch64")]
unsafe fn fit_linear_neon(data: &[i32]) -> (i32, i32) {
    use core::arch::aarch64::{
        vaddq_s64, vdupq_n_s64, vget_high_s32, vget_low_s32, vgetq_lane_s64, vld1q_s32, vmovl_s32,
        vmull_s32,
    };

    let n = data.len();
    if n < 8 {
        return fit_linear_fixed(data);
    }

    let n64 = n as i64;
    let sum_x = (n64 * (n64 - 1)) >> 1;
    let n_sq = n64 as i128 * n64 as i128;
    let denominator = ((n_sq * (n_sq - 1)) / 12) as i64;

    if denominator == 0 {
        return fit_linear_fixed(data);
    }

    let mut sum_y_vec = vdupq_n_s64(0);
    let mut sum_xy_vec = vdupq_n_s64(0);

    let ptr = data.as_ptr();
    let mut i = 0usize;
    let mut sum_y: i64 = 0;
    let mut sum_xy: i64 = 0;

    // NEON: 4要素ずつ処理 (i32→i64拡張)
    while i + 4 <= n {
        let y4 = vld1q_s32(ptr.add(i));
        let x_vals: [i32; 4] = [i as i32, (i + 1) as i32, (i + 2) as i32, (i + 3) as i32];
        let x4 = vld1q_s32(x_vals.as_ptr());

        // sum_y: i32→i64 拡張して加算
        let y_lo = vmovl_s32(vget_low_s32(y4));
        let y_hi = vmovl_s32(vget_high_s32(y4));
        sum_y_vec = vaddq_s64(sum_y_vec, y_lo);
        sum_y_vec = vaddq_s64(sum_y_vec, y_hi);

        // sum_xy: vmull_s32 で 32×32→64 積を計算して累積
        let xy_lo = vmull_s32(vget_low_s32(y4), vget_low_s32(x4));
        let xy_hi = vmull_s32(vget_high_s32(y4), vget_high_s32(x4));
        sum_xy_vec = vaddq_s64(sum_xy_vec, xy_lo);
        sum_xy_vec = vaddq_s64(sum_xy_vec, xy_hi);

        i += 4;
    }

    // 水平加算
    sum_y += vgetq_lane_s64(sum_y_vec, 0) + vgetq_lane_s64(sum_y_vec, 1);
    sum_xy += vgetq_lane_s64(sum_xy_vec, 0) + vgetq_lane_s64(sum_xy_vec, 1);

    // 端数
    while i < n {
        let y = *ptr.add(i) as i64;
        let x = i as i64;
        sum_y += y;
        sum_xy += x * y;
        i += 1;
    }

    let slope_num = (n64 * sum_xy).wrapping_sub(sum_x * sum_y);
    let slope = (slope_num << Q16_SHIFT) / denominator;
    let sum_y_fixed = sum_y << Q16_SHIFT;
    let slope_term = slope.wrapping_mul(sum_x);
    let intercept = (sum_y_fixed.wrapping_sub(slope_term)) / n64;

    (slope as i32, intercept as i32)
}

// ── E9: 差分エンコーディング ──────────────────────────────────────────

/// Delta-encode a batch of coefficient pairs for improved compressibility.
///
/// Takes a sequence of (slope, intercept) pairs and encodes each as the
/// difference from the previous pair. The first pair is stored as-is.
///
/// # Returns
///
/// Delta-encoded pairs. First element = original, rest = deltas.
#[cfg(feature = "std")]
#[must_use]
pub fn delta_encode_coefficients(coeffs: &[(i32, i32)]) -> Vec<(i32, i32)> {
    if coeffs.is_empty() {
        return Vec::new();
    }

    let mut encoded = Vec::with_capacity(coeffs.len());
    encoded.push(coeffs[0]);

    for i in 1..coeffs.len() {
        let ds = coeffs[i].0.wrapping_sub(coeffs[i - 1].0);
        let di = coeffs[i].1.wrapping_sub(coeffs[i - 1].1);
        encoded.push((ds, di));
    }

    encoded
}

/// Decode delta-encoded coefficient pairs back to absolute values.
#[cfg(feature = "std")]
#[must_use]
pub fn delta_decode_coefficients(encoded: &[(i32, i32)]) -> Vec<(i32, i32)> {
    if encoded.is_empty() {
        return Vec::new();
    }

    let mut decoded = Vec::with_capacity(encoded.len());
    decoded.push(encoded[0]);

    for i in 1..encoded.len() {
        let prev = decoded[i - 1];
        let s = prev.0.wrapping_add(encoded[i].0);
        let intercept = prev.1.wrapping_add(encoded[i].1);
        decoded.push((s, intercept));
    }

    decoded
}

/// Compute the byte savings from delta encoding.
///
/// Returns `(original_bytes, delta_bytes)` — estimated sizes if each
/// coefficient were stored as a variable-length integer (zigzag + varint).
#[cfg(feature = "std")]
#[must_use]
pub fn delta_encoding_savings(coeffs: &[(i32, i32)]) -> (usize, usize) {
    const fn varint_size(val: i32) -> usize {
        // Zigzag encoding size
        let zigzag = ((val >> 31) ^ (val << 1)) as u32;
        match zigzag {
            0..=127 => 1,
            128..=16383 => 2,
            16_384..=2_097_151 => 3,
            2_097_152..=268_435_455 => 4,
            _ => 5,
        }
    }

    let original: usize = coeffs
        .iter()
        .map(|&(s, i)| varint_size(s) + varint_size(i))
        .sum();

    let deltas = delta_encode_coefficients(coeffs);
    let delta: usize = deltas
        .iter()
        .map(|&(s, i)| varint_size(s) + varint_size(i))
        .sum();

    (original, delta)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fit_linear_perfect() {
        // y = 100x + 50
        let data: [i32; 5] = [50, 150, 250, 350, 450];
        let (slope, intercept) = fit_linear_fixed(&data);

        // slope should be ~100 in Q16.16 = 100 * 65536 = 6_553_600
        assert!((slope - 6_553_600).abs() < 100);
        // intercept should be ~50 in Q16.16 = 50 * 65536 = 3_276_800
        assert!((intercept - 3_276_800).abs() < 100);
    }

    #[test]
    fn test_fit_constant() {
        let data: [i32; 4] = [100, 100, 100, 100];
        let (slope, intercept) = fit_linear_fixed(&data);

        // slope should be ~0
        assert!(slope.abs() < 100);
        // intercept should be ~100 in Q16.16
        assert!((intercept - 6_553_600).abs() < 100);
    }

    #[test]
    fn test_evaluate() {
        let slope = 6_553_600; // 100.0 in Q16.16
        let intercept = 3_276_800; // 50.0 in Q16.16

        let y = evaluate_linear_fixed(slope, intercept, 3);
        // y = 100 * 3 + 50 = 350 in Q16.16 = 22_937_600
        assert!((y - 22_937_600).abs() < 1000);
    }

    #[test]
    fn test_q16_conversion() {
        assert_eq!(int_to_q16(100), 6_553_600);
        assert_eq!(q16_to_int(6_553_600), 100);
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
        assert!((slope4 - 655_360).abs() < 100); // slope = 10

        // Test with 5 elements (4 + 1 remainder)
        let data5: [i32; 5] = [10, 20, 30, 40, 50];
        let (slope5, _) = fit_linear_fixed(&data5);
        assert!((slope5 - 655_360).abs() < 100); // slope = 10

        // Test with 7 elements (4 + 3 remainder)
        let data7: [i32; 7] = [10, 20, 30, 40, 50, 60, 70];
        let (slope7, _) = fit_linear_fixed(&data7);
        assert!((slope7 - 655_360).abs() < 100); // slope = 10
    }

    // ── New tests ─────────────────────────────────────────────

    #[test]
    fn test_two_element_data() {
        // Minimal valid input for fit_linear_fixed
        let data: [i32; 2] = [0, 100];
        let (slope, intercept) = fit_linear_fixed(&data);
        // slope = 100, intercept = 0
        assert!((slope - int_to_q16(100)).abs() < 200);
        assert!(intercept.abs() < 200);
    }

    #[test]
    fn test_negative_slope() {
        // y = -50x + 1000  → data: 1000, 950, 900, 850, 800
        let data: [i32; 5] = [1000, 950, 900, 850, 800];
        let (slope, _intercept) = fit_linear_fixed(&data);
        // slope should be negative: -50 in Q16.16 = -3_276_800
        assert!(slope < 0);
        assert!((slope + 3_276_800).abs() < 500);
    }

    #[test]
    fn test_negative_intercept() {
        // y = 10x - 100  → data: -100, -90, -80, -70, -60
        let data: [i32; 5] = [-100, -90, -80, -70, -60];
        let (slope, intercept) = fit_linear_fixed(&data);
        // slope ≈ 10, intercept ≈ -100
        assert!((slope - int_to_q16(10)).abs() < 500);
        assert!(intercept < 0);
        assert!((intercept + int_to_q16(100)).abs() < 500);
    }

    #[test]
    fn test_three_element_data() {
        // y = 5x + 0  → data: 0, 5, 10
        let data: [i32; 3] = [0, 5, 10];
        let (slope, intercept) = fit_linear_fixed(&data);
        assert!((slope - int_to_q16(5)).abs() < 200);
        assert!(intercept.abs() < 200);
    }

    #[test]
    fn test_evaluate_at_zero() {
        // At x=0, result = intercept
        let slope = int_to_q16(50);
        let intercept = int_to_q16(200);
        let y = evaluate_linear_fixed(slope, intercept, 0);
        assert_eq!(y, intercept);
    }

    #[test]
    fn test_evaluate_negative_x() {
        // slope=10, intercept=100, x=-2 → y = -20 + 100 = 80 (in Q16)
        let slope = int_to_q16(10);
        let intercept = int_to_q16(100);
        let y = evaluate_linear_fixed(slope, intercept, -2);
        let expected = int_to_q16(80);
        assert!((y - expected).abs() < 10);
    }

    #[test]
    fn test_q16_to_int_negative() {
        // -100 in Q16.16
        let q = int_to_q16(-100);
        assert_eq!(q16_to_int(q), -100);
    }

    #[test]
    fn test_q16_to_int_zero() {
        assert_eq!(q16_to_int(0), 0);
        assert_eq!(int_to_q16(0), 0);
    }

    #[test]
    fn test_fit_constant_fixed_empty() {
        let result = fit_constant_fixed(&[]);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_fit_constant_fixed_single() {
        let data = [500];
        let mean = fit_constant_fixed(&data);
        assert_eq!(q16_to_int(mean), 500);
    }

    #[test]
    fn test_fit_constant_fixed_multiple() {
        // Mean of [100, 200, 300] = 200
        let data = [100, 200, 300];
        let mean = fit_constant_fixed(&data);
        assert!((q16_to_int(mean) - 200).abs() <= 1);
    }

    #[test]
    fn test_fit_constant_fixed_unrolled() {
        // 8 elements: exercises the 4x unrolled path + remainder
        let data = [10, 20, 30, 40, 50, 60, 70, 80];
        let mean = fit_constant_fixed(&data);
        // mean = 45
        assert!((q16_to_int(mean) - 45).abs() <= 1);
    }

    #[test]
    fn test_compute_residual_error_perfect_fit() {
        // Perfect linear data → residual should be zero (or very small)
        let data = [0, 10, 20, 30, 40];
        let (slope, intercept) = fit_linear_fixed(&data);
        let err = compute_residual_error(&data, slope, intercept);
        assert!(err < 1000, "residual error for perfect fit: {err}");
    }

    #[test]
    fn test_compute_residual_error_constant_vs_linear() {
        // For linear data, constant model has higher residual than linear model
        let data = [0, 100, 200, 300, 400];
        let constant_mean = fit_constant_fixed(&data);
        let (slope, intercept) = fit_linear_fixed(&data);

        let err_const = compute_residual_error(&data, 0, constant_mean);
        let err_linear = compute_residual_error(&data, slope, intercept);
        assert!(err_linear < err_const);
    }

    #[test]
    fn test_should_use_linear_for_trending_data() {
        // Strongly linear data should prefer linear model
        let data = [0, 100, 200, 300, 400, 500, 600];
        assert!(should_use_linear(&data));
    }

    #[test]
    fn test_should_use_linear_for_constant_data() {
        // Constant data should not prefer linear model
        let data = [100, 100, 100, 100, 100, 100];
        assert!(!should_use_linear(&data));
    }

    #[test]
    fn test_should_use_linear_too_short() {
        // Less than 3 elements always returns false
        assert!(!should_use_linear(&[]));
        assert!(!should_use_linear(&[1]));
        assert!(!should_use_linear(&[1, 2]));
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_q16_to_f32_positive() {
        // 1.0 in Q16.16 = 65536 → q16_to_f32 should give 1.0
        let q = Q16_ONE;
        let f = q16_to_f32(q);
        assert!((f - 1.0).abs() < 1e-5);
    }

    #[allow(clippy::float_cmp)]
    #[cfg(feature = "std")]
    #[test]
    fn test_q16_to_f32_zero() {
        assert_eq!(q16_to_f32(0), 0.0);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_q16_to_f32_half() {
        // 0.5 in Q16.16 = 32768
        let q = Q16_ONE >> 1;
        let f = q16_to_f32(q);
        assert!((f - 0.5).abs() < 1e-5);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_q16_to_f32_negative() {
        // -1.0 in Q16.16 = -65536
        let q = -Q16_ONE;
        let f = q16_to_f32(q);
        assert!((f + 1.0).abs() < 1e-5);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_large_dataset_consistency() {
        // 100-element linear dataset: slope=1, intercept=0
        let data: Vec<i32> = (0..100).collect();
        let (slope, intercept) = fit_linear_fixed(&data);
        // slope ≈ 1.0 in Q16.16 = 65536
        assert!((slope - 65536).abs() < 500);
        assert!(intercept.abs() < 500);
    }

    #[test]
    fn test_fit_linear_all_zeros() {
        let data = [0i32; 8];
        let (slope, intercept) = fit_linear_fixed(&data);
        assert_eq!(slope, 0);
        assert_eq!(intercept, 0);
    }

    #[test]
    fn test_evaluate_linear_fixed_large_x() {
        // Verify no panic/crash with a larger x value
        let slope = int_to_q16(1);
        let intercept = int_to_q16(0);
        let y = evaluate_linear_fixed(slope, intercept, 1000);
        // y ≈ 1000 in Q16.16
        assert!((q16_to_int(y) - 1000).abs() <= 1);
    }

    // ── E1: 多項式フィット テスト ─────────────────────────────────────

    #[test]
    fn test_fit_quadratic_perfect() {
        // y = x² → data: 0, 1, 4, 9, 16, 25
        let data = [0, 1, 4, 9, 16, 25];
        let (a, b, c) = fit_quadratic_fixed(&data);
        // a ≈ 1.0 (Q16.16 = 65536)
        assert!((a - int_to_q16(1)).abs() < 500, "a = {a} (expected ~65536)");
        // b ≈ 0
        assert!(b.abs() < 500, "b = {b} (expected ~0)");
        // c ≈ 0
        assert!(c.abs() < 500, "c = {c} (expected ~0)");
    }

    #[test]
    fn test_fit_quadratic_with_linear() {
        // y = 2x² + 3x + 10  → data: 10, 15, 24, 37, 54
        let data = [10, 15, 24, 37, 54];
        let (a, b, c) = fit_quadratic_fixed(&data);
        assert!((a - int_to_q16(2)).abs() < 1000, "a = {a}");
        assert!((b - int_to_q16(3)).abs() < 1000, "b = {b}");
        assert!((c - int_to_q16(10)).abs() < 1000, "c = {c}");
    }

    #[test]
    fn test_evaluate_quadratic() {
        let a = int_to_q16(1); // x²
        let b = int_to_q16(2); // 2x
        let c = int_to_q16(3); // +3
                               // At x=3: 9 + 6 + 3 = 18
        let y = evaluate_quadratic_fixed(a, b, c, 3);
        assert!((q16_to_int(y) - 18).abs() <= 1);
    }

    #[test]
    fn test_fit_quadratic_fallback_2pts() {
        let data = [10, 20];
        let (a, b, _c) = fit_quadratic_fixed(&data);
        assert_eq!(a, 0); // 2次係数は0に退化
        assert!((b - int_to_q16(10)).abs() < 500);
    }

    #[test]
    fn test_fit_quadratic_empty() {
        let data: [i32; 0] = [];
        let (a, b, c) = fit_quadratic_fixed(&data);
        assert_eq!(a, 0);
        assert_eq!(b, 0);
        assert_eq!(c, 0);
    }

    #[test]
    fn test_fit_cubic_perfect() {
        // y = x³ → data: 0, 1, 8, 27, 64, 125
        let data = [0, 1, 8, 27, 64, 125];
        let (a, b, _c, _d) = fit_cubic_fixed(&data);
        // a ≈ 1.0 (x³ 係数)
        assert!(
            (a - int_to_q16(1)).abs() < 2000,
            "a = {a} (expected ~65536)"
        );
        // b ≈ 0 (x² 係数)
        assert!(b.abs() < 2000, "b = {b} (expected ~0)");
    }

    #[test]
    fn test_fit_cubic_fallback_3pts() {
        // 3点 → 2次退化
        let data = [0, 1, 4];
        let (a, _b, _c, _d) = fit_cubic_fixed(&data);
        assert_eq!(a, 0);
    }

    #[test]
    fn test_evaluate_cubic() {
        let a = int_to_q16(1);
        let b = 0;
        let c = 0;
        let d = int_to_q16(5);
        // At x=2: 8 + 0 + 0 + 5 = 13
        let y = evaluate_cubic_fixed(a, b, c, d, 2);
        assert!((q16_to_int(y) - 13).abs() <= 1);
    }

    // ── E2: 区間分割フィット テスト ───────────────────────────────────

    #[cfg(feature = "std")]
    #[test]
    fn test_piecewise_linear_single_segment() {
        // 完全な線形データ → 1セグメント
        let data: Vec<i32> = (0..20).map(|x| x * 10).collect();
        let segments = fit_piecewise_linear(&data, i64::MAX, 4);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start, 0);
        assert_eq!(segments[0].end, 20);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_piecewise_linear_split() {
        // ステップ関数: 0..10 → 値0, 10..20 → 値1000
        let mut data = vec![0i32; 10];
        data.extend(vec![1000i32; 10]);
        let segments = fit_piecewise_linear(&data, 100, 4);
        assert!(segments.len() >= 2, "segments: {}", segments.len());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_piecewise_linear_empty() {
        let segments = fit_piecewise_linear(&[], 100, 4);
        assert!(segments.is_empty());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_piecewise_linear_short() {
        let data = [1, 2, 3];
        let segments = fit_piecewise_linear(&data, 0, 4);
        assert_eq!(segments.len(), 1);
    }

    // ── E3: 外れ値除去テスト ──────────────────────────────────────────

    #[cfg(feature = "std")]
    #[test]
    fn test_filter_outliers_basic() {
        // 通常データ + 1個の外れ値
        let data = [100, 101, 99, 100, 102, 100, 9999, 100, 101, 99];
        let filtered = filter_outliers_mad(&data, 3);
        assert_eq!(filtered.len(), 10);
        // 外れ値 9999 が中央値に置換されている
        assert!(filtered[6] < 200, "outlier was replaced: {}", filtered[6]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_filter_outliers_no_outliers() {
        let data = [10, 11, 10, 11, 10];
        let filtered = filter_outliers_mad(&data, 3);
        assert_eq!(filtered, data);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_filter_outliers_short() {
        let data = [1, 2];
        let filtered = filter_outliers_mad(&data, 3);
        assert_eq!(filtered, data);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_fit_linear_robust() {
        // y = 10x + 外れ値
        let mut data: Vec<i32> = (0..20).map(|x| x * 10).collect();
        data[10] = 99999; // 外れ値
        let (slope, _intercept) = fit_linear_robust(&data, 3);
        // 外れ値除去後、slope ≈ 10
        assert!((slope - int_to_q16(10)).abs() < 20000, "slope = {slope}");
    }

    // ── E5: SIMD テスト ───────────────────────────────────────────────

    #[test]
    fn test_fit_linear_simd_matches_scalar() {
        let data = [50, 150, 250, 350, 450, 550, 650, 750, 850, 950];
        let (slope_scalar, intercept_scalar) = fit_linear_fixed(&data);
        let (slope_simd, intercept_simd) = fit_linear_simd(&data);
        assert!(
            (slope_simd - slope_scalar).abs() < 100,
            "slope: simd={slope_simd} scalar={slope_scalar}"
        );
        assert!(
            (intercept_simd - intercept_scalar).abs() < 100,
            "intercept: simd={intercept_simd} scalar={intercept_scalar}"
        );
    }

    #[test]
    fn test_fit_linear_simd_large() {
        // 大きなデータセット
        let data: [i32; 16] = [
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ];
        let (slope_scalar, intercept_scalar) = fit_linear_fixed(&data);
        let (slope_simd, intercept_simd) = fit_linear_simd(&data);
        assert!(
            (slope_simd - slope_scalar).abs() < 100,
            "slope: simd={slope_simd} scalar={slope_scalar}"
        );
        assert!(
            (intercept_simd - intercept_scalar).abs() < 100,
            "intercept: simd={intercept_simd} scalar={intercept_scalar}"
        );
    }

    #[test]
    fn test_fit_linear_simd_small_fallback() {
        // 少量データは scalar にフォールバック
        let data = [100, 200, 300];
        let (slope, intercept) = fit_linear_simd(&data);
        assert!((slope - int_to_q16(100)).abs() < 500);
        assert!((intercept - int_to_q16(100)).abs() < 500);
    }

    // ── E9: 差分エンコーディング テスト ───────────────────────────────

    #[cfg(feature = "std")]
    #[test]
    fn test_delta_encode_decode_roundtrip() {
        let coeffs = vec![(100, 200), (110, 210), (120, 220), (130, 230)];
        let encoded = delta_encode_coefficients(&coeffs);
        let decoded = delta_decode_coefficients(&encoded);
        assert_eq!(decoded, coeffs);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_delta_encode_first_preserved() {
        let coeffs = vec![(1000, 2000), (1005, 2010)];
        let encoded = delta_encode_coefficients(&coeffs);
        assert_eq!(encoded[0], (1000, 2000)); // 最初はそのまま
        assert_eq!(encoded[1], (5, 10)); // 差分
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_delta_encode_empty() {
        let encoded = delta_encode_coefficients(&[]);
        assert!(encoded.is_empty());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_delta_encoding_savings() {
        // 緩やかに変化する係数 → 差分は小さく、varint サイズも小さい
        let coeffs: Vec<(i32, i32)> = (0..100)
            .map(|i| (int_to_q16(100 + i), int_to_q16(50 + i)))
            .collect();
        let (original, delta) = delta_encoding_savings(&coeffs);
        assert!(delta <= original, "delta={delta} original={original}");
    }

    // ── Q16 conversion tests (Python binding logic coverage) ──────

    #[test]
    fn test_q16_roundtrip_positive() {
        for v in [1, 10, 100, 1000, 32767] {
            let q = int_to_q16(v);
            assert_eq!(q16_to_int(q), v, "roundtrip failed for {v}");
        }
    }

    #[test]
    fn test_q16_roundtrip_negative() {
        for v in [-1, -10, -100, -1000, -32768] {
            let q = int_to_q16(v);
            assert_eq!(q16_to_int(q), v, "roundtrip failed for {v}");
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_q16_to_f32_roundtrip() {
        // 整数値は f32 経由でも正確に戻る
        for v in [0, 1, -1, 100, -100] {
            let q = int_to_q16(v);
            let f = q16_to_f32(q);
            assert!(
                (f - v as f32).abs() < 1e-3,
                "f32 roundtrip failed for {v}: got {f}",
            );
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_q16_to_f32_fractional() {
        // 0.25 in Q16.16 = 16384
        let q = Q16_ONE / 4;
        let f = q16_to_f32(q);
        assert!((f - 0.25).abs() < 1e-5);
    }

    #[test]
    fn test_compute_residual_error_empty() {
        let err = compute_residual_error(&[], 0, 0);
        assert_eq!(err, 0);
    }

    #[test]
    fn test_compute_residual_error_single() {
        let data = [100];
        let slope = 0;
        let intercept = int_to_q16(100);
        let err = compute_residual_error(&data, slope, intercept);
        assert!(err < 100, "err = {err}");
    }

    // ── 追加テスト: Q16定数 ────────────────────────────────────────────

    #[test]
    fn test_q16_shift_and_one_constants() {
        assert_eq!(Q16_SHIFT, 16);
        assert_eq!(Q16_ONE, 65536);
        assert_eq!(Q16_ONE, 1 << Q16_SHIFT);
    }

    // ── 追加テスト: fit_linear_fixed ─────────────────────────────────

    #[test]
    fn test_fit_linear_exactly_8_elements() {
        // 8要素: 4xループ丁度2周回
        let data = [0i32, 10, 20, 30, 40, 50, 60, 70];
        let (slope, intercept) = fit_linear_fixed(&data);
        assert!((slope - int_to_q16(10)).abs() < 500, "slope={slope}");
        assert!(intercept.abs() < 500, "intercept={intercept}");
    }

    #[test]
    fn test_fit_linear_exactly_9_elements() {
        // 9要素: 4x*2 + 1余り
        let data = [0i32, 5, 10, 15, 20, 25, 30, 35, 40];
        let (slope, intercept) = fit_linear_fixed(&data);
        assert!((slope - int_to_q16(5)).abs() < 300);
        assert!(intercept.abs() < 300);
    }

    #[test]
    fn test_fit_linear_large_values() {
        // Q16.16の範囲内 (≤32767) の大きな値
        let data = [1000i32, 2000, 3000, 4000, 5000];
        let (slope, _intercept) = fit_linear_fixed(&data);
        // slope ≈ 1000 in Q16.16 > 0
        assert!(slope > 0, "slope must be positive, got {slope}");
    }

    #[test]
    fn test_fit_linear_temperature_sensor() {
        // 温度センサー模擬: 25.00°C から0.1°Cずつ上昇 (×100 整数表現)
        let data = [2500i32, 2510, 2520, 2530, 2540, 2550, 2560, 2570];
        let (slope, intercept) = fit_linear_fixed(&data);
        // slope ≈ 10 (Q16.16)
        assert!((slope - int_to_q16(10)).abs() < 500, "slope={slope}");
        // intercept ≈ 2500 (Q16.16)
        assert!(
            (intercept - int_to_q16(2500)).abs() < 1000,
            "intercept={intercept}"
        );
    }

    #[test]
    fn test_fit_linear_pressure_sensor() {
        // 気圧センサー: 101325 Pa から一定下降
        let data = [10132i32, 10130, 10128, 10126, 10124, 10122];
        let (slope, _intercept) = fit_linear_fixed(&data);
        assert!(slope < 0, "pressure drop: slope should be negative");
    }

    #[test]
    fn test_fit_linear_noisy_linear() {
        // ノイズ込み線形データ: y ≈ 100x + 50 ± 5
        let data = [52i32, 148, 253, 347, 452, 548, 653, 747];
        let (slope, _intercept) = fit_linear_fixed(&data);
        // slope ≈ 100 (±10%)
        assert!(
            (slope - int_to_q16(100)).abs() < int_to_q16(15),
            "slope={slope}"
        );
    }

    // ── 追加テスト: evaluate_linear_fixed ────────────────────────────

    #[test]
    fn test_evaluate_linear_fixed_symmetry() {
        // f(x) と f(-x) の対称性チェック: slope=10, intercept=0
        let slope = int_to_q16(10);
        let intercept = 0;
        let y_pos = evaluate_linear_fixed(slope, intercept, 5);
        let y_neg = evaluate_linear_fixed(slope, intercept, -5);
        assert_eq!(y_pos, int_to_q16(50));
        assert_eq!(y_neg, -int_to_q16(50));
    }

    #[test]
    fn test_evaluate_linear_fixed_zero_slope() {
        // slope=0 のとき、常に intercept を返す
        let intercept = int_to_q16(999);
        for x in [-10, 0, 10, 100] {
            let y = evaluate_linear_fixed(0, intercept, x);
            assert_eq!(y, intercept, "x={x}");
        }
    }

    // ── 追加テスト: fit_constant_fixed ────────────────────────────────

    #[test]
    fn test_fit_constant_fixed_four_elements() {
        // 丁度4要素: unrolled pathのみ
        let data = [10i32, 20, 30, 40];
        let mean = fit_constant_fixed(&data);
        assert!((q16_to_int(mean) - 25).abs() <= 1);
    }

    #[test]
    fn test_fit_constant_fixed_five_elements() {
        // 5要素: 4+1
        let data = [10i32, 20, 30, 40, 50];
        let mean = fit_constant_fixed(&data);
        assert!((q16_to_int(mean) - 30).abs() <= 1);
    }

    #[test]
    fn test_fit_constant_fixed_negative_values() {
        // 負の値を含む平均
        let data = [-100i32, -50, 0, 50, 100];
        let mean = fit_constant_fixed(&data);
        assert!(q16_to_int(mean).abs() <= 1);
    }

    // ── 追加テスト: fit_quadratic_fixed ──────────────────────────────

    #[test]
    fn test_fit_quadratic_negative_a() {
        // y = -x² + 100  → data: 100, 99, 96, 91, 84
        let data = [100i32, 99, 96, 91, 84];
        let (a, _b, c) = fit_quadratic_fixed(&data);
        assert!(a < 0, "a should be negative, got {a}");
        assert!((c - int_to_q16(100)).abs() < 1000, "c={c}");
    }

    #[test]
    fn test_fit_quadratic_single_point() {
        let data = [42i32];
        let (a, b, c) = fit_quadratic_fixed(&data);
        assert_eq!(a, 0);
        assert_eq!(b, 0);
        assert_eq!(q16_to_int(c), 42);
    }

    #[test]
    fn test_fit_quadratic_all_same() {
        // 定数データ → a=0, b=0, c=value
        let data = [7i32; 6];
        let (a, b, c) = fit_quadratic_fixed(&data);
        assert_eq!(a, 0, "a={a}");
        assert!(b.abs() < 500, "b={b}");
        assert!((c - int_to_q16(7)).abs() < 500, "c={c}");
    }

    #[test]
    fn test_evaluate_quadratic_at_zero() {
        // x=0 → c
        let a = int_to_q16(5);
        let b = int_to_q16(3);
        let c = int_to_q16(7);
        let y = evaluate_quadratic_fixed(a, b, c, 0);
        assert_eq!(y, c);
    }

    #[test]
    fn test_evaluate_quadratic_at_one() {
        // x=1: a+b+c
        let a = int_to_q16(2);
        let b = int_to_q16(3);
        let c = int_to_q16(4);
        // y = 2*1 + 3*1 + 4 = 9
        let y = evaluate_quadratic_fixed(a, b, c, 1);
        assert!((q16_to_int(y) - 9).abs() <= 1);
    }

    #[test]
    fn test_evaluate_quadratic_negative_x() {
        // x=-2: a*4 - b*2 + c = 1*4 - 2*2 + 1 = 1
        let a = int_to_q16(1);
        let b = int_to_q16(2);
        let c = int_to_q16(1);
        let y = evaluate_quadratic_fixed(a, b, c, -2);
        // 4 - 4 + 1 = 1
        assert!((q16_to_int(y) - 1).abs() <= 1);
    }

    // ── 追加テスト: fit_cubic_fixed ──────────────────────────────────

    #[test]
    fn test_fit_cubic_fallback_1pt() {
        let data = [99i32];
        let (a, b, _c, d) = fit_cubic_fixed(&data);
        assert_eq!(a, 0);
        assert_eq!(b, 0);
        assert_eq!(q16_to_int(d), 99);
    }

    #[test]
    fn test_fit_cubic_fallback_2pts() {
        let data = [0i32, 10];
        let (a, b, _c, _d) = fit_cubic_fixed(&data);
        assert_eq!(a, 0, "cubic term must be 0 for 2 pts");
        assert_eq!(b, 0, "quadratic term must be 0 for 2 pts");
    }

    #[test]
    fn test_fit_cubic_constant_data() {
        let data = [5i32; 5];
        let (a, _b, _c, d) = fit_cubic_fixed(&data);
        assert_eq!(a, 0, "a={a}");
        assert!((d - int_to_q16(5)).abs() < 500, "d={d}");
    }

    #[test]
    fn test_evaluate_cubic_at_zero() {
        let a = int_to_q16(1);
        let b = int_to_q16(2);
        let c = int_to_q16(3);
        let d = int_to_q16(10);
        // x=0: d
        let y = evaluate_cubic_fixed(a, b, c, d, 0);
        assert_eq!(y, d);
    }

    #[test]
    fn test_evaluate_cubic_at_negative_x() {
        // y = x³: at x=-2 → -8
        let a = int_to_q16(1);
        let b = 0;
        let c = 0;
        let d = 0;
        let y = evaluate_cubic_fixed(a, b, c, d, -2);
        assert!((q16_to_int(y) - (-8)).abs() <= 1);
    }

    #[test]
    fn test_evaluate_cubic_consistency_with_quadratic() {
        // When a=0, cubic should match quadratic
        let b = int_to_q16(2);
        let c = int_to_q16(3);
        let d = int_to_q16(5);
        for x in [0i32, 1, 2, -1, -2] {
            let yq = evaluate_quadratic_fixed(b, c, d, x);
            let yc = evaluate_cubic_fixed(0, b, c, d, x);
            assert_eq!(yc, yq, "mismatch at x={x}");
        }
    }

    // ── 追加テスト: should_use_linear ─────────────────────────────────

    #[test]
    fn test_should_use_linear_noisy_constant() {
        // ほぼ定数（±1）のデータ → linear を使わない
        let data = [100i32, 101, 100, 99, 100, 101, 100, 99];
        assert!(!should_use_linear(&data));
    }

    #[test]
    fn test_should_use_linear_exactly_3_elements() {
        // 3要素の完全線形データ
        let data = [0i32, 100, 200];
        assert!(should_use_linear(&data));
    }

    // ── 追加テスト: compute_residual_error ───────────────────────────

    #[test]
    fn test_compute_residual_error_noisy() {
        // ノイズのある線形データ: 完全フィットよりも残差が大きい
        let perfect = [0i32, 10, 20, 30, 40];
        let noisy = [0i32, 12, 18, 33, 37];
        let (sp, ip) = fit_linear_fixed(&perfect);
        let (sn, inn) = fit_linear_fixed(&noisy);
        let err_perfect = compute_residual_error(&perfect, sp, ip);
        let err_noisy = compute_residual_error(&noisy, sn, inn);
        // ノイジーデータの残差が大きいか、同等
        assert!(err_noisy >= err_perfect);
    }

    // ── 追加テスト: std-only 機能 ─────────────────────────────────────

    #[cfg(feature = "std")]
    #[test]
    fn test_filter_outliers_mad_all_same() {
        // 全要素同一 (MAD=0) → そのまま返す
        let data = [42i32; 8];
        let filtered = filter_outliers_mad(&data, 3);
        assert_eq!(filtered, data);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_filter_outliers_mad_multiple_outliers() {
        // 複数の外れ値
        let data = [100i32, 101, 9999, 99, 100, -9999, 101, 100];
        let filtered = filter_outliers_mad(&data, 3);
        assert!(filtered[2] < 200, "upper outlier replaced: {}", filtered[2]);
        assert!(
            filtered[5] > -200,
            "lower outlier replaced: {}",
            filtered[5]
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_filter_outliers_mad_k1_aggressive() {
        // k=1: より積極的に除去
        let data = [100i32, 101, 102, 200, 99, 100];
        let filtered = filter_outliers_mad(&data, 1);
        // 200 は外れ値として置換される
        assert!(filtered[3] < 200, "filtered[3]={}", filtered[3]);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_filter_outliers_mad_exact_3_elements() {
        // 3要素での動作
        let data = [10i32, 100, 10];
        let filtered = filter_outliers_mad(&data, 2);
        assert_eq!(filtered.len(), 3);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_fit_linear_robust_no_outliers() {
        // 外れ値なし: 通常フィットと同じ結果
        let data: Vec<i32> = (0..10).map(|x: i32| x * 50).collect();
        let (s_robust, i_robust) = fit_linear_robust(&data, 3);
        let (s_plain, i_plain) = fit_linear_fixed(&data);
        assert!((s_robust - s_plain).abs() < 200);
        assert!((i_robust - i_plain).abs() < 200);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_piecewise_linear_coverage_fields() {
        // セグメントのフィールドが正しい
        let data: Vec<i32> = (0..10).map(|x| x * 5).collect();
        let segments = fit_piecewise_linear(&data, i64::MAX, 2);
        assert!(!segments.is_empty());
        let seg = &segments[0];
        assert_eq!(seg.start, 0);
        assert_eq!(seg.end, 10);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_piecewise_linear_coverage_ordering() {
        // セグメントが昇順であること
        let mut data = vec![0i32; 10];
        data.extend(vec![500i32; 10]);
        let segments = fit_piecewise_linear(&data, 0, 2);
        for w in segments.windows(2) {
            assert!(w[0].end <= w[1].start, "segments overlap or unordered");
        }
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_piecewise_linear_single_element() {
        let data = [42i32];
        let segments = fit_piecewise_linear(&data, 100, 4);
        // min_segment_len=4 → data.len()<4, 単一セグメントを返す
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start, 0);
        assert_eq!(segments[0].end, 1);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_delta_encode_single_element() {
        let coeffs = vec![(500i32, 1000i32)];
        let encoded = delta_encode_coefficients(&coeffs);
        assert_eq!(encoded.len(), 1);
        assert_eq!(encoded[0], (500, 1000));
        let decoded = delta_decode_coefficients(&encoded);
        assert_eq!(decoded, coeffs);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_delta_encode_negative_values() {
        let coeffs = vec![(-100i32, -200i32), (-90, -180), (-80, -160)];
        let encoded = delta_encode_coefficients(&coeffs);
        assert_eq!(encoded[0], (-100, -200));
        assert_eq!(encoded[1].0, 10); // delta = -90 - (-100) = 10
        let decoded = delta_decode_coefficients(&encoded);
        assert_eq!(decoded, coeffs);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_delta_decode_empty() {
        let decoded = delta_decode_coefficients(&[]);
        assert!(decoded.is_empty());
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_delta_encoding_savings_constant_coefficients() {
        // 変化しない係数 → delta=0 → 圧縮効率が高い
        let coeffs: Vec<(i32, i32)> = vec![(int_to_q16(10), int_to_q16(5)); 20];
        let (original, delta) = delta_encoding_savings(&coeffs);
        assert!(delta <= original, "delta={delta} original={original}");
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_delta_encoding_savings_single() {
        let coeffs = vec![(int_to_q16(50), int_to_q16(25))];
        let (original, delta) = delta_encoding_savings(&coeffs);
        assert_eq!(original, delta); // 1要素はデルタ変換しない
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_fit_linear_simd_zero_data() {
        let data = [0i32; 8];
        let (slope, intercept) = fit_linear_simd(&data);
        assert_eq!(slope, 0);
        assert_eq!(intercept, 0);
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_large_dataset_q16_accuracy() {
        // 1000要素の完全線形データ: 精度確認
        let data: Vec<i32> = (0..1000).map(|x: i32| x * 3 + 7).collect();
        let (slope, intercept) = fit_linear_fixed(&data);
        // slope ≈ 3, intercept ≈ 7
        assert!((slope - int_to_q16(3)).abs() < 1000, "slope={slope}");
        assert!(
            (intercept - int_to_q16(7)).abs() < 1000,
            "intercept={intercept}"
        );
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_model_selection_sensor_window() {
        // センサーウィンドウ: 直線的な温度変化 → linear を使う
        let window: Vec<i32> = (0..16).map(|x: i32| 2000 + x * 5).collect();
        assert!(should_use_linear(&window));
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_model_selection_stable_sensor() {
        // 安定したセンサー → constant を使う
        let window: Vec<i32> = (0..16).map(|_| 2500).collect();
        assert!(!should_use_linear(&window));
    }
}
