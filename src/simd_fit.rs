//! SIMD-accelerated linear fitting (`fit_linear_simd`).

use crate::q16_linear::fit_linear_fixed;
// Only the x86_64 / aarch64 kernels shift into Q16.16; the scalar fallback
// on other targets (thumbv7em etc.) delegates to `fit_linear_fixed`.
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
use crate::q16_linear::Q16_SHIFT;

pub fn fit_linear_simd(data: &[i32]) -> (i32, i32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("sse2") {
            // SAFETY: SSE2 サポートを上記で確認済み。data は有効なスライス。
            return unsafe { fit_linear_sse2(data) };
        }
        return fit_linear_fixed(data);
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
#[inline]
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
