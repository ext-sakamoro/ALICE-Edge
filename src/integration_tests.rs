//! Integration tests spanning multiple modules.

#![allow(
    clippy::float_cmp,
    clippy::unreadable_literal,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    clippy::explicit_iter_loop,
    clippy::bool_to_int_with_if,
    clippy::approx_constant,
    clippy::cast_lossless,
    clippy::redundant_clone,
    clippy::format_collect,
    clippy::similar_names,
    clippy::needless_collect,
    clippy::iter_cloned_collect,
    clippy::suboptimal_flops,
    clippy::should_panic_without_expect,
    clippy::manual_range_contains
)]

use crate::adaptive_polyfit::*;
use crate::constant_fit::*;
#[cfg(feature = "std")]
use crate::delta::*;
// `fit_piecewise_linear` is used by tests that only compile under `feature = "std"`;
// cargo fix removed this import under default features, breaking the std-feature build.
#[cfg(feature = "std")]
use crate::piecewise::*;
use crate::q16_linear::*;
#[cfg(feature = "std")]
use crate::robust::*;
use crate::simd_fit::*;

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

// fuzz_polyfit (2026-09-16): the fraction-free elimination in the cubic fit
// multiplied i128 entries by the running product of pivots and overflowed on
// long windows with large samples; the quadratic accumulators were i64.
// Both now use checked arithmetic and degrade one order on overflow.
#[cfg(feature = "std")]
#[test]
fn test_fit_cubic_exact_integer_polynomial() {
    // y = 2x³ - 3x² + 5x + 7 on 0..12: every coefficient is exactly
    // representable in Q16.16, and the normal equations are consistent, so
    // the solve must reproduce them to integer-division rounding.
    let data: Vec<i32> = (0..12)
        .map(|x| 2 * x * x * x - 3 * x * x + 5 * x + 7)
        .collect();
    let (a, b, c, d) = fit_cubic_fixed(&data);
    for (got, want) in [(a, 2), (b, -3), (c, 5), (d, 7)] {
        assert!(
            (got - int_to_q16(want)).abs() <= 1,
            "coefficient {got} != {} (Q16.16 of {want})",
            int_to_q16(want)
        );
    }
}

#[cfg(feature = "std")]
#[test]
fn test_fit_quadratic_exact_integer_polynomial() {
    // y = 3x² - 4x + 9 on 0..20
    let data: Vec<i32> = (0..20).map(|x| 3 * x * x - 4 * x + 9).collect();
    let (a, b, c) = fit_quadratic_fixed(&data);
    for (got, want) in [(a, 3), (b, -4), (c, 9)] {
        assert!(
            (got - int_to_q16(want)).abs() <= 1,
            "coefficient {got} != Q16.16 of {want}"
        );
    }
}

#[cfg(feature = "std")]
#[test]
fn test_fit_polynomial_extreme_inputs_do_not_panic() {
    // 4096 samples alternating at the i32 limits: Σx⁶·y exceeds i128, so the
    // cubic degrades to quadratic (a = 0) and, if that overflows too, to the
    // linear fit. The contract is "never panic, never wrap silently".
    let data: Vec<i32> = (0..4096)
        .map(|i| if i % 2 == 0 { i32::MAX } else { i32::MIN })
        .collect();
    let (a, _b, _c, _d) = fit_cubic_fixed(&data);
    assert_eq!(a, 0, "cubic must degrade, not return a wrapped coefficient");
    let _ = fit_quadratic_fixed(&data);
    let _ = should_use_linear(&data);

    // Exact-integer capacity measured 2026-09-16 (probe over |y| ≤ 1e3..2^31):
    // cubic ≤ 64 samples (32 at full i32 range), quadratic ≤ 4096. A window
    // at that size still gets a real cubic; beyond it the fit degrades.
    let data: Vec<i32> = (0..64i64).map(|x| (x * x * x / 10 + x) as i32).collect();
    let (a, _b, _c, _d) = fit_cubic_fixed(&data);
    assert!(
        a > 0,
        "cubic coefficient of a cubic trend must be positive, got {a}"
    );
    let data: Vec<i32> = (0..4096i64).map(|x| (x * x / 10 + x) as i32).collect();
    let (a, _b, _c) = fit_quadratic_fixed(&data);
    assert!(
        a > 0,
        "quadratic coefficient of a quadratic trend must be positive, got {a}"
    );
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

// fuzz_fit_linear (2026-09-16): `(x - median).abs()` overflowed i32 for
// full-range inputs and the threshold `k * MAD` was truncated to i32.
#[cfg(feature = "std")]
#[test]
fn test_filter_outliers_full_i32_range_no_overflow() {
    let data = [i32::MIN, i32::MAX, 0, i32::MIN, i32::MAX, 1, -1];
    let filtered = filter_outliers_mad(&data, 3);
    assert_eq!(filtered.len(), data.len());
    let mut sorted = data;
    sorted.sort_unstable();
    let median = sorted[data.len() / 2];
    for (&before, &after) in data.iter().zip(&filtered) {
        assert!(after == before || after == median);
    }
}

#[cfg(feature = "std")]
#[test]
fn test_filter_outliers_huge_k_keeps_everything() {
    // median 11, MAD = 1, largest deviation 9988: k * MAD saturates instead
    // of wrapping to a negative threshold (which replaced every sample by
    // the median).
    let data = [10, 11, 10, 11, 10, 12, 9999];
    let filtered = filter_outliers_mad(&data, i32::MAX);
    assert_eq!(filtered, data);
    let filtered = filter_outliers_mad(&data, 20_000);
    assert_eq!(filtered, data);
    // and a threshold below the deviation still replaces the outlier
    let filtered = filter_outliers_mad(&data, 3);
    assert_eq!(filtered, [10, 11, 10, 11, 10, 12, 11]);
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
