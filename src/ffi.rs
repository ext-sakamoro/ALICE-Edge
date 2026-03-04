// SPDX-License-Identifier: MIT
//! C-ABI FFI bindings for ALICE-Edge core functions
//!
//! Exposes Q16.16 fixed-point fitting, evaluation, and utility functions
//! as `extern "C"` for consumption by Unity (C# `DllImport`), UE5 (C++ extern),
//! and any other C-compatible runtime.
//!
//! # Safety
//!
//! All functions accept raw pointers and lengths. Callers must ensure:
//! - `data` is non-null and points to at least `len` contiguous `i32` values
//! - `len > 0` for fitting functions
//!
//! # Feature
//!
//! Enabled with `--features ffi`.

use crate::{
    compute_residual_error, evaluate_cubic_fixed, evaluate_linear_fixed, evaluate_quadratic_fixed,
    fit_constant_fixed, fit_cubic_fixed, fit_linear_fixed, fit_quadratic_fixed, int_to_q16,
    q16_to_f32, q16_to_int, should_use_linear,
};

/// Result of linear fitting: slope + intercept in Q16.16
#[repr(C)]
pub struct AliceLinearResult {
    pub slope: i32,
    pub intercept: i32,
}

/// Result of quadratic fitting: a, b, c in Q16.16
#[repr(C)]
pub struct AliceQuadraticResult {
    pub a: i32,
    pub b: i32,
    pub c: i32,
}

/// Result of cubic fitting: a, b, c, d in Q16.16
#[repr(C)]
pub struct AliceCubicResult {
    pub a: i32,
    pub b: i32,
    pub c: i32,
    pub d: i32,
}

// ── Linear fitting ──────────────────────────────────────────────────────

/// Fit a linear model to sensor data.
///
/// # Safety
///
/// `data` must be non-null, pointing to at least `len` contiguous `i32` values.
#[no_mangle]
pub unsafe extern "C" fn alice_fit_linear(data: *const i32, len: usize) -> AliceLinearResult {
    if data.is_null() || len == 0 {
        return AliceLinearResult {
            slope: 0,
            intercept: 0,
        };
    }
    // SAFETY: Caller guarantees data points to len contiguous i32 values.
    let slice = core::slice::from_raw_parts(data, len);
    let (slope, intercept) = fit_linear_fixed(slice);
    AliceLinearResult { slope, intercept }
}

/// Evaluate a linear model at point x.
#[no_mangle]
pub extern "C" fn alice_evaluate_linear(slope: i32, intercept: i32, x: i32) -> i32 {
    evaluate_linear_fixed(slope, intercept, x)
}

// ── Quadratic fitting ───────────────────────────────────────────────────

/// Fit a quadratic model to sensor data.
///
/// # Safety
///
/// `data` must be non-null, pointing to at least `len` contiguous `i32` values.
#[no_mangle]
pub unsafe extern "C" fn alice_fit_quadratic(data: *const i32, len: usize) -> AliceQuadraticResult {
    if data.is_null() || len == 0 {
        return AliceQuadraticResult { a: 0, b: 0, c: 0 };
    }
    // SAFETY: Caller guarantees data points to len contiguous i32 values.
    let slice = core::slice::from_raw_parts(data, len);
    let (a, b, c) = fit_quadratic_fixed(slice);
    AliceQuadraticResult { a, b, c }
}

/// Evaluate a quadratic model at point x.
#[no_mangle]
pub extern "C" fn alice_evaluate_quadratic(a: i32, b: i32, c: i32, x: i32) -> i32 {
    evaluate_quadratic_fixed(a, b, c, x)
}

// ── Cubic fitting ───────────────────────────────────────────────────────

/// Fit a cubic model to sensor data.
///
/// # Safety
///
/// `data` must be non-null, pointing to at least `len` contiguous `i32` values.
#[no_mangle]
pub unsafe extern "C" fn alice_fit_cubic(data: *const i32, len: usize) -> AliceCubicResult {
    if data.is_null() || len == 0 {
        return AliceCubicResult {
            a: 0,
            b: 0,
            c: 0,
            d: 0,
        };
    }
    // SAFETY: Caller guarantees data points to len contiguous i32 values.
    let slice = core::slice::from_raw_parts(data, len);
    let (a, b, c, d) = fit_cubic_fixed(slice);
    AliceCubicResult { a, b, c, d }
}

/// Evaluate a cubic model at point x.
#[no_mangle]
pub extern "C" fn alice_evaluate_cubic(a: i32, b: i32, c: i32, d: i32, x: i32) -> i32 {
    evaluate_cubic_fixed(a, b, c, d, x)
}

// ── Constant fitting ────────────────────────────────────────────────────

/// Fit a constant model (mean) to sensor data.
///
/// # Safety
///
/// `data` must be non-null, pointing to at least `len` contiguous `i32` values.
#[no_mangle]
pub unsafe extern "C" fn alice_fit_constant(data: *const i32, len: usize) -> i32 {
    if data.is_null() || len == 0 {
        return 0;
    }
    // SAFETY: Caller guarantees data points to len contiguous i32 values.
    let slice = core::slice::from_raw_parts(data, len);
    fit_constant_fixed(slice)
}

// ── Utility functions ───────────────────────────────────────────────────

/// Convert integer to Q16.16 fixed-point.
#[no_mangle]
pub extern "C" fn alice_int_to_q16(i: i32) -> i32 {
    int_to_q16(i)
}

/// Convert Q16.16 fixed-point to integer (truncates).
#[no_mangle]
pub extern "C" fn alice_q16_to_int(q: i32) -> i32 {
    q16_to_int(q)
}

/// Convert Q16.16 fixed-point to f32.
#[no_mangle]
pub extern "C" fn alice_q16_to_f32(q: i32) -> f32 {
    q16_to_f32(q)
}

/// Check if data benefits from linear model vs constant.
///
/// # Safety
///
/// `data` must be non-null, pointing to at least `len` contiguous `i32` values.
#[no_mangle]
pub unsafe extern "C" fn alice_should_use_linear(data: *const i32, len: usize) -> bool {
    if data.is_null() || len == 0 {
        return false;
    }
    // SAFETY: Caller guarantees data points to len contiguous i32 values.
    let slice = core::slice::from_raw_parts(data, len);
    should_use_linear(slice)
}

/// Compute residual error of a linear fit.
///
/// # Safety
///
/// `data` must be non-null, pointing to at least `len` contiguous `i32` values.
#[no_mangle]
pub unsafe extern "C" fn alice_residual_error(
    data: *const i32,
    len: usize,
    slope: i32,
    intercept: i32,
) -> i64 {
    if data.is_null() || len == 0 {
        return 0;
    }
    // SAFETY: Caller guarantees data points to len contiguous i32 values.
    let slice = core::slice::from_raw_parts(data, len);
    compute_residual_error(slice, slope, intercept)
}

// ── Robust / SIMD fitting ───────────────────────────────────────────────

/// Fit a robust linear model (MAD-based outlier removal + least squares).
///
/// `mad_k` controls outlier sensitivity (typically 2-3 in Q16.16).
///
/// # Safety
///
/// `data` must be non-null, pointing to at least `len` contiguous `i32` values.
#[cfg(feature = "std")]
#[no_mangle]
pub unsafe extern "C" fn alice_fit_linear_robust(
    data: *const i32,
    len: usize,
    mad_k: i32,
) -> AliceLinearResult {
    if data.is_null() || len == 0 {
        return AliceLinearResult {
            slope: 0,
            intercept: 0,
        };
    }
    // SAFETY: Caller guarantees data points to len contiguous i32 values.
    let slice = core::slice::from_raw_parts(data, len);
    let (slope, intercept) = crate::fit_linear_robust(slice, mad_k);
    AliceLinearResult { slope, intercept }
}

/// Fit a linear model using SIMD acceleration (auto-fallback on small data).
///
/// # Safety
///
/// `data` must be non-null, pointing to at least `len` contiguous `i32` values.
#[no_mangle]
pub unsafe extern "C" fn alice_fit_linear_simd(data: *const i32, len: usize) -> AliceLinearResult {
    if data.is_null() || len == 0 {
        return AliceLinearResult {
            slope: 0,
            intercept: 0,
        };
    }
    // SAFETY: Caller guarantees data points to len contiguous i32 values.
    let slice = core::slice::from_raw_parts(data, len);
    let (slope, intercept) = crate::fit_linear_simd(slice);
    AliceLinearResult { slope, intercept }
}

// ── Filter / Delta encoding ────────────────────────────────────────────

/// Filter outliers using MAD. Writes filtered data into `out` buffer.
///
/// Returns the number of elements written to `out`.
/// `out_capacity` is the maximum number of elements `out` can hold.
///
/// # Safety
///
/// - `data` must point to at least `len` contiguous `i32` values.
/// - `out` must point to at least `out_capacity` contiguous writable `i32` slots.
#[cfg(feature = "std")]
#[no_mangle]
pub unsafe extern "C" fn alice_filter_outliers_mad(
    data: *const i32,
    len: usize,
    k: i32,
    out: *mut i32,
    out_capacity: usize,
) -> usize {
    if data.is_null() || out.is_null() || len == 0 || out_capacity == 0 {
        return 0;
    }
    // SAFETY: Caller guarantees pointers and sizes.
    let slice = core::slice::from_raw_parts(data, len);
    let filtered = crate::filter_outliers_mad(slice, k);
    let count = filtered.len().min(out_capacity);
    core::ptr::copy_nonoverlapping(filtered.as_ptr(), out, count);
    count
}

/// Delta-encode coefficient pairs in-place.
///
/// `pairs` layout: `[slope0, intercept0, slope1, intercept1, ...]`
/// Encodes deltas in-place and returns the number of pairs processed.
///
/// # Safety
///
/// `pairs` must point to at least `num_pairs * 2` contiguous `i32` values.
#[no_mangle]
pub unsafe extern "C" fn alice_delta_encode(pairs: *mut i32, num_pairs: usize) -> usize {
    if pairs.is_null() || num_pairs < 2 {
        return num_pairs;
    }
    // SAFETY: Caller guarantees pairs points to num_pairs * 2 i32 values.
    let mut prev_s = *pairs;
    let mut prev_i = *pairs.add(1);
    for idx in 1..num_pairs {
        let offset = idx * 2;
        let cur_s = *pairs.add(offset);
        let cur_i = *pairs.add(offset + 1);
        *pairs.add(offset) = cur_s - prev_s;
        *pairs.add(offset + 1) = cur_i - prev_i;
        prev_s = cur_s;
        prev_i = cur_i;
    }
    num_pairs
}

/// Delta-decode coefficient pairs in-place.
///
/// # Safety
///
/// `pairs` must point to at least `num_pairs * 2` contiguous `i32` values.
#[no_mangle]
pub unsafe extern "C" fn alice_delta_decode(pairs: *mut i32, num_pairs: usize) -> usize {
    if pairs.is_null() || num_pairs < 2 {
        return num_pairs;
    }
    // SAFETY: Caller guarantees pairs points to num_pairs * 2 i32 values.
    for idx in 1..num_pairs {
        let offset = idx * 2;
        *pairs.add(offset) += *pairs.add(offset - 2);
        *pairs.add(offset + 1) += *pairs.add(offset - 1);
    }
    num_pairs
}

// ── Security: zeroize ───────────────────────────────────────────────────

/// Securely zero a buffer of i32 values using volatile writes.
///
/// Prevents the compiler from optimizing away the zeroing,
/// ensuring sensitive coefficient data is cleared from memory.
///
/// # Safety
///
/// `buf` must be non-null and point to at least `len` contiguous writable `i32` values.
#[no_mangle]
pub unsafe extern "C" fn alice_zeroize(buf: *mut i32, len: usize) {
    if buf.is_null() || len == 0 {
        return;
    }
    // SAFETY: Caller guarantees buf points to len writable i32 values.
    // Volatile write prevents compiler from optimizing away the zeroing.
    for i in 0..len {
        core::ptr::write_volatile(buf.add(i), 0);
    }
    // Compiler fence to prevent reordering
    core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
}

/// Return the ALICE-Edge library version as a null-terminated C string.
///
/// The returned pointer is valid for the lifetime of the process (static).
#[no_mangle]
pub extern "C" fn alice_edge_version() -> *const u8 {
    c"0.1.0".as_ptr().cast()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_fit_linear() {
        let data = [100, 200, 300, 400, 500];
        // SAFETY: data is a valid local array.
        let result = unsafe { alice_fit_linear(data.as_ptr(), data.len()) };
        assert_ne!(result.slope, 0);
    }

    #[test]
    fn test_ffi_fit_linear_null() {
        let result = unsafe { alice_fit_linear(core::ptr::null(), 0) };
        assert_eq!(result.slope, 0);
        assert_eq!(result.intercept, 0);
    }

    #[test]
    fn test_ffi_evaluate_linear() {
        // x is raw integer index (not Q16.16)
        let val = alice_evaluate_linear(int_to_q16(10), int_to_q16(5), 3);
        // slope(10)*3 + intercept(5) = 35 in Q16.16
        let result = q16_to_int(val);
        assert_eq!(result, 35);
    }

    #[test]
    fn test_ffi_fit_quadratic() {
        let data = [0, 1, 4, 9, 16]; // ~ x^2
        let result = unsafe { alice_fit_quadratic(data.as_ptr(), data.len()) };
        // Just check it doesn't panic and returns something
        let _ = (result.a, result.b, result.c);
    }

    #[test]
    fn test_ffi_fit_cubic() {
        let data = [0, 1, 8, 27, 64, 125, 216, 343];
        let result = unsafe { alice_fit_cubic(data.as_ptr(), data.len()) };
        let _ = (result.a, result.b, result.c, result.d);
    }

    #[test]
    fn test_ffi_fit_constant() {
        let data = [500, 500, 500, 500];
        let mean = unsafe { alice_fit_constant(data.as_ptr(), data.len()) };
        assert_eq!(q16_to_int(mean), 500);
    }

    #[test]
    fn test_ffi_q16_roundtrip() {
        assert_eq!(alice_q16_to_int(alice_int_to_q16(42)), 42);
    }

    #[test]
    fn test_ffi_q16_to_f32() {
        let f = alice_q16_to_f32(alice_int_to_q16(10));
        assert!((f - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_ffi_should_use_linear() {
        let rising = [100, 200, 300, 400, 500];
        let constant = [500, 500, 500, 500, 500];
        let linear = unsafe { alice_should_use_linear(rising.as_ptr(), rising.len()) };
        let not_linear = unsafe { alice_should_use_linear(constant.as_ptr(), constant.len()) };
        assert!(linear);
        assert!(!not_linear);
    }

    #[test]
    fn test_ffi_residual_error() {
        let data = [100, 200, 300, 400, 500];
        let result = unsafe { alice_fit_linear(data.as_ptr(), data.len()) };
        let err = unsafe {
            alice_residual_error(data.as_ptr(), data.len(), result.slope, result.intercept)
        };
        // Perfect linear data should have near-zero residual
        assert!(err < 100);
    }

    #[test]
    fn test_ffi_version() {
        let ptr = alice_edge_version();
        assert!(!ptr.is_null());
        // SAFETY: alice_edge_version returns a valid static string.
        let cstr = unsafe { core::ffi::CStr::from_ptr(ptr as *const core::ffi::c_char) };
        assert_eq!(cstr.to_str().unwrap(), "0.1.0");
    }

    #[test]
    fn test_ffi_evaluate_quadratic() {
        // x is raw integer index (not Q16.16)
        let val = alice_evaluate_quadratic(int_to_q16(1), int_to_q16(2), int_to_q16(3), 2);
        // a*4 + b*2 + c = 1*4 + 2*2 + 3 = 11 in Q16.16
        let result = q16_to_int(val);
        assert_eq!(result, 11);
    }

    #[test]
    fn test_ffi_evaluate_cubic() {
        // evaluate_cubic_fixed expects x as raw integer index, not Q16.16
        let val = alice_evaluate_cubic(
            int_to_q16(1),
            int_to_q16(0),
            int_to_q16(0),
            int_to_q16(0),
            2,
        );
        // a*x^3 = 1*8 = 8 in Q16.16
        let result = q16_to_int(val);
        assert_eq!(result, 8);
    }

    #[test]
    fn test_ffi_fit_linear_robust() {
        let data = [100, 200, 300, 9999, 500]; // 9999 は外れ値
        let result = unsafe { alice_fit_linear_robust(data.as_ptr(), data.len(), int_to_q16(3)) };
        // ロバスト推定は外れ値を除去するためslope != 0
        let _ = (result.slope, result.intercept);
    }

    #[test]
    fn test_ffi_fit_linear_simd() {
        let data = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000];
        let result = unsafe { alice_fit_linear_simd(data.as_ptr(), data.len()) };
        assert_ne!(result.slope, 0);
    }

    #[test]
    fn test_ffi_filter_outliers_mad() {
        let data = [100, 200, 300, 9999, 500];
        let mut out = [0i32; 10];
        let count = unsafe {
            alice_filter_outliers_mad(
                data.as_ptr(),
                data.len(),
                int_to_q16(2),
                out.as_mut_ptr(),
                out.len(),
            )
        };
        assert!(count > 0 && count <= data.len());
    }

    #[test]
    fn test_ffi_delta_encode_decode_roundtrip() {
        // pairs: [(100, 200), (110, 210), (130, 225)]
        let mut pairs = [100, 200, 110, 210, 130, 225];
        let original = pairs;
        unsafe { alice_delta_encode(pairs.as_mut_ptr(), 3) };
        // 最初のペアは変わらない
        assert_eq!(pairs[0], 100);
        assert_eq!(pairs[1], 200);
        // デルタ値
        assert_eq!(pairs[2], 10); // 110 - 100
        assert_eq!(pairs[3], 10); // 210 - 200

        // デコードで元に戻る
        unsafe { alice_delta_decode(pairs.as_mut_ptr(), 3) };
        assert_eq!(pairs, original);
    }

    #[test]
    fn test_ffi_zeroize() {
        let mut buf = [42i32, 100, -1, i32::MAX, i32::MIN];
        unsafe { alice_zeroize(buf.as_mut_ptr(), buf.len()) };
        assert_eq!(buf, [0, 0, 0, 0, 0]);
    }

    #[test]
    fn test_ffi_zeroize_null_safe() {
        // null ポインタでパニックしない
        unsafe { alice_zeroize(core::ptr::null_mut(), 0) };
    }
}
