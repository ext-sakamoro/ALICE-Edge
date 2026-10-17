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
use core::ptr;

/// Result of linear fitting: slope + intercept in Q16.16
#[repr(C)]
pub struct AliceLinearResult {
    pub slope: i32,
    pub intercept: i32,
    /// The producer's `EDGE_SEMANTICS_ID`: pass it back to the
    /// `alice_evaluate_*` / `alice_residual_error` functions with the
    /// coefficients (ALICE-Law G7)
    pub semantics_id: [u8; 32],
}

/// Result of quadratic fitting: a, b, c in Q16.16
#[repr(C)]
pub struct AliceQuadraticResult {
    pub a: i32,
    pub b: i32,
    pub c: i32,
    /// The producer's `EDGE_SEMANTICS_ID`: pass it back to the
    /// `alice_evaluate_*` / `alice_residual_error` functions with the
    /// coefficients (ALICE-Law G7)
    pub semantics_id: [u8; 32],
}

/// Result of cubic fitting: a, b, c, d in Q16.16
#[repr(C)]
pub struct AliceCubicResult {
    pub a: i32,
    pub b: i32,
    pub c: i32,
    pub d: i32,
    /// The producer's `EDGE_SEMANTICS_ID`: pass it back to the
    /// `alice_evaluate_*` / `alice_residual_error` functions with the
    /// coefficients (ALICE-Law G7)
    pub semantics_id: [u8; 32],
}

// ── Linear fitting ──────────────────────────────────────────────────────

/// Fit a linear model to sensor data.
///
/// # Safety
///
/// `data` must be non-null, pointing to at least `len` contiguous `i32` values.
#[no_mangle]
pub unsafe extern "C" fn alice_fit_linear(data: *const i32, len: usize) -> AliceLinearResult {
    ffi_guard(
        AliceLinearResult {
            slope: 0,
            intercept: 0,
            semantics_id: crate::EDGE_SEMANTICS_ID,
        },
        || {
            if data.is_null() || len == 0 {
                return AliceLinearResult {
                    slope: 0,
                    intercept: 0,
                    semantics_id: crate::EDGE_SEMANTICS_ID,
                };
            }
            // SAFETY: Caller guarantees data points to len contiguous i32 values.
            let slice = core::slice::from_raw_parts(data, len);
            let (slope, intercept) = fit_linear_fixed(slice);
            AliceLinearResult {
                slope,
                intercept,
                semantics_id: crate::EDGE_SEMANTICS_ID,
            }
        },
    )
}

/// Evaluate a linear model at point x.
///
/// `semantics_id` points to the 32-byte `EDGE_SEMANTICS_ID` the coefficients
/// came with (`Alice*Result::semantics_id`); the value is written to `out`.
/// Returns 0 (this build's identifier) or 1 (a verified earlier one), -1 for
/// a null pointer and -2 for any other identifier, which also records both
/// identifiers for `alice_edge_last_error` (ALICE-Law G7).
///
/// # Safety
/// `semantics_id` points to 32 readable bytes and `out` to a writable `i32`.
#[no_mangle]
pub unsafe extern "C" fn alice_evaluate_linear(
    slope: i32,
    intercept: i32,
    x: i32,
    semantics_id: *const u8,
    out: *mut i32,
) -> i32 {
    ffi_guard(-1, || {
        if out.is_null() {
            return -1;
        }
        match handshake(semantics_id) {
            Ok(status) => {
                out.write(evaluate_linear_fixed(slope, intercept, x));
                status
            }
            Err(code) => code,
        }
    })
}

// ── Quadratic fitting ───────────────────────────────────────────────────

/// Fit a quadratic model to sensor data.
///
/// # Safety
///
/// `data` must be non-null, pointing to at least `len` contiguous `i32` values.
#[no_mangle]
pub unsafe extern "C" fn alice_fit_quadratic(data: *const i32, len: usize) -> AliceQuadraticResult {
    ffi_guard(
        AliceQuadraticResult {
            a: 0,
            b: 0,
            c: 0,
            semantics_id: crate::EDGE_SEMANTICS_ID,
        },
        || {
            if data.is_null() || len == 0 {
                return AliceQuadraticResult {
                    a: 0,
                    b: 0,
                    c: 0,
                    semantics_id: crate::EDGE_SEMANTICS_ID,
                };
            }
            // SAFETY: Caller guarantees data points to len contiguous i32 values.
            let slice = core::slice::from_raw_parts(data, len);
            let (a, b, c) = fit_quadratic_fixed(slice);
            AliceQuadraticResult {
                a,
                b,
                c,
                semantics_id: crate::EDGE_SEMANTICS_ID,
            }
        },
    )
}

/// Evaluate a quadratic model at point x.
///
/// `semantics_id` points to the 32-byte `EDGE_SEMANTICS_ID` the coefficients
/// came with (`Alice*Result::semantics_id`); the value is written to `out`.
/// Returns 0 (this build's identifier) or 1 (a verified earlier one), -1 for
/// a null pointer and -2 for any other identifier, which also records both
/// identifiers for `alice_edge_last_error` (ALICE-Law G7).
///
/// # Safety
/// `semantics_id` points to 32 readable bytes and `out` to a writable `i32`.
#[no_mangle]
pub unsafe extern "C" fn alice_evaluate_quadratic(
    a: i32,
    b: i32,
    c: i32,
    x: i32,
    semantics_id: *const u8,
    out: *mut i32,
) -> i32 {
    ffi_guard(-1, || {
        if out.is_null() {
            return -1;
        }
        match handshake(semantics_id) {
            Ok(status) => {
                out.write(evaluate_quadratic_fixed(a, b, c, x));
                status
            }
            Err(code) => code,
        }
    })
}

// ── Cubic fitting ───────────────────────────────────────────────────────

/// Fit a cubic model to sensor data.
///
/// # Safety
///
/// `data` must be non-null, pointing to at least `len` contiguous `i32` values.
#[no_mangle]
pub unsafe extern "C" fn alice_fit_cubic(data: *const i32, len: usize) -> AliceCubicResult {
    ffi_guard(
        AliceCubicResult {
            a: 0,
            b: 0,
            c: 0,
            d: 0,
            semantics_id: crate::EDGE_SEMANTICS_ID,
        },
        || {
            if data.is_null() || len == 0 {
                return AliceCubicResult {
                    a: 0,
                    b: 0,
                    c: 0,
                    d: 0,
                    semantics_id: crate::EDGE_SEMANTICS_ID,
                };
            }
            // SAFETY: Caller guarantees data points to len contiguous i32 values.
            let slice = core::slice::from_raw_parts(data, len);
            let (a, b, c, d) = fit_cubic_fixed(slice);
            AliceCubicResult {
                a,
                b,
                c,
                d,
                semantics_id: crate::EDGE_SEMANTICS_ID,
            }
        },
    )
}

/// Evaluate a cubic model at point x.
///
/// `semantics_id` points to the 32-byte `EDGE_SEMANTICS_ID` the coefficients
/// came with (`Alice*Result::semantics_id`); the value is written to `out`.
/// Returns 0 (this build's identifier) or 1 (a verified earlier one), -1 for
/// a null pointer and -2 for any other identifier, which also records both
/// identifiers for `alice_edge_last_error` (ALICE-Law G7).
///
/// # Safety
/// `semantics_id` points to 32 readable bytes and `out` to a writable `i32`.
#[no_mangle]
pub unsafe extern "C" fn alice_evaluate_cubic(
    a: i32,
    b: i32,
    c: i32,
    d: i32,
    x: i32,
    semantics_id: *const u8,
    out: *mut i32,
) -> i32 {
    ffi_guard(-1, || {
        if out.is_null() {
            return -1;
        }
        match handshake(semantics_id) {
            Ok(status) => {
                out.write(evaluate_cubic_fixed(a, b, c, d, x));
                status
            }
            Err(code) => code,
        }
    })
}

// ── Constant fitting ────────────────────────────────────────────────────

/// Fit a constant model (mean) to sensor data.
///
/// # Safety
///
/// `data` must be non-null, pointing to at least `len` contiguous `i32` values.
#[no_mangle]
pub const unsafe extern "C" fn alice_fit_constant(data: *const i32, len: usize) -> i32 {
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
pub const extern "C" fn alice_int_to_q16(i: i32) -> i32 {
    int_to_q16(i)
}

/// Convert Q16.16 fixed-point to integer (truncates).
#[no_mangle]
pub const extern "C" fn alice_q16_to_int(q: i32) -> i32 {
    q16_to_int(q)
}

/// Convert Q16.16 fixed-point to f32.
#[no_mangle]
pub extern "C" fn alice_q16_to_f32(q: i32) -> f32 {
    ffi_guard(f32::NAN, || q16_to_f32(q))
}

/// Check if data benefits from linear model vs constant.
///
/// # Safety
///
/// `data` must be non-null, pointing to at least `len` contiguous `i32` values.
#[no_mangle]
pub unsafe extern "C" fn alice_should_use_linear(data: *const i32, len: usize) -> bool {
    ffi_guard(false, || {
        if data.is_null() || len == 0 {
            return false;
        }
        // SAFETY: Caller guarantees data points to len contiguous i32 values.
        let slice = core::slice::from_raw_parts(data, len);
        should_use_linear(slice)
    })
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
    semantics_id: *const u8,
    out: *mut i64,
) -> i32 {
    ffi_guard(-1, || {
        if out.is_null() {
            return -1;
        }
        match handshake(semantics_id) {
            Ok(status) => {
                let error = if data.is_null() || len == 0 {
                    0
                } else {
                    // SAFETY: Caller guarantees data points to len contiguous i32 values.
                    compute_residual_error(core::slice::from_raw_parts(data, len), slope, intercept)
                };
                out.write(error);
                status
            }
            Err(code) => code,
        }
    })
}

// ── Status codes ────────────────────────────────────────────────────────

/// Status: success.
pub const ALICE_EDGE_OK: i32 = 0;
/// Status: a required pointer was null (`data` with `len > 0`, `out` with a
/// non-zero capacity, or an out-parameter).
pub const ALICE_EDGE_ERR_NULL_POINTER: i32 = -1;
/// Status: the outlier factor `k` was zero or negative. `k` is a plain
/// integer multiplier of the MAD (`3` = three MADs, not Q16.16).
pub const ALICE_EDGE_ERR_NON_POSITIVE_K: i32 = -2;
/// Status: the function panicked; the message is in `alice_edge_last_error`.
pub const ALICE_EDGE_ERR_PANIC: i32 = -3;

/// Map a refused robust call to its status and record the message for
/// `alice_edge_last_error`.
#[cfg(feature = "std")]
fn robust_status(err: crate::RobustError) -> i32 {
    guard::set_last_error(format!("alice_edge: {err}"));
    match err {
        crate::RobustError::NonPositiveK { .. } => ALICE_EDGE_ERR_NON_POSITIVE_K,
        // `RobustError` is `#[non_exhaustive]` only towards other crates;
        // inside this crate the match is exhaustive, so a new variant has to
        // be given its own status here
    }
}

/// The input window, or `None` when `data` is null with `len > 0`.
///
/// # Safety
///
/// When `len > 0`, `data` must point to at least `len` contiguous `i32`.
#[cfg(feature = "std")]
unsafe fn input_slice<'a>(data: *const i32, len: usize) -> Option<&'a [i32]> {
    if len == 0 {
        Some(&[])
    } else if data.is_null() {
        None
    } else {
        // SAFETY: Caller guarantees data points to len contiguous i32 values.
        Some(core::slice::from_raw_parts(data, len))
    }
}

// ── Robust / SIMD fitting ───────────────────────────────────────────────

/// Fit a robust linear model (MAD outlier rejection on the residuals of a
/// least-squares fit, outliers filled with the line's prediction, refit).
///
/// `k` is a **plain integer multiplier** of the MAD (`3` = three MADs; it is
/// not Q16.16). The fit is written to `*out`.
///
/// Returns [`ALICE_EDGE_OK`], or:
/// - [`ALICE_EDGE_ERR_NON_POSITIVE_K`] when `k <= 0` (message in
///   `alice_edge_last_error`)
/// - [`ALICE_EDGE_ERR_NULL_POINTER`] when `out` is null, or `data` is null
///   with `len > 0`
/// - [`ALICE_EDGE_ERR_PANIC`] when the body panicked
///
/// On every status other than `ALICE_EDGE_OK` a non-null `*out` is set to
/// `{0, 0}`. `len == 0` (or fewer than 3 samples) gives the plain fit.
///
/// # Safety
///
/// - when `len > 0`, `data` must point to at least `len` contiguous `i32`
/// - `out` must be null or point to a writable `AliceLinearResult`
#[cfg(feature = "std")]
#[no_mangle]
pub unsafe extern "C" fn alice_fit_linear_robust(
    data: *const i32,
    len: usize,
    k: i32,
    out: *mut AliceLinearResult,
) -> i32 {
    if out.is_null() {
        return ALICE_EDGE_ERR_NULL_POINTER;
    }
    // SAFETY: `out` is non-null and caller-owned.
    out.write(AliceLinearResult {
        slope: 0,
        intercept: 0,
        semantics_id: crate::EDGE_SEMANTICS_ID,
    });
    ffi_guard(ALICE_EDGE_ERR_PANIC, || {
        // SAFETY: forwarded caller guarantee on `data` / `len`.
        let Some(slice) = input_slice(data, len) else {
            return ALICE_EDGE_ERR_NULL_POINTER;
        };
        // unit-test builds only: lets a test make the robust fit panic here
        #[cfg(test)]
        guard::panic_hook::fire_if_armed();
        match crate::fit_linear_robust(slice, k) {
            Ok((slope, intercept)) => {
                // SAFETY: `out` is non-null and caller-owned.
                out.write(AliceLinearResult {
                    slope,
                    intercept,
                    semantics_id: crate::EDGE_SEMANTICS_ID,
                });
                ALICE_EDGE_OK
            }
            Err(e) => robust_status(e),
        }
    })
}

/// Fit a linear model using SIMD acceleration (auto-fallback on small data).
///
/// # Safety
///
/// `data` must be non-null, pointing to at least `len` contiguous `i32` values.
#[no_mangle]
pub unsafe extern "C" fn alice_fit_linear_simd(data: *const i32, len: usize) -> AliceLinearResult {
    ffi_guard(
        AliceLinearResult {
            slope: 0,
            intercept: 0,
            semantics_id: crate::EDGE_SEMANTICS_ID,
        },
        || {
            if data.is_null() || len == 0 {
                return AliceLinearResult {
                    slope: 0,
                    intercept: 0,
                    semantics_id: crate::EDGE_SEMANTICS_ID,
                };
            }
            // SAFETY: Caller guarantees data points to len contiguous i32 values.
            let slice = core::slice::from_raw_parts(data, len);
            let (slope, intercept) = crate::fit_linear_simd(slice);
            AliceLinearResult {
                slope,
                intercept,
                semantics_id: crate::EDGE_SEMANTICS_ID,
            }
        },
    )
}

// ── Filter / Delta encoding ────────────────────────────────────────────

/// Filter outliers using MAD: a sample `x` is replaced by the median when
/// `|x − median| > k · MAD`. Writes the filtered window into `out`.
///
/// `k` is a **plain integer multiplier** of the MAD (`3` = three MADs; it is
/// not Q16.16). `*out_written` receives the number of elements written,
/// `min(len, out_capacity)`.
///
/// Returns [`ALICE_EDGE_OK`], or:
/// - [`ALICE_EDGE_ERR_NON_POSITIVE_K`] when `k <= 0` (message in
///   `alice_edge_last_error`)
/// - [`ALICE_EDGE_ERR_NULL_POINTER`] when `out_written` is null, `data` is
///   null with `len > 0`, or `out` is null with `out_capacity > 0`
/// - [`ALICE_EDGE_ERR_PANIC`] when the body panicked
///
/// On every status other than `ALICE_EDGE_OK` a non-null `*out_written` is 0
/// and `out` is not written. Fewer than 3 samples pass the window through
/// unchanged.
///
/// # Safety
///
/// - when `len > 0`, `data` must point to at least `len` contiguous `i32`
/// - when `out_capacity > 0`, `out` must point to at least `out_capacity`
///   contiguous writable `i32` slots
/// - `out_written` must be null or point to a writable `size_t`
#[cfg(feature = "std")]
#[no_mangle]
pub unsafe extern "C" fn alice_filter_outliers_mad(
    data: *const i32,
    len: usize,
    k: i32,
    out: *mut i32,
    out_capacity: usize,
    out_written: *mut usize,
) -> i32 {
    if out_written.is_null() {
        return ALICE_EDGE_ERR_NULL_POINTER;
    }
    // SAFETY: `out_written` is non-null and caller-owned.
    out_written.write(0);
    ffi_guard(ALICE_EDGE_ERR_PANIC, || {
        if out.is_null() && out_capacity > 0 {
            return ALICE_EDGE_ERR_NULL_POINTER;
        }
        // SAFETY: forwarded caller guarantee on `data` / `len`.
        let Some(slice) = input_slice(data, len) else {
            return ALICE_EDGE_ERR_NULL_POINTER;
        };
        // unit-test builds only: lets a test make the filter panic here
        #[cfg(test)]
        guard::panic_hook::fire_if_armed();
        match crate::filter_outliers_mad(slice, k) {
            Ok(filtered) => {
                let count = filtered.len().min(out_capacity);
                if count > 0 {
                    // SAFETY: `out` holds at least `out_capacity >= count` slots.
                    core::ptr::copy_nonoverlapping(filtered.as_ptr(), out, count);
                }
                // SAFETY: `out_written` is non-null and caller-owned.
                out_written.write(count);
                ALICE_EDGE_OK
            }
            Err(e) => robust_status(e),
        }
    })
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
    ffi_guard(0, || {
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
    })
}

/// Delta-decode coefficient pairs in-place.
///
/// # Safety
///
/// `pairs` must point to at least `num_pairs * 2` contiguous `i32` values.
#[no_mangle]
pub unsafe extern "C" fn alice_delta_decode(pairs: *mut i32, num_pairs: usize) -> usize {
    ffi_guard(0, || {
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
    })
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
    ffi_guard((), || {
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
    });
}

/// Return the ALICE-Edge library version as a null-terminated C string.
///
/// The returned pointer is valid for the lifetime of the process (static).
/// The string is the crate version from `Cargo.toml`.
/// Writes this build's 32-byte `EDGE_SEMANTICS_ID` to `out` (ALICE-Law G7).
/// Returns 0, or -1 when `out` is null or `len` is below 32.
///
/// # Safety
/// `out` points to `len` writable bytes.
#[no_mangle]
pub unsafe extern "C" fn alice_edge_semantics_id(out: *mut u8, len: usize) -> i32 {
    ffi_guard(-1, || {
        if out.is_null() || len < 32 {
            return -1;
        }
        ptr::copy_nonoverlapping(crate::EDGE_SEMANTICS_ID.as_ptr(), out, 32);
        0
    })
}

/// Reads the 32-byte identifier that came with Law-bearing values and checks
/// it: `Ok(0)` this build's, `Ok(1)` a verified earlier one; `Err(-1)` null,
/// `Err(-2)` any other (recording both identifiers for `alice_edge_last_error`
/// with `std`).
unsafe fn handshake(semantics_id: *const u8) -> Result<i32, i32> {
    if semantics_id.is_null() {
        #[cfg(feature = "std")]
        guard::set_last_error("alice_edge: semantics identifier is null");
        return Err(-1);
    }
    let mut id = [0u8; 32];
    ptr::copy_nonoverlapping(semantics_id, id.as_mut_ptr(), 32);
    match crate::semantics::check_semantics(&id) {
        Ok(crate::semantics::IdStatus::Current) => Ok(0),
        Ok(crate::semantics::IdStatus::VerifiedPrevious(_)) => Ok(1),
        Err(m) => {
            #[cfg(feature = "std")]
            guard::set_last_error(format!("alice_edge: {m}"));
            #[cfg(not(feature = "std"))]
            let _ = m;
            Err(-2)
        }
    }
}

#[no_mangle]
pub const extern "C" fn alice_edge_version() -> *const u8 {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr()
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
        let mut val = 0;
        assert_eq!(
            unsafe {
                alice_evaluate_linear(
                    int_to_q16(10),
                    int_to_q16(5),
                    3,
                    crate::EDGE_SEMANTICS_ID.as_ptr(),
                    &mut val,
                )
            },
            0
        );
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
        let mut err = 0;
        assert_eq!(
            unsafe {
                alice_residual_error(
                    data.as_ptr(),
                    data.len(),
                    result.slope,
                    result.intercept,
                    result.semantics_id.as_ptr(),
                    &mut err,
                )
            },
            0
        );
        // Perfect linear data should have near-zero residual
        assert!(err < 100);
    }

    #[test]
    fn test_ffi_version() {
        let ptr = alice_edge_version();
        assert!(!ptr.is_null());
        // SAFETY: alice_edge_version returns a valid static string.
        let cstr = unsafe { core::ffi::CStr::from_ptr(ptr.cast::<core::ffi::c_char>()) };
        assert_eq!(cstr.to_str().unwrap(), env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn test_ffi_evaluate_quadratic() {
        // x is raw integer index (not Q16.16)
        let mut val = 0;
        assert_eq!(
            unsafe {
                alice_evaluate_quadratic(
                    int_to_q16(1),
                    int_to_q16(2),
                    int_to_q16(3),
                    2,
                    crate::EDGE_SEMANTICS_ID.as_ptr(),
                    &mut val,
                )
            },
            0
        );
        // a*4 + b*2 + c = 1*4 + 2*2 + 3 = 11 in Q16.16
        let result = q16_to_int(val);
        assert_eq!(result, 11);
    }

    #[test]
    fn test_ffi_evaluate_cubic() {
        // evaluate_cubic_fixed expects x as raw integer index, not Q16.16
        let mut val = 0;
        let status = unsafe {
            alice_evaluate_cubic(
                int_to_q16(1),
                int_to_q16(0),
                int_to_q16(0),
                int_to_q16(0),
                2,
                crate::EDGE_SEMANTICS_ID.as_ptr(),
                &mut val,
            )
        };
        assert_eq!(status, 0);
        // a*x^3 = 1*8 = 8 in Q16.16
        let result = q16_to_int(val);
        assert_eq!(result, 8);
    }

    /// A ramp `y = 100·(i + 1)` with one gross outlier at `i = 3`, and the
    /// same ramp without it.
    #[cfg(feature = "std")]
    fn ramp_with_outlier() -> ([i32; 12], [i32; 12]) {
        let clean: [i32; 12] = core::array::from_fn(|i| 100 * (i as i32 + 1));
        let mut spiked = clean;
        spiked[3] = 9999;
        (clean, spiked)
    }

    #[cfg(feature = "std")]
    fn ffi_robust(data: &[i32], k: i32) -> (i32, AliceLinearResult) {
        let mut out = AliceLinearResult {
            slope: 1,
            intercept: 1,
            semantics_id: crate::EDGE_SEMANTICS_ID,
        };
        // SAFETY: data is a valid local slice, out a valid local struct.
        let status = unsafe { alice_fit_linear_robust(data.as_ptr(), data.len(), k, &mut out) };
        (status, out)
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_ffi_fit_linear_robust() {
        let (clean, spiked) = ramp_with_outlier();
        let plain_clean = crate::fit_linear_fixed(&clean);
        assert_eq!(plain_clean, (int_to_q16(100), int_to_q16(100)));
        // the outlier pulls the plain fit off the line
        assert_ne!(crate::fit_linear_fixed(&spiked), plain_clean);

        // k is a plain multiplier: 3 = three MADs. The FFI result is the
        // Rust robust fit (checked against the documented rule in
        // tests/robust_contract.rs) ...
        let (status, out) = ffi_robust(&spiked, 3);
        assert_eq!(status, ALICE_EDGE_OK);
        let robust = (out.slope, out.intercept);
        assert_eq!(Ok(robust), crate::fit_linear_robust(&spiked, 3));
        // ... and the outlier is removed. Measured: plain fit slope −67.8,
        // robust fit (97.3, 127.6) against the true (100, 100). The residual
        // iteration stops once the outlier set is stable, so a 12-sample
        // window keeps a few percent of bias from the first fit; the bounds
        // below are 3 in slope and 30 in intercept, while leaving the outlier
        // in (or passing k as Q16.16, which keeps it) gives slope −67.8.
        assert!(
            (robust.0 - plain_clean.0).abs() <= int_to_q16(3),
            "robust slope {} vs {}",
            robust.0,
            plain_clean.0
        );
        assert!(
            (robust.1 - plain_clean.1).abs() <= int_to_q16(30),
            "robust intercept {} vs {}",
            robust.1,
            plain_clean.1
        );
        // the same k as Q16.16 (3 << 16) is a threshold of 196608 MADs:
        // nothing is filtered and the plain fit comes back
        let (status, as_q16) = ffi_robust(&spiked, int_to_q16(3));
        assert_eq!(status, ALICE_EDGE_OK);
        assert_eq!(
            (as_q16.slope, as_q16.intercept),
            crate::fit_linear_fixed(&spiked)
        );

        // k <= 0 is refused with its status, `*out` zeroed, message recorded
        for k in [0, -1, -5, i32::MIN] {
            guard::clear_last_error();
            let (status, out) = ffi_robust(&spiked, k);
            assert_eq!(status, ALICE_EDGE_ERR_NON_POSITIVE_K, "k = {k}");
            assert_eq!((out.slope, out.intercept), (0, 0), "k = {k}");
            let msg = guard::take_last_error().expect("message recorded");
            assert!(msg.contains(&k.to_string()), "{msg}");
        }

        // null out-parameter / null data with a length
        // SAFETY: the function checks both pointers before using them.
        unsafe {
            assert_eq!(
                alice_fit_linear_robust(spiked.as_ptr(), spiked.len(), 3, ptr::null_mut()),
                ALICE_EDGE_ERR_NULL_POINTER
            );
        }
        let mut out = AliceLinearResult {
            slope: 1,
            intercept: 1,
            semantics_id: crate::EDGE_SEMANTICS_ID,
        };
        // SAFETY: as above.
        let status = unsafe { alice_fit_linear_robust(ptr::null(), 4, 3, &mut out) };
        assert_eq!(status, ALICE_EDGE_ERR_NULL_POINTER);
        assert_eq!((out.slope, out.intercept), (0, 0));
        // an empty window is valid: the plain fit of nothing
        // SAFETY: len == 0, data is not read.
        let status = unsafe { alice_fit_linear_robust(ptr::null(), 0, 3, &mut out) };
        assert_eq!(status, ALICE_EDGE_OK);
    }

    #[test]
    fn test_ffi_fit_linear_simd() {
        let data = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000];
        let result = unsafe { alice_fit_linear_simd(data.as_ptr(), data.len()) };
        assert_ne!(result.slope, 0);
    }

    #[cfg(feature = "std")]
    fn ffi_filter(data: &[i32], k: i32, out: &mut [i32]) -> (i32, usize) {
        let mut written = usize::MAX;
        // SAFETY: data / out are valid local slices, written a valid local.
        let status = unsafe {
            alice_filter_outliers_mad(
                data.as_ptr(),
                data.len(),
                k,
                out.as_mut_ptr(),
                out.len(),
                &mut written,
            )
        };
        (status, written)
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_ffi_filter_outliers_mad() {
        // median 300, MAD 100: the outlier is 9699 away and is replaced by
        // the median with k = 2 (a plain multiplier), every other sample kept
        let data = [100, 200, 300, 9999, 500];
        let mut out = [0i32; 10];
        let (status, written) = ffi_filter(&data, 2, &mut out);
        assert_eq!(status, ALICE_EDGE_OK);
        assert_eq!(written, data.len());
        assert_eq!(out[..written], [100, 200, 300, 300, 500]);

        // k <= 0 is refused with its status, nothing written
        for k in [0, -1, i32::MIN] {
            let mut out = [7i32; 5];
            let (status, written) = ffi_filter(&data, k, &mut out);
            assert_eq!(status, ALICE_EDGE_ERR_NON_POSITIVE_K, "k = {k}");
            assert_eq!(written, 0, "k = {k}");
            assert_eq!(out, [7; 5], "k = {k}: out must not be written");
        }

        // a short buffer receives the first `out_capacity` elements
        let mut short = [0i32; 2];
        assert_eq!(ffi_filter(&data, 2, &mut short), (ALICE_EDGE_OK, 2));
        assert_eq!(short, [100, 200]);

        // null out_written
        // SAFETY: the function checks the pointer before using it.
        let status = unsafe {
            alice_filter_outliers_mad(
                data.as_ptr(),
                data.len(),
                2,
                out.as_mut_ptr(),
                out.len(),
                ptr::null_mut(),
            )
        };
        assert_eq!(status, ALICE_EDGE_ERR_NULL_POINTER);
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

    fn foreign_id() -> [u8; 32] {
        let mut id = crate::EDGE_SEMANTICS_ID;
        id[7] ^= 0x02;
        id
    }

    #[cfg(feature = "std")]
    fn assert_last_error_names_both() {
        let s = alice_edge_last_error();
        assert!(!s.is_null(), "a mismatch records both identifiers");
        let msg = unsafe { std::ffi::CStr::from_ptr(s) }
            .to_string_lossy()
            .into_owned();
        unsafe { alice_edge_free_error_string(s) };
        let hex = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
        assert!(msg.contains(&hex(&foreign_id())), "{msg}");
        assert!(msg.contains(&hex(&crate::EDGE_SEMANTICS_ID)), "{msg}");
    }

    /// Every fit result carries this build's identifier, and every function
    /// that takes coefficients back refuses another identifier (ALICE-Law G7).
    #[test]
    fn coefficients_with_another_identifier_are_refused() {
        let data = [100i32, 110, 120, 130];
        let r = unsafe { alice_fit_linear(data.as_ptr(), data.len()) };
        assert_eq!(r.semantics_id, crate::EDGE_SEMANTICS_ID);
        let q = unsafe { alice_fit_quadratic(data.as_ptr(), data.len()) };
        assert_eq!(q.semantics_id, crate::EDGE_SEMANTICS_ID);
        let c = unsafe { alice_fit_cubic(data.as_ptr(), data.len()) };
        assert_eq!(c.semantics_id, crate::EDGE_SEMANTICS_ID);
        let bad = foreign_id();
        let (mut v, mut e) = (7i32, 7i64);
        unsafe {
            assert_eq!(
                alice_evaluate_linear(r.slope, r.intercept, 1, bad.as_ptr(), &mut v),
                -2
            );
            #[cfg(feature = "std")]
            assert_last_error_names_both();
            assert_eq!(
                alice_evaluate_quadratic(q.a, q.b, q.c, 1, bad.as_ptr(), &mut v),
                -2
            );
            #[cfg(feature = "std")]
            assert_last_error_names_both();
            assert_eq!(
                alice_evaluate_cubic(c.a, c.b, c.c, c.d, 1, bad.as_ptr(), &mut v),
                -2
            );
            #[cfg(feature = "std")]
            assert_last_error_names_both();
            assert_eq!(
                alice_residual_error(
                    data.as_ptr(),
                    data.len(),
                    r.slope,
                    r.intercept,
                    bad.as_ptr(),
                    &mut e
                ),
                -2
            );
            #[cfg(feature = "std")]
            assert_last_error_names_both();
            assert_eq!((v, e), (7, 7), "a refused call writes nothing");
            assert_eq!(
                alice_evaluate_linear(r.slope, r.intercept, 1, ptr::null(), &mut v),
                -1
            );
            assert_eq!(
                alice_evaluate_linear(
                    r.slope,
                    r.intercept,
                    1,
                    r.semantics_id.as_ptr(),
                    ptr::null_mut()
                ),
                -1
            );
        }
    }

    #[test]
    fn semantics_id_is_written_and_short_buffers_are_refused() {
        let mut out = [0u8; 32];
        assert_eq!(unsafe { alice_edge_semantics_id(out.as_mut_ptr(), 32) }, 0);
        assert_eq!(out, crate::EDGE_SEMANTICS_ID);
        assert_eq!(unsafe { alice_edge_semantics_id(out.as_mut_ptr(), 31) }, -1);
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

// ============================================================================
// Panic isolation (see `guard`)
// ============================================================================

/// Message of the most recent FFI error on this thread — a panic caught at
/// the boundary, or an argument refused by a status-returning function such
/// as `alice_fit_linear_robust` (NUL-terminated, owned by the callee — release with
/// `alice_edge_free_error_string`), or null if none Always null without the
/// `std` feature (bare-metal has no unwinding: a panic reaches the panic
/// handler instead of this slot).
#[no_mangle]
pub extern "C" fn alice_edge_last_error() -> *mut core::ffi::c_char {
    ffi_guard(ptr::null_mut(), || {
        #[cfg(feature = "std")]
        {
            match guard::take_last_error() {
                Some(msg) => {
                    std::ffi::CString::new(msg).map_or(ptr::null_mut(), std::ffi::CString::into_raw)
                }
                None => ptr::null_mut(),
            }
        }
        #[cfg(not(feature = "std"))]
        {
            ptr::null_mut()
        }
    })
}

/// Clear the most recent FFI error message (no-op without `std`).
#[no_mangle]
pub extern "C" fn alice_edge_clear_last_error() {
    ffi_guard((), guard::clear_last_error);
}

/// Release a string returned by `alice_edge_last_error`.
///
/// # Safety
/// `s` must be null or a pointer returned by `alice_edge_last_error` (freed once).
#[no_mangle]
pub unsafe extern "C" fn alice_edge_free_error_string(s: *mut core::ffi::c_char) {
    ffi_guard((), || {
        #[cfg(feature = "std")]
        if !s.is_null() {
            drop(std::ffi::CString::from_raw(s));
        }
        #[cfg(not(feature = "std"))]
        let _ = s;
    });
}

/// Panic isolation for the C ABI.
///
/// With `std`: a panic that reaches an `extern "C"` boundary aborts the whole
/// process (Rust 1.81+), taking the host down with it Every exported function
/// therefore runs its body through [`guard::ffi_guard`]: a panic is caught
/// inside the function, its message is stored in a thread-local slot the host
/// reads with `alice_edge_last_error`, and the function returns the caller's
/// sentinel Requires `panic = "unwind"` (the default)
///
/// Without `std` (bare-metal firmware): there is no unwinding runtime, so
/// `ffi_guard` simply runs the body and a panic goes to the crate's panic
/// handler (halt / reset), which is the embedded contract anyway
pub(crate) mod guard {
    #[cfg(feature = "std")]
    use std::cell::RefCell;
    #[cfg(feature = "std")]
    use std::panic::{catch_unwind, AssertUnwindSafe};

    #[cfg(feature = "std")]
    thread_local! {
        static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
    }

    /// Record an error message for `alice_edge_last_error`.
    #[cfg(feature = "std")]
    pub fn set_last_error(msg: impl Into<String>) {
        LAST_ERROR.with(|slot| *slot.borrow_mut() = Some(msg.into()));
    }

    /// Take the most recent error message (leaves the slot empty).
    #[cfg(feature = "std")]
    pub fn take_last_error() -> Option<String> {
        LAST_ERROR.with(|slot| slot.borrow_mut().take())
    }

    /// Clear the most recent error message.
    #[cfg(feature = "std")]
    pub fn clear_last_error() {
        LAST_ERROR.with(|slot| *slot.borrow_mut() = None);
    }

    /// Clear the most recent error message (no slot without `std`).
    #[cfg(not(feature = "std"))]
    pub fn clear_last_error() {}

    /// Run `body`, converting a panic into `default` plus a recorded message.
    ///
    /// The closure is treated as unwind-safe: every FFI body only touches its
    /// arguments and caller-owned buffers, so no partially-updated shared state
    /// is observable afterwards.
    #[cfg(feature = "std")]
    #[inline]
    pub fn ffi_guard<T>(default: T, body: impl FnOnce() -> T) -> T {
        match catch_unwind(AssertUnwindSafe(body)) {
            Ok(v) => v,
            Err(payload) => {
                let msg = payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_string())
                    .or_else(|| payload.downcast_ref::<String>().cloned())
                    .unwrap_or_else(|| "panic with non-string payload".to_string());
                set_last_error(format!("alice_edge FFI panic: {msg}"));
                default
            }
        }
    }

    /// Without `std` there is nothing to catch: run the body directly.
    #[cfg(not(feature = "std"))]
    #[inline]
    pub fn ffi_guard<T>(_default: T, body: impl FnOnce() -> T) -> T {
        body()
    }

    /// Unit-test hook for the panic path of the robust FFI wrappers.
    ///
    /// No input is known to make `fit_linear_robust` or `filter_outliers_mad`
    /// panic in both build profiles, so the `ALICE_EDGE_ERR_PANIC` path cannot
    /// be reached through the data. When armed, the next robust wrapper call
    /// on this thread panics inside its `ffi_guard` closure, at the point
    /// where the fit runs. Compiled only for this crate's unit tests
    /// (`cfg(test)`): library, release and integration-test builds contain
    /// neither the hook nor its call sites.
    #[cfg(all(test, feature = "std"))]
    pub mod panic_hook {
        use std::cell::Cell;

        /// The panic message the hook raises.
        pub const MESSAGE: &str = "alice_edge unit-test hook: forced panic in the robust wrapper";

        thread_local! {
            static ARMED: Cell<bool> = const { Cell::new(false) };
        }

        /// Make the next robust wrapper call on this thread panic.
        pub fn arm() {
            ARMED.with(|a| a.set(true));
        }

        /// Panic once if armed (disarms first).
        pub fn fire_if_armed() {
            if ARMED.with(|a| a.replace(false)) {
                panic!("{MESSAGE}");
            }
        }
    }

    #[cfg(all(test, feature = "std"))]
    mod tests {
        use super::*;

        #[test]
        fn panic_becomes_default_and_message() {
            clear_last_error();
            let v = ffi_guard(-1i32, || -> i32 { panic!("boom {}", 42) });
            assert_eq!(v, -1);
            let msg = take_last_error().expect("message recorded");
            assert!(msg.contains("boom 42"), "{msg}");
            assert!(take_last_error().is_none(), "take clears the slot");
        }

        #[test]
        fn ok_call_leaves_slot_untouched() {
            clear_last_error();
            assert_eq!(ffi_guard(0, || 5), 5);
            assert!(take_last_error().is_none());
        }
    }
}

use guard::ffi_guard;

#[cfg(all(test, feature = "std"))]
mod guard_ffi_tests {
    use super::*;

    /// panic が sentinel + `alice_edge_last_error` の message に変換され、host が
    /// 文字列を取得 / 解放できること (extern "C" 3 本の end-to-end)
    #[test]
    fn last_error_roundtrip_through_c_abi() {
        alice_edge_clear_last_error();
        assert!(alice_edge_last_error().is_null(), "初期状態は null");
        let v = ffi_guard(0usize, || -> usize { panic!("ffi test panic") });
        assert_eq!(v, 0);
        let s = alice_edge_last_error();
        assert!(!s.is_null());
        let msg = unsafe { std::ffi::CStr::from_ptr(s) }
            .to_string_lossy()
            .into_owned();
        assert!(msg.contains("ffi test panic"), "{msg}");
        unsafe { alice_edge_free_error_string(s) };
        assert!(alice_edge_last_error().is_null(), "take で slot は空になる");
    }

    /// The message from `alice_edge_last_error`, read and freed through the
    /// C ABI.
    fn last_error_text() -> Option<String> {
        let s = alice_edge_last_error();
        if s.is_null() {
            return None;
        }
        // SAFETY: a non-null result is a CString owned by the caller.
        let msg = unsafe { std::ffi::CStr::from_ptr(s) }
            .to_string_lossy()
            .into_owned();
        // SAFETY: returned by alice_edge_last_error, freed once.
        unsafe { alice_edge_free_error_string(s) };
        Some(msg)
    }

    /// A panic inside the robust fit comes back as `ALICE_EDGE_ERR_PANIC`
    /// with the panic message, not as `ALICE_EDGE_OK`; `*out` holds the
    /// documented `{0, 0}` (written before the guarded body runs), not a fit.
    #[test]
    fn robust_fit_panic_is_reported_as_err_panic() {
        let data = [100, 200, 300, 9999, 500, 600, 700];
        alice_edge_clear_last_error();
        guard::panic_hook::arm();
        let mut out = AliceLinearResult {
            slope: 1,
            intercept: 1,
            semantics_id: crate::EDGE_SEMANTICS_ID,
        };
        // SAFETY: data is a valid local slice, out a valid local struct.
        let status = unsafe { alice_fit_linear_robust(data.as_ptr(), data.len(), 3, &mut out) };
        assert_eq!(status, ALICE_EDGE_ERR_PANIC);
        assert_eq!(
            (out.slope, out.intercept),
            (0, 0),
            "out holds the zeroed result"
        );
        let msg = last_error_text().expect("panic message recorded");
        assert!(!msg.is_empty());
        assert!(msg.contains(guard::panic_hook::MESSAGE), "{msg}");

        // the hook fired once: the next call is the normal fit
        // SAFETY: as above.
        let status = unsafe { alice_fit_linear_robust(data.as_ptr(), data.len(), 3, &mut out) };
        assert_eq!(status, ALICE_EDGE_OK);
        assert_eq!(
            Ok((out.slope, out.intercept)),
            crate::fit_linear_robust(&data, 3)
        );
        assert!(
            last_error_text().is_none(),
            "a successful call records nothing"
        );
    }

    /// The same for the MAD filter: `ALICE_EDGE_ERR_PANIC`, the message, `*out_written`
    /// 0 and `out` not written.
    #[test]
    fn filter_panic_is_reported_as_err_panic() {
        let data = [100, 200, 300, 9999, 500];
        alice_edge_clear_last_error();
        guard::panic_hook::arm();
        let mut out = [7i32; 5];
        let mut written = usize::MAX;
        // SAFETY: data / out are valid local slices, written a valid local.
        let status = unsafe {
            alice_filter_outliers_mad(
                data.as_ptr(),
                data.len(),
                2,
                out.as_mut_ptr(),
                out.len(),
                &mut written,
            )
        };
        assert_eq!(status, ALICE_EDGE_ERR_PANIC);
        assert_eq!(written, 0);
        assert_eq!(out, [7; 5], "out must not be written");
        let msg = last_error_text().expect("panic message recorded");
        assert!(msg.contains(guard::panic_hook::MESSAGE), "{msg}");
    }
}
