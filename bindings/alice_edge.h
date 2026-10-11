// SPDX-License-Identifier: MIT
// ALICE-Edge C/C++ Header — Auto-generated from ffi.rs
// Author: Moroya Sakamoto

#ifndef ALICE_EDGE_H
#define ALICE_EDGE_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ── Result structs ──────────────────────────────────────────────────────

typedef struct {
    int32_t slope;
    int32_t intercept;
} AliceLinearResult;

typedef struct {
    int32_t a;
    int32_t b;
    int32_t c;
} AliceQuadraticResult;

typedef struct {
    int32_t a;
    int32_t b;
    int32_t c;
    int32_t d;
} AliceCubicResult;

// ── Status codes ────────────────────────────────────────────────────────
//
// Returned by alice_fit_linear_robust and alice_filter_outliers_mad. On any
// status other than ALICE_EDGE_OK the message is available through
// alice_edge_last_error() (std builds).

#define ALICE_EDGE_OK                  0   // success
#define ALICE_EDGE_ERR_NULL_POINTER   (-1) // a required pointer was null
#define ALICE_EDGE_ERR_NON_POSITIVE_K (-2) // outlier factor k <= 0
#define ALICE_EDGE_ERR_PANIC          (-3) // the call panicked
#define ALICE_EDGE_ERR_WINDOW_TOO_LONG (-4) // len > ALICE_EDGE_ROBUST_MAX_N

// Longest window alice_fit_linear_robust and alice_filter_outliers_mad accept.
#define ALICE_EDGE_ROBUST_MAX_N 1024

// ── Linear fitting ──────────────────────────────────────────────────────

// Fit a linear model to sensor data (Q16.16 fixed-point).
// data: pointer to i32 array, len: number of elements.
AliceLinearResult alice_fit_linear(const int32_t* data, size_t len);

// Evaluate y = slope * x + intercept (all Q16.16, x is raw integer index).
int32_t alice_evaluate_linear(int32_t slope, int32_t intercept, int32_t x);

// ── Quadratic fitting ───────────────────────────────────────────────────

// Fit a quadratic model y = ax^2 + bx + c (Q16.16).
AliceQuadraticResult alice_fit_quadratic(const int32_t* data, size_t len);

// Evaluate y = ax^2 + bx + c (Q16.16, x is raw integer index).
int32_t alice_evaluate_quadratic(int32_t a, int32_t b, int32_t c, int32_t x);

// ── Cubic fitting ───────────────────────────────────────────────────────

// Fit a cubic model y = ax^3 + bx^2 + cx + d (Q16.16).
AliceCubicResult alice_fit_cubic(const int32_t* data, size_t len);

// Evaluate y = ax^3 + bx^2 + cx + d (Q16.16, x is raw integer index).
int32_t alice_evaluate_cubic(int32_t a, int32_t b, int32_t c, int32_t d, int32_t x);

// ── Constant fitting ────────────────────────────────────────────────────

// Fit a constant model (mean) to sensor data. Returns Q16.16.
int32_t alice_fit_constant(const int32_t* data, size_t len);

// ── Utility functions ───────────────────────────────────────────────────

// Convert integer to Q16.16 fixed-point.
int32_t alice_int_to_q16(int32_t i);

// Convert Q16.16 fixed-point to integer (truncates fractional part).
int32_t alice_q16_to_int(int32_t q);

// Convert Q16.16 fixed-point to float.
float alice_q16_to_f32(int32_t q);

// Check if data benefits from linear model vs constant.
bool alice_should_use_linear(const int32_t* data, size_t len);

// Compute residual error of a linear fit (returns Q32.32).
int64_t alice_residual_error(const int32_t* data, size_t len,
                             int32_t slope, int32_t intercept);

// ── Robust / SIMD fitting ───────────────────────────────────────────────

// Robust linear fit: repeated-median start, MAD outlier rejection on the
// residuals (scale floor of one count), least-squares refit of the kept
// samples until the outlier set is stable.
// k: outlier factor, a PLAIN INTEGER multiplier of the MAD (3 = three MADs;
//    not Q16.16). k <= 0 returns ALICE_EDGE_ERR_NON_POSITIVE_K.
// len > ALICE_EDGE_ROBUST_MAX_N (1024) returns ALICE_EDGE_ERR_WINDOW_TOO_LONG.
// out: receives the fit (Q16.16 slope / intercept); set to {0, 0} on error.
// len == 0 (data may be NULL) is the plain fit of
// nothing, ALICE_EDGE_OK.
// Returns ALICE_EDGE_OK, ALICE_EDGE_ERR_NON_POSITIVE_K,
// ALICE_EDGE_ERR_WINDOW_TOO_LONG,
// ALICE_EDGE_ERR_NULL_POINTER (out NULL, or data NULL with len > 0) or
// ALICE_EDGE_ERR_PANIC.
int32_t alice_fit_linear_robust(const int32_t* data, size_t len, int32_t k,
                                AliceLinearResult* out);

// SIMD-accelerated linear fit (auto-fallback on small data).
AliceLinearResult alice_fit_linear_simd(const int32_t* data, size_t len);

// ── Filter / Delta encoding ────────────────────────────────────────────

// Filter outliers using MAD: x is replaced by the median when
// |x - median| > k * max(MAD, 1). Writes min(len, out_capacity) elements to out and
// that count to *out_written.
// k: outlier factor, a PLAIN INTEGER multiplier of the MAD (3 = three MADs;
//    not Q16.16). k <= 0 returns ALICE_EDGE_ERR_NON_POSITIVE_K.
// len > ALICE_EDGE_ROBUST_MAX_N (1024) returns ALICE_EDGE_ERR_WINDOW_TOO_LONG.
// Fewer than 3 samples pass the window through unchanged. On error *out_written is 0 and out is not written.
// Returns ALICE_EDGE_OK, ALICE_EDGE_ERR_NON_POSITIVE_K,
// ALICE_EDGE_ERR_WINDOW_TOO_LONG,
// ALICE_EDGE_ERR_NULL_POINTER (out_written NULL, data NULL with len > 0, or
// out NULL with out_capacity > 0) or ALICE_EDGE_ERR_PANIC.
int32_t alice_filter_outliers_mad(const int32_t* data, size_t len, int32_t k,
                                  int32_t* out, size_t out_capacity,
                                  size_t* out_written);

// Delta-encode coefficient pairs in-place. Layout: [s0,i0,s1,i1,...].
size_t alice_delta_encode(int32_t* pairs, size_t num_pairs);

// Delta-decode coefficient pairs in-place.
size_t alice_delta_decode(int32_t* pairs, size_t num_pairs);

// ── Security ────────────────────────────────────────────────────────────

// Securely zero a buffer (volatile write, not optimized away).
void alice_zeroize(int32_t* buf, size_t len);

// Return library version as null-terminated C string.
const char* alice_edge_version(void);

// ── Errors ──────────────────────────────────────────────────────────────

// Message of the most recent error on this thread (a caught panic, or an
// argument refused with a status code), or NULL if none. Owned by the
// caller: release it with alice_edge_free_error_string. Always NULL in
// no_std builds.
char* alice_edge_last_error(void);

// Clear the most recent error message.
void alice_edge_clear_last_error(void);

// Release a string returned by alice_edge_last_error (NULL is accepted).
void alice_edge_free_error_string(char* s);

#ifdef __cplusplus
}
#endif

#endif // ALICE_EDGE_H
