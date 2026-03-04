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

// Robust linear fit with MAD-based outlier removal.
// mad_k: outlier sensitivity in Q16.16 (typically int_to_q16(2) or int_to_q16(3)).
AliceLinearResult alice_fit_linear_robust(const int32_t* data, size_t len, int32_t mad_k);

// SIMD-accelerated linear fit (auto-fallback on small data).
AliceLinearResult alice_fit_linear_simd(const int32_t* data, size_t len);

// ── Filter / Delta encoding ────────────────────────────────────────────

// Filter outliers using MAD. Writes result to out, returns count written.
size_t alice_filter_outliers_mad(const int32_t* data, size_t len, int32_t k,
                                 int32_t* out, size_t out_capacity);

// Delta-encode coefficient pairs in-place. Layout: [s0,i0,s1,i1,...].
size_t alice_delta_encode(int32_t* pairs, size_t num_pairs);

// Delta-decode coefficient pairs in-place.
size_t alice_delta_decode(int32_t* pairs, size_t num_pairs);

// ── Security ────────────────────────────────────────────────────────────

// Securely zero a buffer (volatile write, not optimized away).
void alice_zeroize(int32_t* buf, size_t len);

// Return library version as null-terminated C string.
const char* alice_edge_version(void);

#ifdef __cplusplus
}
#endif

#endif // ALICE_EDGE_H
