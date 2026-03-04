// SPDX-License-Identifier: MIT
//! `PyO3` Python Bindings for ALICE-Edge
//!
//! Embedded Model Generator for IoT/Raspberry Pi.
//! Fixed-point least squares fitting exposed to Python + `NumPy`.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

// ============================================================================
// Core Functions
// ============================================================================

/// Fit a linear model y = mx + b using fixed-point least squares.
///
/// Returns (slope, intercept) in Q16.16 format.
/// Input: numpy array of int32 sensor readings.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn fit_linear<'py>(py: Python<'py>, data: PyReadonlyArray1<'py, i32>) -> PyResult<(i32, i32)> {
    let slice = data
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(py.detach(|| crate::fit_linear_fixed(slice)))
}

/// Fit a constant model (mean) in Q16.16 format.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn fit_constant<'py>(py: Python<'py>, data: PyReadonlyArray1<'py, i32>) -> PyResult<i32> {
    let slice = data
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(py.detach(|| crate::fit_constant_fixed(slice)))
}

/// Evaluate y = slope * x + intercept in Q16.16.
#[pyfunction]
fn evaluate_linear(slope: i32, intercept: i32, x: i32) -> i32 {
    crate::evaluate_linear_fixed(slope, intercept, x)
}

/// Compute residual error (sum of squared differences).
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn compute_error<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, i32>,
    slope: i32,
    intercept: i32,
) -> PyResult<i64> {
    let slice = data
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(py.detach(|| crate::compute_residual_error(slice, slope, intercept)))
}

/// Model selection: returns True if linear model is significantly better.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn should_use_linear<'py>(py: Python<'py>, data: PyReadonlyArray1<'py, i32>) -> PyResult<bool> {
    let slice = data
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(py.detach(|| crate::should_use_linear(slice)))
}

// ============================================================================
// Batch API (multiple sensor streams, GIL released)
// ============================================================================

/// Fit linear models for multiple sensor streams at once.
///
/// Input: (`num_streams`, `num_samples`) int32 array
/// Output: (`num_streams`, 2) int32 array — columns are [slope, intercept] in Q16.16
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn fit_linear_batch<'py>(
    py: Python<'py>,
    streams: PyReadonlyArray2<'py, i32>,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let shape = streams.as_array();
    let num_streams = shape.shape()[0];
    let num_samples = shape.shape()[1];
    let slice = streams
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;

    let result = py.detach(|| {
        let mut out = Vec::with_capacity(num_streams * 2);
        for i in 0..num_streams {
            let row = &slice[i * num_samples..(i + 1) * num_samples];
            let (slope, intercept) = crate::fit_linear_fixed(row);
            out.push(slope);
            out.push(intercept);
        }
        out
    });

    let arr = Array2::from_shape_vec((num_streams, 2), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// Evaluate linear models for multiple streams at multiple x positions.
///
/// coeffs: (`num_streams`, 2) — [slope, intercept] per stream
/// `x_values`: (`num_points`,) — x positions to evaluate
/// Output: (`num_streams`, `num_points`) — predicted y in Q16.16
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn evaluate_linear_batch<'py>(
    py: Python<'py>,
    coeffs: PyReadonlyArray2<'py, i32>,
    x_values: PyReadonlyArray1<'py, i32>,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let c_shape = coeffs.as_array();
    let num_streams = c_shape.shape()[0];
    let c_slice = coeffs
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let x_slice = x_values
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let num_points = x_slice.len();

    let result = py.detach(|| {
        let mut out = Vec::with_capacity(num_streams * num_points);
        for i in 0..num_streams {
            let slope = c_slice[i * 2];
            let intercept = c_slice[i * 2 + 1];
            for &x in x_slice {
                out.push(crate::evaluate_linear_fixed(slope, intercept, x));
            }
        }
        out
    });

    let arr = Array2::from_shape_vec((num_streams, num_points), result)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

// ============================================================================
// Conversion Utilities
// ============================================================================

/// Convert Q16.16 fixed-point to float.
#[pyfunction]
fn q16_to_f32(q: i32) -> f32 {
    const INV_Q16_ONE: f32 = 1.0 / (1i32 << 16) as f32;
    q as f32 * INV_Q16_ONE
}

/// Convert integer to Q16.16 fixed-point.
#[pyfunction]
fn int_to_q16(i: i32) -> i32 {
    crate::int_to_q16(i)
}

/// Convert Q16.16 to integer (truncate).
#[pyfunction]
fn q16_to_int(q: i32) -> i32 {
    crate::q16_to_int(q)
}

/// Batch convert Q16.16 array to float array.
#[pyfunction]
#[allow(clippy::needless_pass_by_value)]
fn q16_to_f32_batch<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, i32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let s = values
        .as_slice()
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    let inv_scale = 1.0 / crate::Q16_ONE as f32;
    let result = py.detach(|| {
        s.iter()
            .map(|&q| q as f32 * inv_scale)
            .collect::<Vec<f32>>()
    });
    Ok(result.into_pyarray(py))
}

// ============================================================================
// Module
// ============================================================================

/// # Errors
///
/// Returns `PyResult::Err` if any function or constant fails to register with the Python module.
#[pymodule]
pub fn alice_edge(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core
    m.add_function(wrap_pyfunction!(fit_linear, m)?)?;
    m.add_function(wrap_pyfunction!(fit_constant, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_linear, m)?)?;
    m.add_function(wrap_pyfunction!(compute_error, m)?)?;
    m.add_function(wrap_pyfunction!(should_use_linear, m)?)?;

    // Batch API
    m.add_function(wrap_pyfunction!(fit_linear_batch, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_linear_batch, m)?)?;

    // Conversions
    m.add_function(wrap_pyfunction!(q16_to_f32, m)?)?;
    m.add_function(wrap_pyfunction!(int_to_q16, m)?)?;
    m.add_function(wrap_pyfunction!(q16_to_int, m)?)?;
    m.add_function(wrap_pyfunction!(q16_to_f32_batch, m)?)?;

    // Constants
    m.add("Q16_SHIFT", crate::Q16_SHIFT)?;
    m.add("Q16_ONE", crate::Q16_ONE)?;

    Ok(())
}

// PyO3テストは Python ランタイムが必要なため、ラップ先のRustコアロジックを
// 直接テストする。PyO3関数は薄いラッパーなので、コア関数の正確性を確認。
#[cfg(test)]
mod tests {
    #[test]
    fn test_fit_linear_core() {
        let data = [100, 200, 300, 400, 500];
        let (slope, intercept) = crate::fit_linear_fixed(&data);
        assert_ne!(slope, 0);
        assert_ne!(intercept, 0);
    }

    #[test]
    fn test_fit_constant_core() {
        let data = [500, 500, 500, 500];
        let mean = crate::fit_constant_fixed(&data);
        assert_eq!(crate::q16_to_int(mean), 500);
    }

    #[test]
    fn test_evaluate_linear_core() {
        let val = crate::evaluate_linear_fixed(crate::int_to_q16(10), crate::int_to_q16(5), 3);
        assert_eq!(crate::q16_to_int(val), 35);
    }

    #[test]
    fn test_compute_error_core() {
        let data = [100, 200, 300, 400, 500];
        let (slope, intercept) = crate::fit_linear_fixed(&data);
        let err = crate::compute_residual_error(&data, slope, intercept);
        assert!(err < 100); // 完全な線形データは残差ほぼ0
    }

    #[test]
    fn test_should_use_linear_core() {
        let rising = [100, 200, 300, 400, 500];
        let constant = [500, 500, 500, 500, 500];
        assert!(crate::should_use_linear(&rising));
        assert!(!crate::should_use_linear(&constant));
    }

    #[test]
    fn test_q16_conversion_roundtrip() {
        for val in [-100, -1, 0, 1, 42, 1000] {
            assert_eq!(crate::q16_to_int(crate::int_to_q16(val)), val);
        }
    }

    #[test]
    fn test_q16_to_f32_accuracy() {
        let q = crate::int_to_q16(10);
        let f = crate::q16_to_f32(q);
        assert!((f - 10.0).abs() < 0.001);
    }

    #[test]
    fn test_batch_fit_logic() {
        // バッチAPIのコアロジック: 複数ストリームを順次フィット
        let streams: Vec<Vec<i32>> = vec![vec![100, 200, 300], vec![500, 500, 500]];
        let mut results = Vec::new();
        for stream in &streams {
            let (s, i) = crate::fit_linear_fixed(stream);
            results.push((s, i));
        }
        assert_ne!(results[0].0, 0); // 上昇データ: slope != 0
        assert_eq!(crate::q16_to_int(results[1].0), 0); // 定数データ: slope ≈ 0
    }
}
