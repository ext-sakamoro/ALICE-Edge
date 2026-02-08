//! PyO3 Python Bindings for ALICE-Edge
//!
//! Embedded Model Generator for IoT/Raspberry Pi.
//! Fixed-point least squares fitting exposed to Python + NumPy.

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};

// ============================================================================
// Core Functions
// ============================================================================

/// Fit a linear model y = mx + b using fixed-point least squares.
///
/// Returns (slope, intercept) in Q16.16 format.
/// Input: numpy array of int32 sensor readings.
#[pyfunction]
fn fit_linear<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, i32>,
) -> PyResult<(i32, i32)> {
    let slice = data.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(py.detach(|| crate::fit_linear_fixed(slice)))
}

/// Fit a constant model (mean) in Q16.16 format.
#[pyfunction]
fn fit_constant<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, i32>,
) -> PyResult<i32> {
    let slice = data.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(py.detach(|| crate::fit_constant_fixed(slice)))
}

/// Evaluate y = slope * x + intercept in Q16.16.
#[pyfunction]
fn evaluate_linear(slope: i32, intercept: i32, x: i32) -> i32 {
    crate::evaluate_linear_fixed(slope, intercept, x)
}

/// Compute residual error (sum of squared differences).
#[pyfunction]
fn compute_error<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, i32>,
    slope: i32,
    intercept: i32,
) -> PyResult<i64> {
    let slice = data.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(py.detach(|| crate::compute_residual_error(slice, slope, intercept)))
}

/// Model selection: returns True if linear model is significantly better.
#[pyfunction]
fn should_use_linear<'py>(
    py: Python<'py>,
    data: PyReadonlyArray1<'py, i32>,
) -> PyResult<bool> {
    let slice = data.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(py.detach(|| crate::should_use_linear(slice)))
}

// ============================================================================
// Batch API (multiple sensor streams, GIL released)
// ============================================================================

/// Fit linear models for multiple sensor streams at once.
///
/// Input: (num_streams, num_samples) int32 array
/// Output: (num_streams, 2) int32 array — columns are [slope, intercept] in Q16.16
#[pyfunction]
fn fit_linear_batch<'py>(
    py: Python<'py>,
    streams: PyReadonlyArray2<'py, i32>,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let shape = streams.as_array();
    let num_streams = shape.shape()[0];
    let num_samples = shape.shape()[1];
    let slice = streams.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;

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
/// coeffs: (num_streams, 2) — [slope, intercept] per stream
/// x_values: (num_points,) — x positions to evaluate
/// Output: (num_streams, num_points) — predicted y in Q16.16
#[pyfunction]
fn evaluate_linear_batch<'py>(
    py: Python<'py>,
    coeffs: PyReadonlyArray2<'py, i32>,
    x_values: PyReadonlyArray1<'py, i32>,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let c_shape = coeffs.as_array();
    let num_streams = c_shape.shape()[0];
    let c_slice = coeffs.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let x_slice = x_values.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
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
    q as f32 / crate::Q16_ONE as f32
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
fn q16_to_f32_batch<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, i32>,
) -> PyResult<Bound<'py, PyArray1<f32>>> {
    let s = values.as_slice().map_err(|e| PyValueError::new_err(e.to_string()))?;
    let scale = crate::Q16_ONE as f32;
    let result = py.detach(|| s.iter().map(|&q| q as f32 / scale).collect::<Vec<f32>>());
    Ok(result.into_pyarray(py))
}

// ============================================================================
// Module
// ============================================================================

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
