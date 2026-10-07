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
//! ```rust
//! use alice_edge::{evaluate_linear_fixed, fit_linear_fixed, q16_to_int};
//!
//! // sensor readings (temperature × 100): 25.00 °C rising 0.10 °C per sample
//! let samples = [2500, 2510, 2520, 2530, 2540];
//!
//! // on the device: two Q16.16 numbers instead of five samples
//! let (slope, intercept) = fit_linear_fixed(&samples);
//! assert_eq!((slope, intercept), (10 << 16, 2500 << 16));
//!
//! // on the receiver: reconstruct the value at any sample index
//! let y3 = evaluate_linear_fixed(slope, intercept, 3);
//! assert_eq!(q16_to_int(y3), 2530);
//! ```
//!
//! With the `law` feature, `law::linear_law` returns the same fit together
//! with its evidence, measured residual, valid range and provenance.
//!
//! # Related Projects
//!
//! - [ALICE-Zip](https://github.com/ext-sakamoro/ALICE-Zip) - Full compression engine
//! - [ALICE-DB](https://github.com/ext-sakamoro/ALICE-DB) - Model-based database
//! - [ALICE-Streaming-Protocol](https://github.com/ext-sakamoro/ALICE-Streaming-Protocol) - Video streaming

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
#[cfg(feature = "law")]
pub mod law;
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

// D-split modules (extracted from monolithic lib.rs).
pub mod adaptive_polyfit;
pub mod constant_fit;
#[cfg(feature = "std")]
pub mod delta;
#[cfg(feature = "ml")]
pub mod inference_trait;
pub mod piecewise;
pub mod q16_linear;
#[cfg(feature = "std")]
pub mod robust;
pub mod simd_fit;

#[cfg(test)]
mod integration_tests;

// Backward-compat re-exports.
pub use crate::adaptive_polyfit::*;
pub use crate::constant_fit::*;
#[cfg(feature = "std")]
pub use crate::delta::*;
#[cfg(feature = "ml")]
pub use crate::inference_trait::*;
pub use crate::piecewise::*;
pub use crate::q16_linear::*;
#[cfg(feature = "std")]
pub use crate::robust::*;
pub use crate::simd_fit::*;
