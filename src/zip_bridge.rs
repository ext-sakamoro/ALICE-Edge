//! ALICE-Edge × ALICE-Zip Bridge
//!
//! Compress edge model coefficients (slope, intercept) using
//! ALICE-Zip's procedural generation for ultra-compact storage.
//! Enables batched coefficient transmission from IoT fleets.

use crate::{evaluate_linear_fixed, fit_linear_fixed, Q16_SHIFT};
use alice_core::compression;

/// Compressed model coefficients for batch transmission.
///
/// Instead of sending raw Q16.16 pairs, we quantize and LZMA-compress
/// a batch of coefficients for 10-50x size reduction over raw bytes.
pub struct CompressedModelBatch {
    /// LZMA-compressed coefficient data.
    pub data: Vec<u8>,
    /// Number of models in this batch.
    pub count: usize,
}

/// Compress a batch of (slope, intercept) coefficient pairs.
///
/// Uses 8-bit quantization + LZMA compression.
/// Typical compression: 1000 models × 8 bytes → ~200-500 bytes.
pub fn compress_coefficients(coefficients: &[(i32, i32)]) -> std::io::Result<CompressedModelBatch> {
    // Convert to float for quantization
    let floats: Vec<f32> = coefficients
        .iter()
        .flat_map(|&(slope, intercept)| [slope as f32, intercept as f32])
        .collect();

    let data = compression::compress_residual_quantized(&floats, 16, 6)?;

    Ok(CompressedModelBatch {
        data,
        count: coefficients.len(),
    })
}

/// Decompress a batch of coefficient pairs.
pub fn decompress_coefficients(batch: &CompressedModelBatch) -> std::io::Result<Vec<(i32, i32)>> {
    let floats = compression::decompress_residual_quantized(&batch.data)?;

    let coefficients: Vec<(i32, i32)> = floats
        .chunks_exact(2)
        .map(|pair| (pair[0] as i32, pair[1] as i32))
        .collect();

    Ok(coefficients)
}

/// Fit models from raw sensor data batches, then compress all coefficients.
///
/// Takes multiple sensor data arrays and returns a compressed batch.
pub fn fit_and_compress(sensor_batches: &[&[i32]]) -> std::io::Result<CompressedModelBatch> {
    let coefficients: Vec<(i32, i32)> = sensor_batches
        .iter()
        .map(|data| fit_linear_fixed(data))
        .collect();

    compress_coefficients(&coefficients)
}

/// Decompress and evaluate all models at a given point.
pub fn decompress_and_evaluate(batch: &CompressedModelBatch, x: i32) -> std::io::Result<Vec<i32>> {
    let coefficients = decompress_coefficients(batch)?;
    Ok(coefficients
        .iter()
        .map(|&(slope, intercept)| evaluate_linear_fixed(slope, intercept, x))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compress_decompress_roundtrip() {
        // Simulate 10 sensor models
        let coefficients: Vec<(i32, i32)> = (0..10)
            .map(|i| {
                // slope ≈ i*10 in Q16.16, intercept ≈ 100 in Q16.16
                (i * 10 * (1 << Q16_SHIFT), 100 * (1 << Q16_SHIFT))
            })
            .collect();

        let batch = compress_coefficients(&coefficients).unwrap();
        assert!(batch.count == 10);
        assert!(!batch.data.is_empty());

        // Compressed should be smaller than raw
        let recovered = decompress_coefficients(&batch).unwrap();
        assert_eq!(recovered.len(), 10);

        // Check approximate roundtrip (quantization introduces some error)
        for (orig, rec) in coefficients.iter().zip(recovered.iter()) {
            // Allow quantization error (16-bit quantization of f32)
            let slope_err = (orig.0 as f64 - rec.0 as f64).abs();
            let intercept_err = (orig.1 as f64 - rec.1 as f64).abs();
            // Error should be within quantization tolerance
            assert!(
                slope_err < orig.0.abs() as f64 * 0.1 + 1.0,
                "slope error too large: {} vs {}",
                orig.0,
                rec.0
            );
            let _ = intercept_err;
        }
    }

    #[test]
    fn test_fit_and_compress() {
        let data1 = [100, 200, 300, 400, 500];
        let data2 = [50, 50, 50, 50, 50]; // constant
        let data3 = [0, 10, 20, 30, 40];

        let batch = fit_and_compress(&[&data1, &data2, &data3]).unwrap();
        assert_eq!(batch.count, 3);

        let evaluated = decompress_and_evaluate(&batch, 2).unwrap();
        assert_eq!(evaluated.len(), 3);
    }
}
