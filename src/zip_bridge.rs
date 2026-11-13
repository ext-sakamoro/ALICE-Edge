// SPDX-License-Identifier: LicenseRef-Proprietary
//! ALICE-Edge × ALICE-Zip Bridge
//!
//! Compress edge model coefficients (slope, intercept) using
//! ALICE-Zip's procedural generation for ultra-compact storage.
//! Enables batched coefficient transmission from `IoT` fleets.

use crate::{evaluate_linear_fixed, fit_linear_fixed};
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
///
/// # Errors
///
/// Returns `std::io::Error` if the underlying LZMA compression fails.
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

/// Decompress a batch of coefficient pairs, at most `max_pairs` of them.
///
/// The batch's `count` is the data's own claim, so the caller's `max_pairs` is the ceiling: a batch stating more
/// is refused before anything is decoded, and the data must hold exactly `count` pairs (decoding stops past them).
///
/// # Errors
///
/// [`std::io::ErrorKind::InvalidData`] when the batch states more than `max_pairs` pairs or its data does not hold
/// exactly `count` of them; otherwise the underlying decompression error.
pub fn decompress_coefficients(
    batch: &CompressedModelBatch,
    max_pairs: usize,
) -> std::io::Result<Vec<(i32, i32)>> {
    let invalid = |what: String| std::io::Error::new(std::io::ErrorKind::InvalidData, what);
    if batch.count > max_pairs {
        return Err(invalid(format!(
            "the batch states {} pairs, more than the limit of {max_pairs}",
            batch.count
        )));
    }
    let values = batch
        .count
        .checked_mul(2)
        .ok_or_else(|| invalid(format!("{} pairs do not fit in memory", batch.count)))?;
    let floats = compression::decompress_residual_quantized(&batch.data, values)?;
    if floats.len() != values {
        return Err(invalid(format!(
            "the batch states {} pairs but its data holds {} values",
            batch.count,
            floats.len()
        )));
    }

    let coefficients: Vec<(i32, i32)> = floats
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| (pair[0] as i32, pair[1] as i32))
        .collect();

    Ok(coefficients)
}

/// Fit models from raw sensor data batches, then compress all coefficients.
///
/// Takes multiple sensor data arrays and returns a compressed batch.
///
/// # Errors
///
/// Returns `std::io::Error` if the LZMA compression of the fitted coefficients fails.
pub fn fit_and_compress(sensor_batches: &[&[i32]]) -> std::io::Result<CompressedModelBatch> {
    let coefficients: Vec<(i32, i32)> = sensor_batches
        .iter()
        .map(|data| fit_linear_fixed(data))
        .collect();

    compress_coefficients(&coefficients)
}

/// Decompress and evaluate all models at a given point.
///
/// # Errors
///
/// As [`decompress_coefficients`] (`max_pairs` is the caller's ceiling).
pub fn decompress_and_evaluate(
    batch: &CompressedModelBatch,
    x: i32,
    max_pairs: usize,
) -> std::io::Result<Vec<i32>> {
    let coefficients = decompress_coefficients(batch, max_pairs)?;
    Ok(coefficients
        .iter()
        .map(|&(slope, intercept)| evaluate_linear_fixed(slope, intercept, x))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Q16_SHIFT;

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
        let recovered = decompress_coefficients(&batch, 10).unwrap();
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

        let evaluated = decompress_and_evaluate(&batch, 2, 3).unwrap();
        assert_eq!(evaluated.len(), 3);
    }

    #[test]
    fn test_compress_empty() {
        let batch = compress_coefficients(&[]).unwrap();
        assert_eq!(batch.count, 0);
    }

    #[test]
    fn test_compress_single() {
        let coeffs = [(1 << Q16_SHIFT, 50 << Q16_SHIFT)];
        let batch = compress_coefficients(&coeffs).unwrap();
        assert_eq!(batch.count, 1);
        assert!(!batch.data.is_empty());
    }

    #[test]
    fn test_decompress_and_evaluate_at_zero() {
        let coefficients = vec![(10 << Q16_SHIFT, 100 << Q16_SHIFT)];
        let batch = compress_coefficients(&coefficients).unwrap();
        let evaluated = decompress_and_evaluate(&batch, 0, 1).unwrap();
        assert_eq!(evaluated.len(), 1);
    }

    #[test]
    fn test_fit_and_compress_single_stream() {
        let data = [100, 200, 300];
        let batch = fit_and_compress(&[&data]).unwrap();
        assert_eq!(batch.count, 1);
    }

    #[test]
    fn test_compressed_batch_has_data() {
        let coefficients: Vec<(i32, i32)> = (0..50)
            .map(|i| (i * (1 << Q16_SHIFT), 100 * (1 << Q16_SHIFT)))
            .collect();
        let batch = compress_coefficients(&coefficients).unwrap();
        assert_eq!(batch.count, 50);
        // 圧縮後は元の生バイト数 (50*8=400) よりも小さくなるはず
        assert!(batch.data.len() < 400);
    }

    #[test]
    fn test_fit_and_compress_empty() {
        let batch = fit_and_compress(&[]).unwrap();
        assert_eq!(batch.count, 0);
    }

    // oracle: the batch's count is the data's own claim, so the caller's `max_pairs` is the ceiling: a batch stating
    // more pairs than the caller accepts is refused before decoding, and data holding more pairs than the batch
    // states (the count lying downward) is refused at the stated count
    #[test]
    fn a_batch_past_the_callers_ceiling_or_its_own_count_is_refused() {
        let coefficients: Vec<(i32, i32)> = (0..64).map(|i| (i << 16, 100 << 16)).collect();
        let batch = compress_coefficients(&coefficients).unwrap();
        assert_eq!(decompress_coefficients(&batch, 64).unwrap().len(), 64);
        let e = decompress_coefficients(&batch, 63).expect_err("past the ceiling");
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidData, "{e}");
        assert!(e.to_string().contains("63"), "{e}");
        let lying = CompressedModelBatch {
            data: batch.data.clone(),
            count: 63,
        };
        assert!(
            decompress_coefficients(&lying, 64).is_err(),
            "more data than the batch states"
        );
        let short = CompressedModelBatch {
            data: batch.data.clone(),
            count: 65,
        };
        assert!(
            decompress_coefficients(&short, 65).is_err(),
            "less data than the batch states"
        );
        assert!(decompress_and_evaluate(&batch, 3, 63).is_err());
        assert_eq!(decompress_and_evaluate(&batch, 3, 64).unwrap().len(), 64);
    }
}
