//! ALICE-Codec bridge: Wavelet denoising for sensor data
//!
//! Applies integer wavelet denoising to noisy sensor readings
//! before linear model fitting, improving compression accuracy
//! for sensors with high noise floors.
//!
//! # Pipeline
//!
//! ```text
//! Noisy Sensor Data → Wavelet1D Forward → Threshold High-Freq → Inverse → fit_linear_fixed
//! ```
//!
//! # Example
//!
//! ```ignore
//! use alice_edge::codec_bridge::fit_denoised;
//!
//! // Noisy temperature readings (25.0°C rising, with noise)
//! let noisy = [2500, 2515, 2508, 2522, 2530, 2548, 2539, 2560];
//! let (slope, intercept) = fit_denoised(&noisy, 50);
//! // slope/intercept are cleaner than fit_linear_fixed(&noisy)
//! ```

use alice_codec::Wavelet1D;
use crate::fit_linear_fixed;

/// Denoise sensor data with Wavelet1D, then fit a linear model.
///
/// # Arguments
///
/// * `data` - Raw sensor readings (i32)
/// * `threshold` - Wavelet coefficient threshold; coefficients with
///   absolute value below this are zeroed (denoised). Higher values
///   remove more noise but may lose signal detail.
///
/// # Returns
///
/// `(slope, intercept)` in Q16.16 fixed-point, same as `fit_linear_fixed`.
pub fn fit_denoised(data: &[i32], threshold: i32) -> (i32, i32) {
    if data.len() < 4 {
        // Too short for meaningful wavelet transform; fall back
        return fit_linear_fixed(data);
    }

    let denoised = wavelet_denoise(data, threshold);
    fit_linear_fixed(&denoised)
}

/// Apply Wavelet1D denoising to a sensor data slice.
///
/// Performs forward CDF 5/3 wavelet transform, thresholds
/// high-frequency coefficients, then inverse-transforms.
///
/// The input is padded to a power of 2 internally and the
/// result is truncated back to the original length.
pub fn wavelet_denoise(data: &[i32], threshold: i32) -> Vec<i32> {
    let n = data.len();
    if n < 2 {
        return data.to_vec();
    }

    // Pad to power of 2
    let padded_len = n.next_power_of_two();
    let mut buf = Vec::with_capacity(padded_len);
    buf.extend_from_slice(data);
    // Mirror-pad to reduce boundary artifacts
    for i in 0..(padded_len - n) {
        let idx = n - 1 - (i % n);
        buf.push(data[idx]);
    }

    // Forward wavelet
    let wavelet = Wavelet1D::cdf53();
    wavelet.forward(&mut buf);

    // Threshold high-frequency half (detail coefficients)
    let half = padded_len / 2;
    let abs_threshold = threshold.unsigned_abs();
    for coeff in buf[half..].iter_mut() {
        if (*coeff as i64).unsigned_abs() < abs_threshold as u64 {
            *coeff = 0;
        }
    }

    // Inverse wavelet
    wavelet.inverse(&mut buf);

    // Truncate to original length
    buf.truncate(n);
    buf
}

/// Compute residual error reduction achieved by denoising.
///
/// Returns `(error_raw, error_denoised)` — both in Q32.32.
/// A lower `error_denoised` indicates the wavelet denoising helped.
pub fn denoising_benefit(data: &[i32], threshold: i32) -> (i64, i64) {
    let (slope_raw, intercept_raw) = fit_linear_fixed(data);
    let error_raw = crate::compute_residual_error(data, slope_raw, intercept_raw);

    let denoised = wavelet_denoise(data, threshold);
    let (slope_dn, intercept_dn) = fit_linear_fixed(&denoised);
    let error_dn = crate::compute_residual_error(&denoised, slope_dn, intercept_dn);

    (error_raw, error_dn)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wavelet_denoise_preserves_length() {
        let data = [100, 200, 300, 400, 500, 600, 700, 800];
        let denoised = wavelet_denoise(&data, 10);
        assert_eq!(denoised.len(), data.len());
    }

    #[test]
    fn test_fit_denoised_short_data() {
        let data = [100, 200];
        let (slope, intercept) = fit_denoised(&data, 10);
        // Should fall back to fit_linear_fixed
        let (slope_raw, intercept_raw) = fit_linear_fixed(&data);
        assert_eq!(slope, slope_raw);
        assert_eq!(intercept, intercept_raw);
    }

    #[test]
    fn test_denoising_benefit() {
        // Linear data with noise
        let data = [100, 210, 290, 410, 490, 610, 690, 810];
        let (err_raw, err_dn) = denoising_benefit(&data, 30);
        // Denoised error should be <= raw error
        assert!(err_dn <= err_raw);
    }
}
