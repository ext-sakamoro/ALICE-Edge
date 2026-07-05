//! Delta encoding for coefficient compression (`delta_encode_coefficients` / `delta_decode_coefficients` / `delta_encoding_savings`).

extern crate alloc;
use alloc::vec::Vec;

pub fn delta_encode_coefficients(coeffs: &[(i32, i32)]) -> Vec<(i32, i32)> {
    if coeffs.is_empty() {
        return Vec::new();
    }

    let mut encoded = Vec::with_capacity(coeffs.len());
    encoded.push(coeffs[0]);

    for i in 1..coeffs.len() {
        let ds = coeffs[i].0.wrapping_sub(coeffs[i - 1].0);
        let di = coeffs[i].1.wrapping_sub(coeffs[i - 1].1);
        encoded.push((ds, di));
    }

    encoded
}

/// Decode delta-encoded coefficient pairs back to absolute values.
#[cfg(feature = "std")]
#[must_use]
pub fn delta_decode_coefficients(encoded: &[(i32, i32)]) -> Vec<(i32, i32)> {
    if encoded.is_empty() {
        return Vec::new();
    }

    let mut decoded = Vec::with_capacity(encoded.len());
    decoded.push(encoded[0]);

    for i in 1..encoded.len() {
        let prev = decoded[i - 1];
        let s = prev.0.wrapping_add(encoded[i].0);
        let intercept = prev.1.wrapping_add(encoded[i].1);
        decoded.push((s, intercept));
    }

    decoded
}

/// Compute the byte savings from delta encoding.
///
/// Returns `(original_bytes, delta_bytes)` — estimated sizes if each
/// coefficient were stored as a variable-length integer (zigzag + varint).
#[cfg(feature = "std")]
#[must_use]
pub fn delta_encoding_savings(coeffs: &[(i32, i32)]) -> (usize, usize) {
    const fn varint_size(val: i32) -> usize {
        // Zigzag encoding size
        let zigzag = ((val >> 31) ^ (val << 1)) as u32;
        match zigzag {
            0..=127 => 1,
            128..=16383 => 2,
            16_384..=2_097_151 => 3,
            2_097_152..=268_435_455 => 4,
            _ => 5,
        }
    }

    let original: usize = coeffs
        .iter()
        .map(|&(s, i)| varint_size(s) + varint_size(i))
        .sum();

    let deltas = delta_encode_coefficients(coeffs);
    let delta: usize = deltas
        .iter()
        .map(|&(s, i)| varint_size(s) + varint_size(i))
        .sum();

    (original, delta)
}
