//! Delta coding of (slope, intercept) pairs is a bijection on the full i32
//! range (wrapping arithmetic): decode(encode(x)) == x, and the savings
//! estimate must never panic.
#![no_main]

use alice_edge::{delta_decode_coefficients, delta_encode_coefficients, delta_encoding_savings};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|coeffs: Vec<(i32, i32)>| {
    let encoded = delta_encode_coefficients(&coeffs);
    assert_eq!(encoded.len(), coeffs.len());
    let decoded = delta_decode_coefficients(&encoded);
    assert_eq!(decoded, coeffs, "delta roundtrip is not the identity");
    let _ = delta_encoding_savings(&coeffs);
});
