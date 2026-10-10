//! Coefficient batches across the alice-zip 0.7 → 0.8 codec change, and
//! between the 0.8 and 0.9 writers and readers.
//!
//! alice-zip 0.8 writes the quantised residual container with deflate and a
//! recorded codec byte; 0.7 wrote LZMA with no codec byte. The writer and
//! reader of the old format here are the published `alice-zip` 0.7.0 itself
//! (dev-dependency `alice_zip_07`), called with the arguments
//! `zip_bridge::compress_coefficients` used before the change (16 bits, level 6).
#![cfg(feature = "zip")]

use alice_edge::zip_bridge::{
    compress_coefficients, decompress_coefficients, CompressedModelBatch,
};

/// Coefficient pairs spanning the Q16.16 range a fit produces, with no
/// pattern a codec could exploit perfectly
fn pairs(n: usize, salt: i64) -> Vec<(i32, i32)> {
    (0..n as i64)
        .map(|i| {
            let s = ((i * 7919 + salt * 104_729) % 65_536 - 32_768) as i32;
            let c = ((i * i * 31 + salt) % 2_000_000 - 1_000_000) as i32;
            (s, c)
        })
        .collect()
}

/// What the bridge decodes back, for comparison with the batch written by the
/// previous release (16-bit quantisation is lossy, so the reference is the
/// old reader's own output, not the input)
fn old_round_trip(input: &[(i32, i32)]) -> Vec<(i32, i32)> {
    let floats: Vec<f32> = input
        .iter()
        .flat_map(|&(a, b)| [a as f32, b as f32])
        .collect();
    let bytes = alice_zip_07::compression::compress_residual_quantized(&floats, 16, 6).unwrap();
    alice_zip_07::compression::decompress_residual_quantized(&bytes)
        .unwrap()
        .chunks_exact(2)
        .map(|p| (p[0] as i32, p[1] as i32))
        .collect()
}

#[test]
fn a_batch_written_before_the_codec_change_still_decodes_to_the_same_values() {
    for (n, salt) in [(1, 0), (2, 1), (100, 2), (1000, 3)] {
        let input = pairs(n, salt);
        let floats: Vec<f32> = input
            .iter()
            .flat_map(|&(a, b)| [a as f32, b as f32])
            .collect();
        let old = CompressedModelBatch {
            data: alice_zip_07::compression::compress_residual_quantized(&floats, 16, 6).unwrap(),
            count: n,
        };
        assert_eq!(
            decompress_coefficients(&old).unwrap(),
            old_round_trip(&input),
            "{n} pairs written by alice-zip 0.7 decode differently now"
        );
    }
}

#[test]
fn a_new_batch_round_trips_with_the_same_values_as_before() {
    // quantisation, not the codec, decides the values: the new writer gives
    // back exactly what the old writer + reader gave back
    for (n, salt) in [(1, 4), (2, 5), (100, 6), (1000, 7)] {
        let input = pairs(n, salt);
        let batch = compress_coefficients(&input).unwrap();
        assert_eq!(batch.count, n);
        assert_eq!(
            decompress_coefficients(&batch).unwrap(),
            old_round_trip(&input)
        );
    }
}

#[test]
fn a_receiver_still_on_alice_zip_0_7_refuses_a_new_batch() {
    // the documented behaviour for a fleet that upgrades senders first: the old
    // reader fails, it never hands back numbers
    let mut refused = 0;
    for (n, salt) in [(1, 8), (2, 9), (3, 10), (100, 11), (1000, 12), (5000, 13)] {
        let batch = compress_coefficients(&pairs(n, salt)).unwrap();
        assert!(
            alice_zip_07::compression::decompress_residual_quantized(&batch.data).is_err(),
            "alice-zip 0.7 decoded a new batch of {n} pairs instead of refusing it"
        );
        refused += 1;
    }
    assert_eq!(refused, 6);
}

/// Encode with the published alice-zip 0.8.0, the arguments the bridge uses
fn batch_08(input: &[(i32, i32)]) -> CompressedModelBatch {
    let floats: Vec<f32> = input
        .iter()
        .flat_map(|&(a, b)| [a as f32, b as f32])
        .collect();
    CompressedModelBatch {
        data: alice_zip_08::compression::compress_residual_quantized(&floats, 16, 6).unwrap(),
        count: input.len(),
    }
}

#[test]
fn batches_written_by_alice_zip_0_8_and_0_9_are_interchangeable() {
    // 0.9 changed a quantisation rounding in its Python package; the Rust
    // writer the bridge uses must still give the same bytes and values, so a
    // fleet with both versions reads each other's batches
    for (n, salt) in [(1, 14), (2, 15), (100, 16), (1000, 17)] {
        let input = pairs(n, salt);
        let old = batch_08(&input);
        let new = compress_coefficients(&input).unwrap();
        assert_eq!(
            new.data, old.data,
            "{n} pairs: the 0.9 bytes differ from 0.8"
        );
        let read_by_08: Vec<(i32, i32)> =
            alice_zip_08::compression::decompress_residual_quantized(&new.data)
                .unwrap()
                .chunks_exact(2)
                .map(|p| (p[0] as i32, p[1] as i32))
                .collect();
        assert_eq!(decompress_coefficients(&old).unwrap(), read_by_08);
        assert_eq!(read_by_08, old_round_trip(&input));
    }
}
