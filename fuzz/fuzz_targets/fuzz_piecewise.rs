//! Piecewise-linear segmentation: the returned segments must tile
//! `0..data.len()` exactly (contiguous, in order, no gaps or overlap) for any
//! error budget and minimum segment length, and the call must not panic.
#![no_main]

use alice_edge::fit_piecewise_linear;
use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

#[derive(Arbitrary, Debug)]
struct Input {
    data: Vec<i32>,
    max_error_q32: i64,
    min_segment_len: u8,
}

fuzz_target!(|input: Input| {
    let segments = fit_piecewise_linear(
        &input.data,
        input.max_error_q32,
        usize::from(input.min_segment_len),
    );

    if input.data.is_empty() {
        return;
    }

    let mut expected_start = 0usize;
    for seg in &segments {
        assert_eq!(seg.start, expected_start, "segment does not start where the previous ended");
        assert!(seg.end > seg.start, "empty segment {seg:?}");
        expected_start = seg.end;
    }
    assert_eq!(expected_start, input.data.len(), "segments do not cover the whole input");
});
