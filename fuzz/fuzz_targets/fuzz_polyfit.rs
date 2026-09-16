//! Quadratic / cubic Q16.16 fits and the linear-vs-polynomial selector must
//! accept any sample vector (length 0..N, full i32 range) without panicking.
//! An arithmetic overflow here is a real finding: the crate targets no_std
//! MCUs where a panic is an abort.
#![no_main]

use alice_edge::{fit_cubic_fixed, fit_quadratic_fixed, should_use_linear};
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: Vec<i32>| {
    let _ = should_use_linear(&data);
    let _ = fit_quadratic_fixed(&data);
    let _ = fit_cubic_fixed(&data);
});
