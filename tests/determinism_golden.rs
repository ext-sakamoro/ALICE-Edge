//! Cross-platform golden hashes for every module whose output is transmitted.
//!
//! `tests/analytic_oracle.rs` and `tests/edge_law.rs` check that the numbers
//! are *right*; this file checks that they are the **same bits** on every
//! target. The two properties are independent, and for this crate the second
//! one is the load-bearing half: a window of samples is reduced on the device
//! to coefficients, and those coefficients are what leaves the device. If two
//! devices fitting the same samples disagree in the last ulp, they transmit
//! two different models, and a receiver that re-derives the fit to audit it
//! disagrees with both.
//!
//! Why bit-exactness holds here:
//!
//! 1. The integer fits (`fit_linear_fixed`, `fit_constant_fixed`,
//!    `fit_quadratic_fixed`, `fit_cubic_fixed`, `fit_linear_simd`) are exact
//!    `i64` / `i128` arithmetic with explicit wrapping / checked operations,
//!    so they have no platform freedom at all.
//! 2. `+ - * /` and `sqrt` are required by IEEE 754 to be correctly rounded,
//!    so they produce identical bits on every target Rust supports (x86_64
//!    with SSE2, aarch64, thumbv7em with an FPU, wasm32).
//! 3. Every float transcendental goes through [`alice_det_math`], whose
//!    kernels are built from those operations only, in a fixed evaluation
//!    order. `mul_add` is a fused multiply-add with a single rounding by IEEE
//!    754, and a target without an FMA instruction gets the correctly rounded
//!    software `fma`, so it is bit-identical everywhere and is used freely.
//!    Integer powers are written out rather than taken with `powi`, whose
//!    multiplication tree has an unspecified association order.
//! 4. `clippy.toml` `disallowed-methods` rejects the inherent `f32` / `f64`
//!    forms of all of the above, and the gates run clippy with `--all-targets
//!    ... -D warnings`, so a regression is a compile error rather than a
//!    silent divergence.
//!
//! What this file does **not** cover: the two bridges whose arithmetic is
//! owned by another crate, `object_classifier`'s ternary matrix-vector kernel
//! (`alice_ml`) and `sdf_compress` (`alice_sdf`). Only the parts computed here
//! are hashed — the feature extractor, not the kernel. Those crates carry
//! their own gates.
//!
//! Each scenario drives one module through its public entry points with fixed
//! inputs, serialises every output bit for bit (`to_bits().to_le_bytes()`,
//! little-endian, so the hash does not depend on the host byte order) and
//! compares the SHA-256 with a constant recorded on macOS `aarch64`. CI runs
//! this file on macOS `aarch64`, Linux `x86_64`, Linux `aarch64` and Windows
//! `x86_64`; a mismatch on any of them means a platform-dependent operation
//! crept in.
//!
//! Updating a golden (only after an intentional algorithm change): run the
//! failing test, copy the `actual` hex into the constant, and record the
//! change in CHANGELOG under `[Unreleased] / Changed`. A mismatch that is
//! *not* explained by a deliberate change in this repository is a defect:
//! find the operation that left the list above instead of re-recording.

use core::fmt::Write as _;

use sha2::{Digest, Sha256};

// ---------------------------------------------------------------------------
// Byte sink
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Sink(Vec<u8>);

impl Sink {
    fn f32(&mut self, v: f32) {
        self.0.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    // only the `law` scenario serialises `f64`: the fits themselves are
    // integer, and `sensor_fusion` is `f32`
    #[cfg(feature = "law")]
    fn f64(&mut self, v: f64) {
        self.0.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    fn i32(&mut self, v: i32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn i64(&mut self, v: i64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn usize(&mut self, v: usize) {
        self.0.extend_from_slice(&(v as u64).to_le_bytes());
    }
    fn bool(&mut self, v: bool) {
        self.0.push(u8::from(v));
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn finish(self) -> String {
        let mut hex = String::with_capacity(64);
        for b in Sha256::digest(&self.0) {
            write!(hex, "{b:02x}").expect("writing to a String cannot fail");
        }
        hex
    }
}

/// A hash is only evidence if bytes went into it. Every scenario asserts a
/// non-zero, expected payload length before comparing, so a scenario that
/// silently stopped exercising its module (an API that started returning an
/// empty vector, a loop whose bound became 0, a `#[cfg]` that stopped
/// matching) fails here rather than passing with the hash of an empty buffer.
fn assert_golden(scenario: &str, sink: Sink, min_bytes: usize, expected: &str) {
    let bytes = sink.len();
    assert!(
        bytes >= min_bytes,
        "scenario `{scenario}` serialised {bytes} bytes, expected at least \
         {min_bytes}: the scenario stopped exercising the module, so its hash \
         proves nothing"
    );
    let actual = sink.finish();
    assert_eq!(
        actual, expected,
        "\n\nGolden hash mismatch for scenario `{scenario}` ({bytes} bytes).\n\
         actual:   {actual}\n\
         expected: {expected}\n\n\
         If an algorithm in this repository changed on purpose, update the\n\
         GOLDEN_* constant in tests/determinism_golden.rs and record it in\n\
         CHANGELOG. If it did not, a platform-dependent float operation was\n\
         introduced: check clippy.toml disallowed-methods and that every\n\
         transcendental goes through alice_det_math.\n"
    );
}

/// Deterministic pseudo-random sample window: integer mixing only, so the
/// inputs are the same bits on every target.
fn prand_i32(i: u64, span: i32) -> i32 {
    let mut h = i
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(0x1234_5678_9ABC_DEF0);
    h ^= h >> 33;
    h = h.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    h ^= h >> 29;
    ((h >> 40) as i32) % span
}

/// Deterministic pseudo-random `f32` in `[0, 1)`: an integer divided once by
/// a power of two, so the inputs are themselves bit-identical everywhere.
fn prand_f32(i: u64) -> f32 {
    prand_i32(i, 1 << 24) as f32 / (1u32 << 24) as f32
}

/// The sample windows every fit scenario is driven with: degenerate lengths,
/// exact lines and polynomials, noise, and the Q16 boundary.
fn windows() -> Vec<Vec<i32>> {
    let mut out: Vec<Vec<i32>> = vec![
        vec![],
        vec![42],
        vec![0, 0],
        vec![2500, 2510],
        vec![2500, 2510, 2520, 2530, 2540],
        vec![0, 1, 4, 9, 16, 25, 36, 49],
        vec![0, 1, 8, 27, 64, 125, 216, 343, 512, 729],
        vec![-32768, 0, 32767],
        vec![i32::MIN, 0, i32::MAX],
        vec![7; 33],
    ];
    // a 64-sample ramp with integer noise, and the same ramp with one outlier
    let ramp: Vec<i32> = (0..64).map(|i| 100 * i + prand_i32(i as u64, 7)).collect();
    let mut spiked = ramp.clone();
    spiked[23] = 30_000;
    out.push(ramp);
    out.push(spiked);
    // an exact two-segment broken line (break at 40)
    out.push(
        (0..80)
            .map(|x: i32| if x < 40 { 3 * x } else { 120 - 2 * (x - 40) })
            .collect(),
    );
    out
}

// ---------------------------------------------------------------------------
// 1. q16_linear — the fit that is transmitted, its evaluation and conversions
// ---------------------------------------------------------------------------

const GOLDEN_Q16_LINEAR: &str = "105580c224a8b5fe8ac11762db84662a699ad8aed740a52892f10ef92892041d";

#[test]
fn golden_q16_linear() {
    use alice_edge::q16_linear::{
        evaluate_linear_fixed, fit_linear_fixed, int_to_q16, q16_to_f32, q16_to_int, Q16_ONE,
        Q16_SHIFT,
    };
    let mut s = Sink::default();
    s.i32(Q16_ONE);
    s.i32(Q16_SHIFT);

    for w in windows() {
        let (slope, intercept) = fit_linear_fixed(&w);
        s.i32(slope);
        s.i32(intercept);
        s.usize(w.len());
        // reconstruct inside the window and one step past its end
        for x in [0i32, 1, 7, 39, 63] {
            s.i32(evaluate_linear_fixed(slope, intercept, x));
            s.i32(q16_to_int(evaluate_linear_fixed(slope, intercept, x)));
            s.f32(q16_to_f32(evaluate_linear_fixed(slope, intercept, x)));
        }
    }

    // the conversions on their own, including the Q16 boundary
    for v in [0i32, 1, -1, 100, -100, 32_767, -32_768] {
        s.i32(int_to_q16(v));
        s.i32(q16_to_int(int_to_q16(v)));
        s.f32(q16_to_f32(int_to_q16(v)));
    }
    for q in [0i32, 1, -1, 65_536, -65_536, i32::MAX, i32::MIN] {
        s.i32(q16_to_int(q));
        s.f32(q16_to_f32(q));
    }

    assert_golden("q16_linear", s, 1_000, GOLDEN_Q16_LINEAR);
}

// ---------------------------------------------------------------------------
// 2. adaptive_polyfit — quadratic / cubic fits and their evaluators
// ---------------------------------------------------------------------------

const GOLDEN_ADAPTIVE_POLYFIT: &str =
    "b4a125049e4d8d4ae0278243c32cc972ec46c81cb8f90295695ac8c638d815b5";

#[test]
fn golden_adaptive_polyfit() {
    use alice_edge::adaptive_polyfit::{
        evaluate_cubic_fixed, evaluate_quadratic_fixed, fit_cubic_fixed, fit_quadratic_fixed,
        should_use_linear,
    };
    let mut s = Sink::default();

    for w in windows() {
        s.bool(should_use_linear(&w));

        let (a, b, c) = fit_quadratic_fixed(&w);
        s.i32(a);
        s.i32(b);
        s.i32(c);

        let (ca, cb, cc, cd) = fit_cubic_fixed(&w);
        s.i32(ca);
        s.i32(cb);
        s.i32(cc);
        s.i32(cd);

        // `evaluate_*` is only defined while the reconstruction stays inside
        // the Q16 range; outside it the sum wraps in release and overflows in
        // debug (tests/panic_contract.rs pins both), so the evaluators are
        // driven here with the small coefficients of a short exact window
        if w.len() == 8 {
            for x in 0i32..8 {
                s.i32(evaluate_quadratic_fixed(a, b, c, x));
                s.i32(evaluate_cubic_fixed(ca, cb, cc, cd, x));
            }
        }
    }

    assert_golden("adaptive_polyfit", s, 430, GOLDEN_ADAPTIVE_POLYFIT);
}

// ---------------------------------------------------------------------------
// 3. constant_fit — the mean fit and the residual used to choose a model
// ---------------------------------------------------------------------------

const GOLDEN_CONSTANT_FIT: &str =
    "6f3c6335869a8c6320ba5972a9007a76180aee43ddaaa6be3df785a91baee705";

#[test]
fn golden_constant_fit() {
    use alice_edge::constant_fit::{compute_residual_error, fit_constant_fixed};
    use alice_edge::q16_linear::fit_linear_fixed;
    let mut s = Sink::default();

    for w in windows() {
        let mean = fit_constant_fixed(&w);
        s.i32(mean);
        s.i64(compute_residual_error(&w, 0, mean));
        let (slope, intercept) = fit_linear_fixed(&w);
        s.i64(compute_residual_error(&w, slope, intercept));
    }

    assert_golden("constant_fit", s, 250, GOLDEN_CONSTANT_FIT);
}

// ---------------------------------------------------------------------------
// 4. simd_fit — the SSE2 / NEON kernels against the same inputs. The scalar
//    fit is hashed next to them, so a kernel that diverges from it (or from
//    the kernel on another architecture) changes this hash.
// ---------------------------------------------------------------------------

const GOLDEN_SIMD_FIT: &str = "6907b2ffc5067ca5c9a5188a709e5f0a217fe7e436542c5c3c5397dfdf034a9a";

#[test]
fn golden_simd_fit() {
    use alice_edge::q16_linear::fit_linear_fixed;
    use alice_edge::simd_fit::fit_linear_simd;
    let mut s = Sink::default();

    for w in windows() {
        let (ss, si) = fit_linear_simd(&w);
        let (rs, ri) = fit_linear_fixed(&w);
        s.i32(ss);
        s.i32(si);
        s.i32(rs);
        s.i32(ri);
    }
    // lengths around the 8-sample kernel threshold and its remainder handling
    for n in 1usize..40 {
        let w: Vec<i32> = (0..n).map(|i| 13 * i as i32 - 500).collect();
        let (ss, si) = fit_linear_simd(&w);
        s.i32(ss);
        s.i32(si);
    }

    assert_golden("simd_fit", s, 500, GOLDEN_SIMD_FIT);
}

// ---------------------------------------------------------------------------
// 5. sensor_fusion — Kalman 1D / 2D and the inverse-variance fusion. All
//    float, all basic operations plus the module's own Newton square root.
// ---------------------------------------------------------------------------

const GOLDEN_SENSOR_FUSION: &str =
    "976837f60532e082cbbb29a9d987b08080f8cdb8172fba329f66cb875ac3eacf";

#[test]
fn golden_sensor_fusion() {
    use alice_edge::sensor_fusion::{
        FusedSensor, FusionConfig, KalmanFilter1D, KalmanFilter2D, SensorInput,
    };
    let mut s = Sink::default();

    let mut k1 = KalmanFilter1D::new(0.0, 1.0, 0.01, 0.1);
    for i in 0..200u64 {
        s.f32(k1.filter(25.0 + prand_f32(i) * 2.0 - 1.0));
        s.f32(k1.estimate());
        s.f32(k1.error());
    }

    let mut k2 = KalmanFilter2D::new(0.05, 0.01, 0.5);
    for i in 0..200u64 {
        let (p, v) = k2.filter(f32::from(i as u16) * 0.25 + prand_f32(i + 999) - 0.5);
        s.f32(p);
        s.f32(v);
    }

    let cfg = FusionConfig::default();
    s.f32(cfg.process_noise);
    s.f32(cfg.default_measurement_noise);
    s.f32(cfg.outlier_threshold);
    let mut fused = FusedSensor::new(cfg);
    for i in 0..200u64 {
        let base = 100.0 + prand_f32(i + 4_242) * 4.0;
        let inputs = [
            SensorInput {
                value: base,
                noise: 0.25,
            },
            SensorInput {
                value: base + 0.5,
                noise: 1.0,
            },
            // an occasional gross outlier, to pin the rejection branch
            SensorInput {
                value: if i % 17 == 0 { 1.0e4 } else { base - 0.25 },
                noise: 4.0,
            },
        ];
        s.f32(fused.fuse(&inputs));
    }
    s.f32(fused.estimate());
    s.u64(fused.fusion_count());
    s.u64(fused.rejected_count());

    assert_golden("sensor_fusion", s, 4_800, GOLDEN_SENSOR_FUSION);
}

// ---------------------------------------------------------------------------
// 6. ring_buffer — the window the fits are fed from (integer, order-sensitive)
// ---------------------------------------------------------------------------

const GOLDEN_RING_BUFFER: &str = "224f9f81c7e54478be79641e2b6067e90b803e8acb062014c6a639502377f288";

#[test]
fn golden_ring_buffer() {
    use alice_edge::ring_buffer::RingBuffer;
    let mut s = Sink::default();

    let mut rb = RingBuffer::<i32, 16>::new();
    for i in 0..40u64 {
        rb.push(prand_i32(i, 10_000));
        s.usize(rb.len());
        s.bool(rb.is_empty());
        s.bool(rb.is_full());
        s.usize(rb.capacity());
        s.i32(rb.first().copied().unwrap_or(i32::MIN));
        s.i32(rb.last().copied().unwrap_or(i32::MIN));
        for v in rb.iter() {
            s.i32(*v);
        }
    }
    for idx in 0..20 {
        s.i32(rb.get(idx).copied().unwrap_or(i32::MIN));
    }
    rb.clear();
    s.usize(rb.len());
    s.bool(rb.is_empty());

    assert_golden("ring_buffer", s, 3_200, GOLDEN_RING_BUFFER);
}

// ---------------------------------------------------------------------------
// 7. the transcendental kernels at the arguments the simulated sensor and
//    depth-camera paths feed them. Those two paths are drivers (they sleep,
//    and one timestamps from the wall clock), so they are not replayed here;
//    their arithmetic is.
// ---------------------------------------------------------------------------

const GOLDEN_DET_MATH: &str = "4fb30a451a9aae88529c0f66ef1a94f9b3e400cf360cd6e7398a4d9622b318ab";

#[test]
fn golden_det_math_kernels() {
    let mut s = Sink::default();

    // sensors: t = i / 1000 and t · 0.1 over a 2000-sample run
    for i in 0..2_000u32 {
        let t = f32::from(i as u16) * (1.0 / 1000.0);
        let (sin_t, cos_t) = alice_det_math::sin_cos(t);
        s.f32(sin_t);
        s.f32(cos_t);
        s.f32(alice_det_math::sin(t));
        s.f32(alice_det_math::cos(t));
        s.f32(alice_det_math::sin(f32::from(i as u16) * 0.1));
        s.f32(alice_det_math::cos(f32::from(i as u16) * 0.1));
    }

    // depth camera: a 10x10 grid times 8 frames
    for frame in 0..8u32 {
        let frame_f = frame as f32 * 0.1;
        for iy in 0..10 {
            for ix in 0..10 {
                let x = (ix as f32 - 4.5) * 0.1;
                let z = (iy as f32 - 4.5) * 0.1;
                s.f32(alice_det_math::sin(x * core::f32::consts::PI + frame_f));
                s.f32(alice_det_math::cos(z * 2.71 + frame_f));
            }
        }
    }

    // classifier: ln of a count, exp of a centred logit
    for n in [0u32, 1, 2, 7, 100, 5_000, 100_000] {
        s.f32(alice_det_math::ln(n as f32 + 1.0));
    }
    for i in 0..256u32 {
        let logit = (i as f32) * (1.0 / 16.0) - 8.0;
        s.f32(alice_det_math::exp(logit));
    }

    assert_golden("det_math", s, 55_000, GOLDEN_DET_MATH);
}

// ---------------------------------------------------------------------------
// 8. robust (feature `std`) — residual MAD rejection and the refit
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
const GOLDEN_ROBUST: &str = "30f195a6dab847aee37f1e75d1a2eb36ea24687078198c32d49aee67762ee76a";

#[cfg(feature = "std")]
#[test]
fn golden_robust() {
    use alice_edge::robust::{filter_outliers_mad, fit_linear_robust};
    let mut s = Sink::default();

    for w in windows() {
        for k in [0i32, 1, 3, 10, i32::MAX] {
            let kept = filter_outliers_mad(&w, k);
            s.usize(kept.len());
            for v in &kept {
                s.i32(*v);
            }
            let (slope, intercept) = fit_linear_robust(&w, k);
            s.i32(slope);
            s.i32(intercept);
        }
    }

    assert_golden("robust", s, 6_500, GOLDEN_ROBUST);
}

// ---------------------------------------------------------------------------
// 9. piecewise (feature `std`) — breakpoint search and per-segment fits
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
const GOLDEN_PIECEWISE: &str = "a9706b8473b6d6c50334f9ee9e6726f9eb44409e13118a403c511843db8b49f7";

#[cfg(feature = "std")]
#[test]
fn golden_piecewise() {
    use alice_edge::piecewise::fit_piecewise_linear;
    let mut s = Sink::default();

    for w in windows() {
        for (max_err, min_len) in [(0i64, 0usize), (0, 4), (1_000, 4), (1 << 40, 8)] {
            let segs = fit_piecewise_linear(&w, max_err, min_len);
            s.usize(segs.len());
            for seg in &segs {
                s.usize(seg.start);
                s.usize(seg.end);
                s.i32(seg.slope);
                s.i32(seg.intercept);
            }
        }
    }

    assert_golden("piecewise", s, 5_500, GOLDEN_PIECEWISE);
}

// ---------------------------------------------------------------------------
// 10. delta (feature `std`) — the coefficient stream as it is packed
// ---------------------------------------------------------------------------

#[cfg(feature = "std")]
const GOLDEN_DELTA: &str = "c25c2ed884ea0d109d267e48a0a0c08f13c32291415002519af3a3ebea452d89";

#[cfg(feature = "std")]
#[test]
fn golden_delta() {
    use alice_edge::delta::{
        delta_decode_coefficients, delta_encode_coefficients, delta_encoding_savings,
    };
    use alice_edge::q16_linear::fit_linear_fixed;
    let mut s = Sink::default();

    let coeffs: Vec<(i32, i32)> = windows()
        .iter()
        .map(|w| fit_linear_fixed(w))
        .chain((0..32u64).map(|i| (prand_i32(i, 1 << 20), prand_i32(i + 64, 1 << 20))))
        .collect();
    let encoded = delta_encode_coefficients(&coeffs);
    for (a, b) in &encoded {
        s.i32(*a);
        s.i32(*b);
    }
    let decoded = delta_decode_coefficients(&encoded);
    for (a, b) in &decoded {
        s.i32(*a);
        s.i32(*b);
    }
    let (raw, packed) = delta_encoding_savings(&coeffs);
    s.usize(raw);
    s.usize(packed);

    assert_golden("delta", s, 730, GOLDEN_DELTA);
}

// ---------------------------------------------------------------------------
// 11. law (feature `law`) — the entry point that turns a fit into a law. This
//     is the path whose bits identify the transmitted model, so it is pinned
//     coefficient by coefficient, residual included.
// ---------------------------------------------------------------------------

#[cfg(feature = "law")]
const GOLDEN_LAW: &str = "53583c42b85669e3ecb5f701e23329ba759071d821ddb7efa06ac75424b17000";

#[cfg(feature = "law")]
#[test]
fn golden_law() {
    use alice_edge::law::{linear_law, sample_points, IngestPolicy, Provenance, Verdict};
    let mut s = Sink::default();

    for w in windows() {
        for (x, y) in sample_points(&w) {
            s.f64(x);
            s.f64(y);
        }
        match linear_law(&w, Provenance::new("golden", "fit_linear_fixed")) {
            Err(_) => s.bool(false),
            Ok(law) => {
                s.bool(true);
                for c in law.coefficients() {
                    s.f64(*c);
                }
                let r = law.residual();
                s.u64(r.n as u64);
                s.f64(r.rms);
                s.f64(r.max_abs);
                let d = law.domain();
                s.f64(d.lo);
                s.f64(d.hi);
                // evaluation inside the window, at both ends and outside it
                let span = (w.len() - 1) as f64;
                for frac in [0.0f64, 0.25, 0.5, 0.75, 1.0] {
                    match law.evaluate(span * frac) {
                        Ok(v) => {
                            s.bool(true);
                            s.f64(v);
                        }
                        Err(_) => s.bool(false),
                    }
                }
                s.bool(law.evaluate(span + 1.0).is_ok());
                s.bool(law.evaluate(-1.0).is_ok());
                s.bool(law.evaluate(f64::NAN).is_ok());

                // the verdict on a second window of the same sensor
                let policy = IngestPolicy {
                    abs_tolerance: 1.0,
                    break_factor: 4.0,
                };
                let next = sample_points(&w);
                // tag + the figure the variant carries: the verdict is part
                // of what a receiver acts on, so both are pinned
                let (tag, figure) = match law.ingest(&next, &policy) {
                    Verdict::NoEvidence => (0u8, f64::NEG_INFINITY),
                    Verdict::OutOfRange { outside } => (1, outside as f64),
                    Verdict::Supports { rms } => (2, rms),
                    Verdict::ParameterUpdate { previous_rms, .. } => (3, previous_rms),
                    Verdict::ResidualGrew { rms } => (4, rms),
                    Verdict::Breaks { rms } => (5, rms),
                };
                s.i32(i32::from(tag));
                s.f64(figure);
            }
        }
    }

    assert_golden("law", s, 5_600, GOLDEN_LAW);
}

// ---------------------------------------------------------------------------
// 12. object_classifier (feature `ml`) — the feature extractor. The ternary
//     matrix-vector kernel it feeds lives in `alice_ml` and is gated there.
// ---------------------------------------------------------------------------

#[cfg(feature = "ml")]
const GOLDEN_OBJECT_FEATURES: &str =
    "3e328c9cf3e28445f519846ddfed922f9b40df20618dd1811e2d0451935105af";

#[cfg(feature = "ml")]
#[test]
fn golden_object_features() {
    use alice_edge::object_classifier::{ObjectClass, SdfFeatures};
    let mut s = Sink::default();

    for kind in 0u8..5 {
        for i in 0..16u64 {
            let bounds = [
                prand_f32(i) * 4.0 + 0.01,
                prand_f32(i + 100) * 4.0 + 0.01,
                prand_f32(i + 200) * 4.0 + 0.01,
            ];
            let params = [prand_f32(i + 300), prand_f32(i + 400), prand_f32(i + 500)];
            let f = SdfFeatures::from_primitive(kind, &params, bounds, (i as usize) * 97);
            for v in f.features {
                s.f32(v);
            }
            let g = SdfFeatures::from_svo_stats(
                (i as u32) * 13 + 1,
                (i as u32) * 7,
                (i as u32) % 12,
                bounds,
            );
            for v in g.features {
                s.f32(v);
            }
        }
    }
    for id in 0u8..8 {
        s.i32(ObjectClass::from_id(id) as i32);
    }

    assert_golden("object_features", s, 10_200, GOLDEN_OBJECT_FEATURES);
}
