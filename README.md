# ALICE-Edge

[日本語](README_JP.md)

[![CI](https://github.com/ext-sakamoro/ALICE-Edge/actions/workflows/ci.yml/badge.svg)](https://github.com/ext-sakamoro/ALICE-Edge/actions/workflows/ci.yml)
[![Security](https://github.com/ext-sakamoro/ALICE-Edge/actions/workflows/security-audit.yml/badge.svg)](https://github.com/ext-sakamoro/ALICE-Edge/actions/workflows/security-audit.yml)
[![Fuzz](https://github.com/ext-sakamoro/ALICE-Edge/actions/workflows/fuzz.yml/badge.svg)](https://github.com/ext-sakamoro/ALICE-Edge/actions/workflows/fuzz.yml)

Embedded model fitting for sensor data: "don't send data, send the law".

A window of integer sensor samples is reduced on the device to a few Q16.16
fixed-point coefficients (a line, a constant, a quadratic or cubic, a
piecewise line, a robust line), and only the coefficients are transmitted.
The core is `#![no_std]` and needs no FPU; its only dependency is
[`alice-det-math`](https://crates.io/crates/alice-det-math), itself
dependency-free `no_std`, which keeps the float transcendentals bit-exact
across targets (see [Determinism](#determinism)). Optional features add
sensor drivers, MQTT, a dashboard, a C ABI, Python bindings and bridges to
other ALICE crates.

With the `law` feature an Edge fit is returned as an
`alice_zip::law::SignalLaw`: the transmitted coefficients together with the
samples they were fitted from, the residual measured against those samples,
the valid range and the provenance.

License: MIT OR Apache-2.0

## Contents

- [What it is not for](#what-it-is-not-for)
- [Installation](#installation)
- [Example](#example)
- [Fits as laws (`law` feature)](#fits-as-laws-law-feature)
- [Features](#features)
- [Q16.16 format](#q1616-format)
- [Determinism](#determinism)
- [Platforms](#platforms)
- [Bindings](#bindings)
- [Minimum supported Rust version](#minimum-supported-rust-version)
- [Building and testing](#building-and-testing)
- [Related crates](#related-crates)
- [License](#license)

## What it is not for

- General regression: the fits take samples at the implicit positions
  `x = 0, 1, …, n − 1`, and the Q16.16 contract covers `|y| < 2¹⁵`; values
  outside it wrap (the `law` residual makes such a wrap visible)
- Lossless storage: the coefficients describe the trend of a window, the
  samples themselves are not recoverable from them
- Encryption or authentication of what is transmitted
- A hardware abstraction layer: the `sensors-hw` drivers cover a few I2C /
  SPI / GPIO / UART sensors on Linux through `rppal` and `serialport`

## Installation

```bash
cargo add alice-edge
# with the law entry point (no_std + alloc)
cargo add alice-edge --features law
```

## Example

```rust
use alice_edge::{evaluate_linear_fixed, fit_linear_fixed, q16_to_int};

// sensor readings (temperature × 100): 25.00 °C rising 0.10 °C per sample
let samples = [2500, 2510, 2520, 2530, 2540];

// on the device: two Q16.16 numbers instead of five samples
let (slope, intercept) = fit_linear_fixed(&samples);
assert_eq!((slope, intercept), (10 << 16, 2500 << 16));

// on the receiver: reconstruct the value at any sample index
let y3 = evaluate_linear_fixed(slope, intercept, 3);
assert_eq!(q16_to_int(y3), 2530);
```

## Fits as laws (`law` feature)

`law::linear_law(samples, provenance)` runs `fit_linear_fixed` and returns
the result as an `alice_zip::law::SignalLaw` (re-exported from
`alice_edge::law`):

| item | value |
|------|-------|
| coefficients | the Edge fit, converted without refitting (below) |
| valid range | `[0, n − 1]`; `evaluate` outside it returns `LawError::OutOfRange` |
| evidence | `(i, samples[i])` for every sample |
| residual | RMS and maximum of `samples[i] − f(i)`, measured against the samples, so it includes the Q16 truncation of the fit |
| provenance | given by the caller |

`SignalLaw` stores ascending coefficients in `u = x / (n − 1)`, so the Edge
coefficients `(slope_q, intercept_q)` become

```text
c0 = intercept_q / 2¹⁶
c1 = slope_q · (n − 1) / 2¹⁶
```

Both are exact in `f64` for windows shorter than 2²² samples, and
`evaluate(i)` differs from Edge's own reconstruction
`(slope_q · i + intercept_q) / 2¹⁶` by at most `4ε(|c0| + |c1|)`
(`ε = 2⁻⁵²`), far below one Q16 step. A later window is judged with
`SignalLaw::ingest` (`law::sample_points` turns samples into `(x, y)`
points): `Supports`, `ParameterUpdate`, `ResidualGrew`, `Breaks`,
`OutOfRange` or `NoEvidence`.

```rust
use alice_edge::law::{linear_law, sample_points, IngestPolicy, Provenance, Verdict};

let prov = Provenance::new("sensor 3, window 17", "fit_linear_fixed");
let law = linear_law(&[1, 3, 5, 7, 9], prov).unwrap();
assert_eq!(law.coefficients(), &[1.0, 8.0]); // y = 1 + 2x = 1 + 8u
assert!(law.evaluate(4.5).is_err());          // beyond the window

let policy = IngestPolicy { abs_tolerance: 0.5, break_factor: 4.0 };
assert!(matches!(law.ingest(&sample_points(&[1, 3, 5, 7, 9]), &policy), Verdict::Supports { .. }));
assert!(matches!(law.ingest(&sample_points(&[0, 1, 4, 9, 16]), &policy), Verdict::Breaks { .. }));
```

The module needs `alloc` but not `std`; CI builds it for
`thumbv7em-none-eabihf`. A runnable example is `examples/edge_law.rs`
(`cargo run --example edge_law --features law`), and the closed-form checks
are in `tests/edge_law.rs`.

## Features

| Feature | Default | Dependencies | Description |
|---------|---------|--------------|-------------|
| *(none)* | yes | none | `no_std` core: linear / constant / quadratic / cubic / piecewise fits, Q16.16 helpers, ring buffer, Kalman filters and sensor fusion |
| `std` | no | | host builds: delta coding, robust fit, OTA, telemetry, watchdog |
| `law` | no | alice-zip (without `std`) | fits as `SignalLaw` (`no_std` + `alloc`) |
| `zip` | no | alice-zip (`std`, `lzma`) | residual compression bridge |
| `codec` | no | alice-codec | wavelet denoising bridge |
| `db` | no | alice-db (`fs`) | coefficient persistence bridge; every write and read returns ALICE-DB's error (a refused read is an error, not an empty result; the model approximation only with `ResidualPolicy::ModelOnly`) |
| `ml` | no | alice-ml | 1.58-bit ternary object classification |
| `sdf` | no | alice-sdf | point cloud to SDF compression |
| `depth-camera` | no | rusb | USB depth camera capture |
| `asp` | no | libasp (implies `sdf`, `ml`) | ALICE Streaming Protocol bridge |
| `edge-pipeline` | no | | `depth-camera` + `sdf` + `ml` + `asp` |
| `sensors` | no | serde | simulated sensor drivers |
| `sensors-hw` | no | rppal, serialport | I2C / SPI / GPIO / UART drivers (Linux) |
| `mqtt` | no | rumqttc | MQTT publish (AWS IoT Core, Azure IoT Hub, Mosquitto) |
| `dashboard` | no | alice-analytics | HyperLogLog / Count-Min sketch dashboard |
| `ffi` | no | | C ABI (22 functions), see `bindings/` |
| `pyo3` | no | pyo3, numpy | Python bindings |

Sensor pin assignments are in the `sensors` module documentation.

## Q16.16 format

```text
16 bits integer | 16 bits fraction, range −32768.0 … +32767.99998
2550 (25.50 °C × 100) → 2550 · 65536 = 167 116 800
```

`fit_linear_fixed` returns `(slope, intercept)` in Q16.16;
`evaluate_linear_fixed(slope, intercept, x)` takes `x` as an integer sample
index and returns Q16.16.

## Determinism

The coefficients are what leaves the device, so two devices fitting the same
samples have to produce the same bits. Where that holds, and where it does
not:

| path | arithmetic | bit-exact across targets |
|------|-----------|--------------------------|
| `fit_linear_fixed`, `fit_linear_simd`, `fit_constant_fixed`, `fit_quadratic_fixed`, `fit_cubic_fixed`, the evaluators, `piecewise`, `robust`, `delta`, `ring_buffer` | integer `i64` / `i128`, explicit wrapping or checked | yes, with no float involved at all |
| `sensor_fusion` (Kalman 1D / 2D, inverse-variance fusion), `q16_to_f32`, `law` (`f64` coefficients and residual) | IEEE 754 `+ - * /` and `sqrt` only | yes — IEEE 754 requires these to be correctly rounded |
| the `sin` / `cos` / `ln` / `exp` used by the simulated sensors, the depth-camera simulation and the classifier's feature extractor | [`alice-det-math`](https://crates.io/crates/alice-det-math), whose kernels are built from the operations above in a fixed evaluation order | yes |
| `object_classifier`'s ternary matrix-vector kernel, `sdf_compress` | owned by `alice-ml` / `alice-sdf` | those crates' own guarantee, not re-stated here |

Nothing in the crate calls the platform `libm`: `clippy.toml`
`disallowed-methods` rejects the inherent `f32` / `f64` transcendentals
(including `powi`, whose multiplication tree has an unspecified association
order) for the library, the tests, the examples and the benches alike, and the
gates run clippy with `--all-targets ... -D warnings`. `sqrt` and `mul_add`
are deliberately *not* rejected: IEEE 754 requires both to be correctly
rounded, so both are bit-identical everywhere.

`tests/determinism_golden.rs` drives each module through its public entry
points with fixed inputs, serialises every output with `to_bits()` and
compares a SHA-256 against a constant; CI runs it on `x86_64` Linux and
Windows and `aarch64` Linux and macOS, so a platform-dependent operation is a
test failure rather than a silent divergence. Each scenario also asserts a
minimum payload length, so a scenario that stopped exercising its module
fails instead of passing with the hash of an empty buffer.

<!-- claim-test: golden_q16_linear, golden_adaptive_polyfit, golden_constant_fit, golden_simd_fit, golden_sensor_fusion, golden_ring_buffer, golden_det_math_kernels, golden_robust, golden_piecewise, golden_delta, golden_law, golden_object_features -->

Results are recorded under a name for that arithmetic: `alice_edge::SEMANTICS_ID`
(re-exported from `alice-det-math`). Store it next to coefficients that leave
the device; two results compare as numbers only when they carry the same
value. With the `law` feature, `SignalLaw::law_id(&alice_edge::SEMANTICS_ID)`
gives one identifier for the law and the arithmetic together. Both are pinned
in `tests/determinism_golden.rs`, the law identifier against values computed
outside this toolchain from the encoding alice-zip publishes.

<!-- claim-test: golden_semantics_id, golden_law_id -->

Out of scope: targets that compute `f32` in a wider register and round once
(`i586` and older x86 without SSE2), and builds that enable fast-math or
otherwise let the compiler reassociate float arithmetic. Neither is built in
CI.

Degenerate inputs are covered separately. `tests/panic_contract.rs` states,
for each entry point, which of `Err` / an early return / a specific value / a
panic is the contract for an empty window, a single sample, a zero divisor, an
out-of-range or negative argument, a non-finite argument and an integer
overflow, and asserts that one. The overflow cases are asserted in both build
profiles, because Rust panics on overflow with `debug_assertions` and wraps
without it.

<!-- claim-test: an_empty_window_gives_the_zero_fit_from_every_entry_point, every_division_by_a_derived_number_guards_its_zero, the_polynomial_evaluators_overflow_in_debug_and_wrap_in_release, samples_outside_the_q16_range_wrap_rather_than_saturate, the_simd_kernels_agree_with_the_scalar_fit_bit_for_bit -->

## Platforms

CI tests on `x86_64` Linux and Windows and `aarch64` Linux and macOS, builds
the `no_std` core (alone, with `ffi`, and with `law`) for
`thumbv7em-none-eabihf`, and lints `sensors-hw` on Linux. Other targets are
not built in CI.

## Bindings

- C / C++: `bindings/alice_edge.h`, built with `cargo build --release --features ffi`
- Unity (C#): `bindings/AliceEdge.cs` (`DllImport` wrapper)
- Python: `maturin develop --features pyo3`

The C ABI catches panics (with `std`) and reports them through
`alice_edge_last_error()`.

The robust entry points (`fit_linear_robust`, `filter_outliers_mad`) take the
outlier factor `k` as a plain integer multiplier of the MAD (`3` means three
MADs; it is not Q16.16) and refuse `k <= 0` with
`RobustError::NonPositiveK`. Through the C ABI both return a status code
(`ALICE_EDGE_OK`, or `ALICE_EDGE_ERR_NON_POSITIVE_K` for `k <= 0`) and write
their result through an out-parameter; the C# wrapper raises
`AliceEdgeException`.

<!-- claim-test: a_non_positive_mad_factor_is_refused, non_positive_k_is_refused_by_both_functions -->

## Minimum supported Rust version

1.87 (`rust-version` in `Cargo.toml`). CI compiles the library on exactly
that toolchain, without features and with the docs.rs feature set.

## Building and testing

```bash
cargo test --lib --no-default-features   # no_std core
cargo test --features std,law            # lib, oracles, doc tests
cargo test --test edge_law --features law
cargo test --test determinism_golden --features std,law,ml   # cross-platform bit-exactness
cargo test --test panic_contract --features std,law,ml       # degenerate inputs
cargo test --release --test panic_contract --features std    # the overflow paths wrap here
cargo bench --no-run                    # Criterion benches in benches/
cargo build --lib --target thumbv7em-none-eabihf --no-default-features --features law
scripts/preflight.sh            # every CI gate locally
scripts/preflight.sh --quick    # static checks, clippy, builds, docs, cargo test --lib
```

## Related crates

- [ALICE-Zip](https://github.com/ext-sakamoro/ALICE-Zip) — `SignalLaw` and residual compression
- [ALICE-DB](https://github.com/ext-sakamoro/ALICE-DB) — model-based storage
- [ALICE-Codec](https://github.com/ext-sakamoro/ALICE-Codec) — wavelet coding
- [ALICE-Analytics](https://github.com/ext-sakamoro/ALICE-Analytics) — sketches for the dashboard
- [ALICE-Streaming-Protocol](https://github.com/ext-sakamoro/ALICE-Streaming-Protocol) — streaming bridge

<!-- law-integration:begin law=41db3adbbe1e04ab4a27e3ce1951bfd75389b817 -->
## ALICE-Law integration

Generated by ALICE-Law `scripts/gen-law-readme-block.py` from ALICE-Law `41db3adbbe1e` and this repository's `alice-law-conformance.toml`; do not edit it by hand.

| | |
|---|---|
| Release train | alice-edge 0.2.1 (published), law dependency: yes |
| Arithmetic | alice-det-math 0.5.0 in the train, arithmetic identifier `135c407d4249c1c200e4d1be79a12e848ce5c355fc3a918da965849e697ef000`; this crate requires `alice-det-math = "0.5.1"` |
| End to end check | covered (`alice-edge = "0.2"`), digest `966a78784bb4e3341ac6627d519db896f398176b71c2402b188294db6c8e3728`, the same on aarch64 and x86_64 |

### Conformance (G1-G14)

| gate | status | note |
|---|---|---|
| G1 no-libm-direct | unverified | unverified: no banned call is known, and every feature is now linted (edge-pipeline and pyo3 were not, until the two clippy steps at ci.yml lines 232 and 239), … |
| G2 ieee-max-min | known unmet | unmet: no ban on f32::max / f32::min / f64::max / f64::min in clippy.toml, no Law helper and no default-deny manifest of raw_* sites. Library sites, measured … |
| G3 no-outer-fneg-of-fma | unverified | unverified: no violation is known; the only mul_add sites are src/sensors.rs:839 (lat_deg.mul_add(-100.0, lat_raw)) and src/sensors.rs:848 … |
| G4 debug-release-arch-goldens | known unmet | unmet: (1) no signed-zero or exact-cancellation case is checked against fixed bit patterns (tests/ has no -0.0 literal and no cancellation case; … |
| G5 release-profile-test-leg | known unmet | unmet: the release profile runs only tests/panic_contract.rs (.github/workflows/ci.yml:101 std and :159 full feature set, on the four-OS matrix including the … |
| G6 semantics-id | known unmet | unmet: src/lib.rs:151 re-exports alice-det-math's identifier instead of computing one that folds this crate's own pins (the Q16.16 normal equations, … |
| G7 handshake-carries-id | unverified | unverified: the boundaries within one exchange are listed but not ruled on: src/ffi.rs (22 #[no_mangle] extern "C" functions taking Q16.16 coefficients and … |
| G8 persisted-semantics-id | unverified | unverified: the durable forms are listed but not ruled on: src/object_classifier.rs:287 save_weights and :309 load_weights (float classifier weights in a file, … |
| G9 persisted-meaning-change-is-major | unverified | unverified: depends on G6 and G8; this crate defines no identifier of its own whose history could be audited (src/lib.rs:151 re-exports alice-det-math's, whose … |
| G10 no-run-varying-artifacts | pass | the 24 production .elapsed() reads sit under scoped clippy::disallowed_methods allows with a reason (stage latency and frame pacing in edge_pipeline.rs, … |
| G11 gate-self-tests | known unmet | unmet: the clippy ban that carries G1 and the compiled layer of G10 (clippy.toml lines 22 to 96, run by .github/workflows/ci.yml lines 217 to 227) has no … |
| G12 public-repo-text-rules | pass | the workflow fetches ALICE-Law's scripts/public_text_check.py at the revision in .github/public-text-pin (ca96d9214ffe0da43e9cb034417a4f6c041ba4be) and refuses … |
| G13 no-unspecified-order | unverified | unverified: no violation is known since d06f4da: src/edge_pipeline.rs:368 (voxel_grid_downsample) and src/telemetry.rs:158 (TelemetryHub, read by … |
| G14 float-environment-invariants | unverified | unverified: no flag is known; every place a flag can come from was read at this commit: Cargo.toml [profile.release] and [profile.bench] set opt-level, lto and … |

### Features and the guarantee

| feature | guarantee | note |
|---|---|---|
| `asp` | not declared |  |
| `codec` | not declared |  |
| `dashboard` | not declared |  |
| `db` | not declared |  |
| `depth-camera` | not declared |  |
| `edge-pipeline` | not declared |  |
| `ffi` | not declared |  |
| `law` | not declared |  |
| `ml` | not declared |  |
| `mqtt` | not declared |  |
| `pyo3` | not declared |  |
| `sdf` | not declared |  |
| `sensors` | not declared |  |
| `sensors-hw` | not declared |  |
| `std` | not declared |  |
| `zip` | not declared |  |
<!-- law-integration:end -->

## License

MIT OR Apache-2.0 ([LICENSE](LICENSE), [LICENSE-APACHE](LICENSE-APACHE))
