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
| `db` | no | alice-db | coefficient persistence bridge |
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

## License

MIT OR Apache-2.0 ([LICENSE](LICENSE), [LICENSE-APACHE](LICENSE-APACHE))
