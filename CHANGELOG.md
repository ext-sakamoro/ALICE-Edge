# Changelog

All notable changes to ALICE-Edge will be documented in this file.

## [Unreleased]

### Fixed
- `filter_outliers_mad`: deviations were computed as `(x - median).abs()` in `i32` (overflow for full-range samples) and the threshold `k * MAD` was truncated to `i32`, so a large `k` wrapped negative and replaced every sample by the median. Deviations and the threshold are now `i64` with saturating multiply (found by `fuzz_fit_linear`)
- `fit_cubic_fixed`: the fraction-free elimination multiplied `i128` entries by the running product of pivots and overflowed (panic in debug, wrapped coefficients in release) on windows longer than a few dozen samples. It now uses Bareiss elimination (exact division, entries bounded by minors) with checked arithmetic and degrades to the quadratic fit on overflow. `fit_quadratic_fixed` accumulators moved from `i64` to checked `i128` and degrade to the linear fit on overflow. Measured exact-integer capacity: cubic ≤ 64 samples (32 at full `i32` range), quadratic ≤ 4096 (found by `fuzz_polyfit`)
- `asp` feature implies `ml`: `asp_bridge` uses `object_classifier::ObjectClass`, so `--features asp` alone did not compile (found by the feature powerset job)
- `q16_to_f32` is no longer gated on `std` (it is plain `f32` arithmetic), so `--features ffi` compiles in `no_std` and on `thumbv7em-none-eabihf`
- rustdoc: three doc comments used `[name:type]` byte-layout notation that rustdoc parsed as broken intra-doc links
- `fit_linear_simd`: `is_x86_feature_detected!` is a std macro, so `--no-default-features` did not compile on x86_64 (the old CI step was `continue-on-error`); `no_std` now uses the compile-time `cfg!(target_feature = "sse2")`
- `zip` feature: `zip_bridge` uses `compress_residual_quantized` / `decompress_residual_quantized`, which the crates.io `alice-zip` crate did not provide until 0.5.0 (they only existed in the `libalice` CLI crate); the feature compiled in CI solely because the sibling was stubbed. Now `alice-zip = "0.5"` with the `lzma` feature, the CI stub is removed and `cargo test --features std,zip` runs in CI
- `db` feature: `alice-db` is a crates.io dependency (0.2.0-beta.2); the CI stub declared version 0.1.0 and could not satisfy `^0.2.0-beta.1`, so dependency resolution failed in every CI job
- All ALICE sibling dependencies (`alice-codec` / `alice-sdf` / `alice-ml` / `alice-analytics`) come from crates.io; the CI "dependency stubs" step (empty sibling crates) is removed, so the feature builds now compile against the real crates
- `dashboard`: `CountMinSketch` / `HyperLogLog` are imported from `alice_analytics::sketch` (they are not re-exported at the crate root); this only compiled against the empty stub

### Added
- `PiecewiseSegment` derives `Debug`, `Clone`, `PartialEq`, `Eq` (not `Copy`: adding it is a semver-major change per cargo-semver-checks)
- CI: `.github/workflows/security-audit.yml` (cargo audit / cargo deny / cargo machete / coverage / semver-checks / stub guard), `.github/workflows/fuzz.yml` (4 libFuzzer targets under `fuzz/`), `deny.toml`, `scripts/preflight.sh` (local reproduction of every CI gate)
- CI: `msrv` job (`rust-version = "1.87"`, compiled on that toolchain), `feature-powerset` job (`cargo hack --depth 2`, 80 combinations), `no_std` unit tests, `thumbv7em` build with `ffi`, `sensors-hw` check on Linux, examples build
- Regression tests for the two fuzz findings (`test_filter_outliers_*`, `test_fit_*_exact_integer_polynomial`, `test_fit_polynomial_extreme_inputs_do_not_panic`)
- `asp` feature restored (`libasp` 1.0 on crates.io), implies `sdf`; `edge-pipeline` includes it again

### Changed
- CI: every `continue-on-error` removed from `ci.yml`; clippy runs with `-D warnings` on `no_std`, `std` and the full feature set (it was `--cap-lints warn` and non-blocking); rustdoc `-D warnings` also covers the docs.rs feature set
- `rust-version = "1.87"` and `resolver = "3"` declared; `rust-toolchain.toml` pinned to 1.98.1 (was 1.92.0, below the 1.93 floor of cargo-semver-checks 0.50)
- `[package.metadata.docs.rs] features` lists the full feature set (everything except `pyo3` and `sensors-hw`)
- `pyo3` / `numpy` 0.27 → 0.29 (RUSTSEC-2026-0176 / RUSTSEC-2026-0177), `rumqttc` 0.24 → 0.25 (rustls 0.23)

### Removed
- `serde_json` dependency (declared under `sensors`, never used)

### Security
- `deny.toml` / `cargo audit` ignore list documents the advisories that cannot be fixed from this crate: rustls-webpki 0.102 (pinned by rumqttc 0.25.1, `mqtt` feature only) and rkyv 0.7 (alice-db 0.2.0-beta.2, `db` feature only)

## [0.1.0] - 2026-02-23

### Added
- Core `fit_linear_fixed` / `evaluate_linear_fixed` — O(1) x-sum least squares in Q16.16 fixed-point
- `zip_bridge` — (feature `zip`) ALICE-Zip coefficient compression
- `codec_bridge` — (feature `codec`) Wavelet denoising for sensor data
- `db_bridge` — (feature `db`) ALICE-DB model coefficient persistence
- `asp_bridge` — (feature `asp`) ALICE Streaming Protocol bridge
- `depth_capture` — (feature `depth-camera`) Dolphin D5 Lite depth camera driver via `rusb`
- `sdf_compress` — (feature `sdf`) SDF compression for point clouds
- `object_classifier` — (feature `ml`) 1.58-bit ternary object classification
- `edge_pipeline` — (feature `edge-pipeline`) Full depth-camera → SDF → ML → ASP pipeline
- `sensors` — (feature `sensors`) Simulated sensor drivers (GPIO, I2C, SPI, UART)
- `mqtt_bridge` — (feature `mqtt`) MQTT publish to AWS IoT / Azure / local broker
- `dashboard` — (feature `dashboard`) ALICE-Analytics metrics bridge
- `python` — (feature `pyo3`) Python bindings
- `no_std` by default (runs on Cortex-M, ESP32, AVR, RISC-V)
- 31 unit tests
