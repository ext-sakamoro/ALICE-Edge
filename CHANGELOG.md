# Changelog

All notable changes to ALICE-Edge will be documented in this file.

## [Unreleased]

### Fixed
- `zip` feature: `zip_bridge` uses `compress_residual_quantized` / `decompress_residual_quantized`, which the crates.io `alice-zip` crate did not provide until 0.5.0 (they only existed in the `libalice` CLI crate); the feature compiled in CI solely because the sibling was stubbed. Now `alice-zip = "0.5"` with the `lzma` feature, the CI stub is removed and `cargo test --features std,zip` runs in CI
- `db` feature: `alice-db` is a crates.io dependency (0.2.0-beta.2); the CI stub declared version 0.1.0 and could not satisfy `^0.2.0-beta.1`, so dependency resolution failed in every CI job
- All ALICE sibling dependencies (`alice-codec` / `alice-sdf` / `alice-ml` / `alice-analytics`) come from crates.io; the CI "dependency stubs" step (empty sibling crates) is removed, so the feature builds now compile against the real crates
- `dashboard`: `CountMinSketch` / `HyperLogLog` are imported from `alice_analytics::sketch` (they are not re-exported at the crate root); this only compiled against the empty stub

### Added
- `asp` feature restored (`libasp` 1.0 on crates.io), implies `sdf`; `edge-pipeline` includes it again

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
