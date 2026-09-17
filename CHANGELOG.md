# Changelog

All notable changes to ALICE-Edge will be documented in this file.

## [Unreleased]

### Added
- `tests/analytic_oracle.rs` — 閉形式 oracle 9 本 (CLAUDE.md § 解析解突合テスト規律、2026-09-17): 整数直線 / 2 次 / 3 次多項式の exact 復元 (Q16 dyadic、SIMD path bit 一致、f64 正規方程式との一致)、折れ線 2 本の exact 分割、robust fit の外れ値耐性 (1 %)、delta coding 往復、Kalman 1D の Riccati 漸化式逐語 + 定常解 P* = (−Q + √(Q²+4QR))/2、Kalman 2D の等速復元 (dt 独立)、逆分散重み融合 + 10σ 外れ値棄却、ring buffer FIFO、`sdf` feature: 球面点群 → 球 primitive (中心 / 半径 2 cm 以内) CI に oracle step (既存 test step は `--lib` で tests/ が走っていなかった)

### Fixed (oracle 先行 red 2 → 修正)
- **`fit_piecewise_linear` が折れ点で切れない**: 「1 本の直線 fit の最大残差地点で分割」は 2 直線を 1 本で fit した残差が端点で最大になるため、exact な 2 直線 (折れ点 40) を 39 / 43 / … と 5 segment 以上に刻んでいた → 左右 2 本の残差和が最小の k (optimal single break) で分割、exact な折れ線は SSE 0 の折れ点で 2 segment
- **`fit_linear_robust` が外れ値の「修復」で自ら bias を入れていた**: 生値の global median と MAD (傾向のある系列では MAD ≈ n/4·slope で粗い外れ値しか捕まらない) で検出し、捕まえた点を global median に置換 → ramp `y = x` の x = 10 に外れ値 1 個で slope 0.98 (2 % 低下) → 残差 MAD + 直線予測値で補完 + mask 収束まで反復 (1 % 以内)

### Fixed
- **FFI 14 関数の panic 隔離** (`src/ffi.rs`): 全 `extern "C"` を `ffi_guard(sentinel, || ..)` で包み、panic は host を落とさず sentinel (zero `Alice*Result` / NaN / false / 0 / ()) + `alice_edge_last_error()` (新規、`alice_edge_clear_last_error` / `alice_edge_free_error_string` も) で通知 `alice_delta_encode` / `_decode` の panic 時 sentinel は 0 (`num_pairs` を返すと「全件処理した」と誤読される) `[profile.release] panic = "abort"` を撤去 (abort では `catch_unwind` が機能しない) release profile で guard test 通過

### Changed
- `ffi` の panic 隔離は `std` feature 時のみ有効 (bare-metal `no_std + ffi` には unwinding runtime が無く panic は panic handler で halt するので、`ffi_guard` は本体を直接呼び、`alice_edge_last_error` は常に null) thumbv7em の `no_std + ffi` build は従来通り通る

### Fixed
- `filter_outliers_mad`: deviations were computed as `(x - median).abs()` in `i32` (overflow for full-range samples) and the threshold `k * MAD` was truncated to `i32`, so a large `k` wrapped negative and replaced every sample by the median. Deviations and the threshold are now `i64` with saturating multiply (found by `fuzz_fit_linear`)
- `fit_cubic_fixed`: the fraction-free elimination multiplied `i128` entries by the running product of pivots and overflowed (panic in debug, wrapped coefficients in release) on windows longer than a few dozen samples. It now uses Bareiss elimination (exact division, entries bounded by minors) with checked arithmetic and degrades to the quadratic fit on overflow. `fit_quadratic_fixed` accumulators moved from `i64` to checked `i128` and degrade to the linear fit on overflow. Measured exact-integer capacity: cubic ≤ 64 samples (32 at full `i32` range), quadratic ≤ 4096 (found by `fuzz_polyfit`)
- `asp` feature implies `ml`: `asp_bridge` uses `object_classifier::ObjectClass`, so `--features asp` alone did not compile (found by the feature powerset job)
- `q16_to_f32` is no longer gated on `std` (it is plain `f32` arithmetic), so `--features ffi` compiles in `no_std` and on `thumbv7em-none-eabihf`
- rustdoc: three doc comments used `[name:type]` byte-layout notation that rustdoc parsed as broken intra-doc links
- `sensors-hw` did not compile (the feature was never built in CI): `Bme280Sensor::new` opened the I2C bus inside a `const fn` and panicked on failure (now opened in `init()` with a `SensorError`), `Spi::transfer` takes separate read / write buffers in rppal 0.19, and the GPS hardware loop ignored `interval` (now the minimum spacing between accepted NMEA fixes). Verified on a Raspberry Pi (aarch64 Linux) and gated by a `clippy -D warnings` step on the ubuntu job
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
