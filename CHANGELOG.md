# Changelog

All notable changes to ALICE-Edge will be documented in this file.

## [Unreleased]

### Fixed
- C ABI の `alice_edge_version()` は版を文字列 `"0.1.0"` で埋め込んでいたので、0.1.1 でも `"0.1.0"` を返していた `Cargo.toml` の版 (`CARGO_PKG_VERSION`) から作るようにし、試験も同じ値と比べる
- C ABI の文書が `alice_fit_linear_robust` の `mad_k` を Q16.16 (`int_to_q16(2)` / `int_to_q16(3)`) と説明していたが、実装は素の整数として MAD に掛けていた 文書どおり `int_to_q16(3)` (= 196608) を渡すと閾値が MAD の 196608 倍になり、外れ値は 1 つも除外されなかった `src/ffi.rs` / `bindings/alice_edge.h` / `bindings/AliceEdge.cs` を素の整数の説明に直し、FFI の試験 `test_ffi_fit_linear_robust` は外れ値を含む ramp で外れ値が除外されること (Rust の `fit_linear_robust` と一致し、真の直線から slope 3 / intercept 30 以内) と、Q16.16 の `k` では素の fit が返ることを確かめる (以前は何も検査していなかった)
- `cargo test` (feature 指定なし) が build できなかった `tests/analytic_oracle.rs` が `std` でだけ在る `piecewise::fit_piecewise_linear` を cfg 無しで import していた import と該当の試験に `#[cfg(feature = "std")]` を付け、CI と `scripts/preflight.sh` に全 integration test を feature 無しで build する step (`cargo test --tests --no-run --no-default-features`) を足した CI は試験 target を feature 付きで個別に回していたので出なかった

### Added
- `src/ffi.rs` unit tests for the panic path of `alice_fit_linear_robust` and `alice_filter_outliers_mad`: a panic inside the guarded body returns `ALICE_EDGE_ERR_PANIC` (not `ALICE_EDGE_OK`), `alice_edge_last_error()` carries the panic message, `*out` is the documented `{0, 0}` / `*out_written` is 0 and `out` is not written. No input is known to make the robust fit panic in both build profiles, so the panic is raised by a hook compiled only for the crate's unit tests (`cfg(test)`); library and release builds contain neither the hook nor its call sites
- `scripts/ffi_status_check.py`: the C ABI status codes (`ALICE_EDGE_OK` / `ERR_NULL_POINTER` / `ERR_NON_POSITIVE_K` / `ERR_PANIC`) must have the same name and value in `src/ffi.rs`, `bindings/alice_edge.h` and `bindings/AliceEdge.cs` (`AliceStatus`); each file must define exactly 4, so a missing or renamed definition fails instead of comparing nothing. `scripts/test_ffi_status_check.py` covers swapped values in each file, a missing entry, an extra entry and a renamed class. Run in the CI docs lint job on three OS and in `scripts/preflight.sh`
- `law` feature and `law::linear_law(samples, provenance)`: the `fit_linear_fixed` result as an `alice_zip::law::SignalLaw` (alice-zip 0.5.1, used without `std`, so the feature builds for `thumbv7em-none-eabihf`). The Q16.16 coefficients are converted without refitting to the normalised basis `u = x / (n − 1)` (`c0 = intercept_q / 2¹⁶`, `c1 = slope_q · (n − 1) / 2¹⁶`); the valid range is `[0, n − 1]`, the evidence is the samples, and the residual is measured against the samples (it includes the Q16 truncation of the fit and the wrap of samples outside the Q16 range). Fewer than two samples return `LawError::TooFewPoints`
- `law::sample_points(samples)`: samples as `(x, y)` points, for judging a later window with `SignalLaw::ingest`; the law types are re-exported from `alice_edge::law`
- `tests/edge_law.rs`: 10 closed-form checks (exact line, residual of a parabola, Q16 truncation in the residual, agreement with Edge's reconstruction within `4ε(|c0| + |c1|)`, valid range, out-of-range refusal, verdicts `Supports` / `Breaks` / `OutOfRange` / `NoEvidence`, degenerate and full-range inputs) and `examples/edge_law.rs`
- `README_JP.md`
- CI: test matrix on `ubuntu-latest` / `macos-latest` / `windows-latest` / `ubuntu-24.04-arm`, `law` in the thumbv7em build and clippy, `tests/edge_law.rs` step, docs lint job on three OS (`scripts/docs_lint.py` + `scripts/test_docs_lint.py`), `scripts/stub_guard.sh`
- `tests/determinism_golden.rs`: 12 scenarios (`q16_linear`, `adaptive_polyfit`, `constant_fit`, `simd_fit`, `sensor_fusion`, `ring_buffer`, the transcendental kernels at the arguments the sensor and depth-camera simulations feed them, `robust`, `piecewise`, `delta`, `law`, the classifier's feature extractor) that drive a module from its public entry points, serialise every output with `to_bits()` and compare a SHA-256 against a constant. Run on all four OS legs of the CI matrix, so a platform-dependent float operation is a test failure. Each scenario also asserts a minimum payload length, so a scenario that stopped exercising its module fails rather than passing with the hash of an empty buffer
- `tests/panic_contract.rs`: 23 cases stating, per entry point, which of `Err` / an early return / a specific value / a panic is the contract for an empty window, a single sample, a guarded division, an out-of-range or negative argument, a non-finite argument, the Q16.16 wrap and the integer overflow paths, and asserting that one. The overflow cases run in both build profiles (Rust panics on overflow with `debug_assertions` and wraps without it), and every `should_panic` names the message it expects. One case is `#[ignore]`d and records a known defect: `fit_linear_fixed` multiplies by 2¹⁶ with a shift, which does not check the value for overflow, so a window longer than the measured capacity (6410 samples at slope 1, 4870 at slope 3, 2027 at slope 100) returns a wrong slope instead of degrading; `fit_linear_simd` shares it, while `fit_quadratic_fixed` and `fit_cubic_fixed` solve the same system with checked `i128` and are exact
- `clippy.toml` with `disallowed-methods`: the inherent `f32` / `f64` transcendentals (and `powi`, whose multiplication tree has an unspecified association order) are rejected for the library, the tests, the examples and the benches. `sqrt` and `mul_add` are deliberately absent — IEEE 754 requires both to be correctly rounded, so both are bit-identical on every target
- README / README_JP: a `Determinism` section listing, per path, which arithmetic it uses and whether it is bit-exact across targets, together with what is out of scope (x87 without SSE2, fast-math)
- CI and `scripts/preflight.sh`: steps for the determinism goldens (no features / `std` / full feature set) and the panic contract (`std` and full feature set, debug and release). `preflight.sh --quick` runs the goldens and the debug-profile panic contract
- `SEMANTICS_ID` re-exported at the crate root from `alice-det-math`: the identifier of the arithmetic every float transcendental in this crate goes through, to be recorded next to stored or transmitted coefficients and passed to `law::SignalLaw::law_id`. `tests/determinism_golden.rs` pins its hex (`d2209b30…691398e`) and asserts it equals `alice_det_math::SEMANTICS_ID` (`golden_semantics_id`)
- `tests/determinism_golden.rs` `golden_law_id` (feature `law`): the `law_id` of two fixed `linear_law` fits (a rising and a falling window), pinned against values computed outside this toolchain with SHA-256 over the encoding alice-zip publishes and coefficients derived by hand from the samples

### Changed
- **破壊的変更:** `robust::filter_outliers_mad(data, k)` と `robust::fit_linear_robust(data, k)` は `k <= 0` を `Err(RobustError::NonPositiveK { k })` で拒否する 戻り値は `Result<Vec<i32>, RobustError>` / `Result<(i32, i32), RobustError>` に変わる `k` は MAD に掛ける素の整数 (`3` は MAD の 3 倍) で、`k <= 0` では `|偏差| > k · MAD` が中心ちょうど以外のすべての標本を外れ値にし、窓を壊した結果を正常値として返していた 新しい `RobustError` は `#[non_exhaustive]` で `Display` と `std::error::Error` を実装する
- **破壊的変更:** C ABI: `alice_fit_linear_robust(data, len, k, out)` と `alice_filter_outliers_mad(data, len, k, out, out_capacity, out_written)` は状態 code (`int32_t`) を返し、結果は出力引数に書く 状態は `ALICE_EDGE_OK` (0) / `ALICE_EDGE_ERR_NULL_POINTER` (-1) / `ALICE_EDGE_ERR_NON_POSITIVE_K` (-2) / `ALICE_EDGE_ERR_PANIC` (-3) で、`k <= 0` の理由は `alice_edge_last_error()` で読める 以前は `k <= 0` を受け付け、panic 時は 0 の結果 (正常な fit と区別できない) を返していた `bindings/alice_edge.h` に定数と `alice_edge_last_error` / `alice_edge_clear_last_error` / `alice_edge_free_error_string` の宣言を追加し、`bindings/AliceEdge.cs` は新しい signature と、失敗時に `AliceEdgeException` (状態と message を持つ) を投げる `AliceModel.FitLinearRobust` / `AliceModel.FilterOutliersMad` を持つ
- **Breaking (wire format of `zip_bridge` batches):** `alice-zip` を 0.7 から 0.9 に上げた 0.8 から量子化残差の container を deflate で書き、codec の byte を記録する (0.7 は LZMA で codec の byte 無し) 0.7 で書いた batch は今も同じ値に復号できる 新しく書いた batch は受信側に `alice-zip` 0.8 以上が要る (0.8.0 と 0.9.0 は bridge の引数で同じ byte を書き、互いの batch を同じ値に戻す) 0.7 の受信側は error で拒否する (codec の印を長さとして読み「data truncated」を返す、値は返さない) 送信側を先に上げる fleet では受信側を先に上げる `tests/zip_compat.rs` が公開版の `alice-zip` 0.7.0 と 0.8.0 (dev-dependency) を旧い書き手・読み手として、旧 batch の復号・新 batch の往復・0.7 による拒否・0.8 と 0.9 の相互の読み書きの 4 点を確かめる (CI の全 OS、0 件実行で失敗) `zip` feature は旧 batch を読むために `lzma` を有効のまま残す
- `zip_bridge` の doc が 8 bit 量子化と書いていたが実装は 16 bit だった 記述を実装に合わせた
- benchmark target renamed to `edge_bench` (`cargo bench --bench edge_bench`); the MQTT examples default their client id to `alice-edge`
- `alice-det-math` requirement raised from `0.3` to `0.4` and `alice-zip` from `0.5.1` to `0.7`, so every feature set resolves a single `alice-det-math` (0.4.0), the same series as `alice-zip` 0.7, `alice-db` and `alice-analytics` 0.3. 0.4 changes the bits of `atan64` / `atan2_64` only, neither of which this crate calls; the alice-zip API used here (`law::SignalLaw` and `compression::{compress,decompress}_residual_quantized`) is unchanged apart from the added `SignalLaw::law_id`. Every golden hash in `tests/determinism_golden.rs` and every case in `tests/panic_contract.rs` is unchanged
- `zip` feature enables alice-zip's `std` and `lzma` features explicitly (the dependency is declared without features so that `law` can use alice-zip without `std`); the resolved feature set of `zip` is unchanged
- `alice-zip` requirement raised from `0.5` to `0.5.1` (`alice_zip::law` is first published in 0.5.1)
- The crate-level example passes `x` to `evaluate_linear_fixed` as an integer sample index, as documented, and asserts the reconstructed value
- README rewritten: what the crate is not for, installation with `cargo add`, the `law` entry point, features table, platforms tested in CI; benchmark figures without a recorded measurement date are removed
- `cargo audit` keeps its advisory database under `target/advisory-db`
- Every float transcendental goes through `alice-det-math` 0.3 (a new, mandatory, dependency-free `no_std` dependency) instead of the platform `libm`: the four `sin` / `cos` in the simulated sensor and depth-camera paths, the two `ln` and the `exp` in the classifier's feature extractor and softmax, and the trigonometry in `tests/analytic_oracle.rs`. The platform `libm` differs in the last ulp between macOS, glibc, MSVC and wasm, and the coefficients a fit produces are what gets transmitted and stored, so the same samples have to give the same bits on every device. The `std` feature now also enables `alice-det-math/std`
- **Behavior change:** `TernaryClassifier::new` and `TernaryClassifier::from_weights` reject `num_classes = 0` at construction (`num_classes must be at least 1`). Previously only the upper bound was checked, so a zero was accepted and the failure surfaced later inside `classify`, as a generic `index out of bounds` several calls away from the argument that caused it. `from_weights` additionally checks each weight vector's length against the architecture (`w1` is `HIDDEN_DIM · FEATURE_DIM`, `w2` is `HIDDEN_DIM · HIDDEN_DIM`, `w3` is `num_classes · HIDDEN_DIM`) and names the layer and both lengths, where the kernel constructor one level down reported a bare assertion failure. `TernaryClassifier::load_weights` inherits both checks, so a weight file whose header does not match its payload is now refused with a message that says which layer is wrong
- `law::linear_law` documents that the residual, measured against the samples rather than reported by the fit, is an independent check on the fit itself: a window the fit cannot represent shows up as a residual orders of magnitude above the Q16 step
- `tests/determinism_golden.rs` `golden_robust`: `k ∈ {0, 1, 3, 10, i32::MAX}` をまとめた 1 本の hash (`30f195a6…`) を `k` ごとの hash 4 本 (`k = 1` `f7baf928…` / `k = 3` `07caa63a…` / `k = 10` `75950f93…` / `k = i32::MAX` `a8de2b48…`) に分けた 4 本は変更前の実装で記録した値で、変更後も同じ `k <= 0` は全窓で拒否されることを検査する fuzz target `fuzz_fit_linear` と `tests/support/robust_reference.rs` の契約も `k > 0` のみが規則の対象で、`k <= 0` は拒否されることを検査する
- CI (`fuzz.yml`): time-boxed run は blocking になった crash / leak / out-of-memory / 入力ごとの timeout (30 秒) と、実行件数 0 の run は red になる target ごとの実行件数・coverage・結果を step summary に書き、失敗時は crash 入力と log を artifact に上げる 回帰入力は `fuzz/regressions/<target>/` に置き、time-boxed run の前に replay する `ci/**` branch でも走り、branch の run は同時 2 job まで
- CI (`fuzz.yml`): workflow の token を `permissions: contents: read` に絞り、使う action 4 本 (`actions/checkout` v5.1.0 / `dtolnay/rust-toolchain` nightly / `Swatinem/rust-cache` v2.9.2 / `actions/upload-artifact` v6.0.0) を commit SHA で固定した (tag が付け替えられても実行される中身は変わらない)

## [0.1.1] - 2026-09-17

### Added
- `tests/analytic_oracle.rs` — 閉形式 oracle 9 本: 整数直線 / 2 次 / 3 次多項式の exact 復元 (Q16 dyadic、SIMD path bit 一致、f64 正規方程式との一致)、折れ線 2 本の exact 分割、robust fit の外れ値耐性 (1 %)、delta coding 往復、Kalman 1D の Riccati 漸化式逐語 + 定常解 P* = (−Q + √(Q²+4QR))/2、Kalman 2D の等速復元 (dt 独立)、逆分散重み融合 + 10σ 外れ値棄却、ring buffer FIFO、`sdf` feature: 球面点群 → 球 primitive (中心 / 半径 2 cm 以内) CI に oracle step (既存 test step は `--lib` で tests/ が走っていなかった)

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
- `sensors-hw` did not compile (the feature was never built in CI): `Bme280Sensor::new` opened the I2C bus inside a `const fn` and panicked on failure (now opened in `init()` with a `SensorError`), `Spi::transfer` takes separate read / write buffers in rppal 0.19, and the GPS hardware loop ignored `interval` (now the minimum spacing between accepted NMEA fixes). Verified on an aarch64 Linux board and gated by a `clippy -D warnings` step on the ubuntu job
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
