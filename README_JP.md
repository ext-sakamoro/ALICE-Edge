# ALICE-Edge

[English](README.md)

[![CI](https://github.com/ext-sakamoro/ALICE-Edge/actions/workflows/ci.yml/badge.svg)](https://github.com/ext-sakamoro/ALICE-Edge/actions/workflows/ci.yml)
[![Security](https://github.com/ext-sakamoro/ALICE-Edge/actions/workflows/security-audit.yml/badge.svg)](https://github.com/ext-sakamoro/ALICE-Edge/actions/workflows/security-audit.yml)
[![Fuzz](https://github.com/ext-sakamoro/ALICE-Edge/actions/workflows/fuzz.yml/badge.svg)](https://github.com/ext-sakamoro/ALICE-Edge/actions/workflows/fuzz.yml)

センサーデータ向けの組み込みモデル fit: 「データでなく法則を送る」

整数のセンサー sample の窓を、デバイス上で数個の Q16.16 固定小数点係数
(直線 / 定数 / 2 次・3 次 / 折れ線 / 外れ値に強い直線) に縮約し、係数だけを送る
コアは `#![no_std]` で FPU も要らない 依存は
[`alice-det-math`](https://crates.io/crates/alice-det-math) 1 本だけで
(それ自体も依存を持たない `no_std`)、float の超越関数を target 間で bit 一致
させるためにある ([決定論](#決定論) 参照) optional feature で
センサードライバ、MQTT、ダッシュボード、C ABI、Python バインディング、
他の ALICE crate へのブリッジを足せる

`law` feature を有効にすると、Edge の fit を `alice_zip::law::SignalLaw`
として返す 送信する係数に、fit 元の sample、その sample に対して測った残差、
有効範囲、出典を付けたもの

License: MIT OR Apache-2.0

## 目次

- [向かない用途](#向かない用途)
- [インストール](#インストール)
- [使用例](#使用例)
- [fit を法則として扱う (`law` feature)](#fit-を法則として扱う-law-feature)
- [Feature](#feature)
- [Q16.16 形式](#q1616-形式)
- [決定論](#決定論)
- [プラットフォーム](#プラットフォーム)
- [バインディング](#バインディング)
- [最小対応 Rust バージョン](#最小対応-rust-バージョン)
- [ビルドとテスト](#ビルドとテスト)
- [関連 crate](#関連-crate)
- [ライセンス](#ライセンス)

## 向かない用途

- 一般的な回帰: fit は sample を暗黙の位置 `x = 0, 1, …, n − 1` で受け取り、
  Q16.16 の契約は `|y| < 2¹⁵` まで それを超える値は wrap する
  (`law` の残差にはその wrap が現れる)
- 可逆な保存: 係数は窓の傾向を表すもので、sample そのものは係数から復元できない
- 送信内容の暗号化や認証
- ハードウェア抽象化層: `sensors-hw` のドライバは `rppal` と `serialport` による
  Linux 上の I2C / SPI / GPIO / UART センサー数種のみ

## インストール

```bash
cargo add alice-edge
# law の入口を使う場合 (no_std + alloc)
cargo add alice-edge --features law
```

## 使用例

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

## fit を法則として扱う (`law` feature)

`law::linear_law(samples, provenance)` は `fit_linear_fixed` を実行し、
結果を `alice_zip::law::SignalLaw` (`alice_edge::law` から再 export) で返す

| 項目 | 値 |
|------|----|
| 係数 | Edge の fit を再 fit せずに変換したもの (下記) |
| 有効範囲 | `[0, n − 1]` 範囲外の `evaluate` は `LawError::OutOfRange` |
| evidence | 全 sample の `(i, samples[i])` |
| 残差 | `samples[i] − f(i)` の RMS と最大値 sample に対して測るので fit の Q16 切り捨てを含む |
| 出典 | 呼び出し側が与える |

`SignalLaw` は `u = x / (n − 1)` の昇冪係数を持つので、Edge の係数
`(slope_q, intercept_q)` は次のようになる

```text
c0 = intercept_q / 2¹⁶
c1 = slope_q · (n − 1) / 2¹⁶
```

2²² sample 未満の窓ではどちらも `f64` で厳密で、`evaluate(i)` と Edge 自身の復元
`(slope_q · i + intercept_q) / 2¹⁶` の差は `4ε(|c0| + |c1|)` (`ε = 2⁻⁵²`) 以下
Q16 の 1 刻みよりはるかに小さい 後の窓は `SignalLaw::ingest` で判定する
(`law::sample_points` が sample を `(x, y)` 点にする) 結果は `Supports` /
`ParameterUpdate` / `ResidualGrew` / `Breaks` / `OutOfRange` / `NoEvidence`

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

この module は `alloc` を使うが `std` は使わない CI は `thumbv7em-none-eabihf`
向けに build する 実行できる例は `examples/edge_law.rs`
(`cargo run --example edge_law --features law`)、閉形式の検査は
`tests/edge_law.rs` にある

## Feature

| Feature | 既定 | 依存 | 内容 |
|---------|------|------|------|
| *(なし)* | yes | なし | `no_std` コア: 直線 / 定数 / 2 次 / 3 次 / 折れ線 fit、Q16.16 helper、ring buffer、Kalman filter とセンサー融合 |
| `std` | no | | host 向け: delta 符号化、外れ値に強い fit、OTA、telemetry、watchdog |
| `law` | no | alice-zip (`std` なし) | fit を `SignalLaw` で返す (`no_std` + `alloc`) |
| `zip` | no | alice-zip (`std`, `lzma`) | 残差圧縮ブリッジ |
| `codec` | no | alice-codec | wavelet ノイズ除去ブリッジ |
| `db` | no | alice-db | 係数の永続化ブリッジ |
| `ml` | no | alice-ml | 1.58-bit ternary 物体分類 |
| `sdf` | no | alice-sdf | 点群から SDF への圧縮 |
| `depth-camera` | no | rusb | USB depth camera の取り込み |
| `asp` | no | libasp (`sdf`, `ml` を含む) | ALICE Streaming Protocol ブリッジ |
| `edge-pipeline` | no | | `depth-camera` + `sdf` + `ml` + `asp` |
| `sensors` | no | serde | シミュレートしたセンサードライバ |
| `sensors-hw` | no | rppal, serialport | I2C / SPI / GPIO / UART ドライバ (Linux) |
| `mqtt` | no | rumqttc | MQTT publish (AWS IoT Core、Azure IoT Hub、Mosquitto) |
| `dashboard` | no | alice-analytics | HyperLogLog / Count-Min sketch のダッシュボード |
| `ffi` | no | | C ABI (22 関数)、`bindings/` を参照 |
| `pyo3` | no | pyo3, numpy | Python バインディング |

センサーのピン割り当ては `sensors` module の doc にある

## Q16.16 形式

```text
16 bits integer | 16 bits fraction, range −32768.0 … +32767.99998
2550 (25.50 °C × 100) → 2550 · 65536 = 167 116 800
```

`fit_linear_fixed` は `(slope, intercept)` を Q16.16 で返す
`evaluate_linear_fixed(slope, intercept, x)` は `x` を整数の sample index として受け取り
Q16.16 を返す

## 決定論

デバイスから出ていくのは係数なので、同じ sample を当てはめた 2 台は
同じ bit を出す必要がある 成立する範囲と、しない範囲:

| 経路 | 演算 | target 間で bit 一致 |
|------|------|---------------------|
| `fit_linear_fixed` / `fit_linear_simd` / `fit_constant_fixed` / `fit_quadratic_fixed` / `fit_cubic_fixed` / 各 evaluator / `piecewise` / `robust` / `delta` / `ring_buffer` | 整数 `i64` / `i128`、wrapping / checked を明示 | する (float を 1 つも使わない) |
| `sensor_fusion` (Kalman 1D / 2D、逆分散重み融合) / `q16_to_f32` / `law` (`f64` の係数と残差) | IEEE 754 の `+ - * /` と `sqrt` のみ | する (IEEE 754 がこれらを正確丸めと規定) |
| 擬似センサー / 深度カメラのシミュレーションと分類器の特徴抽出が使う `sin` / `cos` / `ln` / `exp` | [`alice-det-math`](https://crates.io/crates/alice-det-math) (上の演算のみを固定した評価順で組んだ kernel) | する |
| `object_classifier` の 3 値行列ベクトル積 kernel / `sdf_compress` | `alice-ml` / `alice-sdf` 側の責任 | 各 crate の保証に従う (ここでは言明しない) |

platform の `libm` はどこからも呼ばない `clippy.toml` の
`disallowed-methods` が `f32` / `f64` の inherent な超越関数 (結合順が
未規定な `powi` を含む) を library / test / example / bench すべてで拒否し、
gate は clippy を `--all-targets ... -D warnings` で走らせる `sqrt` と
`mul_add` は**意図的に拒否していない** IEEE 754 が両方を正確丸めと
規定しているので、どの target でも bit 一致する

`tests/determinism_golden.rs` は各 module を公開入口から固定入力で駆動し、
全出力を `to_bits()` で直列化して SHA-256 を定数と突合する CI は
`x86_64` の Linux と Windows、`aarch64` の Linux と macOS で走らせるので、
platform 依存の演算が入れば silent な乖離でなく test の失敗になる 各
scenario は直列化した byte 数の下限も assert するので、module を駆動しなく
なった scenario は空 buffer の hash で通るのでなく fail する

<!-- claim-test: golden_q16_linear, golden_adaptive_polyfit, golden_constant_fit, golden_simd_fit, golden_sensor_fusion, golden_ring_buffer, golden_det_math_kernels, golden_robust, golden_piecewise, golden_delta, golden_law, golden_object_features -->

結果はその算術の名前と一緒に記録する: `alice_edge::SEMANTICS_ID`
(`alice-det-math` からの再 export) 端末から出る係数の隣に保存し、同じ値を
持つ結果どうしだけを数値として比べる `law` feature では
`SignalLaw::law_id(&alice_edge::SEMANTICS_ID)` が法則と算術をまとめて 1 つの
識別子にする どちらも `tests/determinism_golden.rs` で固定しており、法則の
識別子は alice-zip が公開している符号化からこの toolchain の外で計算した値と
突合している

<!-- claim-test: golden_semantics_id, golden_law_id -->

保証範囲の外: `f32` をより広い register で計算して 1 度だけ丸める target
(`i586` 等、SSE2 のない旧 x86) と、fast-math などで compiler に float の
再結合を許す build どちらも CI では build していない

退化入力は別立てで覆う `tests/panic_contract.rs` は各入口について、空の窓 /
1 sample / 0 除算 / 範囲外・負値 / 非有限 / 整数 overflow のそれぞれで
`Err` / 早期 return / 特定の値 / panic のどれが契約かを明示してから、その 1 つを
assert する overflow は両 profile で assert する (Rust は
`debug_assertions` 付きで panic、無しで wrap するため)

<!-- claim-test: an_empty_window_gives_the_zero_fit_from_every_entry_point, every_division_by_a_derived_number_guards_its_zero, the_polynomial_evaluators_overflow_in_debug_and_wrap_in_release, samples_outside_the_q16_range_wrap_rather_than_saturate, the_simd_kernels_agree_with_the_scalar_fit_bit_for_bit -->

## プラットフォーム

CI は `x86_64` の Linux と Windows、`aarch64` の Linux と macOS で test し、
`no_std` コア (単体、`ffi` 付き、`law` 付き) を `thumbv7em-none-eabihf` 向けに build し、
`sensors-hw` を Linux で lint する それ以外の target は CI で build していない

## バインディング

- C / C++: `bindings/alice_edge.h`、`cargo build --release --features ffi` で build
- Unity (C#): `bindings/AliceEdge.cs` (`DllImport` wrapper)
- Python: `maturin develop --features pyo3`

C ABI は (`std` 時) panic を捕捉し `alice_edge_last_error()` で通知する

## 最小対応 Rust バージョン

1.87 (`Cargo.toml` の `rust-version`) CI はそのバージョンの toolchain で、
feature なしと docs.rs の feature 集合の両方で library を compile する

## ビルドとテスト

```bash
cargo test --lib --no-default-features   # no_std core
cargo test --features std,law            # lib, oracles, doc tests
cargo test --test edge_law --features law
cargo test --test determinism_golden --features std,law,ml   # target 間の bit 一致
cargo test --test panic_contract --features std,law,ml       # 退化入力
cargo test --release --test panic_contract --features std    # overflow は release で wrap
cargo bench --no-run                    # Criterion benches in benches/
cargo build --lib --target thumbv7em-none-eabihf --no-default-features --features law
scripts/preflight.sh            # every CI gate locally
scripts/preflight.sh --quick    # static checks, clippy, builds, docs, cargo test --lib
```

## 関連 crate

- [ALICE-Zip](https://github.com/ext-sakamoro/ALICE-Zip) — `SignalLaw` と残差圧縮
- [ALICE-DB](https://github.com/ext-sakamoro/ALICE-DB) — モデルベースの保存
- [ALICE-Codec](https://github.com/ext-sakamoro/ALICE-Codec) — wavelet 符号化
- [ALICE-Analytics](https://github.com/ext-sakamoro/ALICE-Analytics) — ダッシュボード用 sketch
- [ALICE-Streaming-Protocol](https://github.com/ext-sakamoro/ALICE-Streaming-Protocol) — ストリーミングブリッジ

## ライセンス

MIT OR Apache-2.0 ([LICENSE](LICENSE), [LICENSE-APACHE](LICENSE-APACHE))
