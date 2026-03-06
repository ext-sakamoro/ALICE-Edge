# ALICE-Edge — Claude Code 設定

## プロジェクト概要

Embedded Model Generator - Don't send data, send the law

| 項目 | 値 |
|------|-----|
| クレート名 | `alice-edge` |
| バージョン | 0.1.0 |
| ライセンス | MIT |
| リポジトリ | `ext-sakamoro/ALICE-Edge` |
| テスト数 | 249 (lib 243 + doc 6) |
| clippy (all+pedantic+nursery, all-targets) | 0 warnings |
| cargo doc | 0 warnings |
| Eco-Systemブリッジ | bridge_edge.rs + bridge_edge_ext.rs |

## Feature フラグ

| Feature | 内容 |
|---------|------|
| `std` | ホスト上テスト用（Vec, alloc等） |
| `ffi` | C-ABI FFI exports (`extern "C"`) |
| `edge-pipeline` | depth-camera + sdf + ml + asp |
| `sensors` | センサードライバ（シミュレーション） |
| `sensors-hw` | 実GPIO/I2C/SPI/UART（rppal） |
| `mqtt` | MQTT publish (AWS IoT / Azure / local) |
| `dashboard` | ALICE-Analytics ダッシュボード |
| `ml` | 1.58bit ternary object classification |
| `codec` | Wavelet denoising |
| `zip` | ALICE-Zip 圧縮 |
| `db` | ALICE-DB 永続化 |
| `pyo3` | Python バインディング |

## バインディング

- **C/C++ header**: `bindings/alice_edge.h`
- **Unity C# DllImport**: `bindings/AliceEdge.cs`
- **FFI feature**: `cargo build --release --features ffi` → `libalice_edge.dylib`

## コーディングルール

メインCLAUDE.md「Git Commit設定」参照。日本語コミット・コメント、署名禁止、作成者 `Moroya Sakamoto`。

## ALICE 品質基準

ALICE-KARIKARI.md「100/100品質基準」参照。clippy基準: `pedantic+nursery`

| 指標 | 値 |
|------|-----|
| clippy (pedantic+nursery) | 0 warnings |
| テスト数 | 168 (std) / 98 (no_std) |
| fmt | clean |

## Eco-System パイプライン

本クレートはALICE-Eco-Systemの以下のパスで使用:
- Path A (IoT/Sensor)
- Path C (MoCap)
- Path E (Embedded)
- Path N (Energy)
- Path O (Space)
- Path P (Neural)
- Path Q (Climate)
- Path U (Presence)

## 情報更新ルール

- バージョンアップ時: このCLAUDE.mdのバージョンを更新
- APIの破壊的変更時: ALICE-Eco-Systemブリッジへの影響をメモ
- テスト数/品質の変化時: 品質基準セクションを更新
- 新feature追加時: プロジェクト概要テーブルを更新
