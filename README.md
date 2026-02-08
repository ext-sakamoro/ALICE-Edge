# ALICE-Edge

**Embedded Model Generator** - "Don't send data. Send the law."

<p align="center">
  <em>Ultra-lightweight procedural compression for IoT and embedded systems</em>
</p>

## The Philosophy

Raw sensor data **never leaves the device**. Instead, we fit a mathematical model on-device and transmit only the coefficients.

```
Traditional IoT:  1000 samples × 4 bytes = 4KB transmitted
ALICE-Edge:       1000 samples → y = ax + b → 8 bytes transmitted
                  Compression: 500x
```

## Features

| Feature | Description |
|---------|-------------|
| **`#![no_std]`** | No heap, no OS, no problem |
| **Zero Dependencies** | Pure Rust, nothing else |
| **Fixed-Point Math** | No FPU required (Q16.16 format) |
| **Stack-Only** | Predictable memory usage |
| **1KB Footprint** | Fits in the tiniest MCU |

## Supported Platforms

- **Raspberry Pi Pico (RP2040)** - First-class support
- ARM Cortex-M (M0, M0+, M3, M4, M7)
- ESP32 / ESP8266
- AVR (Arduino)
- RISC-V
- Any platform with Rust `core`

## Quick Start

```toml
# Cargo.toml
[dependencies]
alice-edge = "0.1"
```

```rust
#![no_std]
#![no_main]

use alice_edge::{fit_linear_fixed, evaluate_linear_fixed, q16_to_int};

// Your sensor readings (e.g., temperature × 100)
let samples = [2500, 2510, 2520, 2530, 2540]; // 25.00°C rising

// Fit model ON DEVICE - raw data never leaves!
let (slope, intercept) = fit_linear_fixed(&samples);

// Transmit only 8 bytes
radio_send(&[slope.to_le_bytes(), intercept.to_le_bytes()]);

// On receiver: reconstruct any point
let temp_at_10 = evaluate_linear_fixed(slope, intercept, 10);
let celsius = q16_to_int(temp_at_10) as f32 / 100.0;
```

## API

### Model Fitting

```rust
/// Fit linear model: y = slope × x + intercept
/// Returns (slope, intercept) in Q16.16 fixed-point
pub fn fit_linear_fixed(data: &[i32]) -> (i32, i32);

/// Fit constant model (mean value)
/// Returns mean in Q16.16 fixed-point
pub fn fit_constant_fixed(data: &[i32]) -> i32;
```

### Evaluation

```rust
/// Evaluate model at point x
/// Returns y in Q16.16 fixed-point
pub fn evaluate_linear_fixed(slope: i32, intercept: i32, x: i32) -> i32;
```

### Utilities

```rust
/// Convert Q16.16 to integer
pub const fn q16_to_int(q: i32) -> i32;

/// Convert integer to Q16.16
pub const fn int_to_q16(i: i32) -> i32;

/// Decide if linear model is worth it
pub fn should_use_linear(data: &[i32]) -> bool;
```

## Q16.16 Fixed-Point Format

ALICE-Edge uses Q16.16 fixed-point arithmetic:
- 16 bits for integer part
- 16 bits for fractional part
- Range: -32768.0 to +32767.99998

```
Value 100.5 in Q16.16:
  = 100.5 × 65536
  = 6586368

Convert back:
  6586368 / 65536 = 100.5
```

## Memory Usage

| Function | Stack Usage |
|----------|-------------|
| `fit_linear_fixed` | ~48 bytes |
| `evaluate_linear_fixed` | ~16 bytes |
| `fit_constant_fixed` | ~24 bytes |

## Security Model

```
┌──────────────────────────────────────────────────────────┐
│                    EDGE DEVICE                           │
│  ┌────────────┐    ┌────────────┐    ┌──────────────┐   │
│  │   Sensor   │───▶│ fit_linear │───▶│ Coefficients │───┼──▶ Network
│  │   (raw)    │    │  (model)   │    │   (8 bytes)  │   │
│  └────────────┘    └────────────┘    └──────────────┘   │
│        │                                                 │
│        ▼                                                 │
│   [DISCARDED]  ← Raw data never leaves device!          │
└──────────────────────────────────────────────────────────┘
```

**Privacy by Design**: The raw sensor data is processed and immediately discarded. Only the mathematical model (coefficients) is transmitted. This makes ALICE-Edge ideal for:

- Medical devices (HIPAA compliance)
- Industrial sensors (trade secrets)
- Smart home (privacy)

## Cross-Crate Bridges

ALICE-Edge connects to other ALICE ecosystem crates via feature-gated bridge modules:

| Bridge | Feature | Target Crate | Description |
|--------|---------|--------------|-------------|
| `zip_bridge` | `zip` | [ALICE-Zip](../ALICE-Zip) | ALICE-Zip compression for edge deployment artifact packaging |

### Cargo Profile

Standardized `[profile.bench]` added for consistent benchmarking across ALICE crates.

## Related Projects

| Project | Description |
|---------|-------------|
| [ALICE-Zip](https://github.com/ext-sakamoro/ALICE-Zip) | Core procedural generation engine |
| [ALICE-DB](https://github.com/ext-sakamoro/ALICE-DB) | Model-based time-series database |
| [ALICE-Streaming-Protocol](https://github.com/ext-sakamoro/ALICE-Streaming-Protocol) | Ultra-low bandwidth video streaming |
| [ALICE-Eco-System](https://github.com/ext-sakamoro/ALICE-Eco-System) | Complete Edge-to-Cloud pipeline demo |

All projects share the core philosophy: **encode the generation process, not the data itself**.

## License

MIT License

## Author

Moroya Sakamoto

---

*"The best sensor network is one where data never travels."*
