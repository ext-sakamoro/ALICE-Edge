# Contributing to ALICE-Edge

## Build

```bash
cargo build
```

## Test

```bash
cargo test --features std
```

Note: tests require `std` feature since the core library is `no_std`.

## Lint

```bash
cargo clippy --lib --tests --features std -- -W clippy::all
cargo fmt -- --check
cargo doc --no-deps 2>&1 | grep warning
```

## Optional Features

```bash
# Full edge pipeline (depth-camera + SDF + ML + ASP)
cargo build --features edge-pipeline

# Sensor drivers (simulated)
cargo build --features sensors

# Real hardware GPIO/I2C/SPI (Raspberry Pi)
cargo build --features sensors-hw

# MQTT bridge
cargo build --features mqtt

# Python bindings
cargo build --features pyo3
```

## Design Constraints

- **`no_std` by default**: core runs on 1KB stack (Cortex-M, ESP32, AVR, RISC-V).
- **Q16.16 fixed-point**: all math uses integer arithmetic — no FPU required.
- **O(1) x-sums**: closed-form `Sum(x) = n(n-1)/2`, `Sum(x²) = n(n-1)(2n-1)/6`.
- **Unsafe pointer arithmetic**: zero bounds checks in hot fitting loop.
- **"Don't send data, send the law"**: raw sensor data never leaves the device.
