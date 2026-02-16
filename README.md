# ALICE-Edge

**Embedded Model Generator** - "Don't send data. Send the law."

<p align="center">
  <em>Ultra-lightweight procedural compression for IoT and embedded systems.<br/>
  1000 sensor samples → 8 bytes. Runs on Raspberry Pi 5 to bare-metal MCU.</em>
</p>

## The Philosophy

Raw sensor data **never leaves the device**. Instead, we fit a mathematical model on-device and transmit only the coefficients.

```
Traditional IoT:  1000 samples × 4 bytes = 4,000 bytes transmitted
ALICE-Edge:       1000 samples → y = ax + b → 8 bytes transmitted
                  Compression: 500x
                  Privacy: raw data discarded on-device
```

## Benchmark: Raspberry Pi 5 (Cortex-A76, Measured)

| Metric | Value |
|--------|-------|
| **fit_linear_fixed (1000 samples)** | < 5 µs |
| **Compression ratio** | **500x** (4000 B → 8 B) |
| **Stack usage** | 48 bytes |
| **Binary size** (no_std core) | < 1 KB |
| **Binary size** (full features) | < 100 KB |
| **Memory (RSS)** | < 2 MB |
| **Dependencies** (no_std) | **Zero** |

## Architecture

```
 ┌───────────────────────────────────────────────────────────────────────┐
 │  ALICE-Edge on Raspberry Pi 5                                        │
 │                                                                       │
 │  ┌──────────┐  I2C/SPI/GPIO/UART  ┌──────────────┐                  │
 │  │ BME280   │─────────────────────▶│              │   ┌──────────┐   │
 │  │ DHT22    │                      │ fit_linear   │──▶│ 8 bytes  │   │
 │  │ ADXL345  │                      │ _fixed()     │   │ (slope,  │   │
 │  │ GPS      │                      │              │   │ intercept│   │
 │  └──────────┘                      │ Q16.16       │   └────┬─────┘   │
 │                                    │ Fixed-Point  │        │         │
 │  ┌──────────────┐                  └──────────────┘        │         │
 │  │ ALICE-       │                                          ▼         │
 │  │ Analytics    │◀──latency, compression stats    ┌──────────────┐   │
 │  │ Dashboard    │                                 │ MQTT Publish │   │
 │  │ (HLL, CMS)  │                                 │ AWS IoT Core │   │
 │  └──────────────┘                                 │ Azure IoT Hub│   │
 │                                                   │ Mosquitto    │   │
 │  Raw data → DISCARDED (privacy by design)         └──────────────┘   │
 └───────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Simulated Sensors (any platform)

```bash
git clone https://github.com/ext-sakamoro/ALICE-Edge.git
cd ALICE-Edge

# Run simulated sensor demo (no hardware required)
cargo run --example simulate_sensors --features sensors

# Multi-sensor hub with compression table
cargo run --example multi_sensor_hub --features sensors
```

### 2. Real Hardware on Raspberry Pi 5

```bash
# BME280 (I2C) + DHT22 (GPIO) + ADXL345 (SPI)
cargo run --example bme280_compress --features sensors-hw
```

### 3. MQTT to Cloud

```bash
# Local Mosquitto
cargo run --example mqtt_local --features "sensors,mqtt"

# AWS IoT Core
export AWS_IOT_ENDPOINT="<account>-ats.iot.<region>.amazonaws.com"
cargo run --example mqtt_aws_iot --features "sensors,mqtt"
```

### 4. Dashboard

```bash
cargo run --example dashboard_demo --features "sensors,dashboard"
```

## Wiring Guide (Raspberry Pi 5)

### BME280 (Temperature / Humidity / Pressure) — I2C

```
BME280          Pi 5
──────          ────
VIN  ────────── 3.3V (Pin 1)
GND  ────────── GND  (Pin 6)
SDA  ────────── GPIO 2 / SDA1 (Pin 3)
SCL  ────────── GPIO 3 / SCL1 (Pin 5)
```

### DHT22 (Temperature / Humidity) — GPIO

```
DHT22           Pi 5
──────          ────
VCC  ────────── 3.3V (Pin 1)
GND  ────────── GND  (Pin 9)
DATA ────────── GPIO 4 (Pin 7)
     (10kΩ pull-up between DATA and VCC)
```

### ADXL345 (3-Axis Accelerometer) — SPI

```
ADXL345         Pi 5
──────          ────
VCC  ────────── 3.3V (Pin 17)
GND  ────────── GND  (Pin 20)
CS   ────────── GPIO 8 / CE0 (Pin 24)
SDO  ────────── GPIO 9 / MISO (Pin 21)
SDA  ────────── GPIO 10 / MOSI (Pin 19)
SCL  ────────── GPIO 11 / SCLK (Pin 23)
```

### GPS Module (NEO-6M / NEO-7M) — UART

```
GPS             Pi 5
──────          ────
VCC  ────────── 3.3V (Pin 1)
GND  ────────── GND  (Pin 14)
TX   ────────── GPIO 15 / RXD (Pin 10)
RX   ────────── GPIO 14 / TXD (Pin 8)
```

## API

### Core (no_std, zero dependencies)

```rust
use alice_edge::{fit_linear_fixed, evaluate_linear_fixed, q16_to_int};

// Sensor readings (e.g., temperature × 100)
let samples = [2500, 2510, 2520, 2530, 2540]; // 25.00°C rising

// Fit model ON DEVICE — raw data never leaves!
let (slope, intercept) = fit_linear_fixed(&samples);

// Transmit only 8 bytes
transmit(&slope.to_le_bytes());
transmit(&intercept.to_le_bytes());

// On receiver: reconstruct any point
let temp_at_10 = evaluate_linear_fixed(slope, intercept, 10);
let celsius = q16_to_int(temp_at_10) as f32 / 100.0;
```

### Sensor Drivers (feature: `sensors` / `sensors-hw`)

```rust
use alice_edge::sensors::{Bme280Sensor, SensorDriver};

let mut sensor = Bme280Sensor::new(1, 0x76); // I2C bus 1
sensor.init()?;

let batch = sensor.read_samples(1000)?;
let (slope, intercept) = fit_linear_fixed(&batch.temperature);
// 4000 bytes → 8 bytes (500x compression)
```

### MQTT Bridge (feature: `mqtt`)

```rust
use alice_edge::mqtt_bridge::{MqttConfig, MqttPublisher, CoefficientPayload};

let config = MqttConfig::local("alice-edge-pi5");
let mut publisher = MqttPublisher::new(config)?;

publisher.publish_coefficients("bme280/temperature", &payload)?;
publisher.publish_binary("bme280/temperature/bin", &payload)?;  // 24 bytes
```

### Dashboard (feature: `dashboard`)

```rust
use alice_edge::dashboard::EdgeDashboard;

let mut dashboard = EdgeDashboard::new();
dashboard.record_compression("bme280", 4000, 8, latency_us);

dashboard.print_dashboard();  // Terminal output
let json = dashboard.to_json();  // JSON for API
```

## Feature Flags

| Feature | Dependencies | Description |
|---------|-------------|-------------|
| *(default)* | None | `no_std` core: fit/evaluate/Q16.16 |
| `sensors` | serde, serde_json | Sensor drivers (simulated) |
| `sensors-hw` | rppal, serialport | Real GPIO/I2C/SPI/UART on Pi |
| `mqtt` | rumqttc | MQTT publish to cloud |
| `dashboard` | alice-analytics | HLL/CMS/latency dashboard |
| `pyo3` | pyo3, numpy | Python bindings (zero-copy NumPy) |
| `zip` | alice-zip | ALICE-Zip compression bridge |
| `codec` | alice-codec | Wavelet denoising bridge |
| `db` | alice-db | Coefficient persistence bridge |
| `ml` | alice-ml | 1.58-bit ternary classification |
| `depth-camera` | rusb | Dolphin D5 Lite depth camera |
| `sdf` | alice-sdf | SDF point cloud compression |
| `asp` | libasp | ALICE Streaming Protocol bridge |
| `edge-pipeline` | (all above) | Full depth → SDF → ML pipeline |

## Q16.16 Fixed-Point Format

ALICE-Edge uses Q16.16 fixed-point arithmetic (no FPU required):

```
16 bits integer | 16 bits fraction
Range: -32768.0 to +32767.99998

Value 25.50°C (as 2550 raw):
  Q16.16 = 2550 × 65536 = 167,116,800

  Convert back: 167,116,800 / 65536 = 2550 → 25.50°C
```

## Memory & Stack Usage

| Function | Stack | Description |
|----------|-------|-------------|
| `fit_linear_fixed` | 48 B | O(N) loop, O(1) x-sums |
| `evaluate_linear_fixed` | 16 B | Single MLA instruction |
| `fit_constant_fixed` | 24 B | Mean value |
| `compute_residual_error` | 32 B | Sum of squared errors |

## Project Structure

```
ALICE-Edge/
├── Cargo.toml
├── src/
│   ├── lib.rs              # Core: fit_linear_fixed, Q16.16 (no_std)
│   ├── sensors.rs          # BME280, DHT22, ADXL345, GPS drivers
│   ├── mqtt_bridge.rs      # MQTT publish (AWS IoT, Azure, Mosquitto)
│   ├── dashboard.rs        # ALICE-Analytics dashboard (HLL, CMS)
│   ├── python.rs           # PyO3 bindings (zero-copy NumPy)
│   ├── zip_bridge.rs       # ALICE-Zip compression
│   ├── codec_bridge.rs     # Wavelet denoising
│   ├── db_bridge.rs        # Coefficient persistence
│   ├── ml_bridge.rs        # Ternary neural network (544 B model)
│   ├── depth_capture.rs    # Dolphin D5 Lite depth camera
│   ├── sdf_compress.rs     # SDF point cloud compression
│   ├── object_classifier.rs # Edge object classification
│   ├── asp_bridge.rs       # ALICE Streaming Protocol
│   └── edge_pipeline.rs    # Full depth → SDF → ML pipeline
├── examples/
│   ├── simulate_sensors.rs # Simulated sensor demo (any platform)
│   ├── bme280_compress.rs  # BME280 compression (Pi or simulated)
│   ├── multi_sensor_hub.rs # Multi-sensor compression table
│   ├── mqtt_local.rs       # MQTT to local Mosquitto
│   ├── mqtt_aws_iot.rs     # MQTT to AWS IoT Core
│   ├── dashboard_demo.rs   # ALICE-Analytics dashboard
│   └── python_batch.py     # Python + NumPy batch processing
├── benches/
│   └── pi5_bench.rs        # Criterion benchmarks
└── README.md
```

## Security Model

```
┌──────────────────────────────────────────────────────────────────┐
│                      EDGE DEVICE                                  │
│  ┌────────────┐    ┌────────────┐    ┌──────────────┐            │
│  │   Sensor   │───▶│ fit_linear │───▶│ Coefficients │───▶ Network│
│  │   (raw)    │    │  _fixed()  │    │   (8 bytes)  │            │
│  └────────────┘    └────────────┘    └──────────────┘            │
│        │                                                          │
│        ▼                                                          │
│   [DISCARDED]  ← Raw data NEVER leaves device                    │
└──────────────────────────────────────────────────────────────────┘
```

**Privacy by Design**: Raw sensor data is processed and immediately discarded. Only mathematical coefficients (8 bytes) are transmitted. Ideal for:

- Medical devices (HIPAA compliance)
- Industrial sensors (trade secrets)
- Smart home (user privacy)

## ALICE Ecosystem Integration

| Bridge | Source Crate | Usage |
|--------|-------------|-------|
| Sensor compression | **ALICE-Edge** (core) | Q16.16 linear regression |
| Model storage | **ALICE-DB** | Coefficient time-series |
| Wavelet denoising | **ALICE-Codec** | Pre-processing sensor data |
| Binary compression | **ALICE-Zip** | Artifact packaging |
| Analytics | **ALICE-Analytics** | HLL, CMS, latency tracking |
| Neural classification | **ALICE-ML** | 1.58-bit ternary inference |
| SDF compression | **ALICE-SDF** | Point cloud → SDF |
| Streaming | **ALICE-Streaming-Protocol** | Low-bandwidth video |

## Build

### On Raspberry Pi 5 (recommended)

```bash
# Install Rust (one-time)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

# Clone
git clone https://github.com/ext-sakamoro/ALICE-Edge.git
git clone https://github.com/ext-sakamoro/ALICE-Analytics.git

# Build (release)
cd ALICE-Edge
cargo build --release --features sensors-hw,mqtt,dashboard

# Run benchmarks
cargo bench
```

### Cross-compile (macOS → aarch64 Linux)

```bash
rustup target add aarch64-unknown-linux-gnu
cargo build --release --target aarch64-unknown-linux-gnu --features sensors,mqtt,dashboard
scp target/aarch64-unknown-linux-gnu/release/examples/* pi@raspberrypi:~/
```

## Supported Platforms

| Platform | no_std Core | Sensors (HW) | MQTT | Dashboard |
|----------|:-----------:|:------------:|:----:|:---------:|
| **Raspberry Pi 5** | Y | Y | Y | Y |
| **Raspberry Pi 4/3** | Y | Y | Y | Y |
| Raspberry Pi Pico (RP2040) | Y | - | - | - |
| ARM Cortex-M (M0/M3/M4/M7) | Y | - | - | - |
| ESP32 / ESP8266 | Y | - | - | - |
| RISC-V | Y | - | - | - |
| macOS / Linux (x86_64) | Y | sim | Y | Y |

## License

MIT (Core)

*Note: The core `no_std` library is licensed under MIT. However, enabling certain feature flags (e.g., `dashboard`, `db`, `ml`) links against AGPL-3.0 components from the broader ALICE ecosystem. Binaries built with these features enabled are subject to the terms of the AGPL-3.0 license.*

## Author

Moroya Sakamoto

---

*"The best sensor network is one where data never travels."*
