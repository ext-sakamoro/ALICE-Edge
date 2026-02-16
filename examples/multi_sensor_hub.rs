//! Multi-sensor hub — compress data from multiple sensors
//!
//! Demonstrates a realistic IoT hub scenario with multiple sensor types.
//! Each sensor's data is independently compressed and transmitted.
//!
//! ```bash
//! # Simulated (any platform):
//! cargo run --example multi_sensor_hub --features sensors
//!
//! # Real hardware (Pi 5):
//! cargo run --example multi_sensor_hub --features sensors-hw
//! ```
//!
//! Author: Moroya Sakamoto

use alice_edge::sensors::{SimulatedSensor, SensorDriver};
use alice_edge::{fit_linear_fixed, should_use_linear};
use std::time::Instant;

struct SensorHub {
    sensors: Vec<(&'static str, SimulatedSensor)>,
}

impl SensorHub {
    fn new() -> Self {
        Self {
            sensors: vec![
                ("bme280-temp", SimulatedSensor::new(2500, 5)),   // 25.00°C ±0.05
                ("bme280-humi", SimulatedSensor::new(6000, 20)),  // 60.00% ±0.20
                ("bme280-pres", SimulatedSensor::new(101325, 50)), // 1013.25 hPa ±0.50
                ("dht22-temp", SimulatedSensor::new(2200, 10)),   // 22.00°C ±0.10
                ("dht22-humi", SimulatedSensor::new(4500, 30)),   // 45.00% ±0.30
                ("adxl345-x", SimulatedSensor::new(0, 100)),     // 0g ±1.00 accel
                ("adxl345-y", SimulatedSensor::new(0, 100)),     // 0g ±1.00
                ("adxl345-z", SimulatedSensor::new(9800, 50)),    // ~1g ±0.50
            ],
        }
    }

    fn init_all(&mut self) {
        for (name, sensor) in &mut self.sensors {
            sensor.init().unwrap_or_else(|e| panic!("Failed to init {}: {:?}", name, e));
        }
    }

    fn collect_and_compress(&mut self, samples_per_sensor: usize) {
        let mut total_raw = 0usize;
        let mut total_compressed = 0usize;

        println!("┌──────────────┬──────────┬──────────┬──────┬───────────┬──────────┐");
        println!("│ Sensor       │ Samples  │ Raw (B)  │ Comp │ Ratio     │ Model    │");
        println!("├──────────────┼──────────┼──────────┼──────┼───────────┼──────────┤");

        for (name, sensor) in &mut self.sensors {
            let start = Instant::now();
            let batch = sensor.read_samples(samples_per_sensor).expect("read failed");
            let data = &batch.temperature; // Simulated sensor puts all data in temperature

            let raw_bytes = data.len() * 4;
            let use_linear = should_use_linear(data);
            let compressed_bytes = if use_linear { 8 } else { 4 };
            let model = if use_linear { "linear" } else { "const" };

            let (_slope, _intercept) = fit_linear_fixed(data);
            let _elapsed = start.elapsed();

            total_raw += raw_bytes;
            total_compressed += compressed_bytes;

            println!(
                "│ {:<12} │ {:>8} │ {:>8} │ {:>4} │ {:>7.0}x  │ {:<8} │",
                name,
                data.len(),
                raw_bytes,
                compressed_bytes,
                raw_bytes as f64 / compressed_bytes as f64,
                model,
            );
        }

        println!("├──────────────┼──────────┼──────────┼──────┼───────────┼──────────┤");
        println!(
            "│ TOTAL        │ {:>8} │ {:>8} │ {:>4} │ {:>7.0}x  │          │",
            samples_per_sensor * self.sensors.len(),
            total_raw,
            total_compressed,
            total_raw as f64 / total_compressed as f64,
        );
        println!("└──────────────┴──────────┴──────────┴──────┴───────────┴──────────┘");

        println!(
            "\nBandwidth savings: {} bytes → {} bytes ({:.1}% reduction)",
            total_raw,
            total_compressed,
            (1.0 - total_compressed as f64 / total_raw as f64) * 100.0,
        );
    }
}

fn main() {
    println!("=== ALICE-Edge: Multi-Sensor Hub ===\n");

    let mut hub = SensorHub::new();
    hub.init_all();

    for &n in &[100, 1000] {
        println!("\n--- {} samples per sensor ---\n", n);
        hub.collect_and_compress(n);
    }
}
