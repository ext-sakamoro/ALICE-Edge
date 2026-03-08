//! BME280 sensor compression — real I2C on Pi 5 or simulated
//!
//! Reads temperature, humidity, and pressure from BME280 (I2C),
//! fits linear models, and outputs compressed coefficients.
//!
//! ```bash
//! # On Raspberry Pi 5 (real hardware):
//! cargo run --example bme280_compress --features sensors-hw
//!
//! # On macOS/CI (simulated):
//! cargo run --example bme280_compress --features sensors
//! ```
//!
//! Author: Moroya Sakamoto

#[cfg(feature = "sensors-hw")]
use alice_edge::sensors::Bme280Sensor;
use alice_edge::sensors::SensorDriver;
#[cfg(not(feature = "sensors-hw"))]
use alice_edge::sensors::SimulatedSensor;
use alice_edge::{fit_linear_fixed, Q16_SHIFT};

fn main() {
    println!("=== ALICE-Edge: BME280 Compression Demo ===\n");

    #[cfg(feature = "sensors-hw")]
    let mut sensor = {
        println!("Mode: Real hardware (I2C bus 1, addr 0x76)");
        Bme280Sensor::new(1, 0x76)
    };

    #[cfg(not(feature = "sensors-hw"))]
    let mut sensor = {
        println!("Mode: Simulated (no hardware)");
        SimulatedSensor::new(2500, 5)
    };

    sensor.init().expect("Sensor init failed");

    let sample_counts = [100, 500, 1000];

    for &n in &sample_counts {
        let batch = sensor.read_samples(n).expect("Read failed");

        println!("\n--- {n} samples ---");

        // Temperature channel
        let (t_slope, t_intercept) = fit_linear_fixed(&batch.temperature);
        let raw_bytes = n * 4;
        let compressed_bytes = 8;
        println!(
            "Temperature: slope={:.4} intercept={:.2} | {} → {} bytes ({}x)",
            f64::from(t_slope) / f64::from(1 << Q16_SHIFT),
            f64::from(t_intercept) / f64::from(1 << Q16_SHIFT),
            raw_bytes,
            compressed_bytes,
            raw_bytes / compressed_bytes
        );

        // Humidity channel (if available)
        if !batch.humidity.is_empty() {
            let (h_slope, h_intercept) = fit_linear_fixed(&batch.humidity);
            println!(
                "Humidity:    slope={:.4} intercept={:.2} | {} → {} bytes ({}x)",
                f64::from(h_slope) / f64::from(1 << Q16_SHIFT),
                f64::from(h_intercept) / f64::from(1 << Q16_SHIFT),
                raw_bytes,
                compressed_bytes,
                raw_bytes / compressed_bytes
            );
        }

        // Pressure channel (if available)
        if !batch.pressure.is_empty() {
            let (p_slope, p_intercept) = fit_linear_fixed(&batch.pressure);
            println!(
                "Pressure:    slope={:.4} intercept={:.2} | {} → {} bytes ({}x)",
                f64::from(p_slope) / f64::from(1 << Q16_SHIFT),
                f64::from(p_intercept) / f64::from(1 << Q16_SHIFT),
                raw_bytes,
                compressed_bytes,
                raw_bytes / compressed_bytes
            );
        }
    }

    println!("\nDone!");
}
