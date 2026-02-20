//! Simulated sensor demo — runs on any platform (macOS/Linux/Pi)
//!
//! Demonstrates ALICE-Edge compression without real hardware.
//!
//! ```bash
//! cargo run --example simulate_sensors --features sensors
//! ```
//!
//! Author: Moroya Sakamoto

use alice_edge::sensors::{SensorDriver, SimulatedSensor};
use alice_edge::{fit_linear_fixed, q16_to_int, Q16_SHIFT};

fn main() {
    println!("=== ALICE-Edge: Simulated Sensor Demo ===\n");

    let mut sensor = SimulatedSensor::new(2500, 10); // 25.00°C base, ±0.10 noise
    sensor.init().expect("Failed to init simulated sensor");

    // Collect 1000 samples
    let batch = sensor.read_samples(1000).expect("Failed to read samples");
    let temperature = &batch.temperature;

    println!("Collected {} temperature samples", temperature.len());
    println!(
        "  Range: {:.2}°C — {:.2}°C",
        *temperature.iter().min().unwrap() as f64 / 100.0,
        *temperature.iter().max().unwrap() as f64 / 100.0,
    );

    // Fit linear model
    let (slope, intercept) = fit_linear_fixed(temperature);

    println!("\nModel coefficients (Q16.16):");
    println!(
        "  slope     = {} (≈ {:.4})",
        slope,
        slope as f64 / (1 << Q16_SHIFT) as f64
    );
    println!(
        "  intercept = {} (≈ {:.2})",
        intercept,
        intercept as f64 / (1 << Q16_SHIFT) as f64
    );

    // Compression ratio
    let raw_bytes = temperature.len() * 4;
    let compressed_bytes = 8; // slope + intercept = 2 × i32
    println!("\nCompression:");
    println!(
        "  Raw:        {} bytes ({} samples × 4)",
        raw_bytes,
        temperature.len()
    );
    println!(
        "  Compressed: {} bytes (slope + intercept)",
        compressed_bytes
    );
    println!("  Ratio:      {}x", raw_bytes / compressed_bytes);
    println!(
        "  Savings:    {:.1}%",
        (1.0 - compressed_bytes as f64 / raw_bytes as f64) * 100.0
    );

    // Verify reconstruction
    println!("\nReconstruction verification (first 5 points):");
    for i in 0..5.min(temperature.len()) {
        let predicted = alice_edge::evaluate_linear_fixed(slope, intercept, i as i32);
        let actual = temperature[i];
        let predicted_val = q16_to_int(predicted);
        println!(
            "  x={}: actual={} predicted={} error={}",
            i,
            actual,
            predicted_val,
            (actual - predicted_val).abs()
        );
    }

    println!(
        "\nDone! Transmitted 8 bytes instead of {} bytes.",
        raw_bytes
    );
}
