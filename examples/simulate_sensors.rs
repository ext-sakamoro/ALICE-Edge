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
        f64::from(*temperature.iter().min().unwrap()) / 100.0,
        f64::from(*temperature.iter().max().unwrap()) / 100.0,
    );

    // Fit linear model
    let (slope, intercept) = fit_linear_fixed(temperature);

    println!("\nModel coefficients (Q16.16):");
    println!(
        "  slope     = {} (≈ {:.4})",
        slope,
        f64::from(slope) / f64::from(1 << Q16_SHIFT)
    );
    println!(
        "  intercept = {} (≈ {:.2})",
        intercept,
        f64::from(intercept) / f64::from(1 << Q16_SHIFT)
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
    println!("  Compressed: {compressed_bytes} bytes (slope + intercept)");
    println!("  Ratio:      {}x", raw_bytes / compressed_bytes);
    #[allow(clippy::cast_precision_loss)]
    let savings = (1.0 - compressed_bytes as f64 / raw_bytes as f64) * 100.0;
    println!("  Savings:    {savings:.1}%");

    // Verify reconstruction
    println!("\nReconstruction verification (first 5 points):");
    for (i, &actual) in temperature.iter().enumerate().take(5) {
        #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
        let predicted = alice_edge::evaluate_linear_fixed(slope, intercept, i as i32);
        let predicted_val = q16_to_int(predicted);
        println!(
            "  x={i}: actual={actual} predicted={predicted_val} error={}",
            (actual - predicted_val).abs()
        );
    }

    println!("\nDone! Transmitted 8 bytes instead of {raw_bytes} bytes.");
}
