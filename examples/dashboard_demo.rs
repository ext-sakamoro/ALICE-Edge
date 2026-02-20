//! Dashboard demo — ALICE-Analytics metrics for edge processing
//!
//! Shows real-time monitoring of sensor compression using
//! HyperLogLog (unique sensors), Count-Min Sketch (frequency),
//! and latency percentiles.
//!
//! ```bash
//! cargo run --example dashboard_demo --features "sensors,dashboard"
//! ```
//!
//! Author: Moroya Sakamoto

#[cfg(all(feature = "sensors", feature = "dashboard"))]
fn main() {
    use alice_edge::dashboard::EdgeDashboard;
    use alice_edge::fit_linear_fixed;
    use alice_edge::sensors::{SensorDriver, SimulatedSensor};
    use std::time::Instant;

    println!("=== ALICE-Edge: Dashboard Demo ===\n");

    let mut dashboard = EdgeDashboard::new();

    // Simulate processing from multiple sensors
    let sensor_configs: Vec<(&str, i32, i32, usize)> = vec![
        ("bme280-temp", 2500, 5, 100),
        ("bme280-humidity", 6000, 20, 80),
        ("bme280-pressure", 101325, 50, 60),
        ("dht22-temp", 2200, 10, 50),
        ("dht22-humidity", 4500, 30, 40),
        ("adxl345-x", 0, 100, 200),
        ("adxl345-y", 0, 100, 200),
        ("adxl345-z", 9800, 50, 200),
    ];

    println!("Processing {} sensor types...\n", sensor_configs.len());

    for (sensor_id, base, noise, batches) in &sensor_configs {
        let mut sensor = SimulatedSensor::new(*base, *noise);
        sensor.init().expect("init failed");

        for _ in 0..*batches {
            let batch = sensor.read_samples(1000).expect("read failed");
            let data = &batch.temperature;

            let start = Instant::now();
            let (_slope, _intercept) = fit_linear_fixed(data);
            let latency_us = start.elapsed().as_micros() as u64;

            let raw_bytes = data.len() * 4;
            let compressed_bytes = 8;

            dashboard.record_compression(sensor_id, raw_bytes, compressed_bytes, latency_us);
        }
    }

    // Print dashboard
    dashboard.print_dashboard();

    // Print per-sensor frequency
    println!("\nPer-sensor frequency (Count-Min Sketch estimates):");
    for (sensor_id, _, _, _) in &sensor_configs {
        let freq = dashboard.sensor_frequency(sensor_id);
        println!("  {:<20} : ~{} events", sensor_id, freq);
    }

    // JSON output
    println!("\nJSON output:");
    println!("{}", dashboard.to_json());
}

#[cfg(not(all(feature = "sensors", feature = "dashboard")))]
fn main() {
    eprintln!("This example requires features: sensors, dashboard");
    eprintln!("Run: cargo run --example dashboard_demo --features \"sensors,dashboard\"");
}
