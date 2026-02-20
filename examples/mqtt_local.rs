//! MQTT local publish — send compressed coefficients to local Mosquitto
//!
//! Prerequisites:
//!   sudo apt install mosquitto mosquitto-clients
//!   sudo systemctl start mosquitto
//!
//! Subscribe to watch messages:
//!   mosquitto_sub -t 'alice-edge/#' -v
//!
//! ```bash
//! cargo run --example mqtt_local --features "sensors,mqtt"
//! ```
//!
//! Author: Moroya Sakamoto

#[cfg(all(feature = "sensors", feature = "mqtt"))]
fn main() {
    use alice_edge::mqtt_bridge::{CoefficientPayload, MqttConfig, MqttPublisher};
    use alice_edge::sensors::{SensorDriver, SimulatedSensor};
    use alice_edge::{fit_linear_fixed, Q16_SHIFT};
    use std::time::{SystemTime, UNIX_EPOCH};

    println!("=== ALICE-Edge: MQTT Local Publish ===\n");

    // Connect to local Mosquitto
    let config = MqttConfig::local("alice-edge-pi5");
    let mut publisher = match MqttPublisher::new(config) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to connect: {}", e);
            eprintln!("Is Mosquitto running? sudo systemctl start mosquitto");
            return;
        }
    };

    println!("Connected to localhost:1883");

    // Simulate sensor readings
    let mut sensor = SimulatedSensor::new(2500, 5);
    sensor.init().expect("sensor init");

    let batch = sensor.read_samples(1000).expect("read failed");
    let (slope, intercept) = fit_linear_fixed(&batch.temperature);

    let now_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;

    let payload = CoefficientPayload {
        sensor_id: "bme280".into(),
        channel: "temperature".into(),
        slope_q16: slope,
        intercept_q16: intercept,
        slope_f32: slope as f32 / (1 << Q16_SHIFT) as f32,
        intercept_f32: intercept as f32 / (1 << Q16_SHIFT) as f32,
        sample_count: 1000,
        residual_error: 0.001,
        timestamp_ms: now_ms,
    };

    // Publish JSON
    println!("\nPublishing JSON to alice-edge/bme280/temperature...");
    println!("  Payload: {}", payload.to_json());
    publisher
        .publish_coefficients("bme280/temperature", &payload)
        .expect("publish failed");

    // Publish binary
    println!("\nPublishing binary (24 bytes) to alice-edge/bme280/temperature/bin...");
    publisher
        .publish_binary("bme280/temperature/bin", &payload)
        .expect("publish failed");

    println!(
        "\nStats: {} messages, {} bytes sent",
        publisher.messages_sent, publisher.bytes_sent
    );
    println!("\nDone! Check: mosquitto_sub -t 'alice-edge/#' -v");
}

#[cfg(not(all(feature = "sensors", feature = "mqtt")))]
fn main() {
    eprintln!("This example requires features: sensors, mqtt");
    eprintln!("Run: cargo run --example mqtt_local --features \"sensors,mqtt\"");
}
