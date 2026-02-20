//! MQTT AWS IoT Core publish — send coefficients to AWS cloud
//!
//! Prerequisites:
//!   1. Create an IoT Thing in AWS IoT Core
//!   2. Download device certificates
//!   3. Set environment variables:
//!      export AWS_IOT_ENDPOINT="<account>-ats.iot.<region>.amazonaws.com"
//!      export AWS_IOT_CLIENT_ID="alice-edge-pi5"
//!
//! ```bash
//! cargo run --example mqtt_aws_iot --features "sensors,mqtt"
//! ```
//!
//! Author: Moroya Sakamoto

#[cfg(all(feature = "sensors", feature = "mqtt"))]
fn main() {
    use alice_edge::mqtt_bridge::{CoefficientPayload, MqttConfig, MqttPublisher};
    use alice_edge::sensors::{SensorDriver, SimulatedSensor};
    use alice_edge::{fit_linear_fixed, Q16_SHIFT};
    use std::time::{SystemTime, UNIX_EPOCH};

    println!("=== ALICE-Edge: AWS IoT Core Publish ===\n");

    let endpoint = match std::env::var("AWS_IOT_ENDPOINT") {
        Ok(e) => e,
        Err(_) => {
            eprintln!("Set AWS_IOT_ENDPOINT environment variable");
            eprintln!(
                "Example: export AWS_IOT_ENDPOINT=\"abc123-ats.iot.ap-northeast-1.amazonaws.com\""
            );
            return;
        }
    };
    let client_id = std::env::var("AWS_IOT_CLIENT_ID").unwrap_or_else(|_| "alice-edge-pi5".into());

    let config = MqttConfig::aws_iot(&endpoint, &client_id);
    println!("Endpoint: {}:{}", config.host, config.port);
    println!("Client:   {}", config.client_id);
    println!("TLS:      {}", config.use_tls);

    let mut publisher = match MqttPublisher::new(config) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Connection failed: {}", e);
            return;
        }
    };

    // Simulate sensor data
    let sensors = [
        ("bme280", "temperature", 2500i32, 5i32),
        ("bme280", "humidity", 6000, 20),
        ("dht22", "temperature", 2200, 10),
    ];

    for (sensor_id, channel, base, noise) in &sensors {
        let mut sensor = SimulatedSensor::new(*base, *noise);
        sensor.init().expect("sensor init");

        let batch = sensor.read_samples(1000).expect("read failed");
        let (slope, intercept) = fit_linear_fixed(&batch.temperature);

        let now_ms = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64;

        let payload = CoefficientPayload {
            sensor_id: sensor_id.to_string(),
            channel: channel.to_string(),
            slope_q16: slope,
            intercept_q16: intercept,
            slope_f32: slope as f32 / (1 << Q16_SHIFT) as f32,
            intercept_f32: intercept as f32 / (1 << Q16_SHIFT) as f32,
            sample_count: 1000,
            residual_error: 0.001,
            timestamp_ms: now_ms,
        };

        let topic = format!("{}/{}", sensor_id, channel);
        publisher
            .publish_coefficients(&topic, &payload)
            .expect("publish failed");

        println!(
            "Published {}/{}: slope={:.4} intercept={:.2}",
            sensor_id, channel, payload.slope_f32, payload.intercept_f32,
        );
    }

    println!(
        "\nTotal: {} messages, {} bytes sent to AWS IoT Core",
        publisher.messages_sent, publisher.bytes_sent
    );
}

#[cfg(not(all(feature = "sensors", feature = "mqtt")))]
fn main() {
    eprintln!("This example requires features: sensors, mqtt");
    eprintln!("Run: cargo run --example mqtt_aws_iot --features \"sensors,mqtt\"");
}
