//! MQTT bridge for ALICE-Edge
//!
//! Publishes compressed model coefficients to MQTT brokers.
//! Supports AWS IoT Core, Azure IoT Hub, and standard MQTT 3.1.1/5.0 brokers.
//!
//! # Example
//!
//! ```ignore
//! use alice_edge::mqtt_bridge::{MqttPublisher, MqttConfig};
//!
//! let config = MqttConfig::local("raspberry-pi-5");
//! let mut publisher = MqttPublisher::new(config)?;
//!
//! // Publish compressed coefficients
//! publisher.publish_coefficients("sensor/bme280/temperature", slope, intercept)?;
//! ```
//!
//! Author: Moroya Sakamoto

use rumqttc::{Client, MqttOptions, QoS};
use std::time::Duration;

/// MQTT configuration
#[derive(Debug, Clone)]
pub struct MqttConfig {
    /// Broker host
    pub host: String,
    /// Broker port
    pub port: u16,
    /// Client ID
    pub client_id: String,
    /// Username (optional)
    pub username: Option<String>,
    /// Password (optional)
    pub password: Option<String>,
    /// Use TLS
    pub use_tls: bool,
    /// Keep-alive interval
    pub keep_alive: Duration,
    /// Topic prefix
    pub topic_prefix: String,
}

impl MqttConfig {
    /// Configuration for local Mosquitto broker
    pub fn local(client_id: &str) -> Self {
        Self {
            host: "localhost".into(),
            port: 1883,
            client_id: client_id.into(),
            username: None,
            password: None,
            use_tls: false,
            keep_alive: Duration::from_secs(30),
            topic_prefix: "alice-edge".into(),
        }
    }

    /// Configuration for AWS IoT Core
    ///
    /// Requires certificate-based authentication (set via TLS config).
    /// Endpoint: `<account-id>-ats.iot.<region>.amazonaws.com`
    pub fn aws_iot(endpoint: &str, client_id: &str) -> Self {
        Self {
            host: endpoint.into(),
            port: 8883,
            client_id: client_id.into(),
            username: None,
            password: None,
            use_tls: true,
            keep_alive: Duration::from_secs(30),
            topic_prefix: "alice-edge".into(),
        }
    }

    /// Configuration for Azure IoT Hub
    ///
    /// Endpoint: `<hub-name>.azure-devices.net`
    pub fn azure_iot(hub_name: &str, device_id: &str, sas_token: &str) -> Self {
        Self {
            host: format!("{}.azure-devices.net", hub_name),
            port: 8883,
            client_id: device_id.into(),
            username: Some(format!(
                "{}.azure-devices.net/{}/api-version=2021-04-12",
                hub_name, device_id
            )),
            password: Some(sas_token.into()),
            use_tls: true,
            keep_alive: Duration::from_secs(30),
            topic_prefix: format!("devices/{}/messages/events", device_id),
        }
    }
}

/// Coefficient payload for MQTT transmission
#[derive(Debug, Clone)]
pub struct CoefficientPayload {
    /// Sensor identifier
    pub sensor_id: String,
    /// Channel name (e.g., "temperature", "humidity")
    pub channel: String,
    /// Slope in Q16.16 format
    pub slope_q16: i32,
    /// Intercept in Q16.16 format
    pub intercept_q16: i32,
    /// Slope as float
    pub slope_f32: f32,
    /// Intercept as float
    pub intercept_f32: f32,
    /// Number of samples compressed
    pub sample_count: u32,
    /// Residual error
    pub residual_error: f64,
    /// Timestamp (milliseconds since epoch)
    pub timestamp_ms: u64,
}

impl CoefficientPayload {
    /// Serialize to JSON
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"sensor":"{}","channel":"{}","slope":{},"intercept":{},"slope_q16":{},"intercept_q16":{},"samples":{},"error":{:.6},"ts":{}}}"#,
            self.sensor_id, self.channel,
            self.slope_f32, self.intercept_f32,
            self.slope_q16, self.intercept_q16,
            self.sample_count, self.residual_error,
            self.timestamp_ms
        )
    }

    /// Serialize to compact binary (24 bytes)
    ///
    /// Format: [slope_q16: i32][intercept_q16: i32][sample_count: u32][timestamp: u64][error: f32]
    pub fn to_binary(&self) -> [u8; 24] {
        let mut buf = [0u8; 24];
        buf[0..4].copy_from_slice(&self.slope_q16.to_le_bytes());
        buf[4..8].copy_from_slice(&self.intercept_q16.to_le_bytes());
        buf[8..12].copy_from_slice(&self.sample_count.to_le_bytes());
        buf[12..20].copy_from_slice(&self.timestamp_ms.to_le_bytes());
        buf[20..24].copy_from_slice(&(self.residual_error as f32).to_le_bytes());
        buf
    }
}

/// MQTT publisher for ALICE-Edge coefficients
pub struct MqttPublisher {
    client: Client,
    config: MqttConfig,
    /// Total messages published
    pub messages_sent: u64,
    /// Total bytes published
    pub bytes_sent: u64,
}

impl MqttPublisher {
    /// Create a new MQTT publisher
    pub fn new(config: MqttConfig) -> Result<Self, MqttError> {
        let mut opts = MqttOptions::new(
            &config.client_id,
            &config.host,
            config.port,
        );
        opts.set_keep_alive(config.keep_alive);

        if let (Some(user), Some(pass)) = (&config.username, &config.password) {
            opts.set_credentials(user, pass);
        }

        let (client, mut connection) = Client::new(opts, 64);

        // Spawn connection event loop in background
        std::thread::spawn(move || {
            for _event in connection.iter() {
                // Process events silently
            }
        });

        Ok(Self {
            client,
            config,
            messages_sent: 0,
            bytes_sent: 0,
        })
    }

    /// Publish coefficients as JSON
    pub fn publish_coefficients(
        &mut self,
        topic_suffix: &str,
        payload: &CoefficientPayload,
    ) -> Result<(), MqttError> {
        let topic = format!("{}/{}", self.config.topic_prefix, topic_suffix);
        let json = payload.to_json();
        let bytes = json.len() as u64;

        self.client.publish(&topic, QoS::AtLeastOnce, false, json.as_bytes())
            .map_err(|e| MqttError::Publish(format!("{}", e)))?;

        self.messages_sent += 1;
        self.bytes_sent += bytes;
        Ok(())
    }

    /// Publish coefficients as compact binary (24 bytes per message)
    pub fn publish_binary(
        &mut self,
        topic_suffix: &str,
        payload: &CoefficientPayload,
    ) -> Result<(), MqttError> {
        let topic = format!("{}/{}", self.config.topic_prefix, topic_suffix);
        let binary = payload.to_binary();

        self.client.publish(&topic, QoS::AtLeastOnce, false, &binary)
            .map_err(|e| MqttError::Publish(format!("{}", e)))?;

        self.messages_sent += 1;
        self.bytes_sent += 24;
        Ok(())
    }

    /// Publish a batch of coefficients for multiple channels
    pub fn publish_batch(
        &mut self,
        payloads: &[CoefficientPayload],
    ) -> Result<(), MqttError> {
        for payload in payloads {
            let topic = format!("{}/{}", payload.sensor_id, payload.channel);
            self.publish_coefficients(&topic, payload)?;
        }
        Ok(())
    }
}

/// MQTT errors
#[derive(Debug)]
pub enum MqttError {
    Connection(String),
    Publish(String),
}

impl std::fmt::Display for MqttError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MqttError::Connection(e) => write!(f, "MQTT connection: {}", e),
            MqttError::Publish(e) => write!(f, "MQTT publish: {}", e),
        }
    }
}

impl std::error::Error for MqttError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_payload_json() {
        let payload = CoefficientPayload {
            sensor_id: "bme280".into(),
            channel: "temperature".into(),
            slope_q16: 655360,
            intercept_q16: 163840000,
            slope_f32: 10.0,
            intercept_f32: 2500.0,
            sample_count: 1000,
            residual_error: 0.001,
            timestamp_ms: 1708200000000,
        };
        let json = payload.to_json();
        assert!(json.contains("\"sensor\":\"bme280\""));
        assert!(json.contains("\"slope\":10"));
    }

    #[test]
    fn test_payload_binary() {
        let payload = CoefficientPayload {
            sensor_id: "test".into(),
            channel: "test".into(),
            slope_q16: 100,
            intercept_q16: 200,
            slope_f32: 0.0,
            intercept_f32: 0.0,
            sample_count: 1000,
            residual_error: 0.0,
            timestamp_ms: 0,
        };
        let binary = payload.to_binary();
        assert_eq!(binary.len(), 24);
        assert_eq!(i32::from_le_bytes([binary[0], binary[1], binary[2], binary[3]]), 100);
    }

    #[test]
    fn test_config_local() {
        let config = MqttConfig::local("test-pi");
        assert_eq!(config.host, "localhost");
        assert_eq!(config.port, 1883);
        assert!(!config.use_tls);
    }

    #[test]
    fn test_config_aws() {
        let config = MqttConfig::aws_iot("abc123-ats.iot.ap-northeast-1.amazonaws.com", "pi5");
        assert_eq!(config.port, 8883);
        assert!(config.use_tls);
    }
}
