// SPDX-License-Identifier: MIT
//! MQTT bridge for ALICE-Edge
//!
//! Publishes compressed model coefficients to MQTT brokers.
//! Supports AWS `IoT` Core, Azure `IoT` Hub, and standard MQTT 3.1.1/5.0 brokers.
//!
//! # Example
//!
//! ```no_run
//! use alice_edge::mqtt_bridge::{MqttPublisher, MqttConfig, CoefficientPayload};
//!
//! let config = MqttConfig::local("raspberry-pi-5");
//! let mut publisher = MqttPublisher::new(config).unwrap();
//!
//! // Publish compressed coefficients
//! let payload = CoefficientPayload {
//!     sensor_id: "bme280".into(),
//!     channel: "temperature".into(),
//!     slope_q16: 65536, intercept_q16: 0,
//!     slope_f32: 1.0, intercept_f32: 0.0,
//!     sample_count: 100, residual_error: 0.001,
//!     timestamp_ms: 0,
//! };
//! publisher.publish_coefficients("sensor/bme280/temperature", &payload).unwrap();
//! ```
//!
//! Author: Moroya Sakamoto

use rumqttc::{Client, Event, MqttOptions, Packet, QoS};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// TLS configuration for MQTT connections (AWS `IoT` Core / Azure `IoT` Hub)
#[derive(Debug, Clone)]
pub struct MqttTlsConfig {
    /// CA certificate PEM bytes
    pub ca_cert: Vec<u8>,
    /// Client certificate PEM bytes
    pub client_cert: Vec<u8>,
    /// Client private key PEM bytes
    pub client_key: Vec<u8>,
}

impl MqttTlsConfig {
    /// Load TLS config from PEM file paths
    ///
    /// # Errors
    ///
    /// Returns `MqttError::Connection` if any of the PEM files cannot be read from disk.
    pub fn from_pem_files(
        ca_path: &str,
        cert_path: &str,
        key_path: &str,
    ) -> Result<Self, MqttError> {
        let ca_cert =
            std::fs::read(ca_path).map_err(|e| MqttError::Connection(format!("CA cert: {e}")))?;
        let client_cert = std::fs::read(cert_path)
            .map_err(|e| MqttError::Connection(format!("Client cert: {e}")))?;
        let client_key = std::fs::read(key_path)
            .map_err(|e| MqttError::Connection(format!("Client key: {e}")))?;
        Ok(Self {
            ca_cert,
            client_cert,
            client_key,
        })
    }
}

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
    /// TLS certificate configuration (required when `use_tls` = true for AWS `IoT`)
    pub tls_config: Option<MqttTlsConfig>,
    /// Keep-alive interval
    pub keep_alive: Duration,
    /// Topic prefix
    pub topic_prefix: String,
}

impl MqttConfig {
    /// Configuration for local Mosquitto broker
    #[must_use]
    pub fn local(client_id: &str) -> Self {
        Self {
            host: "localhost".into(),
            port: 1883,
            client_id: client_id.into(),
            username: None,
            password: None,
            use_tls: false,
            tls_config: None,
            keep_alive: Duration::from_secs(30),
            topic_prefix: "alice-edge".into(),
        }
    }

    /// Configuration for AWS `IoT` Core
    ///
    /// Requires certificate-based authentication. Use `with_tls()` to set certificates:
    /// ```no_run
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use alice_edge::mqtt_bridge::{MqttConfig, MqttTlsConfig};
    /// let config = MqttConfig::aws_iot("endpoint", "client")
    ///     .with_tls(MqttTlsConfig::from_pem_files("ca.pem", "cert.pem", "key.pem")?);
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn aws_iot(endpoint: &str, client_id: &str) -> Self {
        Self {
            host: endpoint.into(),
            port: 8883,
            client_id: client_id.into(),
            username: None,
            password: None,
            use_tls: true,
            tls_config: None,
            keep_alive: Duration::from_secs(30),
            topic_prefix: "alice-edge".into(),
        }
    }

    /// Configuration for Azure `IoT` Hub
    ///
    /// Endpoint: `<hub-name>.azure-devices.net`
    #[must_use]
    pub fn azure_iot(hub_name: &str, device_id: &str, sas_token: &str) -> Self {
        Self {
            host: format!("{hub_name}.azure-devices.net"),
            port: 8883,
            client_id: device_id.into(),
            username: Some(format!(
                "{hub_name}.azure-devices.net/{device_id}/api-version=2021-04-12"
            )),
            password: Some(sas_token.into()),
            use_tls: true,
            tls_config: None,
            keep_alive: Duration::from_secs(30),
            topic_prefix: format!("devices/{device_id}/messages/events"),
        }
    }

    /// Set TLS configuration for mTLS connections (AWS `IoT` / Azure `IoT`)
    #[must_use]
    pub fn with_tls(mut self, tls: MqttTlsConfig) -> Self {
        self.tls_config = Some(tls);
        self.use_tls = true;
        self
    }
}

/// Coefficient payload for MQTT transmission
#[derive(Debug, Clone)]
#[cfg_attr(feature = "sensors", derive(serde::Serialize, serde::Deserialize))]
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
    #[must_use]
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"sensor":"{}","channel":"{}","slope":{},"intercept":{},"slope_q16":{},"intercept_q16":{},"samples":{},"error":{:.6},"ts":{}}}"#,
            self.sensor_id,
            self.channel,
            self.slope_f32,
            self.intercept_f32,
            self.slope_q16,
            self.intercept_q16,
            self.sample_count,
            self.residual_error,
            self.timestamp_ms
        )
    }

    /// Serialize to compact binary (24 bytes)
    ///
    /// Format: [`slope_q16`: i32][intercept_q16: i32][`sample_count`: u32][timestamp: u64][error: f32]
    #[must_use]
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
    /// 接続状態（バックグラウンドスレッドが `ConnAck` 受信時に true にセット）
    connected: Arc<AtomicBool>,
    /// Total messages published
    pub messages_sent: u64,
    /// Total bytes published
    pub bytes_sent: u64,
}

impl MqttPublisher {
    /// Create a new MQTT publisher
    ///
    /// TLS 接続を使用する場合、`MqttConfig::tls_config` に証明書を設定すること。
    /// 設定なしで `use_tls = true` の場合、`MqttError::Connection` を返す。
    ///
    /// # Errors
    ///
    /// Returns `MqttError::Connection` if TLS is enabled but no `tls_config` is provided.
    pub fn new(config: MqttConfig) -> Result<Self, MqttError> {
        let mut opts = MqttOptions::new(&config.client_id, &config.host, config.port);
        opts.set_keep_alive(config.keep_alive);

        if let (Some(user), Some(pass)) = (&config.username, &config.password) {
            opts.set_credentials(user, pass);
        }

        // TLS 設定
        if config.use_tls {
            if let Some(ref tls) = config.tls_config {
                let transport =
                    rumqttc::Transport::tls_with_config(rumqttc::TlsConfiguration::Simple {
                        ca: tls.ca_cert.clone(),
                        alpn: None,
                        client_auth: Some((tls.client_cert.clone(), tls.client_key.clone())),
                    });
                opts.set_transport(transport);
            } else {
                return Err(MqttError::Connection(
                    "TLS enabled but no tls_config provided. Use MqttConfig::with_tls() to set certificates.".into(),
                ));
            }
        }

        let (client, mut connection) = Client::new(opts, 64);
        let connected = Arc::new(AtomicBool::new(false));
        let connected_clone = connected.clone();

        // イベントループ: ConnAck 受信で connected=true、切断で false
        std::thread::spawn(move || {
            for event in connection.iter() {
                match event {
                    Ok(Event::Incoming(Packet::ConnAck(_))) => {
                        connected_clone.store(true, Ordering::Release);
                    }
                    Ok(Event::Incoming(Packet::Disconnect)) | Err(_) => {
                        connected_clone.store(false, Ordering::Release);
                    }
                    _ => {}
                }
            }
            // イベントループ終了 → 切断
            connected_clone.store(false, Ordering::Release);
        });

        Ok(Self {
            client,
            config,
            connected,
            messages_sent: 0,
            bytes_sent: 0,
        })
    }

    /// ブローカーに接続済みかどうかを返す
    #[must_use]
    pub fn is_connected(&self) -> bool {
        self.connected.load(Ordering::Acquire)
    }

    /// Publish coefficients as JSON
    ///
    /// # Errors
    ///
    /// Returns `MqttError::Publish` if the MQTT client's send queue is full or the broker rejects the message.
    pub fn publish_coefficients(
        &mut self,
        topic_suffix: &str,
        payload: &CoefficientPayload,
    ) -> Result<(), MqttError> {
        let topic = format!("{}/{}", self.config.topic_prefix, topic_suffix);
        let json = payload.to_json();
        let bytes = json.len() as u64;

        self.client
            .try_publish(&topic, QoS::AtLeastOnce, false, json.as_bytes())
            .map_err(|e| MqttError::Publish(format!("{e}")))?;

        self.messages_sent += 1;
        self.bytes_sent += bytes;
        Ok(())
    }

    /// Publish coefficients as compact binary (24 bytes per message)
    ///
    /// # Errors
    ///
    /// Returns `MqttError::Publish` if the MQTT client's send queue is full or the broker rejects the message.
    pub fn publish_binary(
        &mut self,
        topic_suffix: &str,
        payload: &CoefficientPayload,
    ) -> Result<(), MqttError> {
        let topic = format!("{}/{}", self.config.topic_prefix, topic_suffix);
        let binary = payload.to_binary();

        self.client
            .try_publish(&topic, QoS::AtLeastOnce, false, binary)
            .map_err(|e| MqttError::Publish(format!("{e}")))?;

        self.messages_sent += 1;
        self.bytes_sent += 24;
        Ok(())
    }

    /// Publish a batch of coefficients for multiple channels
    ///
    /// # Errors
    ///
    /// Returns `MqttError::Publish` if publishing any payload in the batch fails; processing stops at the first error.
    pub fn publish_batch(&mut self, payloads: &[CoefficientPayload]) -> Result<(), MqttError> {
        for payload in payloads {
            let topic = format!("{}/{}", payload.sensor_id, payload.channel);
            self.publish_coefficients(&topic, payload)?;
        }
        Ok(())
    }
}

// ── E8: OTA モデル更新 ─────────────────────────────────────────────────

/// OTA model update received via MQTT subscribe.
///
/// Allows cloud-side model coefficients to be pushed to edge devices,
/// enabling dynamic recalibration without redeployment.
#[derive(Debug, Clone)]
pub struct OtaModelUpdate {
    /// Sensor ID this update applies to
    pub sensor_id: String,
    /// Channel name
    pub channel: String,
    /// New slope (Q16.16)
    pub slope_q16: i32,
    /// New intercept (Q16.16)
    pub intercept_q16: i32,
    /// Model version (monotonically increasing)
    pub version: u64,
    /// Timestamp of update
    pub timestamp_ms: u64,
}

impl OtaModelUpdate {
    /// Parse from JSON string
    #[must_use]
    pub fn from_json(json: &str) -> Option<Self> {
        // 軽量パーサー (serde不要)
        fn extract_str<'a>(json: &'a str, key: &str) -> Option<&'a str> {
            let pattern = format!("\"{key}\":\"");
            let start = json.find(&pattern)? + pattern.len();
            let end = json[start..].find('"')? + start;
            Some(&json[start..end])
        }

        fn extract_i32(json: &str, key: &str) -> Option<i32> {
            let pattern = format!("\"{key}\":");
            let start = json.find(&pattern)? + pattern.len();
            let rest = json[start..].trim_start();
            let end = rest
                .find(|c: char| !c.is_ascii_digit() && c != '-')
                .unwrap_or(rest.len());
            rest[..end].parse().ok()
        }

        fn extract_u64(json: &str, key: &str) -> Option<u64> {
            let pattern = format!("\"{key}\":");
            let start = json.find(&pattern)? + pattern.len();
            let rest = json[start..].trim_start();
            let end = rest
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(rest.len());
            rest[..end].parse().ok()
        }

        Some(OtaModelUpdate {
            sensor_id: extract_str(json, "sensor")?.to_string(),
            channel: extract_str(json, "channel")?.to_string(),
            slope_q16: extract_i32(json, "slope_q16")?,
            intercept_q16: extract_i32(json, "intercept_q16")?,
            version: extract_u64(json, "version").unwrap_or(0),
            timestamp_ms: extract_u64(json, "ts").unwrap_or(0),
        })
    }

    /// Parse from binary (24 bytes, same layout as `CoefficientPayload`)
    ///
    /// `24バイトの基本フォーマットのみの場合、sensor_id/channel` は空文字列になる。
    /// 拡張フォーマット（`to_binary_ext` で生成）の場合、末尾に length-prefixed
    /// `sensor_id` と channel が含まれる。
    #[must_use]
    pub fn from_binary(data: &[u8]) -> Option<Self> {
        if data.len() < 24 {
            return None;
        }
        let slope_q16 = i32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let intercept_q16 = i32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let version = u64::from_le_bytes([
            data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15],
        ]);
        let timestamp_ms = u64::from_le_bytes([
            data[16], data[17], data[18], data[19], data[20], data[21], data[22], data[23],
        ]);

        // 拡張フォーマット: 24バイト以降に length-prefixed strings
        let mut sensor_id = String::new();
        let mut channel = String::new();
        let mut offset = 24;
        if offset + 2 <= data.len() {
            let sid_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
            offset += 2;
            if offset + sid_len <= data.len() {
                sensor_id = String::from_utf8_lossy(&data[offset..offset + sid_len]).to_string();
                offset += sid_len;
            }
        }
        if offset + 2 <= data.len() {
            let ch_len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
            offset += 2;
            if offset + ch_len <= data.len() {
                channel = String::from_utf8_lossy(&data[offset..offset + ch_len]).to_string();
            }
        }

        Some(OtaModelUpdate {
            sensor_id,
            channel,
            slope_q16,
            intercept_q16,
            version,
            timestamp_ms,
        })
    }

    /// Serialize to extended binary format with `sensor_id/channel`
    ///
    /// Format: [`slope_q16:i32`][intercept_q16:i32][version:u64][timestamp:u64]
    ///         [`sensor_id_len:u16`][sensor_id:bytes][`channel_len:u16`][channel:bytes]
    #[must_use]
    pub fn to_binary_ext(&self) -> Vec<u8> {
        let sid = self.sensor_id.as_bytes();
        let ch = self.channel.as_bytes();
        let total = 24 + 2 + sid.len() + 2 + ch.len();
        let mut buf = Vec::with_capacity(total);
        buf.extend_from_slice(&self.slope_q16.to_le_bytes());
        buf.extend_from_slice(&self.intercept_q16.to_le_bytes());
        buf.extend_from_slice(&self.version.to_le_bytes());
        buf.extend_from_slice(&self.timestamp_ms.to_le_bytes());
        buf.extend_from_slice(&(sid.len() as u16).to_le_bytes());
        buf.extend_from_slice(sid);
        buf.extend_from_slice(&(ch.len() as u16).to_le_bytes());
        buf.extend_from_slice(ch);
        buf
    }
}

/// OTA モデルストア — デバイス上の最新モデル係数を管理
pub struct OtaModelStore {
    models: std::collections::HashMap<String, OtaModelUpdate>,
}

impl Default for OtaModelStore {
    fn default() -> Self {
        Self::new()
    }
}

impl OtaModelStore {
    #[must_use]
    pub fn new() -> Self {
        Self {
            models: std::collections::HashMap::new(),
        }
    }

    /// モデル更新を適用（バージョンが新しい場合のみ）
    pub fn apply_update(&mut self, update: OtaModelUpdate) -> bool {
        let key = format!("{}:{}", update.sensor_id, update.channel);
        if let Some(existing) = self.models.get(&key) {
            if update.version <= existing.version {
                return false;
            }
        }
        self.models.insert(key, update);
        true
    }

    /// 指定センサー/チャネルの最新モデル係数を取得
    #[must_use]
    pub fn get_model(&self, sensor_id: &str, channel: &str) -> Option<(i32, i32)> {
        let key = format!("{sensor_id}:{channel}");
        self.models
            .get(&key)
            .map(|m| (m.slope_q16, m.intercept_q16))
    }

    /// 登録されたモデル数
    #[must_use]
    pub fn model_count(&self) -> usize {
        self.models.len()
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
            MqttError::Connection(e) => write!(f, "MQTT connection: {e}"),
            MqttError::Publish(e) => write!(f, "MQTT publish: {e}"),
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
        assert_eq!(
            i32::from_le_bytes([binary[0], binary[1], binary[2], binary[3]]),
            100
        );
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

    // ── New tests ─────────────────────────────────────────────

    #[test]
    fn test_config_local_client_id() {
        let config = MqttConfig::local("my-device");
        assert_eq!(config.client_id, "my-device");
        assert_eq!(config.topic_prefix, "alice-edge");
        assert!(config.username.is_none());
        assert!(config.password.is_none());
    }

    #[test]
    fn test_config_local_keep_alive() {
        let config = MqttConfig::local("pi5");
        assert_eq!(config.keep_alive, Duration::from_secs(30));
    }

    #[test]
    fn test_config_aws_no_credentials() {
        let config = MqttConfig::aws_iot("example.iot.region.amazonaws.com", "device-001");
        assert!(config.username.is_none());
        assert!(config.password.is_none());
        assert_eq!(config.client_id, "device-001");
    }

    #[test]
    fn test_config_azure_iot() {
        let config = MqttConfig::azure_iot("myhub", "device-xyz", "SharedAccessSignature sr=...");
        assert_eq!(config.host, "myhub.azure-devices.net");
        assert_eq!(config.port, 8883);
        assert!(config.use_tls);
        assert!(config.username.is_some());
        assert!(config.password.is_some());
        // Username must contain hub and device ID
        let username = config.username.unwrap();
        assert!(username.contains("myhub"));
        assert!(username.contains("device-xyz"));
    }

    #[test]
    fn test_config_azure_topic_prefix() {
        let config = MqttConfig::azure_iot("hub", "dev", "token");
        assert!(config.topic_prefix.contains("dev"));
    }

    #[test]
    fn test_payload_binary_intercept() {
        let payload = CoefficientPayload {
            sensor_id: "x".into(),
            channel: "y".into(),
            slope_q16: 0,
            intercept_q16: -999,
            slope_f32: 0.0,
            intercept_f32: 0.0,
            sample_count: 0,
            residual_error: 0.0,
            timestamp_ms: 0,
        };
        let bin = payload.to_binary();
        let intercept = i32::from_le_bytes([bin[4], bin[5], bin[6], bin[7]]);
        assert_eq!(intercept, -999);
    }

    #[test]
    fn test_payload_binary_sample_count() {
        let payload = CoefficientPayload {
            sensor_id: "s".into(),
            channel: "c".into(),
            slope_q16: 0,
            intercept_q16: 0,
            slope_f32: 0.0,
            intercept_f32: 0.0,
            sample_count: 12345,
            residual_error: 0.0,
            timestamp_ms: 0,
        };
        let bin = payload.to_binary();
        let count = u32::from_le_bytes([bin[8], bin[9], bin[10], bin[11]]);
        assert_eq!(count, 12345);
    }

    #[test]
    fn test_payload_binary_timestamp() {
        let payload = CoefficientPayload {
            sensor_id: "s".into(),
            channel: "c".into(),
            slope_q16: 0,
            intercept_q16: 0,
            slope_f32: 0.0,
            intercept_f32: 0.0,
            sample_count: 0,
            residual_error: 0.0,
            timestamp_ms: 9_999_999_999_000,
        };
        let bin = payload.to_binary();
        let ts = u64::from_le_bytes([
            bin[12], bin[13], bin[14], bin[15], bin[16], bin[17], bin[18], bin[19],
        ]);
        assert_eq!(ts, 9_999_999_999_000);
    }

    #[test]
    fn test_payload_json_contains_channel() {
        let payload = CoefficientPayload {
            sensor_id: "gps".into(),
            channel: "altitude".into(),
            slope_q16: 0,
            intercept_q16: 0,
            slope_f32: 0.0,
            intercept_f32: 0.0,
            sample_count: 50,
            residual_error: 0.0,
            timestamp_ms: 0,
        };
        let json = payload.to_json();
        assert!(json.contains("\"channel\":\"altitude\""));
        assert!(json.contains("\"samples\":50"));
    }

    #[test]
    fn test_payload_json_contains_q16_fields() {
        let payload = CoefficientPayload {
            sensor_id: "imu".into(),
            channel: "accel_z".into(),
            slope_q16: 42,
            intercept_q16: -7,
            slope_f32: 0.0,
            intercept_f32: 0.0,
            sample_count: 0,
            residual_error: 0.0,
            timestamp_ms: 0,
        };
        let json = payload.to_json();
        assert!(json.contains("\"slope_q16\":42"));
        assert!(json.contains("\"intercept_q16\":-7"));
    }

    #[test]
    fn test_mqtt_error_display_connection() {
        let err = MqttError::Connection("refused".into());
        assert_eq!(format!("{}", err), "MQTT connection: refused");
    }

    #[test]
    fn test_mqtt_error_display_publish() {
        let err = MqttError::Publish("queue full".into());
        assert_eq!(format!("{}", err), "MQTT publish: queue full");
    }

    // ── E8: OTA テスト ────────────────────────────────────────────

    #[test]
    fn test_ota_from_json() {
        let json = r#"{"sensor":"bme280","channel":"temperature","slope_q16":655360,"intercept_q16":1638400,"version":1,"ts":1700000000000}"#;
        let update = OtaModelUpdate::from_json(json);
        assert!(update.is_some());
        let u = update.unwrap();
        assert_eq!(u.sensor_id, "bme280");
        assert_eq!(u.channel, "temperature");
        assert_eq!(u.slope_q16, 655360);
        assert_eq!(u.intercept_q16, 1638400);
        assert_eq!(u.version, 1);
    }

    #[test]
    fn test_ota_from_binary() {
        // 拡張バイナリフォーマットでラウンドトリップ
        let update = OtaModelUpdate {
            sensor_id: "s".into(),
            channel: "c".into(),
            slope_q16: 42,
            intercept_q16: -99,
            version: 7,
            timestamp_ms: 12345,
        };
        let bin = update.to_binary_ext();
        let parsed = OtaModelUpdate::from_binary(&bin).unwrap();
        assert_eq!(parsed.slope_q16, 42);
        assert_eq!(parsed.intercept_q16, -99);
        assert_eq!(parsed.version, 7);
        assert_eq!(parsed.timestamp_ms, 12345);
        assert_eq!(parsed.sensor_id, "s");
        assert_eq!(parsed.channel, "c");
    }

    #[test]
    fn test_ota_from_binary_too_short() {
        let data = [0u8; 10];
        assert!(OtaModelUpdate::from_binary(&data).is_none());
    }

    #[test]
    fn test_ota_store_apply_and_get() {
        let mut store = OtaModelStore::new();
        let update = OtaModelUpdate {
            sensor_id: "bme280".into(),
            channel: "temp".into(),
            slope_q16: 100,
            intercept_q16: 200,
            version: 1,
            timestamp_ms: 0,
        };
        assert!(store.apply_update(update));
        assert_eq!(store.model_count(), 1);
        let model = store.get_model("bme280", "temp");
        assert_eq!(model, Some((100, 200)));
    }

    #[test]
    fn test_ota_store_rejects_old_version() {
        let mut store = OtaModelStore::new();
        store.apply_update(OtaModelUpdate {
            sensor_id: "s".into(),
            channel: "c".into(),
            slope_q16: 1,
            intercept_q16: 2,
            version: 5,
            timestamp_ms: 0,
        });
        // 古いバージョンは拒否
        let rejected = !store.apply_update(OtaModelUpdate {
            sensor_id: "s".into(),
            channel: "c".into(),
            slope_q16: 99,
            intercept_q16: 99,
            version: 3,
            timestamp_ms: 0,
        });
        assert!(rejected);
        assert_eq!(store.get_model("s", "c"), Some((1, 2)));
    }

    #[test]
    fn test_ota_store_accepts_newer_version() {
        let mut store = OtaModelStore::new();
        store.apply_update(OtaModelUpdate {
            sensor_id: "s".into(),
            channel: "c".into(),
            slope_q16: 1,
            intercept_q16: 2,
            version: 1,
            timestamp_ms: 0,
        });
        assert!(store.apply_update(OtaModelUpdate {
            sensor_id: "s".into(),
            channel: "c".into(),
            slope_q16: 99,
            intercept_q16: 88,
            version: 2,
            timestamp_ms: 0,
        }));
        assert_eq!(store.get_model("s", "c"), Some((99, 88)));
    }

    #[test]
    fn test_ota_store_get_nonexistent() {
        let store = OtaModelStore::new();
        assert_eq!(store.get_model("missing", "channel"), None);
    }

    // ── A7: OTA binary ext roundtrip テスト ───────────────────────

    #[test]
    fn test_ota_binary_ext_roundtrip() {
        let update = OtaModelUpdate {
            sensor_id: "bme280".into(),
            channel: "temperature".into(),
            slope_q16: 655360,
            intercept_q16: -1638400,
            version: 42,
            timestamp_ms: 1700000000000,
        };
        let bin = update.to_binary_ext();
        let parsed = OtaModelUpdate::from_binary(&bin).unwrap();
        assert_eq!(parsed.sensor_id, "bme280");
        assert_eq!(parsed.channel, "temperature");
        assert_eq!(parsed.slope_q16, 655360);
        assert_eq!(parsed.intercept_q16, -1638400);
        assert_eq!(parsed.version, 42);
        assert_eq!(parsed.timestamp_ms, 1700000000000);
    }

    #[test]
    fn test_ota_binary_ext_empty_strings() {
        let update = OtaModelUpdate {
            sensor_id: String::new(),
            channel: String::new(),
            slope_q16: 1,
            intercept_q16: 2,
            version: 0,
            timestamp_ms: 0,
        };
        let bin = update.to_binary_ext();
        // 24 base + 2 (sid_len=0) + 0 + 2 (ch_len=0) + 0 = 28
        assert_eq!(bin.len(), 28);
        let parsed = OtaModelUpdate::from_binary(&bin).unwrap();
        assert_eq!(parsed.sensor_id, "");
        assert_eq!(parsed.channel, "");
        assert_eq!(parsed.slope_q16, 1);
    }

    // ── A2: TLS config テスト ─────────────────────────────────────

    #[test]
    fn test_aws_iot_without_tls_config_fails() {
        let config = MqttConfig::aws_iot("example.iot.region.amazonaws.com", "device");
        let result = MqttPublisher::new(config);
        assert!(result.is_err());
        if let Err(MqttError::Connection(msg)) = result {
            assert!(msg.contains("tls_config"));
        }
    }

    #[test]
    fn test_with_tls_builder() {
        let config = MqttConfig::local("test").with_tls(MqttTlsConfig {
            ca_cert: b"ca".to_vec(),
            client_cert: b"cert".to_vec(),
            client_key: b"key".to_vec(),
        });
        assert!(config.use_tls);
        assert!(config.tls_config.is_some());
    }
}
