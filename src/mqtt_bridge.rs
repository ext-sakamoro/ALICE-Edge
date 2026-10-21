// SPDX-License-Identifier: MIT
//! MQTT bridge for ALICE-Edge
//!
//! Publishes compressed model coefficients to MQTT brokers.
//! Supports AWS `IoT` Core, Azure `IoT` Hub, and standard MQTT 3.1.1 brokers:
//! publish only, QoS 1, over TCP or mutually authenticated TLS (rustls).
//!
//! # Example
//!
//! ```no_run
//! use alice_edge::mqtt_bridge::{MqttPublisher, MqttConfig, CoefficientPayload};
//!
//! let config = MqttConfig::local("edge-node-1");
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

use std::io::{self, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender, TryRecvError, TrySendError};
use std::sync::Arc;
use std::time::{Duration, Instant};

use rustls::pki_types::pem::PemObject;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};

use crate::mqtt_session::{Session, Status};
use crate::mqtt_wire as wire;

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
#[cfg_attr(feature = "serde-raw", derive(serde::Serialize, serde::Deserialize))]
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
    /// Serialize to JSON, with this build's `EDGE_SEMANTICS_ID` as the
    /// `semantics_id` field (64 lower-case hex digits) so the receiver can check
    /// which arithmetic produced the coefficients (ALICE-Law G7)
    #[must_use]
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"semantics_id":"{}","sensor":"{}","channel":"{}","slope":{},"intercept":{},"slope_q16":{},"intercept_q16":{},"samples":{},"error":{:.6},"ts":{}}}"#,
            hex32(&crate::EDGE_SEMANTICS_ID),
            json_escape(&self.sensor_id),
            json_escape(&self.channel),
            self.slope_f32,
            self.intercept_f32,
            self.slope_q16,
            self.intercept_q16,
            self.sample_count,
            self.residual_error,
            self.timestamp_ms
        )
    }

    /// Parse a payload written by [`CoefficientPayload::to_json`], checking its
    /// `semantics_id` (this build's gives [`IdStatus::Current`], a verified
    /// earlier one [`IdStatus::VerifiedPrevious`]).
    ///
    /// # Errors
    ///
    /// [`PayloadError::Semantics`] (both identifiers) for any other identifier,
    /// [`PayloadError::MissingId`] without one, [`PayloadError::Malformed`] for
    /// text that is not the flat object `to_json` writes.
    pub fn from_json(json: &str) -> Result<(Self, IdStatus), PayloadError> {
        let fields = parse_flat_json(json)?;
        let status = check_json_id(&fields)?;
        Ok((
            Self {
                sensor_id: field_str(&fields, "sensor")?,
                channel: field_str(&fields, "channel")?,
                slope_f32: field_num(&fields, "slope")?,
                intercept_f32: field_num(&fields, "intercept")?,
                slope_q16: field_num(&fields, "slope_q16")?,
                intercept_q16: field_num(&fields, "intercept_q16")?,
                sample_count: field_num(&fields, "samples")?,
                residual_error: field_num(&fields, "error")?,
                timestamp_ms: field_num(&fields, "ts")?,
            },
            status,
        ))
    }

    /// Serialize to compact binary ([`PAYLOAD_BINARY_LEN`] = 62 bytes)
    ///
    /// Format version 1, little-endian: magic `AEMQ`, `u16` version, this
    /// build's 32-byte `EDGE_SEMANTICS_ID`, then
    /// [`slope_q16`: i32][intercept_q16: i32][`sample_count`: u32][timestamp: u64][error: f32]
    /// (the 24-byte body of the earlier, headerless format).
    #[must_use]
    pub fn to_binary(&self) -> [u8; PAYLOAD_BINARY_LEN] {
        let mut buf = [0u8; PAYLOAD_BINARY_LEN];
        buf[0..4].copy_from_slice(&PAYLOAD_MAGIC);
        buf[4..6].copy_from_slice(&PAYLOAD_FORMAT_VERSION.to_le_bytes());
        buf[6..38].copy_from_slice(&crate::EDGE_SEMANTICS_ID);
        let body = &mut buf[38..];
        body[0..4].copy_from_slice(&self.slope_q16.to_le_bytes());
        body[4..8].copy_from_slice(&self.intercept_q16.to_le_bytes());
        body[8..12].copy_from_slice(&self.sample_count.to_le_bytes());
        body[12..20].copy_from_slice(&self.timestamp_ms.to_le_bytes());
        body[20..24].copy_from_slice(&(self.residual_error as f32).to_le_bytes());
        buf
    }

    /// Parse a binary payload written by [`CoefficientPayload::to_binary`],
    /// checking its identifier.
    ///
    /// # Errors
    ///
    /// [`PayloadError::Semantics`] (both identifiers) for any other identifier,
    /// [`PayloadError::BadMagic`] for the earlier 24-byte format without a
    /// header, and the version and length errors.
    pub fn from_binary(data: &[u8]) -> Result<(BinaryCoefficients, IdStatus), PayloadError> {
        let (status, body) = check_binary_header(data, &PAYLOAD_MAGIC)?;
        if body.len() != 24 {
            return Err(PayloadError::Malformed("binary payload length"));
        }
        let mut ts = [0u8; 8];
        ts.copy_from_slice(&body[12..20]);
        Ok((
            BinaryCoefficients {
                slope_q16: i32::from_le_bytes([body[0], body[1], body[2], body[3]]),
                intercept_q16: i32::from_le_bytes([body[4], body[5], body[6], body[7]]),
                sample_count: u32::from_le_bytes([body[8], body[9], body[10], body[11]]),
                timestamp_ms: u64::from_le_bytes(ts),
                residual_error: f32::from_le_bytes([body[20], body[21], body[22], body[23]]),
            },
            status,
        ))
    }
}

/// The coefficients a binary payload carries (the text fields are not in the
/// binary form)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BinaryCoefficients {
    /// Slope in Q16.16 format
    pub slope_q16: i32,
    /// Intercept in Q16.16 format
    pub intercept_q16: i32,
    /// Number of samples compressed
    pub sample_count: u32,
    /// Timestamp (milliseconds since epoch)
    pub timestamp_ms: u64,
    /// Residual error, as the `f32` the binary form stores
    pub residual_error: f32,
}

pub use crate::semantics::IdStatus;
use crate::semantics::SemanticsMismatch;

/// The first four bytes of a binary coefficient payload.
pub const PAYLOAD_MAGIC: [u8; 4] = *b"AEMQ";
/// The first four bytes of a binary OTA model update.
pub const OTA_MAGIC: [u8; 4] = *b"AEOT";
/// The payload format version this release writes and reads.
pub const PAYLOAD_FORMAT_VERSION: u16 = 1;
/// Length of [`CoefficientPayload::to_binary`].
pub const PAYLOAD_BINARY_LEN: usize = 4 + 2 + 32 + 24;

/// Why a received payload was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PayloadError {
    /// A binary payload without the expected magic (the earlier headerless
    /// format, or another message).
    BadMagic,
    /// A format version this release does not read.
    UnsupportedVersion(u16),
    /// A JSON payload without a `semantics_id` field.
    MissingId,
    /// The carried identifier is neither this build's nor a verified earlier one.
    Semantics(SemanticsMismatch),
    /// The bytes or text are not the format written by this module.
    Malformed(&'static str),
}

impl std::fmt::Display for PayloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BadMagic => write!(f, "payload: bad magic (no semantics identifier header)"),
            Self::UnsupportedVersion(v) => write!(f, "payload: unsupported format version {v}"),
            Self::MissingId => write!(f, "payload: no semantics_id field"),
            Self::Semantics(m) => write!(f, "payload: {m}"),
            Self::Malformed(why) => write!(f, "payload malformed: {why}"),
        }
    }
}

impl std::error::Error for PayloadError {}

fn hex32(id: &[u8; 32]) -> String {
    use std::fmt::Write;
    id.iter().fold(String::with_capacity(64), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            c if (c as u32) < 0x20 => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}

enum JsonValue {
    Str(String),
    Num(String),
}

/// Parses a flat JSON object of string and number values (the form this
/// module writes); anything else is `Malformed`.
fn parse_flat_json(text: &str) -> Result<Vec<(String, JsonValue)>, PayloadError> {
    fn string(
        chars: &mut std::iter::Peekable<std::str::Chars<'_>>,
    ) -> Result<String, PayloadError> {
        let mut out = String::new();
        loop {
            match chars
                .next()
                .ok_or(PayloadError::Malformed("unterminated string"))?
            {
                '"' => return Ok(out),
                '\\' => match chars
                    .next()
                    .ok_or(PayloadError::Malformed("unterminated escape"))?
                {
                    '"' => out.push('"'),
                    '\\' => out.push('\\'),
                    '/' => out.push('/'),
                    'n' => out.push('\n'),
                    't' => out.push('\t'),
                    'r' => out.push('\r'),
                    'u' => {
                        let hex: String = (0..4).filter_map(|_| chars.next()).collect();
                        let code = u32::from_str_radix(&hex, 16)
                            .map_err(|_| PayloadError::Malformed("bad \\u escape"))?;
                        out.push(
                            char::from_u32(code)
                                .ok_or(PayloadError::Malformed("bad \\u escape"))?,
                        );
                    }
                    _ => return Err(PayloadError::Malformed("unknown escape")),
                },
                c => out.push(c),
            }
        }
    }
    let mut chars = text.trim().chars().peekable();
    if chars.next() != Some('{') {
        return Err(PayloadError::Malformed("not an object"));
    }
    let mut fields = Vec::new();
    loop {
        while chars.peek().is_some_and(|c| c.is_whitespace()) {
            chars.next();
        }
        match chars.next() {
            Some('}') if fields.is_empty() => break,
            Some('"') => {}
            _ => return Err(PayloadError::Malformed("expected a key")),
        }
        let key = string(&mut chars)?;
        while chars.peek().is_some_and(|c| c.is_whitespace()) {
            chars.next();
        }
        if chars.next() != Some(':') {
            return Err(PayloadError::Malformed("expected ':'"));
        }
        while chars.peek().is_some_and(|c| c.is_whitespace()) {
            chars.next();
        }
        let value = if chars.peek() == Some(&'"') {
            chars.next();
            JsonValue::Str(string(&mut chars)?)
        } else {
            let mut num = String::new();
            while chars
                .peek()
                .is_some_and(|c| c.is_ascii_digit() || "+-.eE".contains(*c))
            {
                num.push(chars.next().unwrap_or('0'));
            }
            if num.is_empty() {
                return Err(PayloadError::Malformed("expected a string or a number"));
            }
            JsonValue::Num(num)
        };
        if fields.iter().any(|(k, _): &(String, JsonValue)| *k == key) {
            return Err(PayloadError::Malformed("repeated key"));
        }
        fields.push((key, value));
        while chars.peek().is_some_and(|c| c.is_whitespace()) {
            chars.next();
        }
        match chars.next() {
            Some(',') => continue,
            Some('}') => break,
            _ => return Err(PayloadError::Malformed("expected ',' or '}'")),
        }
    }
    if chars.any(|c| !c.is_whitespace()) {
        return Err(PayloadError::Malformed("text after the object"));
    }
    Ok(fields)
}

fn field<'a>(fields: &'a [(String, JsonValue)], key: &'static str) -> Option<&'a JsonValue> {
    fields.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn field_str(fields: &[(String, JsonValue)], key: &'static str) -> Result<String, PayloadError> {
    match field(fields, key) {
        Some(JsonValue::Str(s)) => Ok(s.clone()),
        _ => Err(PayloadError::Malformed(key)),
    }
}

fn field_num<T: std::str::FromStr>(
    fields: &[(String, JsonValue)],
    key: &'static str,
) -> Result<T, PayloadError> {
    match field(fields, key) {
        Some(JsonValue::Num(n)) => n.parse().map_err(|_| PayloadError::Malformed(key)),
        _ => Err(PayloadError::Malformed(key)),
    }
}

fn check_json_id(fields: &[(String, JsonValue)]) -> Result<IdStatus, PayloadError> {
    let hex = match field(fields, "semantics_id") {
        Some(JsonValue::Str(s)) => s,
        Some(JsonValue::Num(_)) => return Err(PayloadError::Malformed("semantics_id")),
        None => return Err(PayloadError::MissingId),
    };
    if hex.len() != 64 || !hex.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(PayloadError::Malformed("semantics_id"));
    }
    let mut id = [0u8; 32];
    for (i, b) in id.iter_mut().enumerate() {
        *b = u8::from_str_radix(&hex[2 * i..2 * i + 2], 16)
            .map_err(|_| PayloadError::Malformed("semantics_id"))?;
    }
    crate::semantics::check_semantics(&id).map_err(PayloadError::Semantics)
}

fn check_binary_header<'a>(
    data: &'a [u8],
    magic: &[u8; 4],
) -> Result<(IdStatus, &'a [u8]), PayloadError> {
    if data.len() < 4 || &data[..4] != magic {
        return Err(PayloadError::BadMagic);
    }
    if data.len() < 38 {
        return Err(PayloadError::Malformed("header"));
    }
    let version = u16::from_le_bytes([data[4], data[5]]);
    if version != PAYLOAD_FORMAT_VERSION {
        return Err(PayloadError::UnsupportedVersion(version));
    }
    let mut id = [0u8; 32];
    id.copy_from_slice(&data[6..38]);
    let status = crate::semantics::check_semantics(&id).map_err(PayloadError::Semantics)?;
    Ok((status, &data[38..]))
}

/// The broker's answer to CONNECT, as reported by [`MqttPublisher::connection_state`]
pub use crate::mqtt_wire::ConnectRefused;

/// Connection state reported by [`MqttPublisher::connection_state`]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConnectionState {
    /// No connection yet, or a connection is being opened
    Connecting,
    /// The broker accepted the connection (CONNACK return code 0)
    Connected,
    /// The connection was lost or could not be opened (TCP, TLS or protocol
    /// failure); a new one is attempted after a backoff of 1 s doubling to 32 s
    Disconnected,
    /// The broker refused the last connection with this CONNACK return code;
    /// attempts continue after the backoff
    Refused(ConnectRefused),
}

const STATE_CONNECTING: u8 = 0;
const STATE_CONNECTED: u8 = 1;
const STATE_DISCONNECTED: u8 = 2;
const STATE_REFUSED: u8 = 0x10;

fn encode_state(status: Status) -> u8 {
    match status {
        Status::Connecting => STATE_CONNECTING,
        Status::Connected => STATE_CONNECTED,
        Status::Disconnected => STATE_DISCONNECTED,
        Status::Refused(r) => STATE_REFUSED | r.code(),
    }
}

fn decode_state(value: u8) -> ConnectionState {
    match value {
        STATE_CONNECTED => ConnectionState::Connected,
        STATE_DISCONNECTED => ConnectionState::Disconnected,
        v if v & STATE_REFUSED != 0 => ConnectRefused::from_code(v & 0x0f)
            .map_or(ConnectionState::Connecting, ConnectionState::Refused),
        _ => ConnectionState::Connecting,
    }
}

/// Messages waiting for the network thread (beyond the ones in flight)
const REQUEST_QUEUE: usize = 64;
/// How long one read waits before the thread checks its queue and timers
const IO_POLL: Duration = Duration::from_millis(50);
/// TCP connect and TLS handshake limit
const OPEN_TIMEOUT: Duration = Duration::from_secs(5);

enum Request {
    Publish { topic: String, payload: Vec<u8> },
}

/// How a connection is opened, fixed when the publisher is created: a TLS
/// configuration never falls back to plaintext
enum Transport {
    Plain,
    Tls {
        config: Arc<rustls::ClientConfig>,
        server_name: ServerName<'static>,
    },
}

enum Stream {
    Plain(TcpStream),
    Tls(Box<rustls::StreamOwned<rustls::ClientConnection, TcpStream>>),
}

impl Read for Stream {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        match self {
            Stream::Plain(s) => s.read(buf),
            Stream::Tls(s) => s.read(buf),
        }
    }
}

impl Write for Stream {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        match self {
            Stream::Plain(s) => s.write(buf),
            Stream::Tls(s) => s.write(buf),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match self {
            Stream::Plain(s) => s.flush(),
            Stream::Tls(s) => s.flush(),
        }
    }
}

fn tls_error(what: &str, e: impl std::fmt::Display) -> MqttError {
    MqttError::Connection(format!("TLS {what}: {e}"))
}

/// rustls client configuration with the `ring` provider: the CA PEM is the
/// only trust anchor (no system store) and the client certificate and key are
/// always presented (mutual TLS)
fn tls_transport(tls: &MqttTlsConfig, host: &str) -> Result<Transport, MqttError> {
    let mut roots = rustls::RootCertStore::empty();
    for cert in CertificateDer::pem_slice_iter(&tls.ca_cert) {
        let cert = cert.map_err(|e| tls_error("CA certificate", e))?;
        roots
            .add(cert)
            .map_err(|e| tls_error("CA certificate", e))?;
    }
    if roots.is_empty() {
        return Err(tls_error("CA certificate", "no PEM certificate found"));
    }
    let chain = CertificateDer::pem_slice_iter(&tls.client_cert)
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| tls_error("client certificate", e))?;
    if chain.is_empty() {
        return Err(tls_error("client certificate", "no PEM certificate found"));
    }
    let key =
        PrivateKeyDer::from_pem_slice(&tls.client_key).map_err(|e| tls_error("client key", e))?;
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let config = rustls::ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .map_err(|e| tls_error("protocol versions", e))?
        .with_root_certificates(roots)
        .with_client_auth_cert(chain, key)
        .map_err(|e| tls_error("client certificate", e))?;
    let server_name =
        ServerName::try_from(host.to_owned()).map_err(|e| tls_error("server name", e))?;
    Ok(Transport::Tls {
        config: Arc::new(config),
        server_name,
    })
}

fn connect_tcp(host: &str, port: u16) -> io::Result<TcpStream> {
    let mut last = io::Error::new(
        io::ErrorKind::NotFound,
        format!("{host}:{port} resolved to no address"),
    );
    for addr in (host, port).to_socket_addrs()? {
        match TcpStream::connect_timeout(&addr, OPEN_TIMEOUT) {
            Ok(stream) => return Ok(stream),
            Err(e) => last = e,
        }
    }
    Err(last)
}

impl Transport {
    fn open(&self, host: &str, port: u16) -> io::Result<Stream> {
        let mut tcp = connect_tcp(host, port)?;
        tcp.set_nodelay(true)?;
        tcp.set_write_timeout(Some(OPEN_TIMEOUT))?;
        match self {
            Transport::Plain => {
                tcp.set_read_timeout(Some(IO_POLL))?;
                Ok(Stream::Plain(tcp))
            }
            Transport::Tls {
                config,
                server_name,
            } => {
                tcp.set_read_timeout(Some(OPEN_TIMEOUT))?;
                let mut conn =
                    rustls::ClientConnection::new(Arc::clone(config), server_name.clone())
                        .map_err(io::Error::other)?;
                // the handshake (server certificate verified against the CA)
                // finishes before the first MQTT byte; an error drops the socket
                while conn.is_handshaking() {
                    conn.complete_io(&mut tcp)?;
                }
                tcp.set_read_timeout(Some(IO_POLL))?;
                Ok(Stream::Tls(Box::new(rustls::StreamOwned::new(conn, tcp))))
            }
        }
    }
}

/// The network thread: owns the connection and the session, takes messages
/// from the publisher's queue and publishes the connection state
struct Driver {
    host: String,
    port: u16,
    transport: Transport,
    session: Session,
    requests: Receiver<Request>,
    state: Arc<AtomicU8>,
    stream: Option<Stream>,
}

impl Driver {
    fn run(mut self) {
        #[allow(
            clippy::disallowed_methods,
            reason = "the MQTT network thread's clock: it decides when a PINGREQ is due, when a missing \
                      CONNACK or PINGRESP closes the connection and when a reconnect is attempted; it \
                      never enters a payload, a coefficient or any other computed value"
        )]
        let origin = Instant::now();
        let mut buf = [0u8; 4096];
        loop {
            let now = Self::millis(origin);
            while self.session.can_accept() {
                match self.requests.try_recv() {
                    Ok(Request::Publish { topic, payload }) => {
                        // the publisher checked the topic and size, so this cannot fail
                        if let Ok(bytes) = self.session.publish(now, topic, payload) {
                            self.write(now, &bytes);
                        }
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        self.shutdown();
                        return;
                    }
                }
            }
            if self.stream.is_none() && self.session.should_connect(now) {
                let opened = self.transport.open(&self.host, self.port);
                let now = Self::millis(origin);
                match opened {
                    Ok(stream) => {
                        self.stream = Some(stream);
                        let connect = self.session.on_open(now);
                        self.write(now, &connect);
                    }
                    Err(_) => self.session.on_close(now),
                }
            }
            let now = Self::millis(origin);
            if let Some(stream) = self.stream.as_mut() {
                match stream.read(&mut buf) {
                    Ok(0) => self.close(now),
                    Ok(n) => match self.session.on_bytes(now, &buf[..n]) {
                        Ok(reply) => self.write(now, &reply),
                        Err(_) => self.close(now),
                    },
                    Err(e)
                        if matches!(
                            e.kind(),
                            io::ErrorKind::WouldBlock
                                | io::ErrorKind::TimedOut
                                | io::ErrorKind::Interrupted
                        ) => {}
                    Err(_) => self.close(now),
                }
            } else {
                std::thread::sleep(IO_POLL);
            }
            let now = Self::millis(origin);
            match self.session.tick(now) {
                Ok(bytes) => self.write(now, &bytes),
                Err(_) => self.close(now),
            }
            self.state
                .store(encode_state(self.session.status()), Ordering::Release);
        }
    }

    fn millis(origin: Instant) -> u64 {
        #[allow(
            clippy::disallowed_methods,
            reason = "same clock as Driver::run's origin: timers only"
        )]
        let elapsed = Instant::now().duration_since(origin);
        u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX)
    }

    fn write(&mut self, now: u64, bytes: &[u8]) {
        if bytes.is_empty() {
            return;
        }
        let failed = match self.stream.as_mut() {
            Some(stream) => stream
                .write_all(bytes)
                .and_then(|()| stream.flush())
                .is_err(),
            None => false,
        };
        if failed {
            self.close(now);
        }
    }

    fn close(&mut self, now: u64) {
        if let Some(Stream::Tls(stream)) = self.stream.as_mut() {
            stream.conn.send_close_notify();
            let _ = stream.flush();
        }
        self.stream = None;
        self.session.on_close(now);
    }

    /// The publisher was dropped: say DISCONNECT on an open session and stop
    fn shutdown(&mut self) {
        if self.session.status() == Status::Connected {
            if let Some(stream) = self.stream.as_mut() {
                let _ = stream
                    .write_all(&wire::encode_disconnect())
                    .and_then(|()| stream.flush());
            }
        }
        if let Some(Stream::Tls(stream)) = self.stream.as_mut() {
            stream.conn.send_close_notify();
            let _ = stream.flush();
        }
        self.stream = None;
        self.state.store(STATE_DISCONNECTED, Ordering::Release);
    }
}

/// MQTT publisher for ALICE-Edge coefficients
///
/// Publishes at QoS 1 over MQTT 3.1.1 (TCP, or TLS 1.2 / 1.3 through rustls
/// with mutual authentication). [`MqttPublisher::new`] returns at once; a
/// network thread opens the connection, keeps it alive (PINGREQ after
/// `keep_alive` without traffic), reconnects after a backoff of 1 s doubling
/// to 32 s, and sends every unacknowledged message again after a reconnect.
pub struct MqttPublisher {
    requests: SyncSender<Request>,
    config: MqttConfig,
    /// Connection state, written by the network thread
    state: Arc<AtomicU8>,
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
    /// Returns `MqttError::Connection` if TLS is enabled but no `tls_config` is
    /// provided, if a PEM in `tls_config` does not parse, if the host is not a
    /// valid TLS server name, or if the client identifier, user name or
    /// password cannot be carried by MQTT 3.1.1 (a password needs a user name).
    pub fn new(config: MqttConfig) -> Result<Self, MqttError> {
        let transport = if config.use_tls {
            match &config.tls_config {
                Some(tls) => tls_transport(tls, &config.host)?,
                None => {
                    return Err(MqttError::Connection(
                        "TLS enabled but no tls_config provided. Use MqttConfig::with_tls() to set certificates.".into(),
                    ))
                }
            }
        } else {
            Transport::Plain
        };
        let session = Session::new(&wire::Connect {
            client_id: &config.client_id,
            keep_alive_secs: u16::try_from(config.keep_alive.as_secs()).unwrap_or(u16::MAX),
            clean_session: true,
            user_name: config.username.as_deref(),
            password: config.password.as_deref().map(str::as_bytes),
        })
        .map_err(|e| MqttError::Connection(format!("CONNECT: {e}")))?;

        let (requests, receiver) = sync_channel(REQUEST_QUEUE);
        let state = Arc::new(AtomicU8::new(STATE_CONNECTING));
        let driver = Driver {
            host: config.host.clone(),
            port: config.port,
            transport,
            session,
            requests: receiver,
            state: Arc::clone(&state),
            stream: None,
        };

        // ALICE-Law G13: the client's event loop drives network I/O on its own thread and only sets
        // a flag; nothing it computes is combined into a value (listed in scripts/order_allowlist.toml)
        #[allow(clippy::disallowed_methods)]
        std::thread::spawn(move || {
            driver.run();
        });

        Ok(Self {
            requests,
            config,
            state,
            messages_sent: 0,
            bytes_sent: 0,
        })
    }

    /// ブローカーに接続済みかどうかを返す
    #[must_use]
    pub fn is_connected(&self) -> bool {
        self.connection_state() == ConnectionState::Connected
    }

    /// The connection state, including the broker's refusal code
    #[must_use]
    pub fn connection_state(&self) -> ConnectionState {
        decode_state(self.state.load(Ordering::Acquire))
    }

    fn enqueue(&mut self, topic: String, payload: Vec<u8>) -> Result<(), MqttError> {
        wire::check_publish(&topic, payload.len())
            .map_err(|e| MqttError::Publish(format!("topic {topic:?}: {e}")))?;
        let bytes = payload.len() as u64;
        match self.requests.try_send(Request::Publish { topic, payload }) {
            Ok(()) => {
                self.messages_sent += 1;
                self.bytes_sent += bytes;
                Ok(())
            }
            Err(TrySendError::Full(_)) => Err(MqttError::Publish(format!(
                "send queue full ({REQUEST_QUEUE} messages waiting for the network thread)"
            ))),
            Err(TrySendError::Disconnected(_)) => {
                Err(MqttError::Publish("the network thread has stopped".into()))
            }
        }
    }

    /// Publish coefficients as JSON
    ///
    /// # Errors
    ///
    /// Returns `MqttError::Publish` if the topic is not publishable (empty or
    /// with a `+` / `#` wildcard) or the send queue is full.
    pub fn publish_coefficients(
        &mut self,
        topic_suffix: &str,
        payload: &CoefficientPayload,
    ) -> Result<(), MqttError> {
        let topic = format!("{}/{}", self.config.topic_prefix, topic_suffix);
        self.enqueue(topic, payload.to_json().into_bytes())
    }

    /// Publish coefficients as compact binary ([`PAYLOAD_BINARY_LEN`] bytes per message)
    ///
    /// # Errors
    ///
    /// Returns `MqttError::Publish` if the topic is not publishable (empty or
    /// with a `+` / `#` wildcard) or the send queue is full.
    pub fn publish_binary(
        &mut self,
        topic_suffix: &str,
        payload: &CoefficientPayload,
    ) -> Result<(), MqttError> {
        let topic = format!("{}/{}", self.config.topic_prefix, topic_suffix);
        self.enqueue(topic, payload.to_binary().to_vec())
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
    /// Parse an update from JSON, checking its `semantics_id` field (64
    /// lower-case hex digits: the `EDGE_SEMANTICS_ID` of the build that fitted
    /// the coefficients). `version` and `ts` default to 0.
    ///
    /// # Errors
    ///
    /// [`PayloadError::Semantics`] (both identifiers) for another identifier,
    /// [`PayloadError::MissingId`] without one, [`PayloadError::Malformed`] for
    /// a missing field or text that is not a flat JSON object.
    pub fn from_json(json: &str) -> Result<(Self, IdStatus), PayloadError> {
        let fields = parse_flat_json(json)?;
        let status = check_json_id(&fields)?;
        let opt_u64 = |key| match field(&fields, key) {
            None => Ok(0),
            Some(_) => field_num::<u64>(&fields, key),
        };
        Ok((
            Self {
                sensor_id: field_str(&fields, "sensor")?,
                channel: field_str(&fields, "channel")?,
                slope_q16: field_num(&fields, "slope_q16")?,
                intercept_q16: field_num(&fields, "intercept_q16")?,
                version: opt_u64("version")?,
                timestamp_ms: opt_u64("ts")?,
            },
            status,
        ))
    }

    /// Parse an update written by [`OtaModelUpdate::to_binary_ext`], checking
    /// its identifier.
    ///
    /// # Errors
    ///
    /// [`PayloadError::Semantics`] (both identifiers) for another identifier,
    /// [`PayloadError::BadMagic`] for the earlier headerless format, and the
    /// version and length errors.
    pub fn from_binary(data: &[u8]) -> Result<(Self, IdStatus), PayloadError> {
        let (status, data) = check_binary_header(data, &OTA_MAGIC)?;
        if data.len() < 24 {
            return Err(PayloadError::Malformed("update length"));
        }
        let word = |o: usize| {
            let mut b = [0u8; 8];
            b.copy_from_slice(&data[o..o + 8]);
            u64::from_le_bytes(b)
        };
        let slope_q16 = i32::from_le_bytes([data[0], data[1], data[2], data[3]]);
        let intercept_q16 = i32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        let version = word(8);
        let timestamp_ms = word(16);
        let mut offset = 24;
        let mut text = || -> Result<String, PayloadError> {
            if offset + 2 > data.len() {
                return Err(PayloadError::Malformed("string length"));
            }
            let len = u16::from_le_bytes([data[offset], data[offset + 1]]) as usize;
            offset += 2;
            if offset + len > data.len() {
                return Err(PayloadError::Malformed("string data"));
            }
            let s = std::str::from_utf8(&data[offset..offset + len])
                .map_err(|_| PayloadError::Malformed("string is not UTF-8"))?
                .to_string();
            offset += len;
            Ok(s)
        };
        let sensor_id = text()?;
        let channel = text()?;
        if offset != data.len() {
            return Err(PayloadError::Malformed("bytes after the update"));
        }
        Ok((
            Self {
                sensor_id,
                channel,
                slope_q16,
                intercept_q16,
                version,
                timestamp_ms,
            },
            status,
        ))
    }

    /// Serialize to the binary update format
    ///
    /// Format version 1, little-endian: magic `AEOT`, `u16` version, this
    /// build's 32-byte `EDGE_SEMANTICS_ID`, then
    /// `[slope_q16:i32][intercept_q16:i32][version:u64][timestamp:u64]`
    /// `[sensor_id_len:u16][sensor_id:bytes][channel_len:u16][channel:bytes]`
    #[must_use]
    pub fn to_binary_ext(&self) -> Vec<u8> {
        let sid = self.sensor_id.as_bytes();
        let ch = self.channel.as_bytes();
        let mut buf = Vec::with_capacity(38 + 24 + 4 + sid.len() + ch.len());
        buf.extend_from_slice(&OTA_MAGIC);
        buf.extend_from_slice(&PAYLOAD_FORMAT_VERSION.to_le_bytes());
        buf.extend_from_slice(&crate::EDGE_SEMANTICS_ID);
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
    models: std::collections::BTreeMap<String, OtaModelUpdate>,
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
            models: std::collections::BTreeMap::new(),
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
        assert_eq!(binary.len(), PAYLOAD_BINARY_LEN);
        assert_eq!(
            i32::from_le_bytes([binary[38], binary[39], binary[40], binary[41]]),
            100
        );
    }

    fn sample_payload() -> CoefficientPayload {
        CoefficientPayload {
            sensor_id: "bme\"280\\x".into(),
            channel: "temp\nerature".into(),
            slope_q16: -655_360,
            intercept_q16: 163_840_000,
            slope_f32: -10.0,
            intercept_f32: 2500.5,
            sample_count: 1000,
            residual_error: 0.001_25,
            timestamp_ms: 1_708_200_000_000,
        }
    }

    fn foreign() -> [u8; 32] {
        let mut id = crate::EDGE_SEMANTICS_ID;
        id[9] ^= 0x80;
        id
    }

    fn assert_names_both(e: &PayloadError) {
        let text = e.to_string();
        assert!(text.contains(&hex32(&foreign())), "{text}");
        assert!(text.contains(&hex32(&crate::EDGE_SEMANTICS_ID)), "{text}");
    }

    /// JSON and binary payloads round-trip with this build's identifier, and
    /// the text fields survive quotes, backslashes and newlines.
    #[test]
    fn payloads_round_trip_with_the_identifier() {
        let p = sample_payload();
        let (back, status) = CoefficientPayload::from_json(&p.to_json()).unwrap();
        assert_eq!(status, IdStatus::Current);
        assert_eq!(
            (back.sensor_id.as_str(), back.channel.as_str()),
            (p.sensor_id.as_str(), p.channel.as_str())
        );
        assert_eq!(
            (back.slope_q16, back.intercept_q16, back.sample_count),
            (p.slope_q16, p.intercept_q16, p.sample_count)
        );
        assert_eq!(
            (back.slope_f32.to_bits(), back.intercept_f32.to_bits()),
            (p.slope_f32.to_bits(), p.intercept_f32.to_bits())
        );
        assert_eq!(back.timestamp_ms, p.timestamp_ms);
        let (bin, status) = CoefficientPayload::from_binary(&p.to_binary()).unwrap();
        assert_eq!(status, IdStatus::Current);
        assert_eq!(
            (
                bin.slope_q16,
                bin.intercept_q16,
                bin.sample_count,
                bin.timestamp_ms
            ),
            (p.slope_q16, p.intercept_q16, p.sample_count, p.timestamp_ms)
        );
        let ota = OtaModelUpdate {
            sensor_id: "s".into(),
            channel: "c".into(),
            slope_q16: 3,
            intercept_q16: -4,
            version: 5,
            timestamp_ms: 6,
        };
        let (o, status) = OtaModelUpdate::from_binary(&ota.to_binary_ext()).unwrap();
        assert_eq!(status, IdStatus::Current);
        assert_eq!((o.slope_q16, o.intercept_q16, o.version), (3, -4, 5));
    }

    /// Every reader refuses another identifier naming both (ALICE-Law G7).
    #[test]
    fn payloads_with_another_identifier_are_refused_naming_both() {
        let p = sample_payload();
        let json = p
            .to_json()
            .replace(&hex32(&crate::EDGE_SEMANTICS_ID), &hex32(&foreign()));
        let e = CoefficientPayload::from_json(&json).unwrap_err();
        assert!(matches!(e, PayloadError::Semantics(_)));
        assert_names_both(&e);
        let mut bin = p.to_binary();
        bin[6..38].copy_from_slice(&foreign());
        let e = CoefficientPayload::from_binary(&bin).unwrap_err();
        assert_names_both(&e);
        let ota = OtaModelUpdate {
            sensor_id: "s".into(),
            channel: "c".into(),
            slope_q16: 1,
            intercept_q16: 2,
            version: 0,
            timestamp_ms: 0,
        };
        let mut ob = ota.to_binary_ext();
        ob[6..38].copy_from_slice(&foreign());
        assert_names_both(&OtaModelUpdate::from_binary(&ob).unwrap_err());
        let oj = format!(
            r#"{{"semantics_id":"{}","sensor":"s","channel":"c","slope_q16":1,"intercept_q16":2}}"#,
            hex32(&foreign())
        );
        assert_names_both(&OtaModelUpdate::from_json(&oj).unwrap_err());
    }

    /// A payload without the identifier (the earlier formats) is refused.
    #[test]
    fn payloads_without_the_identifier_are_refused() {
        let p = sample_payload();
        let legacy_bin = &p.to_binary()[38..];
        assert_eq!(
            CoefficientPayload::from_binary(legacy_bin).unwrap_err(),
            PayloadError::BadMagic
        );
        let no_id =
            r#"{"sensor":"s","channel":"c","slope_q16":1,"intercept_q16":2,"version":1,"ts":0}"#;
        assert_eq!(
            OtaModelUpdate::from_json(no_id).unwrap_err(),
            PayloadError::MissingId
        );
        let mut v2 = p.to_binary();
        v2[4] = 2;
        assert_eq!(
            CoefficientPayload::from_binary(&v2).unwrap_err(),
            PayloadError::UnsupportedVersion(2)
        );
        assert!(matches!(
            CoefficientPayload::from_json("[1]"),
            Err(PayloadError::Malformed(_))
        ));
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
        let config =
            MqttConfig::aws_iot("abc123-ats.iot.ap-northeast-1.amazonaws.com", "edge-node");
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
        let config = MqttConfig::local("edge-node");
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
        let intercept = i32::from_le_bytes([bin[42], bin[43], bin[44], bin[45]]);
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
        let count = u32::from_le_bytes([bin[46], bin[47], bin[48], bin[49]]);
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
            bin[50], bin[51], bin[52], bin[53], bin[54], bin[55], bin[56], bin[57],
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
        let json = format!(
            r#"{{"semantics_id":"{}","sensor":"bme280","channel":"temperature","slope_q16":655360,"intercept_q16":1638400,"version":1,"ts":1700000000000}}"#,
            hex32(&crate::EDGE_SEMANTICS_ID)
        );
        let (u, status) = OtaModelUpdate::from_json(&json).unwrap();
        assert_eq!(status, IdStatus::Current);
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
        let parsed = OtaModelUpdate::from_binary(&bin).unwrap().0;
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
        assert_eq!(
            OtaModelUpdate::from_binary(&data).unwrap_err(),
            PayloadError::BadMagic
        );
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
        let parsed = OtaModelUpdate::from_binary(&bin).unwrap().0;
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
        // 38 header + 24 base + 2 (sid_len=0) + 0 + 2 (ch_len=0) + 0 = 66
        assert_eq!(bin.len(), 66);
        let parsed = OtaModelUpdate::from_binary(&bin).unwrap().0;
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
