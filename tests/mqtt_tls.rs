// SPDX-License-Identifier: LicenseRef-Proprietary
//! The MQTT client over mutual TLS against a rustls broker in the test thread,
//! with certificates made here (P-256 keys derived from fixed labels, pure Rust): the happy path, and every way the
//! handshake can fail closing the connection without a byte of MQTT in
//! plaintext.
#![cfg(feature = "mqtt")]

#[path = "support/mqtt_broker.rs"]
mod broker;

use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::sync::Arc;
use std::time::Duration;

use alice_edge::mqtt_bridge::{
    ConnectionState, MqttConfig, MqttError, MqttPublisher, MqttTlsConfig,
};
use p256::ecdsa::{DerSignature, SigningKey};
use p256::pkcs8::{EncodePrivateKey, LineEnding};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};
use sha2::{Digest, Sha256};
use x509_cert::builder::profile::BuilderProfile;
use x509_cert::builder::{Builder, CertificateBuilder};
use x509_cert::der::asn1::{GeneralizedTime, Ia5String, ObjectIdentifier, OctetString};
use x509_cert::der::{Encode, EncodePem};
use x509_cert::ext::pkix::name::GeneralName;
use x509_cert::ext::pkix::{
    AuthorityKeyIdentifier, BasicConstraints, ExtendedKeyUsage, KeyUsage, KeyUsages,
    SubjectAltName, SubjectKeyIdentifier,
};
use x509_cert::ext::{Extension, ToExtension};
use x509_cert::name::Name;
use x509_cert::serial_number::SerialNumber;
use x509_cert::spki::{SubjectPublicKeyInfoOwned, SubjectPublicKeyInfoRef};
use x509_cert::time::{Time, Validity};
use x509_cert::TbsCertificate;

const MQTT_PROTOCOL_NAME: &[u8] = &[0x00, 0x04, b'M', b'Q', b'T', b'T'];

/// The extended key usage of a leaf (RFC 5280 section 4.2.1.12)
#[derive(Clone, Copy, Debug)]
enum Purpose {
    ServerAuth,
    ClientAuth,
}

impl Purpose {
    fn oid(self) -> ObjectIdentifier {
        match self {
            Self::ServerAuth => ObjectIdentifier::new_unwrap("1.3.6.1.5.5.7.3.1"),
            Self::ClientAuth => ObjectIdentifier::new_unwrap("1.3.6.1.5.5.7.3.2"),
        }
    }
}

/// A P-256 key whose scalar is SHA-256 of `label`: the same label gives the
/// same key on every run, a different label a different key. Test-only: the
/// key is public by construction (anyone can recompute it from the label), so
/// it is not secret and must never be used outside these tests
fn key(label: &str) -> SigningKey {
    SigningKey::from_slice(&Sha256::digest(label.as_bytes())).expect("a valid P-256 scalar")
}

/// 2000-01-01 to 2099-12-31, so the certificates hold whenever the tests run
fn validity() -> Validity {
    let at = |secs| {
        Time::GeneralTime(
            GeneralizedTime::from_unix_duration(Duration::from_secs(secs)).expect("in range"),
        )
    };
    Validity::new(at(946_684_800), at(4_102_358_400))
}

/// The two certificate profiles of these tests: a self-signed root CA and a
/// leaf it issues, each with the extensions of a CA and of an end entity
/// (subject and authority key identifiers, basic constraints, key usage)
enum TestProfile {
    Root { subject: Name },
    Leaf { issuer: Name, subject: Name },
}

impl BuilderProfile for TestProfile {
    fn get_issuer(&self, subject: &Name) -> Name {
        match self {
            Self::Root { .. } => subject.clone(),
            Self::Leaf { issuer, .. } => issuer.clone(),
        }
    }

    fn get_subject(&self) -> Name {
        match self {
            Self::Root { subject } | Self::Leaf { subject, .. } => subject.clone(),
        }
    }

    fn build_extensions(
        &self,
        spk: SubjectPublicKeyInfoRef<'_>,
        issuer_spk: SubjectPublicKeyInfoRef<'_>,
        tbs: &TbsCertificate,
    ) -> x509_cert::builder::Result<Vec<Extension>> {
        let subject = tbs.subject();
        let mut extensions = Vec::new();
        let ski = SubjectKeyIdentifier::try_from(spk)?;
        extensions.push((&ski).to_extension(subject, &extensions)?);
        let (ca, usage) = match self {
            Self::Root { .. } => (true, KeyUsages::KeyCertSign | KeyUsages::CRLSign),
            Self::Leaf { .. } => {
                let aki = AuthorityKeyIdentifier::try_from(issuer_spk)?;
                extensions.push((&aki).to_extension(subject, &extensions)?);
                (
                    false,
                    KeyUsages::DigitalSignature | KeyUsages::NonRepudiation,
                )
            }
        };
        let constraints = BasicConstraints {
            ca,
            path_len_constraint: None,
        };
        extensions.push((&constraints).to_extension(subject, &extensions)?);
        extensions.push((&KeyUsage(usage.into())).to_extension(subject, &extensions)?);
        Ok(extensions)
    }
}

struct Ca {
    name: Name,
    key: SigningKey,
    der: CertificateDer<'static>,
    pem: String,
}

struct Leaf {
    cert_pem: String,
    cert_der: CertificateDer<'static>,
    key_pem: String,
    key_der: Vec<u8>,
}

impl Ca {
    fn new(common_name: &str) -> Self {
        let key = key(common_name);
        let name: Name = format!("CN={common_name}").parse().expect("a name");
        let spki = SubjectPublicKeyInfoOwned::from_key(key.verifying_key()).expect("SPKI");
        let cert = CertificateBuilder::new(
            TestProfile::Root {
                subject: name.clone(),
            },
            SerialNumber::from(1u32),
            validity(),
            spki,
        )
        .expect("root profile")
        .build::<_, DerSignature>(&key)
        .expect("self-signed");
        Self {
            name,
            key,
            der: CertificateDer::from(cert.to_der().expect("DER")),
            pem: cert.to_pem(LineEnding::LF).expect("PEM"),
        }
    }

    fn pem(&self) -> String {
        self.pem.clone()
    }

    fn der(&self) -> CertificateDer<'static> {
        self.der.clone()
    }

    fn leaf(&self, names: &[&str], purpose: Purpose) -> Leaf {
        let key = key(&format!("{}/{}/{purpose:?}", self.name, names.join(",")));
        let spki = SubjectPublicKeyInfoOwned::from_key(key.verifying_key()).expect("SPKI");
        let profile = TestProfile::Leaf {
            issuer: self.name.clone(),
            subject: "CN=leaf".parse().expect("a name"),
        };
        let mut builder =
            CertificateBuilder::new(profile, SerialNumber::from(2u32), validity(), spki)
                .expect("leaf profile");
        let alt_names = names
            .iter()
            .map(|name| match name.parse::<std::net::IpAddr>() {
                Ok(std::net::IpAddr::V4(ip)) => {
                    GeneralName::IpAddress(OctetString::new(ip.octets()).expect("octets"))
                }
                Ok(std::net::IpAddr::V6(ip)) => {
                    GeneralName::IpAddress(OctetString::new(ip.octets()).expect("octets"))
                }
                Err(_) => GeneralName::DnsName(Ia5String::new(name).expect("an ASCII name")),
            })
            .collect();
        builder
            .add_extension(&SubjectAltName(alt_names))
            .expect("SAN");
        builder
            .add_extension(&ExtendedKeyUsage(vec![purpose.oid()]))
            .expect("EKU");
        let cert = builder
            .build::<_, DerSignature>(&self.key)
            .expect("signed by the CA");
        Leaf {
            cert_pem: cert.to_pem(LineEnding::LF).expect("PEM"),
            cert_der: CertificateDer::from(cert.to_der().expect("DER")),
            key_pem: key.to_pkcs8_pem(LineEnding::LF).expect("PEM").to_string(),
            key_der: key.to_pkcs8_der().expect("PKCS#8").as_bytes().to_vec(),
        }
    }
}

fn provider() -> Arc<rustls::crypto::CryptoProvider> {
    Arc::new(rustls::crypto::ring::default_provider())
}

/// A broker that presents `server` and accepts only client certificates issued by `client_ca`
fn server_config(server: &Leaf, client_ca: &Ca) -> Arc<rustls::ServerConfig> {
    let mut roots = rustls::RootCertStore::empty();
    roots.add(client_ca.der()).unwrap();
    let verifier =
        rustls::server::WebPkiClientVerifier::builder_with_provider(Arc::new(roots), provider())
            .build()
            .unwrap();
    let key = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(server.key_der.clone()));
    Arc::new(
        rustls::ServerConfig::builder_with_provider(provider())
            .with_safe_default_protocol_versions()
            .unwrap()
            .with_client_cert_verifier(verifier)
            .with_single_cert(vec![server.cert_der.clone()], key)
            .unwrap(),
    )
}

fn client_config(host: &str, port: u16, trusted_ca: &Ca, client: &Leaf) -> MqttConfig {
    let mut config = MqttConfig::aws_iot(host, "edge-tls").with_tls(MqttTlsConfig {
        ca_cert: trusted_ca.pem().into_bytes(),
        client_cert: client.cert_pem.clone().into_bytes(),
        client_key: client.key_pem.clone().into_bytes(),
    });
    config.port = port;
    config
}

/// Every byte the broker's socket received, kept for inspection
struct Recorder {
    inner: TcpStream,
    seen: Vec<u8>,
}

impl Read for Recorder {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = self.inner.read(buf)?;
        self.seen.extend_from_slice(&buf[..n]);
        Ok(n)
    }
}

impl Write for Recorder {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        self.inner.write(buf)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.inner.flush()
    }
}

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

/// Run the broker's side of one handshake; returns the result and the raw bytes received
fn handshake(
    stream: TcpStream,
    config: Arc<rustls::ServerConfig>,
) -> (io::Result<()>, Vec<u8>, rustls::ServerConnection) {
    let mut conn = rustls::ServerConnection::new(config).unwrap();
    let mut recorder = Recorder {
        inner: stream,
        seen: Vec::new(),
    };
    let mut result = Ok(());
    while conn.is_handshaking() {
        if let Err(e) = conn.complete_io(&mut recorder) {
            result = Err(e);
            break;
        }
    }
    (result, recorder.seen, conn)
}

#[test]
fn mutual_tls_carries_connect_and_publish() {
    let ca = Ca::new("test ca");
    let server = ca.leaf(&["127.0.0.1"], Purpose::ServerAuth);
    let client = ca.leaf(&["edge-tls"], Purpose::ClientAuth);
    let (listener, port) = broker::listener();
    let mut publisher = MqttPublisher::new(client_config("127.0.0.1", port, &ca, &client)).unwrap();

    let stream = broker::accept(&listener);
    let (result, seen, conn) = handshake(stream.try_clone().unwrap(), server_config(&server, &ca));
    result.expect("handshake");
    assert_eq!(
        seen.first(),
        Some(&0x16),
        "a TLS handshake record comes first"
    );
    let presented = conn
        .peer_certificates()
        .expect("the client presented a certificate");
    assert_eq!(
        presented[0], client.cert_der,
        "the configured client certificate"
    );

    let mut tls = rustls::StreamOwned::new(conn, stream);
    let connect = broker::read_connect(&mut tls);
    assert_eq!(connect.client_id, "edge-tls");
    broker::connack(&mut tls, 0);
    assert!(broker::wait_until(250, || publisher.is_connected()));
    publisher.publish_binary("s/c", &sample()).unwrap();
    let publish = broker::read_publish(&mut tls);
    assert_eq!(publish.topic, "alice-edge/s/c");
    assert_eq!(publish.payload, sample().to_binary());
}

fn sample() -> alice_edge::mqtt_bridge::CoefficientPayload {
    alice_edge::mqtt_bridge::CoefficientPayload {
        sensor_id: "s".into(),
        channel: "c".into(),
        slope_q16: 1,
        intercept_q16: 2,
        slope_f32: 0.0,
        intercept_f32: 0.0,
        sample_count: 3,
        residual_error: 0.0,
        timestamp_ms: 4,
    }
}

/// The client must refuse the broker and never reach Connected
fn assert_never_connected(publisher: &MqttPublisher) {
    assert!(
        !broker::wait_until(50, || publisher.connection_state()
            == ConnectionState::Connected),
        "the client connected"
    );
}

#[test]
fn a_server_certificate_from_another_ca_is_refused() {
    let ca = Ca::new("broker ca");
    let other = Ca::new("trusted ca");
    let server = ca.leaf(&["127.0.0.1"], Purpose::ServerAuth);
    let client = other.leaf(&["edge-tls"], Purpose::ClientAuth);
    let (listener, port) = broker::listener();
    let publisher = MqttPublisher::new(client_config("127.0.0.1", port, &other, &client)).unwrap();
    let stream = broker::accept(&listener);
    let (result, seen, _) = handshake(stream, server_config(&server, &other));
    assert!(result.is_err(), "the client aborted the handshake");
    assert!(
        !contains(&seen, MQTT_PROTOCOL_NAME),
        "no CONNECT in plaintext"
    );
    assert_never_connected(&publisher);
}

#[test]
fn a_server_certificate_for_another_name_is_refused() {
    let ca = Ca::new("test ca");
    let server = ca.leaf(&["broker.example"], Purpose::ServerAuth);
    let client = ca.leaf(&["edge-tls"], Purpose::ClientAuth);
    let (listener, port) = broker::listener();
    let publisher = MqttPublisher::new(client_config("127.0.0.1", port, &ca, &client)).unwrap();
    let stream = broker::accept(&listener);
    let (result, seen, _) = handshake(stream, server_config(&server, &ca));
    assert!(result.is_err(), "127.0.0.1 is not broker.example");
    assert!(!contains(&seen, MQTT_PROTOCOL_NAME));
    assert_never_connected(&publisher);
}

#[test]
fn a_client_certificate_the_broker_does_not_trust_gets_no_session() {
    let ca = Ca::new("broker ca");
    let rogue = Ca::new("rogue ca");
    let server = ca.leaf(&["127.0.0.1"], Purpose::ServerAuth);
    let client = rogue.leaf(&["edge-tls"], Purpose::ClientAuth);
    let (listener, port) = broker::listener();
    let publisher = MqttPublisher::new(client_config("127.0.0.1", port, &ca, &client)).unwrap();
    let stream = broker::accept(&listener);
    let (result, seen, _) = handshake(stream, server_config(&server, &ca));
    assert!(result.is_err(), "the broker refused the client certificate");
    assert!(!contains(&seen, MQTT_PROTOCOL_NAME));
    assert_never_connected(&publisher);
}

#[test]
fn a_plaintext_broker_only_ever_sees_tls_records() {
    let ca = Ca::new("test ca");
    let client = ca.leaf(&["edge-tls"], Purpose::ClientAuth);
    let (listener, port) = broker::listener();
    let publisher = MqttPublisher::new(client_config("127.0.0.1", port, &ca, &client)).unwrap();
    // two attempts: the retry after the failure is TLS again, never a fallback
    for attempt in 0..2 {
        let mut stream = broker::accept(&listener);
        let mut first = [0u8; 5];
        stream.read_exact(&mut first).unwrap();
        assert_eq!(
            first[0], 0x16,
            "attempt {attempt}: a TLS handshake record, not CONNECT (0x10)"
        );
        // answer like an MQTT broker would (CONNACK) and hang up
        broker::connack(&mut stream, 0);
        let mut rest = Vec::new();
        let _ = stream.read_to_end(&mut rest);
        assert!(
            !contains(&rest, MQTT_PROTOCOL_NAME),
            "attempt {attempt}: no CONNECT in plaintext"
        );
    }
    assert_never_connected(&publisher);
}

#[test]
fn unusable_tls_material_fails_in_new() {
    let ca = Ca::new("test ca");
    let client = ca.leaf(&["edge-tls"], Purpose::ClientAuth);
    let good = MqttTlsConfig {
        ca_cert: ca.pem().into_bytes(),
        client_cert: client.cert_pem.clone().into_bytes(),
        client_key: client.key_pem.clone().into_bytes(),
    };
    let cases: Vec<(&str, MqttTlsConfig)> = vec![
        (
            "no CA",
            MqttTlsConfig {
                ca_cert: Vec::new(),
                ..good.clone()
            },
        ),
        (
            "CA not PEM",
            MqttTlsConfig {
                ca_cert: b"not a certificate".to_vec(),
                ..good.clone()
            },
        ),
        (
            "no client certificate",
            MqttTlsConfig {
                client_cert: Vec::new(),
                ..good.clone()
            },
        ),
        (
            "no client key",
            MqttTlsConfig {
                client_key: Vec::new(),
                ..good.clone()
            },
        ),
        (
            "key of another certificate",
            MqttTlsConfig {
                client_key: ca.leaf(&["x"], Purpose::ClientAuth).key_pem.into_bytes(),
                ..good.clone()
            },
        ),
    ];
    // each case is refused by its own check, named in the message
    let causes = [
        "TLS CA certificate: no PEM certificate found",
        "TLS CA certificate: no PEM certificate found",
        "TLS client certificate: no PEM certificate found",
        "TLS client key: ",
        "TLS client certificate: keys may not be consistent",
    ];
    assert_eq!(cases.len(), causes.len());
    for ((what, tls), cause) in cases.into_iter().zip(causes) {
        let config = MqttConfig::aws_iot("127.0.0.1", "edge-tls").with_tls(tls);
        let err = MqttPublisher::new(config)
            .err()
            .unwrap_or_else(|| panic!("{what}: accepted"));
        assert!(
            matches!(&err, MqttError::Connection(m) if m.contains(cause)),
            "{what}: {err}"
        );
    }
    let mut no_material = MqttConfig::aws_iot("127.0.0.1", "edge-tls");
    no_material.use_tls = true;
    assert!(matches!(
        MqttPublisher::new(no_material),
        Err(MqttError::Connection(_))
    ));
    assert!(
        MqttPublisher::new(MqttConfig::aws_iot("127.0.0.1", "edge-tls").with_tls(good)).is_ok()
    );
}

#[test]
fn the_test_certificates_are_what_the_cases_assume() {
    use rustls::client::danger::ServerCertVerifier;
    use rustls::pki_types::{ServerName, UnixTime};

    let ca = Ca::new("test ca");
    let server = ca.leaf(&["127.0.0.1"], Purpose::ServerAuth);
    // the same labels give the same certificate on every run
    assert_eq!(
        server.cert_der,
        Ca::new("test ca")
            .leaf(&["127.0.0.1"], Purpose::ServerAuth)
            .cert_der
    );
    let mut roots = rustls::RootCertStore::empty();
    roots.add(ca.der()).unwrap();
    let roots = Arc::new(roots);
    let now = UnixTime::now();
    let ip = ServerName::try_from("127.0.0.1").unwrap();
    let servers =
        rustls::client::WebPkiServerVerifier::builder_with_provider(Arc::clone(&roots), provider())
            .build()
            .unwrap();
    let verify_server =
        |leaf: &Leaf| servers.verify_server_cert(&leaf.cert_der, &[], &ip, &[], now);
    assert!(verify_server(&server).is_ok());
    let named = ca.leaf(&["broker.example"], Purpose::ServerAuth);
    assert!(matches!(
        verify_server(&named),
        Err(rustls::Error::InvalidCertificate(
            rustls::CertificateError::NotValidForNameContext { .. }
        ))
    ));
    let foreign = Ca::new("other ca").leaf(&["127.0.0.1"], Purpose::ServerAuth);
    assert!(matches!(
        verify_server(&foreign),
        Err(rustls::Error::InvalidCertificate(
            rustls::CertificateError::UnknownIssuer
        ))
    ));

    let clients = rustls::server::WebPkiClientVerifier::builder_with_provider(roots, provider())
        .build()
        .unwrap();
    let client = ca.leaf(&["edge-tls"], Purpose::ClientAuth);
    assert!(clients
        .verify_client_cert(&client.cert_der, &[], now)
        .is_ok());
    assert!(
        clients
            .verify_client_cert(&server.cert_der, &[], now)
            .is_err(),
        "a server-auth leaf is not a client certificate"
    );
    assert_ne!(client.key_der, server.key_der, "each leaf has its own key");
}

#[test]
fn a_broker_that_drops_tcp_without_close_notify_is_left_at_once() {
    let ca = Ca::new("test ca");
    let server = ca.leaf(&["127.0.0.1"], Purpose::ServerAuth);
    let client = ca.leaf(&["edge-tls"], Purpose::ClientAuth);
    let (listener, port) = broker::listener();
    let publisher = MqttPublisher::new(client_config("127.0.0.1", port, &ca, &client)).unwrap();
    let stream = broker::accept(&listener);
    let (result, _, conn) = handshake(stream.try_clone().unwrap(), server_config(&server, &ca));
    result.expect("handshake");
    let mut tls = rustls::StreamOwned::new(conn, stream);
    broker::read_connect(&mut tls);
    broker::connack(&mut tls, 0);
    assert!(broker::wait_until(250, || publisher.is_connected()));

    // close the TCP connection with no TLS close_notify: every later read of
    // the client's TLS stream fails with UnexpectedEof, which must close the
    // connection now rather than at the keep-alive timeout (30 s)
    drop(tls);
    assert!(
        broker::wait_until(50, || !publisher.is_connected()),
        "still connected 1 s after the broker went away"
    );
    assert!(
        broker::try_accept(&listener, 150).is_some(),
        "a reconnect within 3 s (first backoff 1 s)"
    );
}
