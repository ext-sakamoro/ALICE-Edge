// SPDX-License-Identifier: MIT
//! The MQTT client against a scripted broker over loopback TCP
//! (tests/support/mqtt_broker.rs): what goes over the wire, the CONNACK
//! refusals, QoS 1 redelivery after a reconnect, keep-alive and the errors a
//! publisher reports at once.
#![cfg(feature = "mqtt")]

#[path = "support/mqtt_broker.rs"]
mod broker;

use std::io::Read;
use std::time::Duration;

use alice_edge::mqtt_bridge::{
    CoefficientPayload, ConnectRefused, ConnectionState, MqttConfig, MqttError, MqttPublisher,
};

fn payload() -> CoefficientPayload {
    CoefficientPayload {
        sensor_id: "s".into(),
        channel: "c".into(),
        slope_q16: 65_536,
        intercept_q16: -7,
        slope_f32: 1.0,
        intercept_f32: 0.0,
        sample_count: 10,
        residual_error: 0.5,
        timestamp_ms: 1,
    }
}

fn local(port: u16) -> MqttConfig {
    let mut config = MqttConfig::local("edge-test");
    config.host = "127.0.0.1".into();
    config.port = port;
    config
}

#[test]
fn connect_publish_and_disconnect_on_the_wire() {
    let (listener, port) = broker::listener();
    let mut publisher = MqttPublisher::new(local(port)).unwrap();
    assert_eq!(publisher.connection_state(), ConnectionState::Connecting);
    // queued before the connection exists
    publisher.publish_coefficients("s/c", &payload()).unwrap();

    let mut stream = broker::accept(&listener);
    let connect = broker::read_connect(&mut stream);
    assert_eq!(
        connect,
        broker::Connect {
            flags: 0x02,
            keep_alive: 30,
            client_id: "edge-test".into(),
            user_name: None,
            password: None,
        }
    );
    broker::connack(&mut stream, 0);
    let first = broker::read_publish(&mut stream);
    assert_eq!(first.header, 0x32, "first delivery: DUP clear");
    assert_eq!(first.topic, "alice-edge/s/c");
    assert_eq!(first.packet_id, 1);
    assert_eq!(first.payload, payload().to_json().into_bytes());
    broker::puback(&mut stream, 1);
    assert!(broker::wait_until(250, || publisher.is_connected()));

    publisher.publish_binary("s/c/bin", &payload()).unwrap();
    let second = broker::read_publish(&mut stream);
    assert_eq!(second.packet_id, 2);
    assert_eq!(second.payload, payload().to_binary());
    broker::puback(&mut stream, 2);
    assert_eq!(publisher.messages_sent, 2);
    assert_eq!(
        publisher.bytes_sent,
        (payload().to_json().len() + payload().to_binary().len()) as u64,
        "payload bytes of both messages"
    );

    drop(publisher);
    assert_eq!(
        broker::read_packet(&mut stream),
        (0xe0, Vec::new()),
        "DISCONNECT on drop"
    );
    let mut rest = Vec::new();
    stream.read_to_end(&mut rest).unwrap();
    assert!(rest.is_empty(), "the socket closes after DISCONNECT");
}

#[test]
fn azure_style_user_name_and_password_are_sent() {
    let (listener, port) = broker::listener();
    let mut config = MqttConfig::azure_iot("hub", "device-1", "SharedAccessSignature sr=x");
    config.use_tls = false; // the TLS path is tests/mqtt_tls.rs
    config.host = "127.0.0.1".into();
    config.port = port;
    let _publisher = MqttPublisher::new(config).unwrap();
    let mut stream = broker::accept(&listener);
    let connect = broker::read_connect(&mut stream);
    assert_eq!(connect.flags, 0xc2, "user name + password + clean session");
    assert_eq!(connect.client_id, "device-1");
    assert_eq!(
        connect.user_name.as_deref(),
        Some("hub.azure-devices.net/device-1/api-version=2021-04-12")
    );
    assert_eq!(
        connect.password.as_deref(),
        Some(&b"SharedAccessSignature sr=x"[..])
    );
}

#[test]
fn each_connack_refusal_is_reported_as_its_own_state() {
    let refusals = [
        (1, ConnectRefused::UnacceptableProtocolVersion),
        (2, ConnectRefused::IdentifierRejected),
        (3, ConnectRefused::ServerUnavailable),
        (4, ConnectRefused::BadUserNameOrPassword),
        (5, ConnectRefused::NotAuthorized),
    ];
    for (code, refusal) in refusals {
        let (listener, port) = broker::listener();
        let publisher = MqttPublisher::new(local(port)).unwrap();
        let mut stream = broker::accept(&listener);
        broker::read_connect(&mut stream);
        broker::connack(&mut stream, code);
        assert!(
            broker::wait_until(250, || publisher.connection_state()
                == ConnectionState::Refused(refusal)),
            "code {code}: state {:?}",
            publisher.connection_state()
        );
        assert!(!publisher.is_connected());
        assert_eq!(refusal.code(), code);
    }
}

#[test]
fn an_unacknowledged_publish_is_resent_with_dup_after_a_reconnect() {
    let (listener, port) = broker::listener();
    let mut publisher = MqttPublisher::new(local(port)).unwrap();
    let mut stream = broker::accept(&listener);
    broker::read_connect(&mut stream);
    broker::connack(&mut stream, 0);
    publisher.publish_coefficients("s/c", &payload()).unwrap();
    let first = broker::read_publish(&mut stream);
    assert_eq!(first.header, 0x32);
    drop(stream); // no PUBACK: the broker goes away

    assert!(broker::wait_until(250, || publisher.connection_state()
        == ConnectionState::Disconnected));
    // the first reconnect waits 1 s
    let mut stream = broker::accept(&listener);
    broker::read_connect(&mut stream);
    broker::connack(&mut stream, 0);
    let again = broker::read_publish(&mut stream);
    assert_eq!(again.header, 0x3a, "redelivery sets DUP");
    assert_eq!(again.packet_id, first.packet_id);
    assert_eq!(again.payload, first.payload);
    broker::puback(&mut stream, again.packet_id);
    assert!(broker::wait_until(250, || publisher.is_connected()));
}

#[test]
fn keep_alive_pings_and_a_silent_broker_is_dropped() {
    let (listener, port) = broker::listener();
    let mut config = local(port);
    config.keep_alive = Duration::from_secs(1);
    let publisher = MqttPublisher::new(config).unwrap();
    let mut stream = broker::accept(&listener);
    assert_eq!(broker::read_connect(&mut stream).keep_alive, 1);
    broker::connack(&mut stream, 0);
    assert_eq!(
        broker::read_packet(&mut stream),
        (0xc0, Vec::new()),
        "PINGREQ after 1 s idle"
    );
    std::io::Write::write_all(&mut stream, &[0xd0, 0x00]).unwrap();
    assert_eq!(
        broker::read_packet(&mut stream),
        (0xc0, Vec::new()),
        "PINGREQ again 1 s later"
    );
    // no PINGRESP this time: the client closes after one more interval
    let mut rest = Vec::new();
    stream.read_to_end(&mut rest).unwrap();
    assert!(rest.is_empty());
    assert!(broker::wait_until(250, || publisher.connection_state()
        == ConnectionState::Disconnected));
    // and it connects again
    let mut stream = broker::accept(&listener);
    broker::read_connect(&mut stream);
}

#[test]
fn a_packet_the_client_never_expects_closes_the_connection() {
    let (listener, port) = broker::listener();
    let publisher = MqttPublisher::new(local(port)).unwrap();
    let mut stream = broker::accept(&listener);
    broker::read_connect(&mut stream);
    broker::connack(&mut stream, 0);
    assert!(broker::wait_until(250, || publisher.is_connected()));
    // a PUBLISH from the broker: this client never subscribed
    std::io::Write::write_all(&mut stream, &[0x30, 0x03, 0x00, 0x01, b't']).unwrap();
    let mut rest = Vec::new();
    stream.read_to_end(&mut rest).unwrap();
    assert!(rest.is_empty(), "the client closed without replying");
    assert!(broker::wait_until(250, || !publisher.is_connected()));
}

#[test]
fn invalid_topics_and_credentials_fail_at_once() {
    let (_listener, port) = broker::listener();
    let mut publisher = MqttPublisher::new(local(port)).unwrap();
    for suffix in ["s/#", "s/+/c", "s\0"] {
        let err = publisher
            .publish_coefficients(suffix, &payload())
            .unwrap_err();
        assert!(matches!(err, MqttError::Publish(_)), "{suffix:?}: {err}");
    }
    assert_eq!(publisher.messages_sent, 0);

    let mut config = local(port);
    config.password = Some("pw".into());
    let err = MqttPublisher::new(config)
        .err()
        .expect("a password needs a user name");
    assert!(matches!(err, MqttError::Connection(_)), "{err}");
}

#[test]
fn a_full_queue_is_an_error_not_a_block() {
    // nothing listens on this port: every message stays queued
    let (listener, port) = broker::listener();
    drop(listener);
    let mut publisher = MqttPublisher::new(local(port)).unwrap();
    let mut accepted = 0;
    let err = loop {
        match publisher.publish_coefficients("s/c", &payload()) {
            Ok(()) => accepted += 1,
            Err(e) => break e,
        }
        assert!(
            accepted <= 128,
            "64 in flight + 64 queued is the most the publisher holds"
        );
    };
    assert!(
        accepted >= 64,
        "the queue holds 64 messages, accepted {accepted}"
    );
    assert!(
        matches!(&err, MqttError::Publish(m) if m.contains("queue full")),
        "{err}"
    );
}

#[test]
#[allow(
    clippy::disallowed_methods,
    reason = "measures the wait between two connection attempts; only a lower bound is \
              asserted, so a slow runner cannot make it fail"
)]
fn reconnects_wait_for_the_backoff() {
    use std::time::Instant;
    // the broker hangs up on every connection before CONNACK
    let (listener, port) = broker::listener();
    let _publisher = MqttPublisher::new(local(port)).unwrap();
    let mut gaps = Vec::new();
    drop(broker::accept(&listener));
    for _ in 0..2 {
        let hung_up = Instant::now();
        let stream = broker::try_accept(&listener, 400).expect("a reconnect within 8 s");
        gaps.push(hung_up.elapsed());
        drop(stream);
    }
    // first backoff 1 s, then doubled to 2 s (a reconnect storm would come
    // back within one 50 ms poll)
    assert!(gaps[0] >= Duration::from_millis(950), "gaps {gaps:?}");
    assert!(gaps[1] >= Duration::from_millis(1_950), "gaps {gaps:?}");
}

#[test]
fn a_reset_connection_is_noticed_at_once() {
    let (listener, port) = broker::listener();
    let publisher = MqttPublisher::new(local(port)).unwrap();
    let stream = broker::accept(&listener);
    // wait until the CONNECT is in the broker's receive buffer, then close
    // without reading it: the kernel answers with a reset, not an orderly end
    let mut probe = [0u8; 1];
    assert_eq!(stream.peek(&mut probe).unwrap(), 1);
    drop(stream);
    // the client sees the reset on its next read and reconnects after the
    // first backoff (1 s), long before the CONNACK timeout (5 s) would close it
    let again = broker::try_accept(&listener, 150).expect("a reconnect within 3 s");
    drop(again);
    assert_ne!(publisher.connection_state(), ConnectionState::Connected);
}
