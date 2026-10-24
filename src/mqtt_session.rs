// SPDX-License-Identifier: LicenseRef-Proprietary
//! Connection state of the publish-only MQTT 3.1.1 client, without I/O
//!
//! [`Session`] is driven by the transport loop in [`crate::mqtt_bridge`]: it is
//! told when a connection opens or closes, which bytes arrived and what the
//! time is (milliseconds from any fixed origin), and it answers with the
//! bytes to write and whether to close. Time is an argument, so keep-alive,
//! the CONNACK timeout and the reconnect backoff are tested with a counter in
//! place of a clock.
//!
//! Delivery is QoS 1: a PUBLISH stays in flight until its PUBACK arrives, and
//! every message in flight is sent again (DUP set, same packet identifier)
//! after a reconnect.

use std::collections::VecDeque;

use crate::mqtt_wire::{self as wire, ConnectRefused, Incoming, WireError};

/// First wait before reconnecting, doubled after each failed attempt
pub(crate) const BACKOFF_FIRST_MS: u64 = 1_000;
/// Longest wait before reconnecting
pub(crate) const BACKOFF_MAX_MS: u64 = 32_000;
/// Time allowed from opening a connection to receiving CONNACK
pub(crate) const CONNACK_TIMEOUT_MS: u64 = 5_000;
/// Messages awaiting PUBACK; further messages wait in the publisher's queue
pub(crate) const MAX_IN_FLIGHT: usize = 64;

/// What the session reports to the publisher
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Status {
    /// No connection has been attempted yet, or one is being opened
    Connecting,
    /// CONNACK accepted the connection
    Connected,
    /// The last connection was lost; a new one is attempted after the backoff
    Disconnected,
    /// The last CONNACK refused the connection; attempts continue after the backoff
    Refused(ConnectRefused),
}

/// Why the transport must close
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum CloseReason {
    /// The broker sent bytes this client cannot accept
    Protocol(WireError),
    /// The broker sent a packet out of order (CONNACK twice, PUBACK before CONNACK, ...)
    OutOfOrder(&'static str),
    /// CONNACK refused the connection
    Refused(ConnectRefused),
    /// No CONNACK within [`CONNACK_TIMEOUT_MS`]
    ConnackTimeout,
    /// No PINGRESP within the keep-alive interval
    PingTimeout,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    /// No connection; open one at or after `retry_at`
    Idle {
        retry_at: u64,
    },
    /// Connection open, CONNECT sent, CONNACK not yet received
    AwaitConnAck {
        since: u64,
    },
    Connected,
}

#[derive(Debug, Clone)]
struct Message {
    packet_id: u16,
    topic: String,
    payload: Vec<u8>,
    /// sent at least once (a later send sets DUP)
    sent: bool,
}

/// MQTT session state for one client
#[derive(Debug)]
pub(crate) struct Session {
    connect: Vec<u8>,
    keep_alive_ms: u64,
    phase: Phase,
    started: bool,
    refused: Option<ConnectRefused>,
    backoff_ms: u64,
    next_packet_id: u16,
    in_flight: VecDeque<Message>,
    inbuf: Vec<u8>,
    last_sent: u64,
    ping_sent: Option<u64>,
}

impl Session {
    /// A session whose CONNECT packet is encoded now, so a client identifier,
    /// user name or password that MQTT 3.1.1 cannot carry fails before any I/O
    pub(crate) fn new(connect: &wire::Connect<'_>) -> Result<Self, WireError> {
        Ok(Self {
            connect: wire::encode_connect(connect)?,
            keep_alive_ms: u64::from(connect.keep_alive_secs) * 1_000,
            phase: Phase::Idle { retry_at: 0 },
            started: false,
            refused: None,
            backoff_ms: BACKOFF_FIRST_MS,
            next_packet_id: 1,
            in_flight: VecDeque::new(),
            inbuf: Vec::new(),
            last_sent: 0,
            ping_sent: None,
        })
    }

    pub(crate) fn status(&self) -> Status {
        match self.phase {
            Phase::Connected => Status::Connected,
            _ => match self.refused {
                Some(r) => Status::Refused(r),
                None if !self.started || matches!(self.phase, Phase::AwaitConnAck { .. }) => {
                    Status::Connecting
                }
                None => Status::Disconnected,
            },
        }
    }

    /// Whether the transport should open a connection now
    pub(crate) fn should_connect(&self, now: u64) -> bool {
        matches!(self.phase, Phase::Idle { retry_at } if now >= retry_at)
    }

    /// Whether another message can be taken from the publisher's queue
    pub(crate) fn can_accept(&self) -> bool {
        self.in_flight.len() < MAX_IN_FLIGHT
    }

    /// The transport opened a connection: returns CONNECT to write first
    pub(crate) fn on_open(&mut self, now: u64) -> Vec<u8> {
        self.started = true;
        self.phase = Phase::AwaitConnAck { since: now };
        self.inbuf.clear();
        self.ping_sent = None;
        self.last_sent = now;
        self.connect.clone()
    }

    /// The connection closed or could not be opened: wait for the backoff
    pub(crate) fn on_close(&mut self, now: u64) {
        self.started = true;
        self.phase = Phase::Idle {
            retry_at: now.saturating_add(self.backoff_ms),
        };
        self.backoff_ms = (self.backoff_ms * 2).min(BACKOFF_MAX_MS);
        self.inbuf.clear();
        self.ping_sent = None;
    }

    /// Queue a QoS 1 message; returns the PUBLISH bytes to write when connected
    ///
    /// The caller checks [`Self::can_accept`] first and validates the topic
    /// with [`wire::check_topic`] before queueing it.
    pub(crate) fn publish(
        &mut self,
        now: u64,
        topic: String,
        payload: Vec<u8>,
    ) -> Result<Vec<u8>, WireError> {
        wire::check_topic(&topic)?;
        let packet_id = self.allocate_packet_id();
        let mut message = Message {
            packet_id,
            topic,
            payload,
            sent: false,
        };
        let out = if self.phase == Phase::Connected {
            let bytes = wire::encode_publish(&message.topic, packet_id, &message.payload, false)?;
            message.sent = true;
            self.last_sent = now;
            bytes
        } else {
            Vec::new()
        };
        self.in_flight.push_back(message);
        Ok(out)
    }

    /// Bytes arrived from the broker: returns the bytes to write in reply, or
    /// why the connection must close
    pub(crate) fn on_bytes(&mut self, now: u64, data: &[u8]) -> Result<Vec<u8>, CloseReason> {
        self.inbuf.extend_from_slice(data);
        let mut out = Vec::new();
        loop {
            let (packet, used) = match wire::decode(&self.inbuf) {
                Ok(Some(decoded)) => decoded,
                Ok(None) => return Ok(out),
                Err(e) => return Err(CloseReason::Protocol(e)),
            };
            self.inbuf.drain(..used);
            self.handle(now, packet, &mut out)?;
        }
    }

    fn handle(&mut self, now: u64, packet: Incoming, out: &mut Vec<u8>) -> Result<(), CloseReason> {
        match (self.phase, packet) {
            (
                Phase::AwaitConnAck { .. },
                Incoming::ConnAck {
                    refused: Some(r), ..
                },
            ) => {
                self.refused = Some(r);
                Err(CloseReason::Refused(r))
            }
            (Phase::AwaitConnAck { .. }, Incoming::ConnAck { refused: None, .. }) => {
                self.phase = Phase::Connected;
                self.refused = None;
                self.backoff_ms = BACKOFF_FIRST_MS;
                for message in &mut self.in_flight {
                    let bytes = wire::encode_publish(
                        &message.topic,
                        message.packet_id,
                        &message.payload,
                        message.sent,
                    )
                    .map_err(CloseReason::Protocol)?;
                    out.extend_from_slice(&bytes);
                    message.sent = true;
                }
                self.last_sent = now;
                Ok(())
            }
            (Phase::Connected, Incoming::PubAck { packet_id }) => {
                // an identifier not in flight (a PUBACK repeated after a resend) is ignored
                if let Some(i) = self.in_flight.iter().position(|m| m.packet_id == packet_id) {
                    self.in_flight.remove(i);
                }
                Ok(())
            }
            (Phase::Connected, Incoming::PingResp) => {
                self.ping_sent = None;
                Ok(())
            }
            (Phase::Connected, Incoming::ConnAck { .. }) => {
                Err(CloseReason::OutOfOrder("CONNACK on an open session"))
            }
            (_, Incoming::PubAck { .. }) => Err(CloseReason::OutOfOrder("PUBACK before CONNACK")),
            (_, Incoming::PingResp) => Err(CloseReason::OutOfOrder("PINGRESP before CONNACK")),
            (Phase::Idle { .. }, Incoming::ConnAck { .. }) => {
                Err(CloseReason::OutOfOrder("CONNACK with no connection"))
            }
        }
    }

    /// Time passed: returns the bytes to write (PINGREQ), or why the
    /// connection must close (no CONNACK / no PINGRESP in time)
    pub(crate) fn tick(&mut self, now: u64) -> Result<Vec<u8>, CloseReason> {
        match self.phase {
            Phase::AwaitConnAck { since } if now.saturating_sub(since) >= CONNACK_TIMEOUT_MS => {
                Err(CloseReason::ConnackTimeout)
            }
            Phase::Connected if self.keep_alive_ms > 0 => match self.ping_sent {
                Some(at) if now.saturating_sub(at) >= self.keep_alive_ms => {
                    Err(CloseReason::PingTimeout)
                }
                Some(_) => Ok(Vec::new()),
                None if now.saturating_sub(self.last_sent) >= self.keep_alive_ms => {
                    self.ping_sent = Some(now);
                    self.last_sent = now;
                    Ok(wire::encode_pingreq().to_vec())
                }
                None => Ok(Vec::new()),
            },
            _ => Ok(Vec::new()),
        }
    }

    /// Identifiers run 1..=65535 and wrap to 1, skipping any still in flight
    fn allocate_packet_id(&mut self) -> u16 {
        loop {
            let id = self.next_packet_id;
            self.next_packet_id = if id == u16::MAX { 1 } else { id + 1 };
            if !self.in_flight.iter().any(|m| m.packet_id == id) {
                return id;
            }
        }
    }

    #[cfg(test)]
    fn in_flight_ids(&self) -> Vec<u16> {
        self.in_flight.iter().map(|m| m.packet_id).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONNACK_OK: [u8; 4] = [0x20, 0x02, 0x00, 0x00];

    fn session(keep_alive_secs: u16) -> Session {
        Session::new(&wire::Connect {
            client_id: "c",
            keep_alive_secs,
            clean_session: true,
            user_name: None,
            password: None,
        })
        .unwrap()
    }

    fn connected(keep_alive_secs: u16, now: u64) -> Session {
        let mut s = session(keep_alive_secs);
        s.on_open(now);
        assert_eq!(s.on_bytes(now, &CONNACK_OK), Ok(Vec::new()));
        s
    }

    fn puback(id: u16) -> [u8; 4] {
        let [hi, lo] = id.to_be_bytes();
        [0x40, 0x02, hi, lo]
    }

    #[test]
    fn open_writes_connect_and_connack_connects() {
        let mut s = session(30);
        assert_eq!(s.status(), Status::Connecting);
        assert!(s.should_connect(0));
        let connect = s.on_open(0);
        assert_eq!(connect[0], 0x10);
        assert_eq!(&connect[4..8], b"MQTT");
        assert_eq!(s.status(), Status::Connecting);
        assert!(!s.should_connect(0));
        assert_eq!(s.on_bytes(10, &CONNACK_OK), Ok(Vec::new()));
        assert_eq!(s.status(), Status::Connected);
    }

    #[test]
    fn a_message_queued_before_connack_is_sent_after_it_without_dup() {
        let mut s = session(30);
        assert_eq!(s.publish(0, "t".into(), b"a".to_vec()), Ok(Vec::new()));
        s.on_open(0);
        let out = s.on_bytes(5, &CONNACK_OK).unwrap();
        assert_eq!(out, wire::encode_publish("t", 1, b"a", false).unwrap());
    }

    #[test]
    fn publish_while_connected_writes_at_once_and_puback_clears_it() {
        let mut s = connected(30, 0);
        let out = s.publish(1, "t".into(), b"a".to_vec()).unwrap();
        assert_eq!(out, wire::encode_publish("t", 1, b"a", false).unwrap());
        let out = s.publish(1, "t".into(), b"b".to_vec()).unwrap();
        assert_eq!(out, wire::encode_publish("t", 2, b"b", false).unwrap());
        assert_eq!(s.in_flight_ids(), [1, 2]);
        assert_eq!(s.on_bytes(2, &puback(1)), Ok(Vec::new()));
        assert_eq!(s.in_flight_ids(), [2]);
        // an identifier not in flight is ignored
        assert_eq!(s.on_bytes(2, &puback(9)), Ok(Vec::new()));
        assert_eq!(s.in_flight_ids(), [2]);
    }

    #[test]
    fn unacknowledged_messages_are_resent_with_dup_after_a_reconnect() {
        let mut s = connected(30, 0);
        s.publish(1, "t".into(), b"a".to_vec()).unwrap();
        s.publish(1, "u".into(), b"b".to_vec()).unwrap();
        s.on_bytes(2, &puback(1)).unwrap();
        s.on_close(3);
        assert_eq!(s.status(), Status::Disconnected);
        s.publish(4, "v".into(), b"c".to_vec()).unwrap(); // never sent before
        assert!(!s.should_connect(3 + BACKOFF_FIRST_MS - 1));
        assert!(s.should_connect(3 + BACKOFF_FIRST_MS));
        s.on_open(1_003);
        let out = s.on_bytes(1_004, &CONNACK_OK).unwrap();
        let mut want = wire::encode_publish("u", 2, b"b", true).unwrap();
        want.extend(wire::encode_publish("v", 3, b"c", false).unwrap());
        assert_eq!(out, want);
    }

    #[test]
    fn packet_ids_wrap_from_65535_to_1_and_skip_ids_in_flight() {
        let mut s = connected(30, 0);
        s.next_packet_id = u16::MAX - 1;
        for _ in 0..2 {
            s.publish(0, "t".into(), Vec::new()).unwrap();
        }
        assert_eq!(s.in_flight_ids(), [65_534, 65_535]);
        let out = s.publish(0, "t".into(), Vec::new()).unwrap();
        assert_eq!(
            &out[out.len() - 2..],
            &[0x00, 0x01],
            "65535 wraps to 1, never 0"
        );
        // 2 is still in flight from before: the allocator skips it
        s.next_packet_id = 2;
        s.in_flight.push_back(Message {
            packet_id: 2,
            topic: "t".into(),
            payload: Vec::new(),
            sent: true,
        });
        s.publish(0, "t".into(), Vec::new()).unwrap();
        assert_eq!(s.in_flight_ids(), [65_534, 65_535, 1, 2, 3]);
    }

    #[test]
    fn in_flight_is_bounded() {
        let mut s = connected(30, 0);
        for _ in 0..MAX_IN_FLIGHT {
            assert!(s.can_accept());
            s.publish(0, "t".into(), Vec::new()).unwrap();
        }
        assert!(!s.can_accept());
        s.on_bytes(0, &puback(1)).unwrap();
        assert!(s.can_accept());
    }

    #[test]
    fn keep_alive_sends_pingreq_after_the_interval_without_traffic() {
        let mut s = connected(30, 0);
        assert_eq!(s.tick(29_999), Ok(Vec::new()));
        assert_eq!(s.tick(30_000), Ok(vec![0xc0, 0x00]));
        // waiting for PINGRESP: no second PINGREQ
        assert_eq!(s.tick(45_000), Ok(Vec::new()));
        assert_eq!(s.on_bytes(45_001, &[0xd0, 0x00]), Ok(Vec::new()));
        assert_eq!(s.tick(59_999), Ok(Vec::new()));
        assert_eq!(s.tick(60_000), Ok(vec![0xc0, 0x00]));
    }

    #[test]
    fn traffic_postpones_the_ping() {
        let mut s = connected(30, 0);
        s.publish(20_000, "t".into(), Vec::new()).unwrap();
        assert_eq!(s.tick(30_000), Ok(Vec::new()));
        assert_eq!(s.tick(49_999), Ok(Vec::new()));
        assert_eq!(s.tick(50_000), Ok(vec![0xc0, 0x00]));
    }

    #[test]
    fn a_missing_pingresp_closes_after_one_more_interval() {
        let mut s = connected(10, 0);
        assert_eq!(s.tick(10_000), Ok(vec![0xc0, 0x00]));
        assert_eq!(s.tick(19_999), Ok(Vec::new()));
        assert_eq!(s.tick(20_000), Err(CloseReason::PingTimeout));
    }

    #[test]
    fn keep_alive_zero_never_pings() {
        let mut s = connected(0, 0);
        assert_eq!(s.tick(u64::MAX / 2), Ok(Vec::new()));
    }

    #[test]
    fn connack_must_arrive_within_the_timeout() {
        let mut s = session(30);
        s.on_open(100);
        assert_eq!(s.tick(100 + CONNACK_TIMEOUT_MS - 1), Ok(Vec::new()));
        assert_eq!(
            s.tick(100 + CONNACK_TIMEOUT_MS),
            Err(CloseReason::ConnackTimeout)
        );
    }

    #[test]
    fn reconnect_backoff_doubles_to_its_cap_and_resets_on_connack() {
        let mut s = session(30);
        let mut now = 0;
        let mut waits = Vec::new();
        for _ in 0..8 {
            assert!(s.should_connect(now));
            s.on_open(now);
            s.on_close(now);
            let Phase::Idle { retry_at } = s.phase else {
                panic!("not idle after close")
            };
            waits.push(retry_at - now);
            assert!(!s.should_connect(retry_at - 1));
            now = retry_at;
        }
        assert_eq!(
            waits,
            [1_000, 2_000, 4_000, 8_000, 16_000, 32_000, 32_000, 32_000]
        );
        s.on_open(now);
        s.on_bytes(now, &CONNACK_OK).unwrap();
        s.on_close(now);
        assert_eq!(
            s.phase,
            Phase::Idle {
                retry_at: now + BACKOFF_FIRST_MS
            }
        );
    }

    #[test]
    fn each_refused_code_is_reported_and_kept_until_a_connection_is_accepted() {
        for code in 1u8..=5 {
            let mut s = session(30);
            s.on_open(0);
            let want = ConnectRefused::from_code(code).unwrap();
            assert_eq!(
                s.on_bytes(1, &[0x20, 0x02, 0x00, code]),
                Err(CloseReason::Refused(want))
            );
            assert_eq!(s.status(), Status::Refused(want));
            s.on_close(1);
            assert_eq!(
                s.status(),
                Status::Refused(want),
                "kept through the backoff"
            );
            s.on_open(1_001);
            assert_eq!(
                s.status(),
                Status::Refused(want),
                "kept while the next attempt runs"
            );
            s.on_bytes(1_002, &CONNACK_OK).unwrap();
            assert_eq!(s.status(), Status::Connected);
        }
    }

    #[test]
    fn packets_out_of_order_close_the_connection() {
        let mut s = session(30);
        s.on_open(0);
        assert!(matches!(
            s.on_bytes(0, &puback(1)),
            Err(CloseReason::OutOfOrder(_))
        ));
        let mut s = session(30);
        s.on_open(0);
        assert!(matches!(
            s.on_bytes(0, &[0xd0, 0x00]),
            Err(CloseReason::OutOfOrder(_))
        ));
        let mut s = connected(30, 0);
        assert!(matches!(
            s.on_bytes(0, &CONNACK_OK),
            Err(CloseReason::OutOfOrder(_))
        ));
        let mut s = connected(30, 0);
        assert_eq!(
            s.on_bytes(0, &[0x30, 0x00]),
            Err(CloseReason::Protocol(WireError::UnexpectedPacketType(3)))
        );
    }

    #[test]
    fn bytes_split_across_reads_and_joined_in_one_read_both_decode() {
        let mut s = session(30);
        s.on_open(0);
        assert_eq!(s.on_bytes(0, &CONNACK_OK[..1]), Ok(Vec::new()));
        assert_eq!(s.status(), Status::Connecting);
        assert_eq!(s.on_bytes(0, &CONNACK_OK[1..]), Ok(Vec::new()));
        assert_eq!(s.status(), Status::Connected);
        s.publish(0, "t".into(), Vec::new()).unwrap();
        s.publish(0, "t".into(), Vec::new()).unwrap();
        let mut both = puback(1).to_vec();
        both.extend(puback(2));
        both.extend([0xd0, 0x00]);
        s.ping_sent = Some(0);
        assert_eq!(s.on_bytes(0, &both), Ok(Vec::new()));
        assert!(s.in_flight_ids().is_empty());
        assert_eq!(s.ping_sent, None);
    }

    #[test]
    fn a_close_discards_a_partial_packet() {
        let mut s = session(30);
        s.on_open(0);
        s.on_bytes(0, &CONNACK_OK[..2]).unwrap();
        s.on_close(0);
        s.on_open(BACKOFF_FIRST_MS);
        assert_eq!(s.on_bytes(BACKOFF_FIRST_MS, &CONNACK_OK), Ok(Vec::new()));
        assert_eq!(s.status(), Status::Connected);
    }

    #[test]
    fn invalid_connect_fields_fail_before_any_io() {
        let err = Session::new(&wire::Connect {
            client_id: "c",
            keep_alive_secs: 30,
            clean_session: true,
            user_name: None,
            password: Some(b"pw"),
        })
        .unwrap_err();
        assert_eq!(err, WireError::PasswordWithoutUserName);
        let mut s = connected(30, 0);
        assert_eq!(
            s.publish(0, "a/#".into(), Vec::new()),
            Err(WireError::InvalidTopic)
        );
        assert!(
            s.in_flight_ids().is_empty(),
            "a refused topic takes no packet id"
        );
    }
}
