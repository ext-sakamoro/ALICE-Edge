// SPDX-License-Identifier: LicenseRef-Proprietary
//! MQTT 3.1.1 wire format for the publish-only client in [`crate::mqtt_bridge`]
//!
//! Encodes the packets a publishing client sends (CONNECT, PUBLISH at QoS 1,
//! PINGREQ, DISCONNECT) and decodes the packets a broker sends back to it
//! (CONNACK, PUBACK, PINGRESP). Any other packet type from the broker is an
//! error: this client never subscribes, so a broker has no reason to send one.
//!
//! The file uses no crate paths, so `fuzz/fuzz_targets/fuzz_mqtt_decode.rs`
//! includes it by path and fuzzes the decoder without a public entry point.
//!
//! Reference: MQTT Version 3.1.1, OASIS Standard, 29 October 2014
//! (sections 2.2.3 remaining length, 3.1 CONNECT, 3.2 CONNACK, 3.3 PUBLISH,
//! 3.4 PUBACK, 3.12 PINGREQ, 3.13 PINGRESP, 3.14 DISCONNECT).

use std::fmt;

/// Largest remaining length four bytes can carry (section 2.2.3)
pub(crate) const MAX_REMAINING_LENGTH: usize = 268_435_455;

/// Largest UTF-8 string or binary field (two-byte length prefix)
pub(crate) const MAX_FIELD_LEN: usize = 65_535;

const TYPE_CONNECT: u8 = 1;
const TYPE_CONNACK: u8 = 2;
const TYPE_PUBLISH: u8 = 3;
const TYPE_PUBACK: u8 = 4;
const TYPE_PINGREQ: u8 = 12;
const TYPE_PINGRESP: u8 = 13;
const TYPE_DISCONNECT: u8 = 14;

/// A CONNACK return code other than 0 (connection accepted), section 3.2.2.3
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConnectRefused {
    /// 1: the broker does not support MQTT 3.1.1
    UnacceptableProtocolVersion,
    /// 2: the broker does not accept this client identifier
    IdentifierRejected,
    /// 3: the network connection is up but the MQTT service is unavailable
    ServerUnavailable,
    /// 4: the user name or password is malformed (Azure `IoT` Hub reports an
    /// expired or wrong SAS token this way)
    BadUserNameOrPassword,
    /// 5: the client is not authorised to connect
    NotAuthorized,
}

impl ConnectRefused {
    /// The CONNACK return code (1 to 5)
    #[must_use]
    pub fn code(self) -> u8 {
        match self {
            Self::UnacceptableProtocolVersion => 1,
            Self::IdentifierRejected => 2,
            Self::ServerUnavailable => 3,
            Self::BadUserNameOrPassword => 4,
            Self::NotAuthorized => 5,
        }
    }

    pub(crate) fn from_code(code: u8) -> Option<Self> {
        match code {
            1 => Some(Self::UnacceptableProtocolVersion),
            2 => Some(Self::IdentifierRejected),
            3 => Some(Self::ServerUnavailable),
            4 => Some(Self::BadUserNameOrPassword),
            5 => Some(Self::NotAuthorized),
            _ => None,
        }
    }
}

impl fmt::Display for ConnectRefused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::UnacceptableProtocolVersion => "unacceptable protocol version",
            Self::IdentifierRejected => "identifier rejected",
            Self::ServerUnavailable => "server unavailable",
            Self::BadUserNameOrPassword => "bad user name or password",
            Self::NotAuthorized => "not authorized",
        };
        write!(f, "connection refused ({}): {text}", self.code())
    }
}

/// A packet this client cannot encode or a broker packet it cannot accept
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum WireError {
    /// A length above [`MAX_REMAINING_LENGTH`] cannot be encoded
    RemainingLengthTooLarge(usize),
    /// The fourth length byte still had its continuation bit set
    RemainingLengthTooLong,
    /// A remaining length in more bytes than its value needs (a last byte of 0)
    NonMinimalLength,
    /// A string or binary field longer than [`MAX_FIELD_LEN`]
    FieldTooLong(usize),
    /// A string field containing U+0000
    NulInString,
    /// An empty topic, or one with a `+` / `#` wildcard (not allowed in PUBLISH)
    InvalidTopic,
    /// A password needs a user name in MQTT 3.1.1 (section 3.1.2.9)
    PasswordWithoutUserName,
    /// Packet identifier 0 is reserved (section 2.3.1)
    ZeroPacketId,
    /// The broker sent a packet type this client never expects
    UnexpectedPacketType(u8),
    /// The fixed-header flags of a broker packet are not the reserved value
    BadFlags(u8),
    /// The remaining length of a broker packet is not the one its type has
    BadLength { packet_type: u8, len: usize },
    /// CONNACK acknowledge flags with a reserved bit set
    BadConnackFlags(u8),
    /// CONNACK return code 6 to 255 (reserved)
    ReservedConnackCode(u8),
}

impl fmt::Display for WireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::RemainingLengthTooLarge(n) => {
                write!(f, "packet of {n} bytes exceeds the MQTT maximum")
            }
            Self::RemainingLengthTooLong => write!(f, "remaining length longer than four bytes"),
            Self::NonMinimalLength => write!(f, "remaining length not in its shortest form"),
            Self::FieldTooLong(n) => write!(f, "field of {n} bytes exceeds 65535"),
            Self::NulInString => write!(f, "string contains U+0000"),
            Self::InvalidTopic => write!(f, "topic is empty or contains a wildcard"),
            Self::PasswordWithoutUserName => write!(f, "password without a user name"),
            Self::ZeroPacketId => write!(f, "packet identifier 0"),
            Self::UnexpectedPacketType(t) => {
                write!(f, "unexpected packet type {t} from the broker")
            }
            Self::BadFlags(b) => write!(f, "reserved flags not respected in header byte {b:#04x}"),
            Self::BadLength { packet_type, len } => {
                write!(f, "packet type {packet_type} with remaining length {len}")
            }
            Self::BadConnackFlags(b) => write!(f, "CONNACK acknowledge flags {b:#04x}"),
            Self::ReservedConnackCode(c) => write!(f, "reserved CONNACK return code {c}"),
        }
    }
}

/// CONNECT parameters (no will message; this client never sets one)
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Connect<'a> {
    pub client_id: &'a str,
    pub keep_alive_secs: u16,
    pub clean_session: bool,
    pub user_name: Option<&'a str>,
    pub password: Option<&'a [u8]>,
}

/// A packet from the broker
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Incoming {
    ConnAck {
        session_present: bool,
        refused: Option<ConnectRefused>,
    },
    PubAck {
        packet_id: u16,
    },
    PingResp,
}

/// Append the remaining-length bytes of `len` (section 2.2.3)
pub(crate) fn encode_remaining_length(len: usize, out: &mut Vec<u8>) -> Result<(), WireError> {
    if len > MAX_REMAINING_LENGTH {
        return Err(WireError::RemainingLengthTooLarge(len));
    }
    let mut rest = len;
    loop {
        let mut byte = (rest % 128) as u8;
        rest /= 128;
        if rest > 0 {
            byte |= 0x80;
        }
        out.push(byte);
        if rest == 0 {
            return Ok(());
        }
    }
}

/// `Ok(Some((len, bytes used)))`, or `Ok(None)` while the length is incomplete
///
/// Only the shortest encoding of a length is accepted, so every length has
/// one byte form (MQTT 5.0 section 1.5.5 states it; 3.1.1 defines no other).
pub(crate) fn decode_remaining_length(buf: &[u8]) -> Result<Option<(usize, usize)>, WireError> {
    let mut value = 0usize;
    let mut multiplier = 1usize;
    for (i, &byte) in buf.iter().take(4).enumerate() {
        value += usize::from(byte & 0x7f) * multiplier;
        if byte & 0x80 == 0 {
            if i > 0 && byte == 0 {
                return Err(WireError::NonMinimalLength);
            }
            return Ok(Some((value, i + 1)));
        }
        multiplier *= 128;
    }
    if buf.len() >= 4 {
        Err(WireError::RemainingLengthTooLong)
    } else {
        Ok(None)
    }
}

fn check_string(s: &str) -> Result<(), WireError> {
    if s.len() > MAX_FIELD_LEN {
        return Err(WireError::FieldTooLong(s.len()));
    }
    if s.contains('\0') {
        return Err(WireError::NulInString);
    }
    Ok(())
}

fn put_field(out: &mut Vec<u8>, bytes: &[u8]) -> Result<(), WireError> {
    let len = u16::try_from(bytes.len()).map_err(|_| WireError::FieldTooLong(bytes.len()))?;
    out.extend_from_slice(&len.to_be_bytes());
    out.extend_from_slice(bytes);
    Ok(())
}

fn finish(header: u8, body: &[u8]) -> Result<Vec<u8>, WireError> {
    let mut out = Vec::with_capacity(body.len() + 5);
    out.push(header);
    encode_remaining_length(body.len(), &mut out)?;
    out.extend_from_slice(body);
    Ok(out)
}

/// A topic a client may publish to (section 4.7): not empty, no wildcard
pub(crate) fn check_topic(topic: &str) -> Result<(), WireError> {
    check_string(topic)?;
    if topic.is_empty() || topic.contains('+') || topic.contains('#') {
        return Err(WireError::InvalidTopic);
    }
    Ok(())
}

/// A PUBLISH of this topic and payload size can be encoded
pub(crate) fn check_publish(topic: &str, payload_len: usize) -> Result<(), WireError> {
    check_topic(topic)?;
    let len = 2 + topic.len() + 2 + payload_len;
    if len > MAX_REMAINING_LENGTH {
        return Err(WireError::RemainingLengthTooLarge(len));
    }
    Ok(())
}

/// CONNECT (section 3.1)
pub(crate) fn encode_connect(c: &Connect<'_>) -> Result<Vec<u8>, WireError> {
    check_string(c.client_id)?;
    if c.password.is_some() && c.user_name.is_none() {
        return Err(WireError::PasswordWithoutUserName);
    }
    let mut flags = 0u8;
    if c.clean_session {
        flags |= 0x02;
    }
    if let Some(user) = c.user_name {
        check_string(user)?;
        flags |= 0x80;
    }
    if c.password.is_some() {
        flags |= 0x40;
    }
    let mut body = Vec::new();
    put_field(&mut body, b"MQTT")?;
    body.push(4); // protocol level 4 = MQTT 3.1.1
    body.push(flags);
    body.extend_from_slice(&c.keep_alive_secs.to_be_bytes());
    put_field(&mut body, c.client_id.as_bytes())?;
    if let Some(user) = c.user_name {
        put_field(&mut body, user.as_bytes())?;
    }
    if let Some(password) = c.password {
        put_field(&mut body, password)?;
    }
    finish(TYPE_CONNECT << 4, &body)
}

/// PUBLISH at QoS 1, retain off (section 3.3)
pub(crate) fn encode_publish(
    topic: &str,
    packet_id: u16,
    payload: &[u8],
    dup: bool,
) -> Result<Vec<u8>, WireError> {
    check_topic(topic)?;
    if packet_id == 0 {
        return Err(WireError::ZeroPacketId);
    }
    let mut header = (TYPE_PUBLISH << 4) | 0x02;
    if dup {
        header |= 0x08;
    }
    let mut body = Vec::with_capacity(topic.len() + payload.len() + 4);
    put_field(&mut body, topic.as_bytes())?;
    body.extend_from_slice(&packet_id.to_be_bytes());
    body.extend_from_slice(payload);
    finish(header, &body)
}

/// PINGREQ (section 3.12)
pub(crate) fn encode_pingreq() -> [u8; 2] {
    [TYPE_PINGREQ << 4, 0]
}

/// DISCONNECT (section 3.14)
pub(crate) fn encode_disconnect() -> [u8; 2] {
    [TYPE_DISCONNECT << 4, 0]
}

/// Decode one broker packet from the front of `buf`
///
/// `Ok(Some((packet, bytes used)))`, `Ok(None)` while the packet is
/// incomplete, or an error. The type, flags and length are checked as soon as
/// they are read, so a wrong packet fails without waiting for its body.
pub(crate) fn decode(buf: &[u8]) -> Result<Option<(Incoming, usize)>, WireError> {
    let Some(&header) = buf.first() else {
        return Ok(None);
    };
    let packet_type = header >> 4;
    let expected_len = match packet_type {
        TYPE_CONNACK | TYPE_PUBACK => 2,
        TYPE_PINGRESP => 0,
        other => return Err(WireError::UnexpectedPacketType(other)),
    };
    if header & 0x0f != 0 {
        return Err(WireError::BadFlags(header));
    }
    let Some((len, len_bytes)) = decode_remaining_length(&buf[1..])? else {
        return Ok(None);
    };
    if len != expected_len {
        return Err(WireError::BadLength { packet_type, len });
    }
    let start = 1 + len_bytes;
    let end = start + len;
    if buf.len() < end {
        return Ok(None);
    }
    let body = &buf[start..end];
    let packet = match packet_type {
        TYPE_CONNACK => {
            let flags = body[0];
            if flags & 0xfe != 0 {
                return Err(WireError::BadConnackFlags(flags));
            }
            let code = body[1];
            let refused = if code == 0 {
                None
            } else {
                Some(ConnectRefused::from_code(code).ok_or(WireError::ReservedConnackCode(code))?)
            };
            // a refusal carries session present 0 (section 3.2.2.2)
            if refused.is_some() && flags != 0 {
                return Err(WireError::BadConnackFlags(flags));
            }
            Incoming::ConnAck {
                session_present: flags == 1,
                refused,
            }
        }
        TYPE_PUBACK => {
            let packet_id = u16::from_be_bytes([body[0], body[1]]);
            if packet_id == 0 {
                return Err(WireError::ZeroPacketId);
            }
            Incoming::PubAck { packet_id }
        }
        _ => Incoming::PingResp,
    };
    Ok(Some((packet, end)))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rl(len: usize) -> Vec<u8> {
        let mut out = Vec::new();
        encode_remaining_length(len, &mut out).unwrap();
        out
    }

    #[test]
    fn remaining_length_bytes_at_every_boundary() {
        // section 2.2.3 table: 1 byte up to 127, 2 up to 16 383, 3 up to 2 097 151, 4 up to 268 435 455
        let table: &[(usize, &[u8])] = &[
            (0, &[0x00]),
            (1, &[0x01]),
            (127, &[0x7f]),
            (128, &[0x80, 0x01]),
            (321, &[0xc1, 0x02]),
            (16_383, &[0xff, 0x7f]),
            (16_384, &[0x80, 0x80, 0x01]),
            (2_097_151, &[0xff, 0xff, 0x7f]),
            (2_097_152, &[0x80, 0x80, 0x80, 0x01]),
            (268_435_455, &[0xff, 0xff, 0xff, 0x7f]),
        ];
        for &(len, bytes) in table {
            assert_eq!(rl(len), bytes, "encode {len}");
            assert_eq!(
                decode_remaining_length(bytes),
                Ok(Some((len, bytes.len()))),
                "decode {len}"
            );
        }
    }

    #[test]
    fn remaining_length_out_of_range_is_refused() {
        let mut out = Vec::new();
        assert_eq!(
            encode_remaining_length(268_435_456, &mut out),
            Err(WireError::RemainingLengthTooLarge(268_435_456))
        );
        assert!(out.is_empty());
        assert_eq!(
            decode_remaining_length(&[0x80, 0x80, 0x80, 0x80, 0x01]),
            Err(WireError::RemainingLengthTooLong)
        );
        assert_eq!(
            decode_remaining_length(&[0xff, 0xff, 0xff, 0xff]),
            Err(WireError::RemainingLengthTooLong)
        );
    }

    #[test]
    fn a_remaining_length_longer_than_needed_is_refused() {
        // one encoding per length: a last byte of 0 after a continuation byte adds nothing
        for bytes in [
            &[0x80, 0x00][..],
            &[0xff, 0x00],
            &[0x80, 0x80, 0x00],
            &[0x80, 0x80, 0x80, 0x00],
        ] {
            assert_eq!(
                decode_remaining_length(bytes),
                Err(WireError::NonMinimalLength),
                "{bytes:02x?}"
            );
        }
        assert_eq!(
            decode(&[0xd0, 0x80, 0x00]),
            Err(WireError::NonMinimalLength)
        );
        assert_eq!(
            decode(&[0x40, 0x82, 0x00, 0x00, 0x01]),
            Err(WireError::NonMinimalLength)
        );
    }

    #[test]
    fn remaining_length_waits_for_its_last_byte() {
        assert_eq!(decode_remaining_length(&[]), Ok(None));
        assert_eq!(decode_remaining_length(&[0x80]), Ok(None));
        assert_eq!(decode_remaining_length(&[0x80, 0x80]), Ok(None));
        assert_eq!(decode_remaining_length(&[0x80, 0x80, 0x80]), Ok(None));
    }

    fn connect<'a>(
        client_id: &'a str,
        user: Option<&'a str>,
        password: Option<&'a [u8]>,
    ) -> Connect<'a> {
        Connect {
            client_id,
            keep_alive_secs: 30,
            clean_session: true,
            user_name: user,
            password,
        }
    }

    #[test]
    fn connect_bytes_without_credentials() {
        let got = encode_connect(&connect("a", None, None)).unwrap();
        let want = [
            0x10, 13, // CONNECT, remaining length 13
            0x00, 0x04, b'M', b'Q', b'T', b'T', // protocol name
            0x04, // level 4 = 3.1.1
            0x02, // clean session
            0x00, 30, // keep alive 30 s
            0x00, 0x01, b'a', // client identifier
        ];
        assert_eq!(got, want);
    }

    #[test]
    fn connect_bytes_with_user_name_and_password() {
        let got = encode_connect(&connect("a", Some("u"), Some(b"pw"))).unwrap();
        let want = [
            0x10, 20, 0x00, 0x04, b'M', b'Q', b'T', b'T', 0x04, //
            0xc2, // user name + password + clean session
            0x00, 30, 0x00, 0x01, b'a', //
            0x00, 0x01, b'u', //
            0x00, 0x02, b'p', b'w',
        ];
        assert_eq!(got, want);
        let user_only = encode_connect(&connect("a", Some("u"), None)).unwrap();
        assert_eq!(user_only[9], 0x82);
        let not_clean = encode_connect(&Connect {
            clean_session: false,
            ..connect("a", None, None)
        })
        .unwrap();
        assert_eq!(not_clean[9], 0x00);
        let keep = encode_connect(&Connect {
            keep_alive_secs: 0x1234,
            ..connect("a", None, None)
        })
        .unwrap();
        assert_eq!(&keep[10..12], &[0x12, 0x34]);
    }

    #[test]
    fn connect_with_an_azure_sas_user_name_crosses_the_one_byte_length() {
        // an Azure IoT Hub user name is "<hub>.azure-devices.net/<device>/api-version=..."
        let user = "hub.azure-devices.net/device-1/api-version=2021-04-12";
        let sas = vec![b's'; 60];
        let got = encode_connect(&connect("device-1", Some(user), Some(&sas))).unwrap();
        let body_len = 10 + (2 + 8) + (2 + user.len()) + (2 + sas.len());
        assert_eq!(body_len, 137); // 10 + 10 + 55 + 62
        assert_eq!(&got[..3], &[0x10, 0x89, 0x01]); // 137 = 9 + 1 * 128
        assert_eq!(got.len(), 3 + body_len);
        assert_eq!(&got[got.len() - 62..got.len() - 60], &[0x00, 60]);
    }

    #[test]
    fn connect_refuses_what_3_1_1_cannot_carry() {
        assert_eq!(
            encode_connect(&connect("a", None, Some(b"pw"))),
            Err(WireError::PasswordWithoutUserName)
        );
        assert_eq!(
            encode_connect(&connect("a\0", None, None)),
            Err(WireError::NulInString)
        );
        let long = "x".repeat(65_536);
        assert_eq!(
            encode_connect(&connect(&long, None, None)),
            Err(WireError::FieldTooLong(65_536))
        );
        let pw = vec![0u8; 65_536];
        assert_eq!(
            encode_connect(&connect("a", Some("u"), Some(&pw))),
            Err(WireError::FieldTooLong(65_536))
        );
        assert!(encode_connect(&connect(&"x".repeat(65_535), None, None)).is_ok());
    }

    #[test]
    fn publish_qos1_bytes() {
        let got = encode_publish("a/b", 1, b"x", false).unwrap();
        assert_eq!(
            got,
            [0x32, 8, 0x00, 0x03, b'a', b'/', b'b', 0x00, 0x01, b'x']
        );
        let dup = encode_publish("a/b", 0xbeef, b"x", true).unwrap();
        assert_eq!(
            dup,
            [0x3a, 8, 0x00, 0x03, b'a', b'/', b'b', 0xbe, 0xef, b'x']
        );
    }

    #[test]
    fn publish_length_at_the_127_128_16383_16384_boundaries() {
        // topic "t" (3 bytes) + packet id (2) + payload
        for (payload_len, header) in [
            (122usize, &[0x32u8, 0x7f][..]),
            (123, &[0x32, 0x80, 0x01]),
            (16_378, &[0x32, 0xff, 0x7f]),
            (16_379, &[0x32, 0x80, 0x80, 0x01]),
        ] {
            let payload = vec![0xa5; payload_len];
            let got = encode_publish("t", 7, &payload, false).unwrap();
            assert_eq!(&got[..header.len()], header, "payload {payload_len}");
            assert_eq!(
                &got[header.len()..header.len() + 5],
                &[0x00, 0x01, b't', 0x00, 0x07]
            );
            assert_eq!(got.len(), header.len() + 5 + payload_len);
        }
    }

    #[test]
    fn publish_refuses_bad_topics_and_packet_id_zero() {
        assert_eq!(
            encode_publish("", 1, b"", false),
            Err(WireError::InvalidTopic)
        );
        assert_eq!(
            encode_publish("a/+", 1, b"", false),
            Err(WireError::InvalidTopic)
        );
        assert_eq!(
            encode_publish("a/#", 1, b"", false),
            Err(WireError::InvalidTopic)
        );
        assert_eq!(
            encode_publish("a\0", 1, b"", false),
            Err(WireError::NulInString)
        );
        assert_eq!(
            encode_publish("a", 0, b"", false),
            Err(WireError::ZeroPacketId)
        );
        let long = "t".repeat(65_536);
        assert_eq!(
            encode_publish(&long, 1, b"", false),
            Err(WireError::FieldTooLong(65_536))
        );
    }

    #[test]
    fn check_publish_bounds_the_packet_size() {
        let max_payload = MAX_REMAINING_LENGTH - (2 + 1 + 2);
        assert_eq!(check_publish("t", max_payload), Ok(()));
        assert_eq!(
            check_publish("t", max_payload + 1),
            Err(WireError::RemainingLengthTooLarge(MAX_REMAINING_LENGTH + 1))
        );
        assert_eq!(check_publish("a/+", 0), Err(WireError::InvalidTopic));
    }

    #[test]
    fn ping_and_disconnect_bytes() {
        assert_eq!(encode_pingreq(), [0xc0, 0x00]);
        assert_eq!(encode_disconnect(), [0xe0, 0x00]);
    }

    #[test]
    fn connack_accepted_and_session_present() {
        assert_eq!(
            decode(&[0x20, 0x02, 0x00, 0x00]),
            Ok(Some((
                Incoming::ConnAck {
                    session_present: false,
                    refused: None
                },
                4
            )))
        );
        assert_eq!(
            decode(&[0x20, 0x02, 0x01, 0x00]),
            Ok(Some((
                Incoming::ConnAck {
                    session_present: true,
                    refused: None
                },
                4
            )))
        );
    }

    #[test]
    fn every_refused_connack_code_is_a_distinct_refusal() {
        let mut seen = Vec::new();
        for code in 1u8..=5 {
            let Ok(Some((
                Incoming::ConnAck {
                    refused: Some(r), ..
                },
                4,
            ))) = decode(&[0x20, 0x02, 0x00, code])
            else {
                panic!("code {code} not decoded as a refusal");
            };
            assert_eq!(r.code(), code);
            assert!(!seen.contains(&r), "code {code} maps to an earlier refusal");
            assert!(r.to_string().contains(&format!("({code})")));
            seen.push(r);
        }
        assert_eq!(
            seen,
            [
                ConnectRefused::UnacceptableProtocolVersion,
                ConnectRefused::IdentifierRejected,
                ConnectRefused::ServerUnavailable,
                ConnectRefused::BadUserNameOrPassword,
                ConnectRefused::NotAuthorized,
            ]
        );
        for code in [6u8, 0x80, 0xff] {
            assert_eq!(
                decode(&[0x20, 0x02, 0x00, code]),
                Err(WireError::ReservedConnackCode(code))
            );
        }
    }

    #[test]
    fn connack_with_reserved_bits_is_refused() {
        assert_eq!(
            decode(&[0x21, 0x02, 0x00, 0x00]),
            Err(WireError::BadFlags(0x21))
        );
        assert_eq!(
            decode(&[0x20, 0x02, 0x02, 0x00]),
            Err(WireError::BadConnackFlags(0x02))
        );
        assert_eq!(
            decode(&[0x20, 0x02, 0x80, 0x00]),
            Err(WireError::BadConnackFlags(0x80))
        );
        assert_eq!(
            decode(&[0x20, 0x02, 0x01, 0x05]),
            Err(WireError::BadConnackFlags(0x01))
        );
        assert_eq!(
            decode(&[0x20, 0x03, 0x00, 0x00, 0x00]),
            Err(WireError::BadLength {
                packet_type: 2,
                len: 3
            })
        );
    }

    #[test]
    fn puback_and_pingresp() {
        assert_eq!(
            decode(&[0x40, 0x02, 0x12, 0x34]),
            Ok(Some((Incoming::PubAck { packet_id: 0x1234 }, 4)))
        );
        assert_eq!(
            decode(&[0x40, 0x02, 0xff, 0xff]),
            Ok(Some((Incoming::PubAck { packet_id: 0xffff }, 4)))
        );
        assert_eq!(
            decode(&[0x40, 0x02, 0x00, 0x00]),
            Err(WireError::ZeroPacketId)
        );
        assert_eq!(
            decode(&[0x42, 0x02, 0x00, 0x01]),
            Err(WireError::BadFlags(0x42))
        );
        assert_eq!(
            decode(&[0x40, 0x01, 0x00]),
            Err(WireError::BadLength {
                packet_type: 4,
                len: 1
            })
        );
        assert_eq!(decode(&[0xd0, 0x00]), Ok(Some((Incoming::PingResp, 2))));
        assert_eq!(
            decode(&[0xd0, 0x01, 0x00]),
            Err(WireError::BadLength {
                packet_type: 13,
                len: 1
            })
        );
    }

    #[test]
    fn packets_a_publishing_client_never_receives_are_refused() {
        for packet_type in [0u8, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12, 14, 15] {
            assert_eq!(
                decode(&[packet_type << 4, 0x00]),
                Err(WireError::UnexpectedPacketType(packet_type)),
                "type {packet_type}"
            );
        }
    }

    #[test]
    fn incomplete_packets_wait_and_a_second_packet_is_left_in_place() {
        let whole = [0x20, 0x02, 0x00, 0x00];
        for cut in 0..whole.len() {
            assert_eq!(decode(&whole[..cut]), Ok(None), "prefix {cut}");
        }
        let two = [0x40, 0x02, 0x00, 0x09, 0xd0, 0x00];
        assert_eq!(
            decode(&two),
            Ok(Some((Incoming::PubAck { packet_id: 9 }, 4)))
        );
        assert_eq!(decode(&two[4..]), Ok(Some((Incoming::PingResp, 2))));
        // the length is checked before the body arrives
        assert_eq!(
            decode(&[0x40, 0x05]),
            Err(WireError::BadLength {
                packet_type: 4,
                len: 5
            })
        );
    }

    #[test]
    fn each_error_names_its_cause() {
        let cases = [
            (
                WireError::RemainingLengthTooLarge(268_435_456),
                "packet of 268435456 bytes exceeds the MQTT maximum",
            ),
            (
                WireError::RemainingLengthTooLong,
                "remaining length longer than four bytes",
            ),
            (
                WireError::NonMinimalLength,
                "remaining length not in its shortest form",
            ),
            (
                WireError::FieldTooLong(65_536),
                "field of 65536 bytes exceeds 65535",
            ),
            (WireError::NulInString, "string contains U+0000"),
            (
                WireError::InvalidTopic,
                "topic is empty or contains a wildcard",
            ),
            (
                WireError::PasswordWithoutUserName,
                "password without a user name",
            ),
            (WireError::ZeroPacketId, "packet identifier 0"),
            (
                WireError::UnexpectedPacketType(3),
                "unexpected packet type 3 from the broker",
            ),
            (
                WireError::BadFlags(0x41),
                "reserved flags not respected in header byte 0x41",
            ),
            (
                WireError::BadLength {
                    packet_type: 2,
                    len: 5,
                },
                "packet type 2 with remaining length 5",
            ),
            (
                WireError::BadConnackFlags(0x02),
                "CONNACK acknowledge flags 0x02",
            ),
            (
                WireError::ReservedConnackCode(6),
                "reserved CONNACK return code 6",
            ),
        ];
        for (error, text) in cases {
            assert_eq!(error.to_string(), text);
        }
    }
}
