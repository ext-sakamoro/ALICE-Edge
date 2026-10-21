//! The MQTT 3.1.1 decoder of the publish-only client (`src/mqtt_wire.rs`,
//! included by path: the module is private to the crate) on arbitrary broker
//! bytes. For every input:
//!
//! 1. `decode` never panics and never claims more bytes than it was given;
//!    a decoded packet used at least its 2-byte fixed header.
//! 2. A decoded packet is one of the three a publishing client accepts, with
//!    its rules held: CONNACK refusal codes 1 to 5 and no session-present bit
//!    on a refusal, PUBACK identifier not 0, and the exact byte length
//!    (4 / 4 / 2).
//! 3. Decoding is prefix-stable: every proper prefix of a decoded packet is
//!    "incomplete" (`Ok(None)`) or the same error the decoder reports early
//!    (type, flags and length are checked as soon as they are read), never a
//!    different packet.
//! 4. The remaining-length codec round-trips: a decoded length re-encodes to
//!    the bytes it came from when they were the minimal encoding.
#![no_main]

#[allow(dead_code)]
#[path = "../../src/mqtt_wire.rs"]
mod mqtt_wire;

use libfuzzer_sys::fuzz_target;
use mqtt_wire::{decode, decode_remaining_length, encode_remaining_length, Incoming};

fuzz_target!(|data: &[u8]| {
    match decode(data) {
        Ok(Some((packet, used))) => {
            assert!(used >= 2 && used <= data.len(), "used {used} of {}", data.len());
            match packet {
                Incoming::ConnAck { session_present, refused } => {
                    assert_eq!(used, 4);
                    assert_eq!(data[0], 0x20);
                    if let Some(r) = refused {
                        assert!((1..=5).contains(&r.code()));
                        assert!(!session_present);
                    }
                }
                Incoming::PubAck { packet_id } => {
                    assert_eq!(used, 4);
                    assert_eq!(data[0], 0x40);
                    assert_ne!(packet_id, 0);
                }
                Incoming::PingResp => {
                    assert_eq!(used, 2);
                    assert_eq!(data[0], 0xd0);
                }
            }
            for cut in 0..used {
                if let Ok(Some(_)) = decode(&data[..cut]) {
                    panic!("prefix {cut} of a {used}-byte packet decoded as a packet");
                }
            }
        }
        Ok(None) => {}
        Err(_) => {}
    }
    if let Ok(Some((len, used))) = decode_remaining_length(data) {
        let mut again = Vec::new();
        encode_remaining_length(len, &mut again).expect("a decoded length is encodable");
        assert!(again.len() <= used, "re-encoding is never longer");
        if again.len() == used {
            assert_eq!(&again[..], &data[..used], "the minimal encoding round-trips");
        }
    }
});
