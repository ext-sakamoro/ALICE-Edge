// SPDX-License-Identifier: MIT
//! A scripted MQTT 3.1.1 broker for the client tests, run in the test's own
//! thread. Its packet reader is written from the specification (section
//! 2.2.3 remaining length, 3.1 CONNECT, 3.3 PUBLISH) and shares no code with
//! the client, so the two sides check each other.
#![allow(dead_code)]

use std::io::{self, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

/// Polls in steps of this length when waiting for the client
pub const STEP: Duration = Duration::from_millis(20);

/// A listener on a free loopback port
pub fn listener() -> (TcpListener, u16) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind 127.0.0.1:0");
    listener
        .set_nonblocking(true)
        .expect("non-blocking listener");
    let port = listener.local_addr().expect("local addr").port();
    (listener, port)
}

/// The next connection within `steps` polls, or `None`
pub fn try_accept(listener: &TcpListener, steps: u32) -> Option<TcpStream> {
    for _ in 0..steps {
        match listener.accept() {
            Ok((stream, _)) => {
                stream.set_nonblocking(false).expect("blocking stream");
                stream
                    .set_read_timeout(Some(Duration::from_secs(10)))
                    .expect("read timeout");
                return Some(stream);
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => std::thread::sleep(STEP),
            Err(e) => panic!("accept: {e}"),
        }
    }
    None
}

/// The next connection within 10 s
pub fn accept(listener: &TcpListener) -> TcpStream {
    try_accept(listener, 500).expect("the client connected within 10 s")
}

/// One packet: (fixed-header byte, body)
pub fn read_packet(stream: &mut impl Read) -> (u8, Vec<u8>) {
    let mut byte = [0u8; 1];
    stream.read_exact(&mut byte).expect("packet header");
    let header = byte[0];
    let mut len = 0usize;
    let mut shift = 0;
    loop {
        stream.read_exact(&mut byte).expect("remaining length");
        len |= usize::from(byte[0] & 0x7f) << shift;
        if byte[0] & 0x80 == 0 {
            break;
        }
        shift += 7;
        assert!(shift < 28, "remaining length longer than four bytes");
    }
    let mut body = vec![0u8; len];
    stream.read_exact(&mut body).expect("packet body");
    (header, body)
}

fn field(body: &[u8], at: &mut usize) -> Vec<u8> {
    let len = usize::from(u16::from_be_bytes([body[*at], body[*at + 1]]));
    let value = body[*at + 2..*at + 2 + len].to_vec();
    *at += 2 + len;
    value
}

/// A CONNECT as the broker sees it
#[derive(Debug, PartialEq, Eq)]
pub struct Connect {
    pub flags: u8,
    pub keep_alive: u16,
    pub client_id: String,
    pub user_name: Option<String>,
    pub password: Option<Vec<u8>>,
}

pub fn read_connect(stream: &mut impl Read) -> Connect {
    let (header, body) = read_packet(stream);
    assert_eq!(header, 0x10, "CONNECT header");
    assert_eq!(
        &body[..7],
        &[0x00, 0x04, b'M', b'Q', b'T', b'T', 0x04],
        "protocol name and level 4"
    );
    let flags = body[7];
    let keep_alive = u16::from_be_bytes([body[8], body[9]]);
    let mut at = 10;
    let client_id = String::from_utf8(field(&body, &mut at)).expect("client id is UTF-8");
    let user_name = (flags & 0x80 != 0)
        .then(|| String::from_utf8(field(&body, &mut at)).expect("UTF-8 user name"));
    let password = (flags & 0x40 != 0).then(|| field(&body, &mut at));
    assert_eq!(at, body.len(), "nothing after the CONNECT payload");
    Connect {
        flags,
        keep_alive,
        client_id,
        user_name,
        password,
    }
}

/// A PUBLISH as the broker sees it
#[derive(Debug, PartialEq, Eq)]
pub struct Publish {
    pub header: u8,
    pub topic: String,
    pub packet_id: u16,
    pub payload: Vec<u8>,
}

pub fn read_publish(stream: &mut impl Read) -> Publish {
    let (header, body) = read_packet(stream);
    assert_eq!(header & 0xf0, 0x30, "PUBLISH header, got {header:#04x}");
    assert_eq!(header & 0x06, 0x02, "QoS 1");
    assert_eq!(header & 0x01, 0x00, "retain off");
    let mut at = 0;
    let topic = String::from_utf8(field(&body, &mut at)).expect("UTF-8 topic");
    let packet_id = u16::from_be_bytes([body[at], body[at + 1]]);
    Publish {
        header,
        topic,
        packet_id,
        payload: body[at + 2..].to_vec(),
    }
}

pub fn connack(stream: &mut impl Write, code: u8) {
    stream
        .write_all(&[0x20, 0x02, 0x00, code])
        .expect("write CONNACK");
    stream.flush().expect("flush");
}

pub fn puback(stream: &mut impl Write, packet_id: u16) {
    let [hi, lo] = packet_id.to_be_bytes();
    stream
        .write_all(&[0x40, 0x02, hi, lo])
        .expect("write PUBACK");
    stream.flush().expect("flush");
}

/// Wait up to `steps` polls for `done`; true when it held
pub fn wait_until(steps: u32, mut done: impl FnMut() -> bool) -> bool {
    for _ in 0..steps {
        if done() {
            return true;
        }
        std::thread::sleep(STEP);
    }
    done()
}
