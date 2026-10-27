//! `EDGE_SEMANTICS_ID` is the fold of its pins, every pin is the source
//! it claims to be, and every pin moves the identifier.
//!
//! oracle: the fold is recomputed here from the documented procedure (sort
//! by name, length-prefixed name, 32 digest bytes, SHA-256), the golden
//! entries are read from `tests/determinism_golden.rs` (the source, not a
//! copy), and the upstream entries are compared with the upstream crates'
//! own constants.

use alice_edge::semantics::{EDGE_SEMANTICS_ID, EDGE_SEMANTICS_PINS, PREVIOUS_EDGE_SEMANTICS_IDS};
use sha2::{Digest, Sha256};

fn hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

fn fold(pins: &[(&str, [u8; 32])]) -> [u8; 32] {
    let mut sorted = pins.to_vec();
    sorted.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    assert!(!sorted.is_empty(), "empty table");
    for w in sorted.windows(2) {
        assert_ne!(w[0].0, w[1].0, "repeated name");
    }
    let mut h = Sha256::new();
    for (name, digest) in &sorted {
        assert!(!name.is_empty() && name.is_ascii(), "bad name {name:?}");
        h.update(u32::try_from(name.len()).unwrap().to_be_bytes());
        h.update(name.as_bytes());
        h.update(digest);
    }
    h.finalize().into()
}

fn pin(name: &str) -> [u8; 32] {
    EDGE_SEMANTICS_PINS
        .iter()
        .find(|(n, _)| *n == name)
        .unwrap_or_else(|| panic!("no pin {name}"))
        .1
}

/// The hex in the module documentation, the constant and the fold agree.
#[test]
fn the_identifier_is_the_fold_of_its_pins() {
    assert_eq!(
        hex(&EDGE_SEMANTICS_ID),
        hex(&fold(&EDGE_SEMANTICS_PINS)),
        "EDGE_SEMANTICS_ID is not the fold of its pins"
    );
    let doc = include_str!("../src/semantics.rs");
    assert!(
        doc.contains(&format!("//! {}", hex(&EDGE_SEMANTICS_ID))),
        "the module doc does not show the identifier's hex"
    );
}

/// The golden entries are the digests `tests/determinism_golden.rs` records.
#[test]
fn golden_pins_are_the_recorded_goldens() {
    let src = include_str!("determinism_golden.rs");
    let recorded_after = |key: &str| -> String {
        let at = src.find(key).unwrap_or_else(|| panic!("{key} not found"));
        let quote = at + src[at..].find('"').expect("opening quote") + 1;
        src[quote..quote + 64].to_string()
    };
    let goldens = [
        ("q16_linear", "GOLDEN_Q16_LINEAR"),
        ("adaptive_polyfit", "GOLDEN_ADAPTIVE_POLYFIT"),
        ("constant_fit", "GOLDEN_CONSTANT_FIT"),
        ("simd_fit", "GOLDEN_SIMD_FIT"),
        ("sensor_fusion", "GOLDEN_SENSOR_FUSION"),
        ("ring_buffer", "GOLDEN_RING_BUFFER"),
        ("det_math", "GOLDEN_DET_MATH"),
        ("piecewise", "GOLDEN_PIECEWISE"),
        ("delta", "GOLDEN_DELTA"),
        ("law", "GOLDEN_LAW"),
        ("object_features", "GOLDEN_OBJECT_FEATURES"),
        ("sensors", "GOLDEN_SENSORS"),
        ("depth_capture", "GOLDEN_DEPTH_CAPTURE"),
        ("depth_frames_below_2_13", "GOLDEN_DEPTH_FRAMES_BELOW_2_13"),
        ("depth_frames_from_2_13", "GOLDEN_DEPTH_FRAMES_FROM_2_13"),
    ];
    let mut compared = 0;
    for (name, constant) in goldens {
        let recorded = recorded_after(&format!("const {constant}: &str ="));
        assert_eq!(
            hex(&pin(name)),
            recorded,
            "pin {name} differs from {constant}"
        );
        compared += 1;
    }
    // GOLDEN_ROBUST is one (k, digest) row per k
    let robust = &src[src.find("const GOLDEN_ROBUST:").expect("GOLDEN_ROBUST")..];
    let robust = &robust[..robust.find("];").expect("end of GOLDEN_ROBUST")];
    for (name, k) in [
        ("robust_k1", "1,"),
        ("robust_k3", "3,"),
        ("robust_k10", "10,"),
        ("robust_kmax", "i32::MAX,"),
    ] {
        let at = robust
            .find(&format!("(\n        {k}"))
            .unwrap_or_else(|| panic!("robust row {k} not found"));
        let quote = at + robust[at..].find('"').expect("opening quote") + 1;
        assert_eq!(hex(&pin(name)), &robust[quote..quote + 64], "pin {name}");
        compared += 1;
    }
    assert_eq!(compared, 19);
    let mut names: Vec<_> = EDGE_SEMANTICS_PINS.iter().map(|(n, _)| *n).collect();
    names.sort_unstable();
    let mut expected: Vec<_> = goldens.iter().map(|(n, _)| *n).collect();
    expected.extend([
        "robust_k1",
        "robust_k3",
        "robust_k10",
        "robust_kmax",
        "alice-det-math",
        "alice-zip",
        "alice-db",
        "alice-analytics",
        "alice-sdf",
    ]);
    expected.sort_unstable();
    assert_eq!(names, expected);
}

#[test]
fn det_math_pin_is_the_resolved_arithmetic_identifier() {
    assert_eq!(
        hex(&pin("alice-det-math")),
        hex(&alice_det_math::SEMANTICS_ID)
    );
}

#[cfg(feature = "law")]
#[test]
fn zip_pin_is_the_resolved_zip_identifier() {
    assert_eq!(hex(&pin("alice-zip")), hex(&alice_core::law::SEMANTICS_ID));
}

#[cfg(feature = "db")]
#[test]
fn db_pin_is_the_resolved_db_identifier() {
    assert_eq!(hex(&pin("alice-db")), hex(&alice_db::DB_SEMANTICS_ID));
}

#[cfg(feature = "dashboard")]
#[test]
fn analytics_pin_is_the_resolved_analytics_identifier() {
    assert_eq!(
        hex(&pin("alice-analytics")),
        hex(&alice_analytics::ANALYTICS_SEMANTICS_ID)
    );
}

#[cfg(feature = "sdf")]
#[test]
fn sdf_pin_is_the_resolved_sdf_identifier() {
    assert_eq!(hex(&pin("alice-sdf")), hex(&alice_sdf::SDF_SEMANTICS_ID));
}

/// Changing any single pin (each golden and each upstream identifier) moves
/// the identifier.
#[test]
fn every_pin_moves_the_identifier() {
    for i in 0..EDGE_SEMANTICS_PINS.len() {
        let mut changed = EDGE_SEMANTICS_PINS;
        changed[i].1[31] ^= 1;
        assert_ne!(
            fold(&changed),
            fold(&EDGE_SEMANTICS_PINS),
            "pin {} does not move the identifier",
            changed[i].0
        );
    }
}

/// The pins of `1dd8876b…` (alice-det-math 0.5.1 from crates.io), frozen as
/// they were: the 17 goldens of that build and its 5 upstream identifiers
const PINS_1DD8876B: &[(&str, &str)] = &[
    (
        "q16_linear",
        "105580c224a8b5fe8ac11762db84662a699ad8aed740a52892f10ef92892041d",
    ),
    (
        "adaptive_polyfit",
        "b4a125049e4d8d4ae0278243c32cc972ec46c81cb8f90295695ac8c638d815b5",
    ),
    (
        "constant_fit",
        "6f3c6335869a8c6320ba5972a9007a76180aee43ddaaa6be3df785a91baee705",
    ),
    (
        "simd_fit",
        "6907b2ffc5067ca5c9a5188a709e5f0a217fe7e436542c5c3c5397dfdf034a9a",
    ),
    (
        "sensor_fusion",
        "976837f60532e082cbbb29a9d987b08080f8cdb8172fba329f66cb875ac3eacf",
    ),
    (
        "ring_buffer",
        "224f9f81c7e54478be79641e2b6067e90b803e8acb062014c6a639502377f288",
    ),
    (
        "det_math",
        "4fb30a451a9aae88529c0f66ef1a94f9b3e400cf360cd6e7398a4d9622b318ab",
    ),
    (
        "piecewise",
        "a9706b8473b6d6c50334f9ee9e6726f9eb44409e13118a403c511843db8b49f7",
    ),
    (
        "delta",
        "c25c2ed884ea0d109d267e48a0a0c08f13c32291415002519af3a3ebea452d89",
    ),
    (
        "law",
        "53583c42b85669e3ecb5f701e23329ba759071d821ddb7efa06ac75424b17000",
    ),
    (
        "object_features",
        "1e4d836d63f7395d22b5d5a7b07e97340709e8a6b158e875f7442b44612c261a",
    ),
    (
        "sensors",
        "f40d90cb955cc8dcb3ec9bdc5e47523da138e5e2fde79e675358f8eb1ea2278c",
    ),
    (
        "depth_capture",
        "16d9355a324c9eed84fed8f03b5572579a551f215e3aec071f494a036b208122",
    ),
    (
        "robust_k1",
        "f7baf928bd39a35e1e03ecf93bb7779ebbde6e7c24bb5e28ebf59a1332806e76",
    ),
    (
        "robust_k3",
        "07caa63a49319ad7e8e2399d640779496b71d6cd1add0294cd874a91d469bb83",
    ),
    (
        "robust_k10",
        "75950f93be1f73f55d024000dccf87c9b62b4fc03a93885f11fedb0975aab1ae",
    ),
    (
        "robust_kmax",
        "a8de2b48247fbe7108931a655310ab569dc4f1999aa1ac000ac616004d81b5bd",
    ),
    (
        "alice-det-math",
        "85e0884623b154a3b2b264374907b6cbac09307ca3f9a08339e48c567d3cb315",
    ),
    (
        "alice-zip",
        "51ffeb079769ea34bd3cb3df756b6cb940b8d2d6a6a0f1e1115ab8139415935e",
    ),
    (
        "alice-db",
        "e1a82c6407b1a880267b354647891271929bf2a5765a93dc48944ff64237c4ce",
    ),
    (
        "alice-analytics",
        "f8ded5961a8ef8be2d79ea4bfafd68f494c2443d725fb926cf9775824e043d4d",
    ),
    (
        "alice-sdf",
        "151079595c1e8d8fc661909bac7ea0dde0ba5a8b3ef500019d01c6af64defb69",
    ),
];

/// The depth frame windows as the `1dd8876b…` build computes them (recorded
/// from that build with `golden_depth_frames_below_and_from_2_13`)
const DEPTH_FRAMES_BELOW_2_13_AT_1DD8876B: &str =
    "e74d950a951ef412c5cbf59d61868659920f0504bc7063b9da83f081f23cff1a";
const DEPTH_FRAMES_FROM_2_13_AT_1DD8876B: &str =
    "b9067cdb19ba34efd075a9b3770feb3e53a92a7bb90fd132f122fc548fb013f2";

/// `1dd8876b…` is no longer an identifier of this build, and the reason is
/// exactly this generation's changes: every golden of that build is unchanged,
/// the five upstream identifiers moved, and of the depth frame windows added
/// with it the one below `2^13` has the bits it had under that build while the
/// one from `2^13` on moved. A further change to any pin fails here.
#[test]
fn the_1dd8876b_identifier_is_rejected_because_exactly_these_pins_moved() {
    let unhex = |s: &str| -> [u8; 32] {
        core::array::from_fn(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex"))
    };
    let frozen: Vec<(&str, [u8; 32])> = PINS_1DD8876B.iter().map(|&(n, d)| (n, unhex(d))).collect();
    assert_eq!(
        hex(&fold(&frozen)),
        "1dd8876b42ffcef5dabc506377466e282d290466acd2369d063d4358ab66ef80",
        "the frozen table does not fold to its identifier"
    );
    let mut moved = Vec::new();
    let mut compared = 0usize;
    for (name, digest) in PINS_1DD8876B {
        compared += 1;
        if hex(&pin(name)) != *digest {
            moved.push(*name);
        }
    }
    assert_eq!(compared, 22, "pins compared");
    moved.sort_unstable();
    assert_eq!(
        moved,
        [
            "alice-analytics",
            "alice-db",
            "alice-det-math",
            "alice-sdf",
            "alice-zip"
        ]
    );
    assert_eq!(
        hex(&pin("depth_frames_below_2_13")),
        DEPTH_FRAMES_BELOW_2_13_AT_1DD8876B
    );
    assert_ne!(
        hex(&pin("depth_frames_from_2_13")),
        DEPTH_FRAMES_FROM_2_13_AT_1DD8876B
    );
    assert!(!PREVIOUS_EDGE_SEMANTICS_IDS.contains(&unhex(
        "1dd8876b42ffcef5dabc506377466e282d290466acd2369d063d4358ab66ef80"
    )));
}

#[test]
fn the_previous_list_does_not_contain_the_current_identifier() {
    assert!(!PREVIOUS_EDGE_SEMANTICS_IDS.contains(&EDGE_SEMANTICS_ID));
}
