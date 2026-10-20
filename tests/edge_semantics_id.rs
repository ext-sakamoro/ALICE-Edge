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
    assert_eq!(compared, 17);
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

#[test]
fn the_previous_list_does_not_contain_the_current_identifier() {
    assert!(!PREVIOUS_EDGE_SEMANTICS_IDS.contains(&EDGE_SEMANTICS_ID));
}
