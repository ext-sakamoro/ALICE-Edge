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
        "det-math cos",
        "det-math exp",
        "det-math ln",
        "det-math max",
        "det-math min",
        "det-math sin",
        "alice-zip",
        "alice-db",
        "alice-analytics",
        "alice-sdf",
    ]);
    expected.sort_unstable();
    assert_eq!(names, expected);
}

/// Each `det-math <function>` entry is the resolved `alice-det-math`'s
/// identifier of that function, the functions are the ones this crate calls
/// (`DET_MATH_FUNCTIONS`, in that order in `DET_MATH_PINS`), and no
/// whole-crate `alice-det-math` identifier is folded.
#[test]
fn det_math_pins_are_the_resolved_function_identifiers() {
    use alice_edge::semantics::{DET_MATH_FUNCTIONS, DET_MATH_PINS};
    assert_eq!(
        DET_MATH_FUNCTIONS,
        ["cos", "exp", "ln", "max", "min", "sin"]
    );
    assert_eq!(DET_MATH_PINS.len(), DET_MATH_FUNCTIONS.len());
    let mut compared = 0;
    for (&function, &(name, digest)) in DET_MATH_FUNCTIONS.iter().zip(DET_MATH_PINS) {
        assert_eq!(name, function);
        assert_eq!(
            hex(&digest),
            hex(&alice_det_math::function_pin(name)),
            "{name}"
        );
        assert_eq!(
            hex(&pin(&format!("det-math {name}"))),
            hex(&digest),
            "{name}"
        );
        compared += 1;
    }
    assert_eq!(compared, 6);
    assert!(EDGE_SEMANTICS_PINS
        .iter()
        .all(|(n, _)| *n != "alice-det-math"));
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
        // an entry this build no longer folds (the whole-crate `alice-det-math`
        // identifier, now one entry per called function) counts as moved
        let current = EDGE_SEMANTICS_PINS.iter().find(|(n, _)| n == name);
        if current.is_none_or(|(_, d)| hex(d) != *digest) {
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

/// The pins of `573e1af1…` (alice-det-math 0.5.2 at 2d01c63f, folded as one
/// identifier), frozen as they were: 19 goldens and 5 upstream identifiers
const PINS_573E1AF1: &[(&str, &str)] = &[
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
        "depth_frames_below_2_13",
        "e74d950a951ef412c5cbf59d61868659920f0504bc7063b9da83f081f23cff1a",
    ),
    (
        "depth_frames_from_2_13",
        "a1146387667a53203d869d83e05ae0c7a1e57f84716f578efa5177aac67df764",
    ),
    (
        "alice-det-math",
        "a2a11b9b826227fa7a9ed8ab2c1ce63dbb07ad4c39aa2c46a363ddea6b47dbe9",
    ),
    (
        "alice-zip",
        "17d0df1d7c0a6e63d1a7380129f12386b677ac96e7ac193dece546805f4fe84c",
    ),
    (
        "alice-db",
        "670fa328eed6ddf2e28f96eb5ebabfe602eabd80dd025199edece388241223a7",
    ),
    (
        "alice-analytics",
        "38e4dba4aeda3ea1a291d620c0183470aaffb9c9d3cd9a101b1435b0d93909df",
    ),
    (
        "alice-sdf",
        "ad04549c331ec33377ef1d4419c0c6756cef6a852520e470a53f642257da4938",
    ),
];

/// `573e1af1…` is a previous identifier of this build, and the reason is
/// exactly the move to alice-det-math 0.6: every golden of that build is
/// unchanged, the whole-crate `alice-det-math` entry is replaced by one entry
/// per called function, and the four upstream identifiers moved. A further
/// change to any pin fails here.
#[test]
fn the_573e1af1_identifier_is_previous_because_exactly_these_pins_moved() {
    let unhex = |s: &str| -> [u8; 32] {
        core::array::from_fn(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex"))
    };
    let frozen: Vec<(&str, [u8; 32])> = PINS_573E1AF1.iter().map(|&(n, d)| (n, unhex(d))).collect();
    assert_eq!(
        hex(&fold(&frozen)),
        "573e1af1a6d4556748faeff04c0a4bebe827cc42cbe95c38ef81fa9e09547e8b",
        "the frozen table does not fold to its identifier"
    );
    let mut moved = Vec::new();
    let mut compared = 0usize;
    for (name, digest) in PINS_573E1AF1 {
        compared += 1;
        let current = EDGE_SEMANTICS_PINS.iter().find(|(n, _)| n == name);
        if current.is_none_or(|(_, d)| hex(d) != *digest) {
            moved.push(*name);
        }
    }
    assert_eq!(compared, 24, "pins compared");
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
    let added: Vec<_> = EDGE_SEMANTICS_PINS
        .iter()
        .map(|(n, _)| *n)
        .filter(|n| !PINS_573E1AF1.iter().any(|(f, _)| f == n))
        .collect();
    assert_eq!(
        added,
        [
            "det-math cos",
            "det-math exp",
            "det-math ln",
            "det-math max",
            "det-math min",
            "det-math sin"
        ]
    );
    let id = unhex("573e1af1a6d4556748faeff04c0a4bebe827cc42cbe95c38ef81fa9e09547e8b");
    assert_eq!(
        alice_edge::semantics::check_semantics(&id),
        Ok(alice_edge::semantics::IdStatus::VerifiedPrevious(id))
    );
}

/// The pins of `d4d143d1…` (alice-det-math 0.6 folded per function, alice-zip
/// `65f48962…`), frozen as they were: 19 goldens, 6 alice-det-math functions and
/// 4 upstream identifiers
const PINS_D4D143D1: &[(&str, &str)] = &[
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
        "depth_frames_below_2_13",
        "e74d950a951ef412c5cbf59d61868659920f0504bc7063b9da83f081f23cff1a",
    ),
    (
        "depth_frames_from_2_13",
        "a1146387667a53203d869d83e05ae0c7a1e57f84716f578efa5177aac67df764",
    ),
    (
        "det-math cos",
        "f7627941400acabe93c8dca4090eee12f8910fe4c14a8735ad456cef3c48ed7b",
    ),
    (
        "det-math exp",
        "606918f2d637f46f9a792a29b3ee31a991ef3196064456caa7b3a7ca81700a63",
    ),
    (
        "det-math ln",
        "6fe18bc0fba8b7e3e871c4cbe555ddb40a460be7b3c7690303ee30accdcb7291",
    ),
    (
        "det-math max",
        "8bd32fcb313f7fb3d7a699fc3075273de7e5a4e083912d8dc7f4cc4016fc2286",
    ),
    (
        "det-math min",
        "2939e83813b26c868ec3113fb1c25102f16f28d8f24c343a727d4b1bec5904fe",
    ),
    (
        "det-math sin",
        "ee1b80d12df18ec0de5bce5a6bf05432484dabfc1cee1bb9b3f3291dd862cfec",
    ),
    (
        "alice-zip",
        "65f48962c757ebb619d252f8784f830aca5f3698fbd22f34150f9887863f38dd",
    ),
    (
        "alice-db",
        "62d72c5042a8e0b894a02a655803850557882a94abb0de1fc4455c22ecfbffa8",
    ),
    (
        "alice-analytics",
        "847535174fa305517aa11fe8b98f604d1b1ec48abaa23d2b1dce9ccc8aa8884f",
    ),
    (
        "alice-sdf",
        "0f6ffc20efe7737c4cfbb95d35e74c62edd16c1c0d562faf35a2abdcc5ccb7e7",
    ),
];

/// `d4d143d1…` is the previous identifier, and the reason is exactly
/// alice-zip's identifier moving (the kind tag of its causal 2-D law became an
/// entry) and with it alice-db's and alice-analytics', which fold it: every
/// golden is unchanged and no pin was added or removed, so its pinned bits are
/// reproduced and `check_semantics` accepts a value stamped
/// with it as `VerifiedPrevious`. A further change to any pin fails here.
#[test]
fn the_d4d143d1_identifier_is_previous_because_only_upstream_pins_moved() {
    let unhex = |s: &str| -> [u8; 32] {
        core::array::from_fn(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex"))
    };
    let frozen: Vec<(&str, [u8; 32])> = PINS_D4D143D1.iter().map(|&(n, d)| (n, unhex(d))).collect();
    assert_eq!(
        hex(&fold(&frozen)),
        "d4d143d1d024b2d75778cb8d3667a333eccebc009ef387586cba02c3c8a1c976",
        "the frozen table does not fold to its identifier"
    );
    let mut moved = Vec::new();
    let mut compared = 0usize;
    for (name, digest) in PINS_D4D143D1 {
        compared += 1;
        let current = EDGE_SEMANTICS_PINS.iter().find(|(n, _)| n == name);
        if current.is_none_or(|(_, d)| hex(d) != *digest) {
            moved.push(*name);
        }
    }
    assert_eq!(compared, 29, "pins compared");
    moved.sort_unstable();
    assert_eq!(
        moved,
        ["alice-analytics", "alice-db", "alice-sdf", "alice-zip"]
    );
    assert_eq!(
        EDGE_SEMANTICS_PINS.len(),
        PINS_D4D143D1.len(),
        "no pin added or removed"
    );
    let d4 = unhex("d4d143d1d024b2d75778cb8d3667a333eccebc009ef387586cba02c3c8a1c976");
    assert!(PREVIOUS_EDGE_SEMANTICS_IDS.contains(&d4));
    assert_eq!(
        alice_edge::semantics::check_semantics(&d4),
        Ok(alice_edge::semantics::IdStatus::VerifiedPrevious(d4))
    );
}

/// The pins of `e0a546aa…` (alice-sdf `0f6ffc20…`), frozen as they were: 19 goldens, 6
/// alice-det-math functions and 4 upstream identifiers
const PINS_E0A546AA: &[(&str, &str)] = &[
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
        "depth_frames_below_2_13",
        "e74d950a951ef412c5cbf59d61868659920f0504bc7063b9da83f081f23cff1a",
    ),
    (
        "depth_frames_from_2_13",
        "a1146387667a53203d869d83e05ae0c7a1e57f84716f578efa5177aac67df764",
    ),
    (
        "det-math cos",
        "f7627941400acabe93c8dca4090eee12f8910fe4c14a8735ad456cef3c48ed7b",
    ),
    (
        "det-math exp",
        "606918f2d637f46f9a792a29b3ee31a991ef3196064456caa7b3a7ca81700a63",
    ),
    (
        "det-math ln",
        "6fe18bc0fba8b7e3e871c4cbe555ddb40a460be7b3c7690303ee30accdcb7291",
    ),
    (
        "det-math max",
        "8bd32fcb313f7fb3d7a699fc3075273de7e5a4e083912d8dc7f4cc4016fc2286",
    ),
    (
        "det-math min",
        "2939e83813b26c868ec3113fb1c25102f16f28d8f24c343a727d4b1bec5904fe",
    ),
    (
        "det-math sin",
        "ee1b80d12df18ec0de5bce5a6bf05432484dabfc1cee1bb9b3f3291dd862cfec",
    ),
    (
        "alice-zip",
        "922b28656da64d8651eae0db86d0070078c4b5dc183c4a4d0219107f73ffca78",
    ),
    (
        "alice-db",
        "97d6853c2d8eda9e80fa985081ca8e982d7a4dc9cd276f9b83c95c153f6000bb",
    ),
    (
        "alice-analytics",
        "67650b7f63ebe71670b7fd9b5781fe276f515dbcee4497169cdd74f1c19a08d4",
    ),
    (
        "alice-sdf",
        "0f6ffc20efe7737c4cfbb95d35e74c62edd16c1c0d562faf35a2abdcc5ccb7e7",
    ),
];

/// `e0a546aa…` is a previous identifier, and the reason is exactly alice-sdf's identifier
/// moving (`0f6ffc20…` → `dca9b092…`): every other pin is unchanged and no pin was added or
/// removed, so its pinned bits are reproduced and `check_semantics` accepts a value stamped with
/// it as `VerifiedPrevious`. A further change to any pin fails here.
#[test]
fn the_e0a546aa_identifier_is_previous_because_only_the_alice_sdf_pin_moved() {
    let unhex = |s: &str| -> [u8; 32] {
        core::array::from_fn(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex"))
    };
    let frozen: Vec<(&str, [u8; 32])> = PINS_E0A546AA.iter().map(|&(n, d)| (n, unhex(d))).collect();
    assert_eq!(
        hex(&fold(&frozen)),
        "e0a546aa1a5ce5195e8fa294fb90dbce52470468495900ce503ce09ece90497f",
        "the frozen table does not fold to its identifier"
    );
    let mut moved = Vec::new();
    let mut compared = 0usize;
    for (name, digest) in PINS_E0A546AA {
        compared += 1;
        let current = EDGE_SEMANTICS_PINS.iter().find(|(n, _)| n == name);
        if current.is_none_or(|(_, d)| hex(d) != *digest) {
            moved.push(*name);
        }
    }
    assert_eq!(compared, 29, "pins compared");
    assert_eq!(moved, ["alice-sdf"]);
    assert_eq!(
        EDGE_SEMANTICS_PINS.len(),
        PINS_E0A546AA.len(),
        "no pin added or removed"
    );
    let id = unhex("e0a546aa1a5ce5195e8fa294fb90dbce52470468495900ce503ce09ece90497f");
    assert_eq!(
        alice_edge::semantics::check_semantics(&id),
        Ok(alice_edge::semantics::IdStatus::VerifiedPrevious(id))
    );
}

/// Every identifier of `PREVIOUS_EDGE_SEMANTICS_IDS` names this build's bits: its frozen pin table
/// has every scenario golden of this build unchanged (only upstream and alice-det-math entries may
/// differ). An identifier without a frozen table, or a list that lost one, fails here
#[test]
fn every_previous_identifier_reproduces_this_builds_goldens() {
    let unhex = |s: &str| -> [u8; 32] {
        core::array::from_fn(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).expect("hex"))
    };
    let tables: [(&str, &[(&str, &str)]); 3] = [
        (
            "e0a546aa1a5ce5195e8fa294fb90dbce52470468495900ce503ce09ece90497f",
            PINS_E0A546AA,
        ),
        (
            "d4d143d1d024b2d75778cb8d3667a333eccebc009ef387586cba02c3c8a1c976",
            PINS_D4D143D1,
        ),
        (
            "573e1af1a6d4556748faeff04c0a4bebe827cc42cbe95c38ef81fa9e09547e8b",
            PINS_573E1AF1,
        ),
    ];
    let mut ids = 0usize;
    let mut goldens = 0usize;
    for previous in PREVIOUS_EDGE_SEMANTICS_IDS {
        let (_, table) = tables
            .iter()
            .find(|(id, _)| unhex(id) == *previous)
            .unwrap_or_else(|| panic!("no frozen pin table for {}", hex(previous)));
        assert_eq!(
            hex(&fold(
                &table
                    .iter()
                    .map(|&(n, d)| (n, unhex(d)))
                    .collect::<Vec<_>>()
            )),
            hex(previous)
        );
        for (name, digest) in *table {
            if name.starts_with("alice-") || name.starts_with("det-math ") {
                continue;
            }
            let current = EDGE_SEMANTICS_PINS
                .iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| panic!("the golden {name} of {} is gone", hex(previous)));
            assert_eq!(
                hex(&current.1),
                *digest,
                "golden {name} under {}",
                hex(previous)
            );
            goldens += 1;
        }
        ids += 1;
    }
    assert_eq!(ids, tables.len(), "previous identifiers compared");
    assert!(goldens >= 3 * 19, "goldens compared: {goldens}");
}

#[test]
fn the_previous_list_does_not_contain_the_current_identifier() {
    assert!(!PREVIOUS_EDGE_SEMANTICS_IDS.contains(&EDGE_SEMANTICS_ID));
}
