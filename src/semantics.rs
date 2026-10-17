//! Content hash of this crate's Law-bearing semantics: [`EDGE_SEMANTICS_ID`].
//!
//! [`crate::SEMANTICS_ID`] is the arithmetic identifier of `alice-det-math`,
//! re-exported; it names the transcendental functions, not this crate's own
//! rules (the Q16.16 normal equations, the MAD rejection constants, the
//! segment and delta encodings). This module folds the crate's own Law pins,
//! the cross-platform goldens of `tests/determinism_golden.rs`, with the
//! identifiers of the Law-bearing crates it depends on, into one 32-byte
//! value. Two builds with the same identifier produce the same bits on every
//! pinned scenario; a build that changes any of them, or that resolves an
//! upstream crate with a different identifier, has a different value.
//!
//! # Hex form
//!
//! ```text
//! 1dd8876b42ffcef5dabc506377466e282d290466acd2369d063d4358ab66ef80
//! ```
//!
//! # Entries
//!
//! [`EDGE_SEMANTICS_PINS`]: one entry per scenario golden (`"q16_linear"`,
//! `"adaptive_polyfit"`, `"constant_fit"`, `"simd_fit"`, `"sensor_fusion"`,
//! `"ring_buffer"`, `"det_math"`, `"piecewise"`, `"delta"`, `"law"`,
//! `"object_features"`, `"sensors"`, `"depth_capture"`, some run only with
//! the feature that builds the module) and one per `k` of the robust fit
//! (`"robust_k1"`, `"robust_k3"`, `"robust_k10"`, `"robust_kmax"`), plus the
//! identifiers of the Law-bearing dependencies, each the dependency's own Law
//! identifier: `"alice-det-math"` (its `SEMANTICS_ID`), `"alice-zip"` (its
//! `law::SEMANTICS_ID`, the `law` and `zip` features), `"alice-db"` (its
//! `DB_SEMANTICS_ID`, `db`), `"alice-analytics"` (its `ANALYTICS_SEMANTICS_ID`,
//! `dashboard`) and `"alice-sdf"` (its `SDF_SEMANTICS_ID`, `sdf`). The
//! identifier goldens (`GOLDEN_SEMANTICS_ID`, `GOLDEN_LAW_ID_*`) are not pins:
//! they are functions of the entries.
//!
//! Each upstream identifier folds the arithmetic identifier it was built with
//! (alice-zip's from 0.11, alice-sdf's `SDF_SEMANTICS_BY_DET_MATH` rows, and the
//! two built on alice-zip's), so the arithmetic is covered twice: by the
//! `"alice-det-math"` entry and inside each upstream entry.
//!
//! # Fold
//!
//! The same procedure as the `SEMANTICS_ID` fold of `alice-det-math`: sort
//! the entries by name (ascending byte order), reject an empty table, a
//! repeated name, an empty name or a name that is not ASCII, then SHA-256
//! over, for each entry in that order, the name length as 4 bytes big-endian
//! (`u32`), the name bytes and the 32 digest bytes.
//!
//! # Independent of cargo features
//!
//! The table and the identifier are constants that no `cfg` gates; upstream
//! entries are recorded as hex. `tests/edge_semantics_id.rs` checks them
//! against the resolved constants with the feature that pulls each crate in,
//! so resolving another upstream identifier is a failing test.
//!
//! # Version and toolchain
//!
//! The identifier is a constant of this crate version: a dependency bump or
//! a toolchain change that moved any pinned bits fails a golden, and moving
//! the golden moves the identifier. Earlier identifiers whose pinned bits
//! are still reproduced are listed in [`PREVIOUS_EDGE_SEMANTICS_IDS`].

/// Decodes 64 hex digits into 32 bytes at compile time; anything else is a
/// compile error.
pub(crate) const fn hex32(s: &str) -> [u8; 32] {
    const fn nibble(c: u8) -> u8 {
        match c {
            b'0'..=b'9' => c - b'0',
            b'a'..=b'f' => c - b'a' + 10,
            _ => panic!("not a lower-case hex digit"),
        }
    }
    let b = s.as_bytes();
    assert!(b.len() == 64, "a digest is 64 hex digits");
    let mut out = [0u8; 32];
    let mut i = 0;
    while i < 32 {
        out[i] = nibble(b[2 * i]) * 16 + nibble(b[2 * i + 1]);
        i += 1;
    }
    out
}

/// The pins folded into [`EDGE_SEMANTICS_ID`], as `(name, digest)`.
pub const EDGE_SEMANTICS_PINS: [(&str, [u8; 32]); 22] = [
    (
        "q16_linear",
        hex32("105580c224a8b5fe8ac11762db84662a699ad8aed740a52892f10ef92892041d"),
    ),
    (
        "adaptive_polyfit",
        hex32("b4a125049e4d8d4ae0278243c32cc972ec46c81cb8f90295695ac8c638d815b5"),
    ),
    (
        "constant_fit",
        hex32("6f3c6335869a8c6320ba5972a9007a76180aee43ddaaa6be3df785a91baee705"),
    ),
    (
        "simd_fit",
        hex32("6907b2ffc5067ca5c9a5188a709e5f0a217fe7e436542c5c3c5397dfdf034a9a"),
    ),
    (
        "sensor_fusion",
        hex32("976837f60532e082cbbb29a9d987b08080f8cdb8172fba329f66cb875ac3eacf"),
    ),
    (
        "ring_buffer",
        hex32("224f9f81c7e54478be79641e2b6067e90b803e8acb062014c6a639502377f288"),
    ),
    (
        "det_math",
        hex32("4fb30a451a9aae88529c0f66ef1a94f9b3e400cf360cd6e7398a4d9622b318ab"),
    ),
    (
        "piecewise",
        hex32("a9706b8473b6d6c50334f9ee9e6726f9eb44409e13118a403c511843db8b49f7"),
    ),
    (
        "delta",
        hex32("c25c2ed884ea0d109d267e48a0a0c08f13c32291415002519af3a3ebea452d89"),
    ),
    (
        "law",
        hex32("53583c42b85669e3ecb5f701e23329ba759071d821ddb7efa06ac75424b17000"),
    ),
    (
        "object_features",
        hex32("1e4d836d63f7395d22b5d5a7b07e97340709e8a6b158e875f7442b44612c261a"),
    ),
    (
        "sensors",
        hex32("f40d90cb955cc8dcb3ec9bdc5e47523da138e5e2fde79e675358f8eb1ea2278c"),
    ),
    (
        "depth_capture",
        hex32("16d9355a324c9eed84fed8f03b5572579a551f215e3aec071f494a036b208122"),
    ),
    (
        "robust_k1",
        hex32("f7baf928bd39a35e1e03ecf93bb7779ebbde6e7c24bb5e28ebf59a1332806e76"),
    ),
    (
        "robust_k3",
        hex32("07caa63a49319ad7e8e2399d640779496b71d6cd1add0294cd874a91d469bb83"),
    ),
    (
        "robust_k10",
        hex32("75950f93be1f73f55d024000dccf87c9b62b4fc03a93885f11fedb0975aab1ae"),
    ),
    (
        "robust_kmax",
        hex32("a8de2b48247fbe7108931a655310ab569dc4f1999aa1ac000ac616004d81b5bd"),
    ),
    (
        "alice-det-math",
        hex32("85e0884623b154a3b2b264374907b6cbac09307ca3f9a08339e48c567d3cb315"),
    ),
    (
        "alice-zip",
        hex32("51ffeb079769ea34bd3cb3df756b6cb940b8d2d6a6a0f1e1115ab8139415935e"),
    ),
    (
        "alice-db",
        hex32("e1a82c6407b1a880267b354647891271929bf2a5765a93dc48944ff64237c4ce"),
    ),
    (
        "alice-analytics",
        hex32("f8ded5961a8ef8be2d79ea4bfafd68f494c2443d725fb926cf9775824e043d4d"),
    ),
    (
        "alice-sdf",
        hex32("151079595c1e8d8fc661909bac7ea0dde0ba5a8b3ef500019d01c6af64defb69"),
    ),
];

/// The semantics identifier of this crate: the fold of
/// [`EDGE_SEMANTICS_PINS`] (see the module documentation).
pub const EDGE_SEMANTICS_ID: [u8; 32] =
    hex32("1dd8876b42ffcef5dabc506377466e282d290466acd2369d063d4358ab66ef80");

/// Earlier values of [`EDGE_SEMANTICS_ID`] whose pinned bits this release
/// still reproduces (none yet: this is the first identifier).
pub const PREVIOUS_EDGE_SEMANTICS_IDS: &[[u8; 32]] = &[];

/// How a stored or received identifier relates to this build's
/// [`EDGE_SEMANTICS_ID`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IdStatus {
    /// The identifier is this build's [`EDGE_SEMANTICS_ID`].
    Current,
    /// The identifier is listed in [`PREVIOUS_EDGE_SEMANTICS_IDS`]: an earlier
    /// identifier whose pinned bits this build still reproduces.
    VerifiedPrevious([u8; 32]),
}

/// A stored or received identifier that is neither [`EDGE_SEMANTICS_ID`] nor
/// listed in [`PREVIOUS_EDGE_SEMANTICS_IDS`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SemanticsMismatch {
    /// The identifier the bytes or the peer carried.
    pub stored: [u8; 32],
    /// This build's [`EDGE_SEMANTICS_ID`].
    pub expected: [u8; 32],
}

impl core::fmt::Display for SemanticsMismatch {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "semantics identifier mismatch: stored ")?;
        for b in self.stored {
            write!(f, "{b:02x}")?;
        }
        write!(f, ", expected ")?;
        for b in self.expected {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

#[cfg(feature = "std")]
impl std::error::Error for SemanticsMismatch {}

/// Compares an identifier carried by stored bytes or a peer with this build's.
///
/// # Errors
///
/// [`SemanticsMismatch`] naming both identifiers when `stored` is neither
/// [`EDGE_SEMANTICS_ID`] nor in [`PREVIOUS_EDGE_SEMANTICS_IDS`].
pub fn check_semantics(stored: &[u8; 32]) -> Result<IdStatus, SemanticsMismatch> {
    check_semantics_against(stored, &EDGE_SEMANTICS_ID, PREVIOUS_EDGE_SEMANTICS_IDS)
}

/// [`check_semantics`] against an explicit current identifier and list of
/// earlier ones (the list is empty in this release, so the previous-identifier
/// path is tested through this function).
pub(crate) fn check_semantics_against(
    stored: &[u8; 32],
    current: &[u8; 32],
    previous: &[[u8; 32]],
) -> Result<IdStatus, SemanticsMismatch> {
    if stored == current {
        Ok(IdStatus::Current)
    } else if previous.contains(stored) {
        Ok(IdStatus::VerifiedPrevious(*stored))
    } else {
        Err(SemanticsMismatch {
            stored: *stored,
            expected: *current,
        })
    }
}
