//! Content hash of this crate's Law-bearing semantics: [`EDGE_SEMANTICS_ID`].
//!
//! [`crate::SEMANTICS_ID`] is alice-zip's `law::SEMANTICS_ID`, re-exported (the
//! identifier the Law readers check); it names alice-zip's rules and the arithmetic,
//! not this crate's own rules (the Q16.16 normal equations, the MAD rejection constants, the
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
//! 3c3b3b9ba9080c6437de392d5ed6f92c02a6e8e82636f24343c5499dbeb58ad8
//! ```
//!
//! # Entries
//!
//! [`EDGE_SEMANTICS_PINS`]: one entry per scenario golden (`"q16_linear"`,
//! `"adaptive_polyfit"`, `"constant_fit"`, `"simd_fit"`, `"sensor_fusion"`,
//! `"ring_buffer"`, `"det_math"`, `"piecewise"`, `"delta"`, `"law"`,
//! `"object_features"`, `"sensors"`, `"depth_capture"`, `"depth_frames_below_2_13"`,
//! `"depth_frames_from_2_13"`, some run only with
//! the feature that builds the module) and one per `k` of the robust fit
//! (`"robust_k1"`, `"robust_k3"`, `"robust_k10"`, `"robust_kmax"`), plus the
//! identifiers of the Law-bearing dependencies, each the dependency's own Law
//! identifier: `"alice-zip"` (its
//! `law::SEMANTICS_ID`, the `law` and `zip` features), `"alice-db"` (its
//! `DB_SEMANTICS_ID`, `db`), `"alice-analytics"` (its `ANALYTICS_SEMANTICS_ID`,
//! `dashboard`) and `"alice-sdf"` (its `SDF_SEMANTICS_ID`, `sdf`), and one
//! entry `"det-math <function>"` per `alice-det-math` function this crate calls
//! itself ([`DET_MATH_FUNCTIONS`], the identifiers of [`DET_MATH_PINS`],
//! checked against the resolved `alice-det-math` at compile time). The
//! identifier goldens (`GOLDEN_SEMANTICS_ID`, `GOLDEN_LAW_ID_*`) are not pins:
//! they are functions of the entries.
//!
//! Each upstream identifier folds the `alice-det-math` functions its own
//! crate calls; the `"det-math <function>"` entries cover the ones this crate
//! calls, so a change to any function either side computes with moves this
//! identifier, and a change to a function nobody here calls does not.
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
//! are still reproduced are listed in [`PREVIOUS_EDGE_SEMANTICS_IDS`]:
//! when only an upstream identifier this crate folds moves (a pin-only move,
//! every golden unchanged) the old identifier is listed; when a golden moved
//! (the pinned bits are not reproduced) it is not.

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

/// The `alice-det-math` functions this crate calls itself (ALICE-Law
/// `scripts/fold_set_check.py print`). Each is folded into
/// [`EDGE_SEMANTICS_PINS`] as `det-math <function>`; the upstream entries fold
/// the functions their own crates call.
pub const DET_MATH_FUNCTIONS: &[&str] = &["cos", "exp", "ln", "max", "min", "sin"];

/// The identifier of each function of [`DET_MATH_FUNCTIONS`], in that order.
///
/// The `const _` below compares each with `alice_det_math::function_pin` at
/// compile time, so a resolved `alice-det-math` whose function computes other
/// bits stops the compile instead of reporting this identifier.
pub const DET_MATH_PINS: &[(&str, [u8; 32])] = &[
    (
        "cos",
        hex32("f7627941400acabe93c8dca4090eee12f8910fe4c14a8735ad456cef3c48ed7b"),
    ),
    (
        "exp",
        hex32("606918f2d637f46f9a792a29b3ee31a991ef3196064456caa7b3a7ca81700a63"),
    ),
    (
        "ln",
        hex32("6fe18bc0fba8b7e3e871c4cbe555ddb40a460be7b3c7690303ee30accdcb7291"),
    ),
    (
        "max",
        hex32("8bd32fcb313f7fb3d7a699fc3075273de7e5a4e083912d8dc7f4cc4016fc2286"),
    ),
    (
        "min",
        hex32("2939e83813b26c868ec3113fb1c25102f16f28d8f24c343a727d4b1bec5904fe"),
    ),
    (
        "sin",
        hex32("ee1b80d12df18ec0de5bce5a6bf05432484dabfc1cee1bb9b3f3291dd862cfec"),
    ),
];

const _: () = alice_det_math::assert_function_pins(DET_MATH_PINS);

/// The pins folded into [`EDGE_SEMANTICS_ID`], as `(name, digest)`.
pub const EDGE_SEMANTICS_PINS: [(&str, [u8; 32]); 29] = [
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
        "depth_frames_below_2_13",
        hex32("e74d950a951ef412c5cbf59d61868659920f0504bc7063b9da83f081f23cff1a"),
    ),
    (
        "depth_frames_from_2_13",
        hex32("a1146387667a53203d869d83e05ae0c7a1e57f84716f578efa5177aac67df764"),
    ),
    ("det-math cos", DET_MATH_PINS[0].1),
    ("det-math exp", DET_MATH_PINS[1].1),
    ("det-math ln", DET_MATH_PINS[2].1),
    ("det-math max", DET_MATH_PINS[3].1),
    ("det-math min", DET_MATH_PINS[4].1),
    ("det-math sin", DET_MATH_PINS[5].1),
    (
        "alice-zip",
        hex32("922b28656da64d8651eae0db86d0070078c4b5dc183c4a4d0219107f73ffca78"),
    ),
    (
        "alice-db",
        hex32("97d6853c2d8eda9e80fa985081ca8e982d7a4dc9cd276f9b83c95c153f6000bb"),
    ),
    (
        "alice-analytics",
        hex32("67650b7f63ebe71670b7fd9b5781fe276f515dbcee4497169cdd74f1c19a08d4"),
    ),
    (
        "alice-sdf",
        hex32("db7ff4a4b7daf3f242f353fc7af2c448832e8625c19f2522561d3b83a496f775"),
    ),
];

/// The semantics identifier of this crate: the fold of
/// [`EDGE_SEMANTICS_PINS`] (see the module documentation).
pub const EDGE_SEMANTICS_ID: [u8; 32] =
    hex32("3c3b3b9ba9080c6437de392d5ed6f92c02a6e8e82636f24343c5499dbeb58ad8");

/// Earlier values of [`EDGE_SEMANTICS_ID`] whose pinned bits this release
/// still reproduces
///
/// Not listed: with alice-det-math at 2d01c63f the upstream identifiers moved and
/// this crate's own simulated depth frames move from frame 81920 on (`sin` /
/// `cos` reduce arguments from `2^13` on in double precision), so
/// `1dd8876b…` (alice-det-math 0.5.1) does not name this build's bits.
/// `f58c1c1f…` (alice-sdf `dca9b092…`) is listed: only the alice-sdf pin moved
/// (alice-sdf's bell pull added golden rows and still reproduces `dca9b092…`)
/// and every golden of this crate is unchanged.
/// `e0a546aa…` (alice-sdf `0f6ffc20…`) is listed: only the alice-sdf pin moved
/// (alice-sdf's heightmap face blend moved one of its own goldens, so alice-sdf
/// refuses its earlier identifiers) and every golden of this crate is unchanged.
/// `d4d143d1…` (alice-det-math 0.6 folded per function) is listed: only the
/// alice-zip pin moved (alice-zip added the kind tag of its causal 2-D law) and
/// every golden is unchanged, so its pinned bits are reproduced.
/// `573e1af1…` (alice-det-math 0.5.2 at 2d01c63f, folded as one identifier) is
/// listed: the move to alice-det-math 0.6 replaced the whole-crate det-math
/// entry by one entry per called function and moved the upstream pins, but
/// every scenario golden is unchanged, so its pinned bits are reproduced.
/// `tests/edge_semantics_id.rs` shows which pins moved.
pub const PREVIOUS_EDGE_SEMANTICS_IDS: &[[u8; 32]] = &[
    hex32("f58c1c1f198961401672b63d85e721b2e25948bee2a2caf1ced81e01efbdd678"),
    hex32("e0a546aa1a5ce5195e8fa294fb90dbce52470468495900ce503ce09ece90497f"),
    hex32("d4d143d1d024b2d75778cb8d3667a333eccebc009ef387586cba02c3c8a1c976"),
    hex32("573e1af1a6d4556748faeff04c0a4bebe827cc42cbe95c38ef81fa9e09547e8b"),
];

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
/// earlier ones.
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
