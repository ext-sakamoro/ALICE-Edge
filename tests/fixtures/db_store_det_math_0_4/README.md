A lossless store written by `alice-db` 0.3.0 (`alice-det-math` 0.4, arithmetic
identifier `d2209b30…`): 64 slopes `0.5 sin(0.2 k) + 1` (f32) at `100 k` ms,
one segment `slope/seg_1.rkyv` whose residual is pinned to that identifier.

Read by `tests/db_bridge_errors.rs`: a build that computes with another
arithmetic must return the refusal of ALICE-DB, not an empty result.

Written by `scripts/gen_db_fixture` (`alice-db = "=0.3.0"`,
`alice-det-math = "=0.4.0"`):

    cargo run --manifest-path scripts/gen_db_fixture/Cargo.toml -- \
        tests/fixtures/db_store_det_math_0_4/slope

A rerun reproduces every file byte for byte except the segment's creation time
(3 bytes of `seg_1.rkyv` at offset 360).
