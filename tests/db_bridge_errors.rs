//! `db_bridge::CoefficientStore` returns the errors of ALICE-DB instead of
//! replacing them with an empty result.
//!
//! The fixture `tests/fixtures/db_store_det_math_0_4/slope` is a lossless
//! store written by alice-db 0.3.0 on alice-det-math 0.4 (arithmetic
//! identifier d2209b30…): 64 slopes `0.5 sin(0.2 k) + 1` at `100 k` ms. Its
//! residual was measured under another arithmetic than this build's, so the
//! exact values cannot be reconstructed. Oracle: alice-db 0.4 documents that
//! such a read fails with `InvalidData` carrying `ResidualError::Mismatch`
//! (default `ResidualPolicy::Refuse`), and returns the model only under
//! `ResidualPolicy::ModelOnly`.

#![cfg(feature = "db")]

use alice_edge::db_bridge::{CoefficientStore, CoefficientStoreError, ResidualPolicy, Series};
use std::path::Path;

const OLD_ID: &str = "d2209b30f6f1f45baa1b638bcdfee34ac64773b2e63b9c083b2e77afc691398e";

/// The 64 slopes the fixture was written with (f32 bits, printed by the writer)
const ORIGINAL: [u32; 64] = [
    0x3f800000, 0x3f8cb700, 0x3f98ec3b, 0x3fa4231a, 0x3fade92a, 0x3fb5daa9, 0x3fbba687, 0x3fbf119c,
    0x3fbff904, 0x3fbe5385, 0x3fba31ee, 0x3fb3be68, 0x3fab3aca, 0x3fa0fdf9, 0x3f957072, 0x3f89081c,
    0x3f788732, 0x3f5f4a6d, 0x3f475b7c, 0x3f31aea4, 0x3f1f2118, 0x3f107034, 0x3f0631e7, 0x3f00cebc,
    0x3f007dae, 0x3f0541f8, 0x3f0eeaf6, 0x3f1d160e, 0x3f2f32a7, 0x3f4487ec, 0x3f5c3c1d, 0x3f755d56,
    0x3f87758b, 0x3f93f04b, 0x3f9f9f8e, 0x3faa0c12, 0x3fb2cb75, 0x3fb9846f, 0x3fbdf265, 0x3fbfe822,
    0x3fbf51a5, 0x3fbc34ee, 0x3fb6b1be, 0x3faf005c, 0x3fa56f48, 0x3f9a6026, 0x3f8e43d5, 0x3f8195e9,
    0x3f69afa6, 0x3f511735, 0x3f3a5d84, 0x3f266a83, 0x3f1609c6, 0x3f09e27e, 0x3f0270ae, 0x3f000052,
    0x3f02aa4e, 0x3f0a5374, 0x3f16ad90, 0x3f273a93, 0x3f3b5194, 0x3f52258a, 0x3f6acd80, 0x3f8226e3,
];

fn hex(id: &[u8; 32]) -> String {
    id.iter().map(|b| format!("{b:02x}")).collect()
}

/// A private copy of the fixture as `<tmp>/slope` (a read may not change it)
fn fixture_store() -> tempfile::TempDir {
    let tmp = tempfile::tempdir().expect("temp dir");
    let slope = tmp.path().join("slope");
    std::fs::create_dir_all(&slope).expect("create");
    let fixture =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/db_store_det_math_0_4/slope");
    for e in std::fs::read_dir(&fixture).expect("fixture") {
        let e = e.expect("entry");
        std::fs::copy(e.path(), slope.join(e.file_name())).expect("copy");
    }
    tmp
}

#[test]
fn this_build_computes_with_another_arithmetic_than_the_fixture() {
    assert_ne!(hex(&alice_edge::SEMANTICS_ID), OLD_ID);
}

#[test]
fn a_slope_read_of_foreign_arithmetic_is_an_error_not_an_empty_result() {
    let tmp = fixture_store();
    let store = CoefficientStore::open(tmp.path().to_str().unwrap()).expect("open");
    let err = store
        .query_slopes(0, 10_000)
        .expect_err("the exact slopes cannot be reconstructed, so the read must fail");
    assert_eq!(err.series(), Series::Slope);
    match err.residual() {
        Some(alice_db::ResidualError::Mismatch(m)) => {
            assert_eq!(hex(&m.stored), OLD_ID, "{m:?}");
            assert_eq!(m.running, alice_edge::SEMANTICS_ID, "{m:?}");
        }
        other => panic!("expected ResidualError::Mismatch, got {other:?} ({err})"),
    }
    assert!(err.to_string().starts_with("slope read: "), "{err}");
    // a point inside the range fails the same way
    assert!(store.query_slopes(3_000, 3_000).is_err());
    // the other series of the same store is unaffected and empty
    assert_eq!(
        store.query_intercepts(0, 10_000).expect("intercepts"),
        vec![]
    );
}

#[test]
fn the_model_is_read_only_when_asked_for() {
    let tmp = fixture_store();
    let store =
        CoefficientStore::open_with_policy(tmp.path().to_str().unwrap(), ResidualPolicy::ModelOnly)
            .expect("open");
    let rows = store.query_slopes(0, 10_000).expect("model-only read");
    assert_eq!(rows.len(), 64);
    let mut differ = 0;
    for (k, &(ts, v)) in rows.iter().enumerate() {
        assert_eq!(ts, 100 * k as u64);
        // the model of the segment: finite, but not the recorded value
        // (alice-db documents no error bound for it, so none is asserted)
        assert!(v.is_finite(), "k={k}: {v}");
        differ += usize::from(v.to_bits() != ORIGINAL[k]);
    }
    assert!(
        differ > 0,
        "the model reproduced every recorded bit: the fixture tests nothing"
    );
}

#[test]
fn a_timestamp_above_i64_max_is_refused_before_anything_is_written() {
    let tmp = tempfile::tempdir().expect("temp dir");
    let store = CoefficientStore::open(tmp.path().to_str().unwrap()).expect("open");
    let too_big = u64::MAX;
    let entries = [(100, 1.0, 2.0), (too_big, 3.0, 4.0)];
    match store.record_batch(&entries) {
        Err(CoefficientStoreError::Timestamp { value, .. }) => {
            assert_eq!(value, i128::from(too_big));
        }
        other => panic!("expected Timestamp error, got {other:?}"),
    }
    assert!(matches!(
        store.record(too_big, 1.0, 1.0),
        Err(CoefficientStoreError::Timestamp { .. })
    ));
    assert!(matches!(
        store.query_slopes(0, too_big),
        Err(CoefficientStoreError::Timestamp { .. })
    ));
    store.flush().expect("flush");
    assert_eq!(
        store.query_slopes(0, 1000).expect("read"),
        vec![],
        "nothing was written"
    );
}

/// A write ALICE-DB cannot persist is an error of `record_batch`, for either
/// series. With one series' directory read-only after opening, a batch above
/// the memtable capacity (1000 points by default) has to create a segment
/// file there and fails; the error names that series.
#[cfg(unix)]
#[test]
fn a_batch_the_store_cannot_persist_is_an_error() {
    use std::os::unix::fs::PermissionsExt;
    for (series, name) in [(Series::Slope, "slope"), (Series::Intercept, "intercept")] {
        let tmp = tempfile::tempdir().expect("temp dir");
        let store = CoefficientStore::open(tmp.path().to_str().unwrap()).expect("open");
        let dir = tmp.path().join(name);
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o555)).expect("chmod");
        assert!(
            std::fs::write(dir.join("probe"), b"x").is_err(),
            "a read-only directory accepted a new file (running as root?): this test cannot \
             produce a write error here"
        );
        let entries: Vec<(u64, f32, f32)> = (0..1500u64).map(|i| (i, i as f32, 0.0)).collect();
        let result = store.record_batch(&entries);
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o755)).expect("chmod back");
        match result {
            Err(CoefficientStoreError::Db { series: s, op, .. }) => {
                assert_eq!((s, op), (series, "write"), "{name}");
            }
            other => panic!("{name}: expected a write error, got {other:?}"),
        }
    }
}
