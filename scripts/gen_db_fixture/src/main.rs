//! Writes the lossless store read by `tests/db_bridge_errors.rs`: 64 slopes
//! `0.5 sin(0.2 k) + 1` (f32) at `100 k` ms, one segment, its residual pinned
//! to the arithmetic of alice-det-math 0.4. Prints the values as f32 bits
//! (the `ORIGINAL` table of the test) after reading every one back exactly.

use alice_db::{AliceDB, FitConfig, StorageConfig};

fn main() {
    let out = std::path::PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: gen-db-fixture <output directory>"),
    );
    if out.exists() {
        std::fs::remove_dir_all(&out).expect("remove the old store");
    }
    std::fs::create_dir_all(&out).expect("create the output directory");
    let db = AliceDB::with_config(StorageConfig {
        data_dir: out.clone(),
        enable_wal: false,
        fit_config: FitConfig {
            lossless: true,
            ..FitConfig::default()
        },
        ..StorageConfig::default()
    })
    .expect("open");
    let data: Vec<(i64, f32)> = (0..64i64)
        .map(|k| (k * 100, 0.5 * alice_det_math::sin(0.2 * k as f32) + 1.0))
        .collect();
    db.put_batch(&data).expect("write");
    db.flush().expect("flush");
    for &(t, v) in &data {
        assert_eq!(
            db.get(t).expect("read").map(f32::to_bits),
            Some(v.to_bits()),
            "key {t} does not read back exactly"
        );
    }
    let hex: String = alice_det_math::SEMANTICS_ID
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    println!("arithmetic identifier {hex}");
    for (_, v) in &data {
        print!("0x{:08x}, ", v.to_bits());
    }
    println!();
}
