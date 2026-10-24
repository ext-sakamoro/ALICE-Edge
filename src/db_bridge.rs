// SPDX-License-Identifier: LicenseRef-Proprietary
//! ALICE-DB bridge: Persist model coefficients
//!
//! Stores fitted linear model coefficients (slope, intercept) from
//! edge devices into ALICE-DB for historical analysis and monitoring.
//!
//! # Pipeline
//!
//! ```text
//! Sensor data → fit_linear_fixed() → (slope, intercept)
//! → CoefficientStore.record() → ALICE-DB time-series
//! ```
//!
//! # Errors are returned, never replaced by a default
//!
//! Every write and every read returns a [`CoefficientStoreError`] naming the
//! store (slope or intercept) and carrying the `std::io::Error` of ALICE-DB.
//! A read that ALICE-DB refuses, for example a lossless segment whose residual
//! was measured under another arithmetic (`alice_db::ResidualError`), is an
//! error here too, not an empty result: [`CoefficientStoreError::residual`]
//! returns the typed reason. Reading the model approximation of such data is
//! an explicit choice, [`CoefficientStore::open_with_policy`] with
//! [`ResidualPolicy::ModelOnly`].

use alice_db::{AliceDB, ResidualError, StorageConfig};
use core::fmt;
use std::io;
use std::path::PathBuf;

pub use alice_db::ResidualPolicy;

/// Which of the two series an error belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Series {
    /// `<dir>/slope`
    Slope,
    /// `<dir>/intercept`
    Intercept,
}

impl fmt::Display for Series {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Series::Slope => "slope",
            Series::Intercept => "intercept",
        })
    }
}

/// An error of [`CoefficientStore`].
#[derive(Debug)]
#[non_exhaustive]
pub enum CoefficientStoreError {
    /// ALICE-DB failed or refused the operation on `series`.
    Db {
        /// The series the operation was on
        series: Series,
        /// What was done: "open", "write", "read" or "flush"
        op: &'static str,
        /// The error ALICE-DB returned
        source: io::Error,
    },
    /// A timestamp does not fit the `i64` keys of the store
    /// (above `i64::MAX` on write, negative on read).
    Timestamp {
        /// The series the key belongs to
        series: Series,
        /// The key as given (write) or as stored (read)
        value: i128,
    },
}

impl CoefficientStoreError {
    /// The typed reason when ALICE-DB refused a read of lossless data
    /// (`ResidualError::Mismatch`: measured under another arithmetic,
    /// `ResidualError::Corrupt`: the residual cannot be decoded).
    #[must_use]
    pub fn residual(&self) -> Option<&ResidualError> {
        match self {
            CoefficientStoreError::Db { source, .. } => source
                .get_ref()
                .and_then(|inner| inner.downcast_ref::<ResidualError>()),
            CoefficientStoreError::Timestamp { .. } => None,
        }
    }

    /// The series the error belongs to.
    #[must_use]
    pub fn series(&self) -> Series {
        match self {
            CoefficientStoreError::Db { series, .. }
            | CoefficientStoreError::Timestamp { series, .. } => *series,
        }
    }
}

impl fmt::Display for CoefficientStoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CoefficientStoreError::Db { series, op, source } => {
                write!(f, "{series} {op}: {source}")
            }
            CoefficientStoreError::Timestamp { series, value } => {
                write!(f, "{series}: timestamp {value} is outside 0..=i64::MAX")
            }
        }
    }
}

impl std::error::Error for CoefficientStoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            CoefficientStoreError::Db { source, .. } => Some(source),
            CoefficientStoreError::Timestamp { .. } => None,
        }
    }
}

/// Persistent store for edge model coefficients.
///
/// Stores slope and intercept values from linear fits as time-series,
/// enabling trend analysis and anomaly detection on model parameters.
pub struct CoefficientStore {
    slope_db: AliceDB,
    intercept_db: AliceDB,
}

fn db(series: Series, op: &'static str) -> impl FnOnce(io::Error) -> CoefficientStoreError {
    move |source| CoefficientStoreError::Db { series, op, source }
}

fn key(series: Series, timestamp_ms: u64) -> Result<i64, CoefficientStoreError> {
    i64::try_from(timestamp_ms).map_err(|_| CoefficientStoreError::Timestamp {
        series,
        value: i128::from(timestamp_ms),
    })
}

fn read(
    series: Series,
    rows: io::Result<Vec<(i64, f32)>>,
) -> Result<Vec<(u64, f32)>, CoefficientStoreError> {
    // scan only returns keys inside the requested non-negative range, so the
    // conversion cannot fail today; it is checked rather than cast so that a
    // change on either side can never wrap a key into a timestamp
    rows.map_err(db(series, "read"))?
        .into_iter()
        .map(|(ts, v)| {
            u64::try_from(ts)
                .map(|ts| (ts, v))
                .map_err(|_| CoefficientStoreError::Timestamp {
                    series,
                    value: i128::from(ts),
                })
        })
        .collect()
}

impl CoefficientStore {
    /// Open or create coefficient databases at the given directory.
    ///
    /// Reads refuse lossless data whose exact values cannot be reconstructed
    /// ([`ResidualPolicy::Refuse`], the ALICE-DB default).
    ///
    /// # Errors
    ///
    /// Returns [`CoefficientStoreError::Db`] if either database fails to open.
    pub fn open(dir: &str) -> Result<Self, CoefficientStoreError> {
        Self::open_with_policy(dir, ResidualPolicy::Refuse)
    }

    /// Open with an explicit residual policy. [`ResidualPolicy::ModelOnly`]
    /// reads the model approximation of lossless data whose residual cannot
    /// be applied, instead of an error; the values are then not the recorded
    /// ones, which is why it is never the default.
    ///
    /// # Errors
    ///
    /// Returns [`CoefficientStoreError::Db`] if either database fails to open.
    pub fn open_with_policy(
        dir: &str,
        residual_policy: ResidualPolicy,
    ) -> Result<Self, CoefficientStoreError> {
        Self::open_with_options(dir, residual_policy, 0)
    }

    /// Open with an explicit residual policy and the `created_at` metadata
    /// (Unix milliseconds) that ALICE-DB stamps on every segment this store
    /// flushes. Neither this crate nor ALICE-DB reads the wall clock: the
    /// value is the caller's, and the same samples written with the same
    /// `created_at` give the same bytes on disk. [`CoefficientStore::open`]
    /// and [`CoefficientStore::open_with_policy`] pass `0`.
    ///
    /// # Errors
    ///
    /// Returns [`CoefficientStoreError::Db`] if either database fails to open.
    pub fn open_with_options(
        dir: &str,
        residual_policy: ResidualPolicy,
        created_at: u64,
    ) -> Result<Self, CoefficientStoreError> {
        let open = |series: Series| {
            AliceDB::with_config(StorageConfig {
                data_dir: PathBuf::from(dir).join(series.to_string()),
                residual_policy,
                created_at,
                ..StorageConfig::default()
            })
            .map_err(db(series, "open"))
        };
        Ok(Self {
            slope_db: open(Series::Slope)?,
            intercept_db: open(Series::Intercept)?,
        })
    }

    /// Record a fitted model's coefficients (Q16.16 → f32 conversion).
    ///
    /// Converts Q16.16 fixed-point values to f32 for storage.
    ///
    /// # Errors
    ///
    /// Returns [`CoefficientStoreError`] if the timestamp exceeds `i64::MAX`
    /// or either database write fails.
    pub fn record_q16(
        &self,
        timestamp_ms: u64,
        slope_q16: i32,
        intercept_q16: i32,
    ) -> Result<(), CoefficientStoreError> {
        const INV_Q16_ONE: f32 = 1.0 / (1i32 << 16) as f32;
        let slope_f32 = slope_q16 as f32 * INV_Q16_ONE;
        let intercept_f32 = intercept_q16 as f32 * INV_Q16_ONE;
        self.record(timestamp_ms, slope_f32, intercept_f32)
    }

    /// Record a fitted model's coefficients (already f32).
    ///
    /// # Errors
    ///
    /// Returns [`CoefficientStoreError`] if the timestamp exceeds `i64::MAX`
    /// or either database write fails.
    pub fn record(
        &self,
        timestamp_ms: u64,
        slope: f32,
        intercept: f32,
    ) -> Result<(), CoefficientStoreError> {
        let ts = key(Series::Slope, timestamp_ms)?;
        self.slope_db
            .put(ts, slope)
            .map_err(db(Series::Slope, "write"))?;
        self.intercept_db
            .put(ts, intercept)
            .map_err(db(Series::Intercept, "write"))
    }

    /// Query slope history in a time range.
    ///
    /// # Errors
    ///
    /// Returns [`CoefficientStoreError`] if ALICE-DB fails or refuses the read
    /// (see [`CoefficientStoreError::residual`]); never an empty result in
    /// place of an error.
    pub fn query_slopes(
        &self,
        from_ms: u64,
        to_ms: u64,
    ) -> Result<Vec<(u64, f32)>, CoefficientStoreError> {
        let (from, to) = (key(Series::Slope, from_ms)?, key(Series::Slope, to_ms)?);
        read(Series::Slope, self.slope_db.scan(from, to))
    }

    /// Query intercept history in a time range.
    ///
    /// # Errors
    ///
    /// As [`CoefficientStore::query_slopes`].
    pub fn query_intercepts(
        &self,
        from_ms: u64,
        to_ms: u64,
    ) -> Result<Vec<(u64, f32)>, CoefficientStoreError> {
        let (from, to) = (
            key(Series::Intercept, from_ms)?,
            key(Series::Intercept, to_ms)?,
        );
        read(Series::Intercept, self.intercept_db.scan(from, to))
    }

    /// Record a batch of coefficient pairs.
    ///
    /// Every timestamp is checked before anything is written.
    ///
    /// # Errors
    ///
    /// Returns [`CoefficientStoreError`] if a timestamp exceeds `i64::MAX` or
    /// either database write fails.
    pub fn record_batch(&self, entries: &[(u64, f32, f32)]) -> Result<(), CoefficientStoreError> {
        let mut slopes = Vec::with_capacity(entries.len());
        let mut intercepts = Vec::with_capacity(entries.len());
        for &(ts, s, i) in entries {
            let ts = key(Series::Slope, ts)?;
            slopes.push((ts, s));
            intercepts.push((ts, i));
        }
        self.slope_db
            .put_batch(&slopes)
            .map_err(db(Series::Slope, "write"))?;
        self.intercept_db
            .put_batch(&intercepts)
            .map_err(db(Series::Intercept, "write"))
    }

    /// Flush memtable to disk so that scan/query can read the data.
    ///
    /// # Errors
    ///
    /// Returns [`CoefficientStoreError::Db`] if either database flush fails.
    pub fn flush(&self) -> Result<(), CoefficientStoreError> {
        self.slope_db.flush().map_err(db(Series::Slope, "flush"))?;
        self.intercept_db
            .flush()
            .map_err(db(Series::Intercept, "flush"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A wall-clock name alone collides when tests run in parallel: macOS
    // reports microseconds, so two tests can get the same directory and one
    // fails on the database's write lock. The process id and a per-process
    // counter make every call distinct.
    fn temp_dir() -> String {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir =
            std::env::temp_dir().join(format!("alice_edge_db_test_{}_{}", std::process::id(), n));
        dir.to_string_lossy().into_owned()
    }

    /// The bytes of every file the store wrote, keyed by path under `dir`
    fn stored_bytes(dir: &str) -> Vec<(String, Vec<u8>)> {
        let mut out = Vec::new();
        let mut todo = vec![std::path::PathBuf::from(dir)];
        while let Some(p) = todo.pop() {
            for e in std::fs::read_dir(&p).expect("read_dir") {
                let path = e.expect("entry").path();
                if path.is_dir() {
                    todo.extend([path]);
                } else {
                    let name = path
                        .strip_prefix(dir)
                        .expect("under dir")
                        .to_string_lossy()
                        .into_owned();
                    let bytes = std::fs::read(&path).expect("read");
                    out.extend([(name, bytes)]);
                }
            }
        }
        out.sort();
        out
    }

    fn write_fixed_series(dir: &str, created_at: u64) {
        let store = CoefficientStore::open_with_options(dir, ResidualPolicy::Refuse, created_at)
            .expect("open failed");
        let entries: Vec<(u64, f32, f32)> = (0..64u64)
            .map(|k| {
                (
                    k * 100,
                    k as f32 * 0.25 + 1.0,
                    (k % 7) as f32 - k as f32 * 0.5,
                )
            })
            .collect();
        store.record_batch(&entries).expect("record_batch failed");
        store.flush().expect("flush failed");
    }

    // oracle: ALICE-DB 0.5 stamps segments with StorageConfig::created_at and
    // reads no clock, so the stored bytes are a function of the samples and
    // created_at alone (under alice-db 0.4 the segment files differed between
    // two runs a second apart, the metadata being SystemTime::now())
    #[test]
    fn stored_bytes_are_a_function_of_samples_and_created_at() {
        let (a, b, c) = (temp_dir(), temp_dir(), temp_dir());
        write_fixed_series(&a, 1_700_000_000_000);
        std::thread::sleep(std::time::Duration::from_millis(20));
        write_fixed_series(&b, 1_700_000_000_000);
        write_fixed_series(&c, 1_700_000_000_001);
        let (a, b, c) = (stored_bytes(&a), stored_bytes(&b), stored_bytes(&c));
        assert!(
            a.iter().any(|(_, bytes)| !bytes.is_empty()),
            "nothing was written"
        );
        assert_eq!(a, b, "same samples and created_at must give the same bytes");
        assert_ne!(a, c, "created_at must reach the stored segments");
    }

    #[test]
    fn test_coefficient_store_open() {
        let dir = temp_dir();
        let store = CoefficientStore::open(&dir).expect("open failed");
        store.record(1000, 1.5, 10.0).expect("record failed");
        store.flush().expect("flush failed");
        let slopes = store.query_slopes(0, 2000).expect("query failed");
        assert!(!slopes.is_empty());
    }

    #[test]
    fn test_record_q16_roundtrip() {
        let dir = temp_dir();
        let store = CoefficientStore::open(&dir).expect("open failed");
        store
            .record_q16(100, 65536, -163840)
            .expect("record_q16 failed");
        store.flush().expect("flush failed");
        let slopes = store.query_slopes(0, 200).expect("query failed");
        assert_eq!(slopes.len(), 1);
        assert!((slopes[0].1 - 1.0).abs() < 0.01);
        let intercepts = store.query_intercepts(0, 200).expect("query failed");
        assert!((intercepts[0].1 - (-2.5)).abs() < 0.01);
    }

    #[test]
    fn test_record_batch() {
        let dir = temp_dir();
        let store = CoefficientStore::open(&dir).expect("open failed");
        let entries = vec![(100, 1.0f32, 2.0f32), (200, 3.0, 4.0), (300, 5.0, 6.0)];
        store.record_batch(&entries).expect("record_batch failed");
        store.flush().expect("flush failed");
        let slopes = store.query_slopes(0, 400).expect("query failed");
        assert_eq!(slopes.len(), 3);
    }

    #[test]
    fn test_query_empty_range() {
        let dir = temp_dir();
        let store = CoefficientStore::open(&dir).expect("open failed");
        store.record(100, 1.0, 2.0).expect("record failed");
        store.flush().expect("flush failed");
        let slopes = store.query_slopes(500, 600).expect("query failed");
        assert!(slopes.is_empty());
    }

    #[test]
    fn test_record_multiple_timestamps() {
        let dir = temp_dir();
        let store = CoefficientStore::open(&dir).expect("open failed");
        for ts in 0..10u64 {
            store
                .record(ts * 100, ts as f32, -(ts as f32))
                .expect("record failed");
        }
        store.flush().expect("flush failed");
        let slopes = store.query_slopes(0, 1000).expect("query failed");
        assert_eq!(slopes.len(), 10);
    }

    #[test]
    fn test_query_intercepts() {
        let dir = temp_dir();
        let store = CoefficientStore::open(&dir).expect("open failed");
        store.record(100, 1.0, 42.0).expect("record failed");
        store.flush().expect("flush failed");
        let intercepts = store.query_intercepts(0, 200).expect("query failed");
        assert_eq!(intercepts.len(), 1);
        assert!((intercepts[0].1 - 42.0).abs() < 0.01);
    }

    #[test]
    fn test_record_q16_zero() {
        let dir = temp_dir();
        let store = CoefficientStore::open(&dir).expect("open failed");
        store.record_q16(0, 0, 0).expect("record_q16 failed");
        store.flush().expect("flush failed");
        let slopes = store.query_slopes(0, 1).expect("query failed");
        assert_eq!(slopes.len(), 1);
        assert!((slopes[0].1).abs() < 0.001);
    }

    #[test]
    fn test_batch_and_query_range() {
        let dir = temp_dir();
        let store = CoefficientStore::open(&dir).expect("open failed");
        let entries: Vec<(u64, f32, f32)> = (0..20).map(|i| (i * 10, i as f32, 0.0)).collect();
        store.record_batch(&entries).expect("record_batch failed");
        store.flush().expect("flush failed");
        let slopes = store.query_slopes(50, 150).expect("query failed");
        assert!(slopes.len() >= 5 && slopes.len() <= 15);
    }
}
