// SPDX-License-Identifier: MIT
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

use alice_db::AliceDB;

/// Persistent store for edge model coefficients.
///
/// Stores slope and intercept values from linear fits as time-series,
/// enabling trend analysis and anomaly detection on model parameters.
pub struct CoefficientStore {
    slope_db: AliceDB,
    intercept_db: AliceDB,
}

impl CoefficientStore {
    /// Open or create coefficient databases at the given directory.
    ///
    /// # Errors
    ///
    /// Returns `Err(String)` if either the slope or intercept database fails to open.
    pub fn open(dir: &str) -> Result<Self, String> {
        let slope_db =
            AliceDB::open(format!("{dir}/slope")).map_err(|e| format!("slope db: {e}"))?;
        let intercept_db =
            AliceDB::open(format!("{dir}/intercept")).map_err(|e| format!("intercept db: {e}"))?;
        Ok(Self {
            slope_db,
            intercept_db,
        })
    }

    /// Record a fitted model's coefficients (Q16.16 → f32 conversion).
    ///
    /// Converts Q16.16 fixed-point values to f32 for storage.
    ///
    /// # Errors
    ///
    /// Returns `Err(String)` if either the slope or intercept database write fails.
    pub fn record_q16(
        &self,
        timestamp_ms: u64,
        slope_q16: i32,
        intercept_q16: i32,
    ) -> Result<(), String> {
        const INV_Q16_ONE: f32 = 1.0 / (1i32 << 16) as f32;
        let slope_f32 = slope_q16 as f32 * INV_Q16_ONE;
        let intercept_f32 = intercept_q16 as f32 * INV_Q16_ONE;
        self.slope_db
            .put(timestamp_ms as i64, slope_f32)
            .map_err(|e| format!("slope write: {e}"))?;
        self.intercept_db
            .put(timestamp_ms as i64, intercept_f32)
            .map_err(|e| format!("intercept write: {e}"))?;
        Ok(())
    }

    /// Record a fitted model's coefficients (already f32).
    ///
    /// # Errors
    ///
    /// Returns `Err(String)` if either the slope or intercept database write fails.
    pub fn record(&self, timestamp_ms: u64, slope: f32, intercept: f32) -> Result<(), String> {
        self.slope_db
            .put(timestamp_ms as i64, slope)
            .map_err(|e| format!("slope write: {e}"))?;
        self.intercept_db
            .put(timestamp_ms as i64, intercept)
            .map_err(|e| format!("intercept write: {e}"))?;
        Ok(())
    }

    /// Query slope history in a time range.
    pub fn query_slopes(&self, from_ms: u64, to_ms: u64) -> Vec<(u64, f32)> {
        self.slope_db
            .scan(from_ms as i64, to_ms as i64)
            .unwrap_or_default()
            .into_iter()
            .map(|(ts, v)| (ts as u64, v))
            .collect()
    }

    /// Query intercept history in a time range.
    pub fn query_intercepts(&self, from_ms: u64, to_ms: u64) -> Vec<(u64, f32)> {
        self.intercept_db
            .scan(from_ms as i64, to_ms as i64)
            .unwrap_or_default()
            .into_iter()
            .map(|(ts, v)| (ts as u64, v))
            .collect()
    }

    /// Record a batch of coefficient pairs.
    pub fn record_batch(&self, entries: &[(u64, f32, f32)]) {
        let slopes: Vec<(i64, f32)> = entries.iter().map(|&(ts, s, _)| (ts as i64, s)).collect();
        let intercepts: Vec<(i64, f32)> =
            entries.iter().map(|&(ts, _, i)| (ts as i64, i)).collect();
        let _ = self.slope_db.put_batch(&slopes);
        let _ = self.intercept_db.put_batch(&intercepts);
    }

    /// Flush memtable to disk so that scan/query can read the data.
    ///
    /// # Errors
    ///
    /// Returns `Err(String)` if either the slope or intercept database flush fails.
    pub fn flush(&self) -> Result<(), String> {
        self.slope_db
            .flush()
            .map_err(|e| format!("slope flush: {e}"))?;
        self.intercept_db
            .flush()
            .map_err(|e| format!("intercept flush: {e}"))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir() -> String {
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        format!("/tmp/alice_edge_db_test_{}", id)
    }

    #[test]
    fn test_coefficient_store_open() {
        let dir = temp_dir();
        let store = CoefficientStore::open(&dir).expect("open failed");
        store.record(1000, 1.5, 10.0).expect("record failed");
        store.flush().expect("flush failed");
        let slopes = store.query_slopes(0, 2000);
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
        let slopes = store.query_slopes(0, 200);
        assert_eq!(slopes.len(), 1);
        assert!((slopes[0].1 - 1.0).abs() < 0.01);
        let intercepts = store.query_intercepts(0, 200);
        assert!((intercepts[0].1 - (-2.5)).abs() < 0.01);
    }

    #[test]
    fn test_record_batch() {
        let dir = temp_dir();
        let store = CoefficientStore::open(&dir).expect("open failed");
        let entries = vec![(100, 1.0f32, 2.0f32), (200, 3.0, 4.0), (300, 5.0, 6.0)];
        store.record_batch(&entries);
        store.flush().expect("flush failed");
        let slopes = store.query_slopes(0, 400);
        assert_eq!(slopes.len(), 3);
    }

    #[test]
    fn test_query_empty_range() {
        let dir = temp_dir();
        let store = CoefficientStore::open(&dir).expect("open failed");
        store.record(100, 1.0, 2.0).expect("record failed");
        store.flush().expect("flush failed");
        let slopes = store.query_slopes(500, 600);
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
        let slopes = store.query_slopes(0, 1000);
        assert_eq!(slopes.len(), 10);
    }

    #[test]
    fn test_query_intercepts() {
        let dir = temp_dir();
        let store = CoefficientStore::open(&dir).expect("open failed");
        store.record(100, 1.0, 42.0).expect("record failed");
        store.flush().expect("flush failed");
        let intercepts = store.query_intercepts(0, 200);
        assert_eq!(intercepts.len(), 1);
        assert!((intercepts[0].1 - 42.0).abs() < 0.01);
    }

    #[test]
    fn test_record_q16_zero() {
        let dir = temp_dir();
        let store = CoefficientStore::open(&dir).expect("open failed");
        store.record_q16(0, 0, 0).expect("record_q16 failed");
        store.flush().expect("flush failed");
        let slopes = store.query_slopes(0, 1);
        assert_eq!(slopes.len(), 1);
        assert!((slopes[0].1).abs() < 0.001);
    }

    #[test]
    fn test_batch_and_query_range() {
        let dir = temp_dir();
        let store = CoefficientStore::open(&dir).expect("open failed");
        let entries: Vec<(u64, f32, f32)> = (0..20).map(|i| (i * 10, i as f32, 0.0)).collect();
        store.record_batch(&entries);
        store.flush().expect("flush failed");
        let slopes = store.query_slopes(50, 150);
        assert!(slopes.len() >= 5 && slopes.len() <= 15);
    }
}
