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
    pub fn open(dir: &str) -> Result<Self, String> {
        let slope_db = AliceDB::open(format!("{}/slope", dir))
            .map_err(|e| format!("slope db: {}", e))?;
        let intercept_db = AliceDB::open(format!("{}/intercept", dir))
            .map_err(|e| format!("intercept db: {}", e))?;
        Ok(Self {
            slope_db,
            intercept_db,
        })
    }

    /// Record a fitted model's coefficients (Q16.16 → f32 conversion).
    ///
    /// Converts Q16.16 fixed-point values to f32 for storage.
    pub fn record_q16(&self, timestamp_ms: u64, slope_q16: i32, intercept_q16: i32) {
        let slope_f32 = slope_q16 as f32 / crate::Q16_ONE as f32;
        let intercept_f32 = intercept_q16 as f32 / crate::Q16_ONE as f32;
        let _ = self.slope_db.put(timestamp_ms as i64, slope_f32);
        let _ = self.intercept_db.put(timestamp_ms as i64, intercept_f32);
    }

    /// Record a fitted model's coefficients (already f32).
    pub fn record(&self, timestamp_ms: u64, slope: f32, intercept: f32) {
        let _ = self.slope_db.put(timestamp_ms as i64, slope);
        let _ = self.intercept_db.put(timestamp_ms as i64, intercept);
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
        let intercepts: Vec<(i64, f32)> = entries.iter().map(|&(ts, _, i)| (ts as i64, i)).collect();
        let _ = self.slope_db.put_batch(&slopes);
        let _ = self.intercept_db.put_batch(&intercepts);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coefficient_store_open() {
        let dir = "/tmp/alice_edge_db_test";
        let result = CoefficientStore::open(dir);
        if result.is_ok() {
            let store = result.unwrap();
            store.record(1000, 1.5, 10.0);
            let slopes = store.query_slopes(0, 2000);
            assert!(!slopes.is_empty());
        }
    }
}
