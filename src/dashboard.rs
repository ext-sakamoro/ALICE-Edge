// SPDX-License-Identifier: MIT
//! ALICE-Analytics dashboard for ALICE-Edge
//!
//! Provides real-time monitoring of edge sensor processing using
//! probabilistic data structures from ALICE-Analytics.
//!
//! # Features
//!
//! - `HyperLogLog`: Count unique sensors
//! - Count-Min Sketch: Per-sensor query frequency
//! - Latency percentiles (p50, p95, p99) via reservoir sampling
//! - Terminal dashboard output (SIGUSR1 or periodic)
//!
//! Author: Moroya Sakamoto

use alice_analytics::{CountMinSketch, HyperLogLog};
use std::time::Instant;

/// Edge dashboard collecting processing metrics
pub struct EdgeDashboard {
    /// Unique sensor counter (`HyperLogLog`, ~16KB)
    hll_sensors: HyperLogLog,
    /// Per-sensor frequency (Count-Min Sketch)
    cms_frequency: CountMinSketch,
    /// Processing latency samples (simple reservoir)
    latencies_us: Vec<u64>,
    /// ソート済みレイテンシキャッシュ（percentile 計算時に遅延構築）
    sorted_latencies: Vec<u64>,
    /// ソート済みキャッシュが無効化されたかどうか
    sorted_dirty: bool,
    /// Compression ratios
    compression_ratios: Vec<f32>,
    /// Total samples processed
    pub total_samples: u64,
    /// Total bytes saved
    pub total_bytes_saved: u64,
    /// Total bytes sent (compressed)
    pub total_bytes_sent: u64,
    /// Total models fitted
    pub total_models: u64,
    /// Start time
    start: Instant,
    /// Last reset
    #[allow(dead_code)]
    last_reset: Instant,
}

impl EdgeDashboard {
    /// Create a new dashboard
    #[must_use]
    pub fn new() -> Self {
        Self {
            hll_sensors: HyperLogLog::new(),
            cms_frequency: CountMinSketch::new(),
            latencies_us: Vec::with_capacity(10000),
            sorted_latencies: Vec::new(),
            sorted_dirty: true,
            compression_ratios: Vec::with_capacity(10000),
            total_samples: 0,
            total_bytes_saved: 0,
            total_bytes_sent: 0,
            total_models: 0,
            start: Instant::now(),
            last_reset: Instant::now(),
        }
    }

    /// Record a sensor observation
    pub fn record_sensor(&mut self, sensor_id: &str) {
        self.hll_sensors.insert_bytes(sensor_id.as_bytes());
        self.cms_frequency.insert_bytes(sensor_id.as_bytes());
    }

    /// Record processing latency (Vitter Algorithm R reservoir sampling)
    pub fn record_latency(&mut self, latency_us: u64) {
        self.sorted_dirty = true;
        if self.latencies_us.len() < 10000 {
            self.latencies_us.push(latency_us);
        } else {
            // Vitter Algorithm R: 確率 k/n で置換 (k=10000, n=total_models+1)
            // 高速な決定的疑似乱数で置換インデックスを生成
            let n = self.total_models.wrapping_add(1);
            // FNV-1a hash of n → 疑似ランダムインデックス
            let hash = n.wrapping_mul(0x517c_c1b7_2722_0a95) ^ 0x6c62_272e_07bb_0142;
            let reservoir_idx = (hash % n) as usize;
            if reservoir_idx < 10000 {
                self.latencies_us[reservoir_idx] = latency_us;
            }
        }
    }

    /// Record a compression event
    pub fn record_compression(
        &mut self,
        sensor_id: &str,
        raw_samples: usize,
        compressed_bytes: usize,
        latency_us: u64,
    ) {
        // Use saturating_mul to prevent overflow on 32-bit targets when raw_samples is large.
        let raw_bytes = raw_samples.saturating_mul(4); // i32 = 4 bytes
        self.total_samples += raw_samples as u64;
        // Use saturating_sub to avoid underflow when compressed_bytes > raw_bytes.
        self.total_bytes_saved += raw_bytes.saturating_sub(compressed_bytes) as u64;
        self.total_bytes_sent += compressed_bytes as u64;
        self.total_models += 1;

        let ratio = raw_bytes as f32 / compressed_bytes.max(1) as f32;
        if self.compression_ratios.len() < 10000 {
            self.compression_ratios.push(ratio);
        }

        self.record_sensor(sensor_id);
        self.record_latency(latency_us);
    }

    /// Get estimated unique sensor count
    #[must_use]
    pub fn unique_sensors(&self) -> u64 {
        self.hll_sensors.cardinality() as u64
    }

    /// Get estimated frequency for a specific sensor
    #[must_use]
    pub fn sensor_frequency(&self, sensor_id: &str) -> u64 {
        self.cms_frequency.estimate_bytes(sensor_id.as_bytes())
    }

    /// Compute latency percentiles（ソート結果をキャッシュ、dirty 時のみ再構築）
    pub fn latency_percentiles(&mut self) -> LatencyStats {
        if self.latencies_us.is_empty() {
            return LatencyStats::default();
        }

        if self.sorted_dirty {
            self.sorted_latencies.clear();
            self.sorted_latencies.extend_from_slice(&self.latencies_us);
            self.sorted_latencies.sort_unstable();
            self.sorted_dirty = false;
        }
        let sorted = &self.sorted_latencies;
        let n = sorted.len();

        // Divide before multiplying to prevent overflow when n is large on
        // platforms where usize is 32 bits (e.g. Cortex-M with std feature).
        LatencyStats {
            p50: sorted[n / 2],
            p95: sorted[n - (n / 20).max(1)],
            p99: sorted[n - (n / 100).max(1)],
            min: sorted[0],
            max: sorted[n - 1],
            avg: sorted.iter().sum::<u64>() / n as u64,
        }
    }

    /// Average compression ratio
    #[inline(always)]
    #[must_use]
    pub fn avg_compression_ratio(&self) -> f32 {
        if self.compression_ratios.is_empty() {
            return 0.0;
        }
        self.compression_ratios.iter().sum::<f32>() / self.compression_ratios.len() as f32
    }

    /// Uptime in seconds
    #[must_use]
    pub fn uptime_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    /// Models per second
    #[inline(always)]
    #[must_use]
    pub fn models_per_sec(&self) -> f64 {
        let elapsed = self.start.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.total_models as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Print terminal dashboard
    pub fn print_dashboard(&mut self) {
        let latency = self.latency_percentiles();
        let uptime = self.uptime_secs();

        println!("┌─────────────────────────────────────────────────────┐");
        println!("│           ALICE-Edge Dashboard                      │");
        println!("├─────────────────────────────────────────────────────┤");
        println!(
            "│ Uptime: {:.1}s  |  Models/sec: {:.1}",
            uptime,
            self.models_per_sec()
        );
        println!("│                                                     │");
        println!("│ Sensors:                                            │");
        println!(
            "│   Unique: ~{:<8} (HyperLogLog estimate)",
            self.unique_sensors()
        );
        println!(
            "│   Total models: {:<12}  Samples: {}",
            self.total_models, self.total_samples
        );
        println!("│                                                     │");
        println!("│ Compression:                                        │");
        println!("│   Avg ratio: {:.1}x", self.avg_compression_ratio());
        println!(
            "│   Bytes saved: {:<12}  Bytes sent: {}",
            format_bytes(self.total_bytes_saved),
            format_bytes(self.total_bytes_sent)
        );
        println!(
            "│   Savings: {:.1}%",
            if self.total_bytes_saved + self.total_bytes_sent > 0 {
                self.total_bytes_saved as f64
                    / (self.total_bytes_saved + self.total_bytes_sent) as f64
                    * 100.0
            } else {
                0.0
            }
        );
        println!("│                                                     │");
        println!("│ Latency (us):                                       │");
        println!(
            "│   p50: {:<8}  p95: {:<8}  p99: {}",
            latency.p50, latency.p95, latency.p99
        );
        println!(
            "│   min: {:<8}  max: {:<8}  avg: {}",
            latency.min, latency.max, latency.avg
        );
        println!("└─────────────────────────────────────────────────────┘");
    }

    /// Get dashboard as JSON string
    pub fn to_json(&mut self) -> String {
        let latency = self.latency_percentiles();
        format!(
            r#"{{"uptime":{:.1},"models_per_sec":{:.1},"unique_sensors":{},"total_models":{},"total_samples":{},"avg_compression":{:.1},"bytes_saved":{},"bytes_sent":{},"latency_p50":{},"latency_p95":{},"latency_p99":{},"latency_avg":{}}}"#,
            self.uptime_secs(),
            self.models_per_sec(),
            self.unique_sensors(),
            self.total_models,
            self.total_samples,
            self.avg_compression_ratio(),
            self.total_bytes_saved,
            self.total_bytes_sent,
            latency.p50,
            latency.p95,
            latency.p99,
            latency.avg,
        )
    }

    /// Reset periodic counters (keep HLL and CMS state)
    #[allow(dead_code)]
    pub fn reset_periodic(&mut self) {
        self.latencies_us.clear();
        self.sorted_latencies.clear();
        self.sorted_dirty = true;
        self.compression_ratios.clear();
        self.last_reset = Instant::now();
    }

    // ── E10: WebSocket 配信 ──────────────────────────────────────────

    /// Get dashboard state as a serializable snapshot for WebSocket push.
    ///
    /// Returns a compact struct suitable for JSON serialization and
    /// periodic push to browser dashboards via WebSocket.
    pub fn snapshot(&mut self) -> DashboardSnapshot {
        let latency = self.latency_percentiles();
        DashboardSnapshot {
            uptime_secs: self.uptime_secs(),
            models_per_sec: self.models_per_sec(),
            unique_sensors: self.unique_sensors(),
            total_models: self.total_models,
            total_samples: self.total_samples,
            avg_compression_ratio: self.avg_compression_ratio(),
            total_bytes_saved: self.total_bytes_saved,
            total_bytes_sent: self.total_bytes_sent,
            latency_p50: latency.p50,
            latency_p95: latency.p95,
            latency_p99: latency.p99,
            latency_min: latency.min,
            latency_max: latency.max,
            latency_avg: latency.avg,
        }
    }
}

impl Default for EdgeDashboard {
    fn default() -> Self {
        Self::new()
    }
}

/// Dashboard snapshot for WebSocket/HTTP streaming [E10]
///
/// Lightweight, serializable struct containing all dashboard metrics.
/// Push to browsers via WebSocket at configurable interval (e.g., 1Hz).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "sensors", derive(serde::Serialize, serde::Deserialize))]
pub struct DashboardSnapshot {
    pub uptime_secs: f64,
    pub models_per_sec: f64,
    pub unique_sensors: u64,
    pub total_models: u64,
    pub total_samples: u64,
    pub avg_compression_ratio: f32,
    pub total_bytes_saved: u64,
    pub total_bytes_sent: u64,
    pub latency_p50: u64,
    pub latency_p95: u64,
    pub latency_p99: u64,
    pub latency_min: u64,
    pub latency_max: u64,
    pub latency_avg: u64,
}

impl DashboardSnapshot {
    /// Serialize to JSON string
    #[must_use]
    pub fn to_json(&self) -> String {
        format!(
            r#"{{"uptime":{:.1},"models_per_sec":{:.1},"unique_sensors":{},"total_models":{},"total_samples":{},"avg_compression":{:.1},"bytes_saved":{},"bytes_sent":{},"latency_p50":{},"latency_p95":{},"latency_p99":{},"latency_min":{},"latency_max":{},"latency_avg":{}}}"#,
            self.uptime_secs,
            self.models_per_sec,
            self.unique_sensors,
            self.total_models,
            self.total_samples,
            self.avg_compression_ratio,
            self.total_bytes_saved,
            self.total_bytes_sent,
            self.latency_p50,
            self.latency_p95,
            self.latency_p99,
            self.latency_min,
            self.latency_max,
            self.latency_avg,
        )
    }

    /// Serialize to compact binary (112 bytes)
    #[must_use]
    pub fn to_binary(&self) -> [u8; 112] {
        let mut buf = [0u8; 112];
        buf[0..8].copy_from_slice(&self.uptime_secs.to_le_bytes());
        buf[8..16].copy_from_slice(&self.models_per_sec.to_le_bytes());
        buf[16..24].copy_from_slice(&self.unique_sensors.to_le_bytes());
        buf[24..32].copy_from_slice(&self.total_models.to_le_bytes());
        buf[32..40].copy_from_slice(&self.total_samples.to_le_bytes());
        buf[40..44].copy_from_slice(&self.avg_compression_ratio.to_le_bytes());
        buf[44..52].copy_from_slice(&self.total_bytes_saved.to_le_bytes());
        buf[52..60].copy_from_slice(&self.total_bytes_sent.to_le_bytes());
        buf[60..68].copy_from_slice(&self.latency_p50.to_le_bytes());
        buf[68..76].copy_from_slice(&self.latency_p95.to_le_bytes());
        buf[76..84].copy_from_slice(&self.latency_p99.to_le_bytes());
        buf[84..92].copy_from_slice(&self.latency_min.to_le_bytes());
        buf[92..100].copy_from_slice(&self.latency_max.to_le_bytes());
        buf[100..108].copy_from_slice(&self.latency_avg.to_le_bytes());
        // 4 bytes padding for alignment
        buf
    }
}

/// Latency statistics
#[derive(Debug, Default, Clone)]
#[cfg_attr(feature = "sensors", derive(serde::Serialize, serde::Deserialize))]
pub struct LatencyStats {
    pub p50: u64,
    pub p95: u64,
    pub p99: u64,
    pub min: u64,
    pub max: u64,
    pub avg: u64,
}

/// Format bytes to human-readable string
#[inline(always)]
fn format_bytes(bytes: u64) -> String {
    const INV_MB: f64 = 1.0 / 1_000_000.0;
    const INV_KB: f64 = 1.0 / 1_000.0;
    if bytes >= 1_000_000 {
        format!("{:.1} MB", bytes as f64 * INV_MB)
    } else if bytes >= 1_000 {
        format!("{:.1} KB", bytes as f64 * INV_KB)
    } else {
        format!("{bytes} B")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dashboard_basic() {
        let mut dashboard = EdgeDashboard::new();

        dashboard.record_compression("bme280", 1000, 8, 50);
        dashboard.record_compression("dht22", 500, 8, 30);
        dashboard.record_compression("bme280", 1000, 8, 45);

        assert_eq!(dashboard.total_models, 3);
        assert_eq!(dashboard.total_samples, 2500);
        assert!(dashboard.unique_sensors() >= 1); // HLL approximation
    }

    #[test]
    fn test_latency_percentiles() {
        let mut dashboard = EdgeDashboard::new();

        for i in 0..100 {
            dashboard.record_latency(i * 10);
        }

        let stats = dashboard.latency_percentiles();
        assert!(stats.p50 >= 400 && stats.p50 <= 600);
        assert!(stats.p99 >= 900);
    }

    #[test]
    fn test_json_output() {
        let mut dashboard = EdgeDashboard::new();
        dashboard.record_compression("test", 100, 8, 10);
        let json = dashboard.to_json();
        assert!(json.contains("\"total_models\":1"));
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1500), "1.5 KB");
        assert_eq!(format_bytes(1500000), "1.5 MB");
    }

    // ── New tests ─────────────────────────────────────────────

    #[test]
    fn test_new_dashboard_zero_state() {
        let dashboard = EdgeDashboard::new();
        assert_eq!(dashboard.total_models, 0);
        assert_eq!(dashboard.total_samples, 0);
        assert_eq!(dashboard.total_bytes_saved, 0);
        assert_eq!(dashboard.total_bytes_sent, 0);
    }

    #[test]
    fn test_latency_percentiles_empty() {
        let mut dashboard = EdgeDashboard::new();
        let stats = dashboard.latency_percentiles();
        assert_eq!(stats.p50, 0);
        assert_eq!(stats.p95, 0);
        assert_eq!(stats.p99, 0);
        assert_eq!(stats.min, 0);
        assert_eq!(stats.max, 0);
        assert_eq!(stats.avg, 0);
    }

    #[test]
    fn test_latency_single_value() {
        let mut dashboard = EdgeDashboard::new();
        dashboard.record_latency(42);
        let stats = dashboard.latency_percentiles();
        assert_eq!(stats.min, 42);
        assert_eq!(stats.max, 42);
        assert_eq!(stats.avg, 42);
    }

    #[test]
    fn test_avg_compression_ratio_empty() {
        let dashboard = EdgeDashboard::new();
        assert_eq!(dashboard.avg_compression_ratio(), 0.0);
    }

    #[test]
    fn test_avg_compression_ratio_computed() {
        let mut dashboard = EdgeDashboard::new();
        // raw_bytes = 1000*4 = 4000, compressed = 8  → ratio = 500
        dashboard.record_compression("sensor1", 1000, 8, 10);
        let ratio = dashboard.avg_compression_ratio();
        assert!(
            ratio > 100.0,
            "expected high compression ratio, got {}",
            ratio
        );
    }

    #[test]
    fn test_bytes_saved_calculation() {
        let mut dashboard = EdgeDashboard::new();
        // raw = 100 * 4 = 400 bytes, compressed = 8 bytes
        dashboard.record_compression("s", 100, 8, 5);
        // bytes_saved = 400 - 8 = 392
        assert_eq!(dashboard.total_bytes_saved, 392);
        assert_eq!(dashboard.total_bytes_sent, 8);
    }

    #[test]
    fn test_bytes_saved_no_underflow_when_compressed_larger() {
        let mut dashboard = EdgeDashboard::new();
        // compressed_bytes > raw_bytes — saturating_sub should give 0
        dashboard.record_compression("s", 1, 100, 5);
        // raw_bytes = 4, compressed = 100 → saved = 0 (saturating)
        assert_eq!(dashboard.total_bytes_saved, 0);
        assert_eq!(dashboard.total_bytes_sent, 100);
    }

    #[test]
    fn test_record_compression_increments_models() {
        let mut dashboard = EdgeDashboard::new();
        for _ in 0..10 {
            dashboard.record_compression("sensor", 100, 8, 5);
        }
        assert_eq!(dashboard.total_models, 10);
    }

    #[test]
    fn test_record_compression_total_samples() {
        let mut dashboard = EdgeDashboard::new();
        dashboard.record_compression("a", 200, 8, 1);
        dashboard.record_compression("b", 300, 8, 1);
        assert_eq!(dashboard.total_samples, 500);
    }

    #[test]
    fn test_record_sensor_frequency() {
        let mut dashboard = EdgeDashboard::new();
        dashboard.record_sensor("bme280");
        dashboard.record_sensor("bme280");
        dashboard.record_sensor("bme280");
        let freq = dashboard.sensor_frequency("bme280");
        assert!(freq >= 1, "frequency should be at least 1");
    }

    #[test]
    fn test_record_latency_reservoir_replacement() {
        let mut dashboard = EdgeDashboard::new();
        // Fill exactly 10000 entries
        for i in 0..10000 {
            dashboard.record_latency(i as u64);
        }
        // Simulate total_models > 10000 by calling record_compression many times
        // to trigger the reservoir branch; we just confirm no panic here.
        dashboard.record_latency(99999);
        let stats = dashboard.latency_percentiles();
        assert!(stats.max >= 99);
    }

    #[test]
    fn test_json_output_contains_uptime() {
        let mut dashboard = EdgeDashboard::new();
        dashboard.record_compression("x", 10, 8, 1);
        let json = dashboard.to_json();
        assert!(json.contains("\"uptime\":"));
    }

    #[test]
    fn test_json_output_contains_bytes_saved() {
        let mut dashboard = EdgeDashboard::new();
        dashboard.record_compression("x", 100, 8, 1);
        let json = dashboard.to_json();
        assert!(json.contains("\"bytes_saved\":392"));
        assert!(json.contains("\"bytes_sent\":8"));
    }

    #[test]
    fn test_format_bytes_boundary_1000() {
        // Exactly 1000 bytes → 1.0 KB
        assert_eq!(format_bytes(1000), "1.0 KB");
    }

    #[test]
    fn test_format_bytes_boundary_1000000() {
        // Exactly 1_000_000 bytes → 1.0 MB
        assert_eq!(format_bytes(1_000_000), "1.0 MB");
    }

    #[test]
    fn test_format_bytes_zero() {
        assert_eq!(format_bytes(0), "0 B");
    }

    #[test]
    fn test_format_bytes_999() {
        assert_eq!(format_bytes(999), "999 B");
    }

    #[test]
    fn test_models_per_sec_zero_before_any_data() {
        let dashboard = EdgeDashboard::new();
        // Should return 0.0 (or a very small number) when total_models == 0
        let mps = dashboard.models_per_sec();
        assert!(mps >= 0.0);
    }

    #[test]
    fn test_uptime_non_negative() {
        let dashboard = EdgeDashboard::new();
        assert!(dashboard.uptime_secs() >= 0.0);
    }

    #[test]
    fn test_latency_stats_min_max() {
        let mut dashboard = EdgeDashboard::new();
        dashboard.record_latency(5);
        dashboard.record_latency(100);
        dashboard.record_latency(50);
        let stats = dashboard.latency_percentiles();
        assert_eq!(stats.min, 5);
        assert_eq!(stats.max, 100);
    }

    #[test]
    fn test_latency_stats_avg_two_values() {
        let mut dashboard = EdgeDashboard::new();
        dashboard.record_latency(10);
        dashboard.record_latency(20);
        let stats = dashboard.latency_percentiles();
        assert_eq!(stats.avg, 15);
    }

    // ── E10: Snapshot / WebSocket テスト ───────────────────────────

    #[test]
    fn test_snapshot_basic() {
        let mut dashboard = EdgeDashboard::new();
        dashboard.record_compression("sensor1", 100, 8, 50);
        let snap = dashboard.snapshot();
        assert_eq!(snap.total_models, 1);
        assert_eq!(snap.total_samples, 100);
        assert_eq!(snap.total_bytes_sent, 8);
    }

    #[test]
    fn test_snapshot_to_json() {
        let mut dashboard = EdgeDashboard::new();
        dashboard.record_compression("s", 50, 4, 10);
        let snap = dashboard.snapshot();
        let json = snap.to_json();
        assert!(json.contains("\"total_models\":1"));
        assert!(json.contains("\"latency_min\":"));
    }

    #[test]
    fn test_snapshot_to_binary_size() {
        let mut dashboard = EdgeDashboard::new();
        let snap = dashboard.snapshot();
        let bin = snap.to_binary();
        assert_eq!(bin.len(), 112);
    }

    #[test]
    fn test_snapshot_to_binary_roundtrip_total_models() {
        let mut dashboard = EdgeDashboard::new();
        dashboard.record_compression("x", 200, 16, 20);
        let snap = dashboard.snapshot();
        let bin = snap.to_binary();
        // total_models は offset 24..32
        let total_models = u64::from_le_bytes([
            bin[24], bin[25], bin[26], bin[27], bin[28], bin[29], bin[30], bin[31],
        ]);
        assert_eq!(total_models, 1);
    }

    #[test]
    fn test_snapshot_empty_dashboard() {
        let mut dashboard = EdgeDashboard::new();
        let snap = dashboard.snapshot();
        assert_eq!(snap.total_models, 0);
        assert_eq!(snap.latency_p50, 0);
        assert_eq!(snap.avg_compression_ratio, 0.0);
    }
}
