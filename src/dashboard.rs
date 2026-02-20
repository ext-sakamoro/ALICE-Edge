//! ALICE-Analytics dashboard for ALICE-Edge
//!
//! Provides real-time monitoring of edge sensor processing using
//! probabilistic data structures from ALICE-Analytics.
//!
//! # Features
//!
//! - HyperLogLog: Count unique sensors
//! - Count-Min Sketch: Per-sensor query frequency
//! - Latency percentiles (p50, p95, p99) via reservoir sampling
//! - Terminal dashboard output (SIGUSR1 or periodic)
//!
//! Author: Moroya Sakamoto

use alice_analytics::{CountMinSketch, HyperLogLog};
use std::time::Instant;

/// Edge dashboard collecting processing metrics
pub struct EdgeDashboard {
    /// Unique sensor counter (HyperLogLog, ~16KB)
    hll_sensors: HyperLogLog,
    /// Per-sensor frequency (Count-Min Sketch)
    cms_frequency: CountMinSketch,
    /// Processing latency samples (simple reservoir)
    latencies_us: Vec<u64>,
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
    pub fn new() -> Self {
        Self {
            hll_sensors: HyperLogLog::new(),
            cms_frequency: CountMinSketch::new(),
            latencies_us: Vec::with_capacity(10000),
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

    /// Record processing latency
    pub fn record_latency(&mut self, latency_us: u64) {
        if self.latencies_us.len() < 10000 {
            self.latencies_us.push(latency_us);
        } else {
            // Reservoir sampling
            let idx = (self.total_models as usize) % 10000;
            self.latencies_us[idx] = latency_us;
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
        let raw_bytes = raw_samples * 4; // i32 = 4 bytes
        self.total_samples += raw_samples as u64;
        self.total_bytes_saved += (raw_bytes - compressed_bytes) as u64;
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
    pub fn unique_sensors(&self) -> u64 {
        self.hll_sensors.cardinality() as u64
    }

    /// Get estimated frequency for a specific sensor
    pub fn sensor_frequency(&self, sensor_id: &str) -> u64 {
        self.cms_frequency.estimate_bytes(sensor_id.as_bytes())
    }

    /// Compute latency percentiles
    pub fn latency_percentiles(&self) -> LatencyStats {
        if self.latencies_us.is_empty() {
            return LatencyStats::default();
        }

        let mut sorted = self.latencies_us.clone();
        sorted.sort_unstable();
        let n = sorted.len();

        LatencyStats {
            p50: sorted[n * 50 / 100],
            p95: sorted[n * 95 / 100],
            p99: sorted[n.saturating_sub(1) * 99 / 100],
            min: sorted[0],
            max: sorted[n - 1],
            avg: sorted.iter().sum::<u64>() / n as u64,
        }
    }

    /// Average compression ratio
    pub fn avg_compression_ratio(&self) -> f32 {
        if self.compression_ratios.is_empty() {
            return 0.0;
        }
        self.compression_ratios.iter().sum::<f32>() / self.compression_ratios.len() as f32
    }

    /// Uptime in seconds
    pub fn uptime_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }

    /// Models per second
    pub fn models_per_sec(&self) -> f64 {
        let elapsed = self.start.elapsed().as_secs_f64();
        if elapsed > 0.0 {
            self.total_models as f64 / elapsed
        } else {
            0.0
        }
    }

    /// Print terminal dashboard
    pub fn print_dashboard(&self) {
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
    pub fn to_json(&self) -> String {
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
        self.compression_ratios.clear();
        self.last_reset = Instant::now();
    }
}

/// Latency statistics
#[derive(Debug, Default, Clone)]
pub struct LatencyStats {
    pub p50: u64,
    pub p95: u64,
    pub p99: u64,
    pub min: u64,
    pub max: u64,
    pub avg: u64,
}

/// Format bytes to human-readable string
fn format_bytes(bytes: u64) -> String {
    if bytes >= 1_000_000 {
        format!("{:.1} MB", bytes as f64 / 1_000_000.0)
    } else if bytes >= 1_000 {
        format!("{:.1} KB", bytes as f64 / 1_000.0)
    } else {
        format!("{} B", bytes)
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
}
