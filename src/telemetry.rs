//! テレメトリ / 監視ハブ
//!
//! 複数エッジデバイスのメトリクス集約と健全性監視。

use std::collections::HashMap;

/// テレメトリ設定。
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    /// デバイス ID。
    pub device_id: String,
    /// 報告間隔 (秒)。
    pub report_interval_secs: u64,
    /// イベントバッファサイズ。
    pub buffer_size: usize,
    /// ハートビート間隔 (秒)。
    pub heartbeat_interval_secs: u64,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            device_id: String::new(),
            report_interval_secs: 60,
            buffer_size: 1000,
            heartbeat_interval_secs: 30,
        }
    }
}

/// デバイスメトリクス。
#[derive(Debug, Clone)]
pub struct DeviceMetrics {
    /// CPU 使用率 (0.0-100.0)。
    pub cpu_usage: f32,
    /// メモリ使用率 (0.0-100.0)。
    pub memory_usage: f32,
    /// 温度 (摂氏)。
    pub temperature: f32,
    /// 稼働時間 (秒)。
    pub uptime_secs: u64,
    /// ディスク使用率 (0.0-100.0)。
    pub disk_usage: f32,
    /// ネットワーク送信バイト数。
    pub network_tx_bytes: u64,
    /// ネットワーク受信バイト数。
    pub network_rx_bytes: u64,
}

impl Default for DeviceMetrics {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0.0,
            temperature: 0.0,
            uptime_secs: 0,
            disk_usage: 0.0,
            network_tx_bytes: 0,
            network_rx_bytes: 0,
        }
    }
}

/// テレメトリイベント。
#[derive(Debug, Clone)]
pub enum TelemetryEvent {
    /// ハートビート。
    Heartbeat {
        /// デバイス ID。
        device_id: String,
        /// タイムスタンプ (Unix 秒)。
        timestamp: u64,
    },
    /// アラート。
    Alert {
        /// デバイス ID。
        device_id: String,
        /// 重大度。
        severity: AlertSeverity,
        /// メッセージ。
        message: String,
        /// タイムスタンプ (Unix 秒)。
        timestamp: u64,
    },
    /// メトリクス報告。
    Metric {
        /// デバイス ID。
        device_id: String,
        /// メトリクスデータ。
        metrics: DeviceMetrics,
        /// タイムスタンプ (Unix 秒)。
        timestamp: u64,
    },
}

/// アラート重大度。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertSeverity {
    /// 情報。
    Info,
    /// 警告。
    Warning,
    /// 危険。
    Critical,
}

impl core::fmt::Display for AlertSeverity {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Info => write!(f, "Info"),
            Self::Warning => write!(f, "Warning"),
            Self::Critical => write!(f, "Critical"),
        }
    }
}

/// 健全性ステータス。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// 正常。
    Healthy,
    /// 劣化。
    Degraded,
    /// 危険。
    Critical,
    /// 不明。
    Unknown,
}

impl core::fmt::Display for HealthStatus {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Healthy => write!(f, "Healthy"),
            Self::Degraded => write!(f, "Degraded"),
            Self::Critical => write!(f, "Critical"),
            Self::Unknown => write!(f, "Unknown"),
        }
    }
}

/// デバイス状態。
#[derive(Debug, Clone)]
struct DeviceState {
    /// 最終ハートビート (Unix 秒)。
    last_heartbeat: u64,
    /// 最新メトリクス。
    metrics: Option<DeviceMetrics>,
    /// 健全性。
    health: HealthStatus,
    /// アラート数。
    alert_count: u64,
}

/// テレメトリハブ (複数デバイス管理)。
#[derive(Debug)]
pub struct TelemetryHub {
    /// デバイス ID → 状態。
    devices: HashMap<String, DeviceState>,
    /// イベントバッファ。
    events: Vec<TelemetryEvent>,
    /// バッファ最大サイズ。
    buffer_size: usize,
    /// ハートビートタイムアウト (秒)。
    heartbeat_timeout_secs: u64,
}

impl Default for TelemetryHub {
    fn default() -> Self {
        Self::new(1000, 120)
    }
}

impl TelemetryHub {
    /// 新しいテレメトリハブを作成。
    #[must_use]
    pub fn new(buffer_size: usize, heartbeat_timeout_secs: u64) -> Self {
        Self {
            devices: HashMap::new(),
            events: Vec::with_capacity(buffer_size.min(1024)),
            buffer_size,
            heartbeat_timeout_secs,
        }
    }

    /// イベントを記録。
    pub fn record(&mut self, event: TelemetryEvent) {
        match &event {
            TelemetryEvent::Heartbeat {
                device_id,
                timestamp,
            } => {
                let state = self
                    .devices
                    .entry(device_id.clone())
                    .or_insert(DeviceState {
                        last_heartbeat: 0,
                        metrics: None,
                        health: HealthStatus::Unknown,
                        alert_count: 0,
                    });
                state.last_heartbeat = *timestamp;
                state.health = HealthStatus::Healthy;
            }
            TelemetryEvent::Alert {
                device_id,
                severity,
                ..
            } => {
                let state = self
                    .devices
                    .entry(device_id.clone())
                    .or_insert(DeviceState {
                        last_heartbeat: 0,
                        metrics: None,
                        health: HealthStatus::Unknown,
                        alert_count: 0,
                    });
                state.alert_count += 1;
                if *severity == AlertSeverity::Critical {
                    state.health = HealthStatus::Critical;
                } else if state.health != HealthStatus::Critical {
                    state.health = HealthStatus::Degraded;
                }
            }
            TelemetryEvent::Metric {
                device_id,
                metrics,
                timestamp,
            } => {
                let state = self
                    .devices
                    .entry(device_id.clone())
                    .or_insert(DeviceState {
                        last_heartbeat: 0,
                        metrics: None,
                        health: HealthStatus::Unknown,
                        alert_count: 0,
                    });
                state.metrics = Some(metrics.clone());
                state.last_heartbeat = *timestamp;
                // 温度に基づく健全性判定
                state.health = if metrics.temperature > 85.0 {
                    HealthStatus::Critical
                } else if metrics.temperature > 70.0 || metrics.cpu_usage > 90.0 {
                    HealthStatus::Degraded
                } else {
                    HealthStatus::Healthy
                };
            }
        }

        if self.events.len() < self.buffer_size {
            self.events.push(event);
        }
    }

    /// バッファをフラッシュし、イベントを返す。
    pub fn flush(&mut self) -> Vec<TelemetryEvent> {
        core::mem::take(&mut self.events)
    }

    /// デバイスの健全性を取得。
    #[must_use]
    pub fn device_health(&self, device_id: &str) -> HealthStatus {
        self.devices
            .get(device_id)
            .map_or(HealthStatus::Unknown, |s| s.health)
    }

    /// デバイスのメトリクスを取得。
    #[must_use]
    pub fn device_metrics(&self, device_id: &str) -> Option<&DeviceMetrics> {
        self.devices.get(device_id).and_then(|s| s.metrics.as_ref())
    }

    /// 登録デバイス数。
    #[must_use]
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// タイムアウトしたデバイスを検出。
    #[must_use]
    pub fn timed_out_devices(&self, now_unix: u64) -> Vec<String> {
        self.devices
            .iter()
            .filter(|(_, s)| {
                s.last_heartbeat > 0
                    && now_unix.saturating_sub(s.last_heartbeat) > self.heartbeat_timeout_secs
            })
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// バッファ内イベント数。
    #[must_use]
    pub const fn event_count(&self) -> usize {
        self.events.len()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let config = TelemetryConfig::default();
        assert_eq!(config.report_interval_secs, 60);
        assert_eq!(config.buffer_size, 1000);
    }

    #[test]
    fn default_metrics() {
        let m = DeviceMetrics::default();
        assert_eq!(m.cpu_usage, 0.0);
        assert_eq!(m.uptime_secs, 0);
    }

    #[test]
    fn alert_severity_display() {
        assert_eq!(AlertSeverity::Info.to_string(), "Info");
        assert_eq!(AlertSeverity::Warning.to_string(), "Warning");
        assert_eq!(AlertSeverity::Critical.to_string(), "Critical");
    }

    #[test]
    fn health_status_display() {
        assert_eq!(HealthStatus::Healthy.to_string(), "Healthy");
        assert_eq!(HealthStatus::Degraded.to_string(), "Degraded");
        assert_eq!(HealthStatus::Critical.to_string(), "Critical");
        assert_eq!(HealthStatus::Unknown.to_string(), "Unknown");
    }

    #[test]
    fn hub_record_heartbeat() {
        let mut hub = TelemetryHub::new(100, 60);
        hub.record(TelemetryEvent::Heartbeat {
            device_id: "dev1".into(),
            timestamp: 1000,
        });
        assert_eq!(hub.device_count(), 1);
        assert_eq!(hub.device_health("dev1"), HealthStatus::Healthy);
    }

    #[test]
    fn hub_record_metric() {
        let mut hub = TelemetryHub::new(100, 60);
        hub.record(TelemetryEvent::Metric {
            device_id: "dev1".into(),
            metrics: DeviceMetrics {
                cpu_usage: 50.0,
                temperature: 45.0,
                ..DeviceMetrics::default()
            },
            timestamp: 1000,
        });
        assert_eq!(hub.device_health("dev1"), HealthStatus::Healthy);
        assert!(hub.device_metrics("dev1").is_some());
    }

    #[test]
    fn hub_high_temperature_degraded() {
        let mut hub = TelemetryHub::new(100, 60);
        hub.record(TelemetryEvent::Metric {
            device_id: "hot".into(),
            metrics: DeviceMetrics {
                temperature: 75.0,
                ..DeviceMetrics::default()
            },
            timestamp: 1000,
        });
        assert_eq!(hub.device_health("hot"), HealthStatus::Degraded);
    }

    #[test]
    fn hub_critical_temperature() {
        let mut hub = TelemetryHub::new(100, 60);
        hub.record(TelemetryEvent::Metric {
            device_id: "burn".into(),
            metrics: DeviceMetrics {
                temperature: 90.0,
                ..DeviceMetrics::default()
            },
            timestamp: 1000,
        });
        assert_eq!(hub.device_health("burn"), HealthStatus::Critical);
    }

    #[test]
    fn hub_alert_critical() {
        let mut hub = TelemetryHub::new(100, 60);
        hub.record(TelemetryEvent::Alert {
            device_id: "dev1".into(),
            severity: AlertSeverity::Critical,
            message: "Overheating".into(),
            timestamp: 1000,
        });
        assert_eq!(hub.device_health("dev1"), HealthStatus::Critical);
    }

    #[test]
    fn hub_flush() {
        let mut hub = TelemetryHub::new(100, 60);
        hub.record(TelemetryEvent::Heartbeat {
            device_id: "dev1".into(),
            timestamp: 1000,
        });
        assert_eq!(hub.event_count(), 1);
        let events = hub.flush();
        assert_eq!(events.len(), 1);
        assert_eq!(hub.event_count(), 0);
    }

    #[test]
    fn hub_timed_out() {
        let mut hub = TelemetryHub::new(100, 60);
        hub.record(TelemetryEvent::Heartbeat {
            device_id: "old".into(),
            timestamp: 100,
        });
        hub.record(TelemetryEvent::Heartbeat {
            device_id: "new".into(),
            timestamp: 500,
        });
        let timed_out = hub.timed_out_devices(500);
        assert!(timed_out.contains(&"old".to_string()));
        assert!(!timed_out.contains(&"new".to_string()));
    }

    #[test]
    fn hub_unknown_device() {
        let hub = TelemetryHub::new(100, 60);
        assert_eq!(hub.device_health("nonexistent"), HealthStatus::Unknown);
        assert!(hub.device_metrics("nonexistent").is_none());
    }

    #[test]
    fn hub_default() {
        let hub = TelemetryHub::default();
        assert_eq!(hub.device_count(), 0);
    }
}
