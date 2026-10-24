// SPDX-License-Identifier: LicenseRef-Proprietary
//! Software Watchdog Timer for ALICE-Edge
//!
//! センサー読み取りやMQTT送信のハングアップを検出するためのウォッチドッグ。
//! 指定タイムアウト以内に `kick()` が呼ばれないと `is_expired()` が true を返す。
//!
//! Author: Moroya Sakamoto

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// ソフトウェアウォッチドッグタイマー
pub struct Watchdog {
    timeout: Duration,
    last_kick: Arc<Mutex<Instant>>,
    running: Arc<AtomicBool>,
}

impl Watchdog {
    /// 指定タイムアウトでウォッチドッグを作成
    #[must_use]
    pub fn new(timeout: Duration) -> Self {
        #[allow(
            clippy::disallowed_methods,
            reason = "liveness/safety mechanism: is_expired() asks whether real \
                       wall-clock time since the last kick exceeds timeout, a \
                       hardware-hang detector that is inherently wall-clock- \
                       driven; the result is consulted live (a bool) and never \
                       stored, transmitted or compared for equality"
        )]
        let now = Instant::now();
        Self {
            timeout,
            last_kick: Arc::new(Mutex::new(now)),
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// ウォッチドッグを開始（バックグラウンドスレッドは不要、ポーリング方式）
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    pub fn start(&self) {
        #[allow(
            clippy::disallowed_methods,
            reason = "same liveness/safety timer as new(), see its reason"
        )]
        let now = Instant::now();
        *self.last_kick.lock().unwrap() = now;
        self.running.store(true, Ordering::Release);
    }

    /// ウォッチドッグを停止
    pub fn stop(&self) {
        self.running.store(false, Ordering::Release);
    }

    /// タイマーをリセット（正常動作中に定期的に呼ぶ）
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    pub fn kick(&self) {
        #[allow(
            clippy::disallowed_methods,
            reason = "same liveness/safety timer as new(), see its reason"
        )]
        let now = Instant::now();
        *self.last_kick.lock().unwrap() = now;
    }

    /// タイムアウトが経過したかどうか
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    #[must_use]
    pub fn is_expired(&self) -> bool {
        if !self.running.load(Ordering::Acquire) {
            return false;
        }
        let last = *self.last_kick.lock().unwrap();
        #[allow(
            clippy::disallowed_methods,
            reason = "reads the same liveness/safety timer as new() (see its \
                       reason): elapsed() measures real wall-clock time since \
                       the last kick; the bool is consulted live, never stored, \
                       transmitted or compared for equality"
        )]
        let since_kick = last.elapsed();
        since_kick > self.timeout
    }

    /// 実行中かどうか
    #[must_use]
    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    /// 設定されたタイムアウト値
    #[must_use]
    pub const fn timeout(&self) -> Duration {
        self.timeout
    }

    /// 最後の kick からの経過時間
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    #[must_use]
    pub fn elapsed_since_kick(&self) -> Duration {
        let last = *self.last_kick.lock().unwrap();
        #[allow(
            clippy::disallowed_methods,
            reason = "reads the same liveness/safety timer as new() (see its \
                       reason) and reports it to the caller for monitoring; \
                       no value this crate computes is derived from it"
        )]
        let since_kick = last.elapsed();
        since_kick
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_watchdog_not_running() {
        let wd = Watchdog::new(Duration::from_secs(5));
        assert!(!wd.is_running());
        assert!(!wd.is_expired());
    }

    #[test]
    fn test_start_stop() {
        let wd = Watchdog::new(Duration::from_secs(5));
        wd.start();
        assert!(wd.is_running());
        wd.stop();
        assert!(!wd.is_running());
        // 停止後は expired にならない
        assert!(!wd.is_expired());
    }

    #[test]
    fn test_kick_resets_timer() {
        let wd = Watchdog::new(Duration::from_millis(50));
        wd.start();
        std::thread::sleep(Duration::from_millis(30));
        wd.kick();
        // kick 直後は expired でない
        assert!(!wd.is_expired());
    }

    #[test]
    fn test_expired_after_timeout() {
        let wd = Watchdog::new(Duration::from_millis(10));
        wd.start();
        std::thread::sleep(Duration::from_millis(30));
        assert!(wd.is_expired());
    }

    #[test]
    fn test_not_expired_before_timeout() {
        let wd = Watchdog::new(Duration::from_secs(10));
        wd.start();
        assert!(!wd.is_expired());
    }

    #[test]
    fn test_timeout_getter() {
        let wd = Watchdog::new(Duration::from_millis(500));
        assert_eq!(wd.timeout(), Duration::from_millis(500));
    }

    #[test]
    fn test_elapsed_since_kick() {
        let wd = Watchdog::new(Duration::from_secs(10));
        wd.start();
        std::thread::sleep(Duration::from_millis(10));
        assert!(wd.elapsed_since_kick() >= Duration::from_millis(5));
    }

    #[test]
    fn test_not_expired_after_stop() {
        let wd = Watchdog::new(Duration::from_millis(1));
        wd.start();
        std::thread::sleep(Duration::from_millis(10));
        wd.stop();
        // 停止後はタイムアウトしていても expired にならない
        assert!(!wd.is_expired());
    }
}
