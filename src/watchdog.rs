// SPDX-License-Identifier: MIT
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
        Self {
            timeout,
            last_kick: Arc::new(Mutex::new(Instant::now())),
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    /// ウォッチドッグを開始（バックグラウンドスレッドは不要、ポーリング方式）
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned.
    pub fn start(&self) {
        *self.last_kick.lock().unwrap() = Instant::now();
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
        *self.last_kick.lock().unwrap() = Instant::now();
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
        last.elapsed() > self.timeout
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
        self.last_kick.lock().unwrap().elapsed()
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
