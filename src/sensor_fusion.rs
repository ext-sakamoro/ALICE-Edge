//! センサーフュージョン (カルマンフィルタ)
//!
//! 複数センサー入力の融合と状態推定。
//! `no_std` 対応。

/// 1次元カルマンフィルタ。
///
/// 単一状態変数 (例: 温度) の推定。
#[derive(Debug, Clone)]
pub struct KalmanFilter1D {
    /// 状態推定値。
    pub x: f32,
    /// 推定誤差共分散。
    pub p: f32,
    /// プロセスノイズ (Q)。
    pub q: f32,
    /// 測定ノイズ (R)。
    pub r: f32,
}

impl KalmanFilter1D {
    /// 新しい 1D カルマンフィルタを作成。
    #[must_use]
    pub const fn new(
        initial_estimate: f32,
        initial_error: f32,
        process_noise: f32,
        measurement_noise: f32,
    ) -> Self {
        Self {
            x: initial_estimate,
            p: initial_error,
            q: process_noise,
            r: measurement_noise,
        }
    }

    /// 予測ステップ。
    pub fn predict(&mut self) {
        // 状態遷移: x_k = x_{k-1} (定常モデル)
        // 共分散: P_k = P_{k-1} + Q
        self.p += self.q;
    }

    /// 更新ステップ。
    pub fn update(&mut self, measurement: f32) {
        // カルマンゲイン: K = P / (P + R)
        let k = self.p / (self.p + self.r);
        // 状態更新: x = x + K * (z - x)
        self.x += k * (measurement - self.x);
        // 共分散更新: P = (1 - K) * P
        self.p *= 1.0 - k;
    }

    /// 予測 + 更新を一度に実行。
    pub fn filter(&mut self, measurement: f32) -> f32 {
        self.predict();
        self.update(measurement);
        self.x
    }

    /// 現在の推定値。
    #[must_use]
    pub const fn estimate(&self) -> f32 {
        self.x
    }

    /// 現在の推定誤差。
    #[must_use]
    pub const fn error(&self) -> f32 {
        self.p
    }
}

/// 2次元カルマンフィルタ (位置 + 速度)。
///
/// 等速モデル: `x_k` = x_{k-1} + v * dt
#[derive(Debug, Clone)]
pub struct KalmanFilter2D {
    /// 位置推定。
    pub position: f32,
    /// 速度推定。
    pub velocity: f32,
    /// 位置誤差共分散。
    p00: f32,
    /// 位置-速度共分散。
    p01: f32,
    /// 速度誤差共分散。
    p11: f32,
    /// プロセスノイズ。
    q: f32,
    /// 測定ノイズ (位置)。
    r: f32,
    /// サンプル間隔 (秒)。
    dt: f32,
}

impl KalmanFilter2D {
    /// 新しい 2D カルマンフィルタを作成。
    #[must_use]
    pub const fn new(dt: f32, process_noise: f32, measurement_noise: f32) -> Self {
        Self {
            position: 0.0,
            velocity: 0.0,
            p00: 1.0,
            p01: 0.0,
            p11: 1.0,
            q: process_noise,
            r: measurement_noise,
            dt,
        }
    }

    /// 予測ステップ。
    pub fn predict(&mut self) {
        // 状態遷移: [pos, vel] = [pos + vel*dt, vel]
        self.position += self.velocity * self.dt;

        // 共分散遷移: P = F*P*F' + Q
        let dt = self.dt;
        let new_p00 = dt.mul_add(dt.mul_add(self.p11, self.p01 + self.p01), self.p00) + self.q;
        let new_p01 = dt.mul_add(self.p11, self.p01);
        let new_p11 = self.p11 + self.q;

        self.p00 = new_p00;
        self.p01 = new_p01;
        self.p11 = new_p11;
    }

    /// 更新ステップ (位置測定)。
    pub fn update(&mut self, measured_position: f32) {
        // イノベーション
        let y = measured_position - self.position;

        // イノベーション共分散: S = P00 + R
        let s = self.p00 + self.r;
        let s_inv = 1.0 / s;

        // カルマンゲイン: K = [P00, P01]' / S
        let k0 = self.p00 * s_inv;
        let k1 = self.p01 * s_inv;

        // 状態更新
        self.position += k0 * y;
        self.velocity += k1 * y;

        // 共分散更新
        let new_p00 = self.p00 - k0 * self.p00;
        let new_p01 = self.p01 - k0 * self.p01;
        let new_p11 = self.p11 - k1 * self.p01;

        self.p00 = new_p00;
        self.p01 = new_p01;
        self.p11 = new_p11;
    }

    /// 予測 + 更新を一度に実行。
    pub fn filter(&mut self, measured_position: f32) -> (f32, f32) {
        self.predict();
        self.update(measured_position);
        (self.position, self.velocity)
    }
}

/// フュージョン設定。
#[derive(Debug, Clone)]
pub struct FusionConfig {
    /// プロセスノイズ。
    pub process_noise: f32,
    /// デフォルト測定ノイズ。
    pub default_measurement_noise: f32,
    /// 外れ値閾値 (シグマ倍)。
    pub outlier_threshold: f32,
}

impl Default for FusionConfig {
    fn default() -> Self {
        Self {
            process_noise: 0.01,
            default_measurement_noise: 1.0,
            outlier_threshold: 3.0,
        }
    }
}

/// 融合センサー入力。
#[derive(Debug, Clone)]
pub struct SensorInput {
    /// 測定値。
    pub value: f32,
    /// 測定ノイズ (分散)。R が小さいほど信頼度が高い。
    pub noise: f32,
}

/// 複数センサーフュージョン。
///
/// 信頼度重み付きで複数センサーの測定値を融合。
#[derive(Debug)]
pub struct FusedSensor {
    /// 内部カルマンフィルタ。
    filter: KalmanFilter1D,
    /// 設定。
    config: FusionConfig,
    /// フュージョン回数。
    fusion_count: u64,
    /// 外れ値として棄却された回数。
    rejected_count: u64,
}

impl FusedSensor {
    /// 新しい融合センサーを作成。
    #[must_use]
    pub const fn new(config: FusionConfig) -> Self {
        let filter = KalmanFilter1D::new(
            0.0,
            100.0,
            config.process_noise,
            config.default_measurement_noise,
        );
        Self {
            filter,
            config,
            fusion_count: 0,
            rejected_count: 0,
        }
    }

    /// 複数センサー入力を融合。
    ///
    /// 重み付き平均を測定値として使用。
    pub fn fuse(&mut self, inputs: &[SensorInput]) -> f32 {
        if inputs.is_empty() {
            return self.filter.estimate();
        }

        self.filter.predict();

        // 信頼度重み付き平均
        let mut total_weight = 0.0f32;
        let mut weighted_sum = 0.0f32;
        let mut valid_count = 0u32;

        for input in inputs {
            // 外れ値検出
            let innovation = (input.value - self.filter.x).abs();
            let std_dev = (self.filter.p + input.noise).sqrt();

            if innovation > self.config.outlier_threshold * std_dev {
                self.rejected_count += 1;
                continue;
            }

            let weight = 1.0 / input.noise;
            total_weight += weight;
            weighted_sum += weight * input.value;
            valid_count += 1;
        }

        if valid_count > 0 && total_weight > 0.0 {
            let fused_measurement = weighted_sum / total_weight;
            let fused_noise = 1.0 / total_weight;
            self.filter.r = fused_noise;
            self.filter.update(fused_measurement);
        }

        self.fusion_count += 1;
        self.filter.estimate()
    }

    /// 現在の推定値。
    #[must_use]
    pub const fn estimate(&self) -> f32 {
        self.filter.estimate()
    }

    /// フュージョン回数。
    #[must_use]
    pub const fn fusion_count(&self) -> u64 {
        self.fusion_count
    }

    /// 棄却回数。
    #[must_use]
    pub const fn rejected_count(&self) -> u64 {
        self.rejected_count
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kalman_1d_constant() {
        let mut kf = KalmanFilter1D::new(0.0, 1.0, 0.01, 0.1);
        // 定常値をフィルタ
        for _ in 0..50 {
            kf.filter(10.0);
        }
        assert!((kf.estimate() - 10.0).abs() < 0.5);
    }

    #[test]
    fn kalman_1d_noisy() {
        let mut kf = KalmanFilter1D::new(0.0, 1.0, 0.01, 1.0);
        let measurements = [10.5, 9.8, 10.2, 10.1, 9.9, 10.3, 10.0];
        for &m in &measurements {
            kf.filter(m);
        }
        // フィルタ値は平均に近いはず (初期値0からの収束途中)
        assert!((kf.estimate() - 10.0).abs() < 2.0);
    }

    #[test]
    fn kalman_1d_error_decreases() {
        let mut kf = KalmanFilter1D::new(0.0, 10.0, 0.01, 0.1);
        let initial_error = kf.error();
        for _ in 0..10 {
            kf.filter(5.0);
        }
        assert!(kf.error() < initial_error);
    }

    #[test]
    fn kalman_2d_constant_velocity() {
        let mut kf = KalmanFilter2D::new(1.0, 0.01, 0.1);
        // 速度 1.0 で移動する物体
        for i in 0..20 {
            kf.filter(i as f32);
        }
        // 速度推定は 1.0 に近いはず
        assert!((kf.velocity - 1.0).abs() < 0.5);
    }

    #[test]
    fn kalman_2d_predict_only() {
        let mut kf = KalmanFilter2D::new(0.1, 0.01, 0.1);
        kf.position = 10.0;
        kf.velocity = 2.0;
        kf.predict();
        assert!((kf.position - 10.2).abs() < 0.01);
    }

    #[test]
    fn fusion_config_default() {
        let config = FusionConfig::default();
        assert_eq!(config.process_noise, 0.01);
        assert_eq!(config.outlier_threshold, 3.0);
    }

    #[test]
    fn fused_sensor_single_input() {
        let mut sensor = FusedSensor::new(FusionConfig::default());
        let inputs = [SensorInput {
            value: 25.0,
            noise: 0.1,
        }];
        for _ in 0..20 {
            sensor.fuse(&inputs);
        }
        assert!((sensor.estimate() - 25.0).abs() < 1.0);
    }

    #[test]
    fn fused_sensor_multiple_inputs() {
        let mut sensor = FusedSensor::new(FusionConfig::default());
        let inputs = [
            SensorInput {
                value: 25.0,
                noise: 0.1,
            },
            SensorInput {
                value: 25.5,
                noise: 0.5,
            },
            SensorInput {
                value: 24.8,
                noise: 0.2,
            },
        ];
        for _ in 0..30 {
            sensor.fuse(&inputs);
        }
        // 低ノイズ入力 (25.0, noise=0.1) に寄った推定
        assert!((sensor.estimate() - 25.0).abs() < 1.0);
    }

    #[test]
    fn fused_sensor_outlier_rejection() {
        let mut sensor = FusedSensor::new(FusionConfig {
            outlier_threshold: 2.0,
            ..FusionConfig::default()
        });
        // まず安定させる
        for _ in 0..20 {
            sensor.fuse(&[SensorInput {
                value: 10.0,
                noise: 0.1,
            }]);
        }
        // 外れ値を入力
        sensor.fuse(&[SensorInput {
            value: 1000.0,
            noise: 0.1,
        }]);
        assert!(sensor.rejected_count() > 0);
        // 推定値は大きく乱れないはず
        assert!((sensor.estimate() - 10.0).abs() < 5.0);
    }

    #[test]
    fn fused_sensor_empty_input() {
        let mut sensor = FusedSensor::new(FusionConfig::default());
        sensor.fuse(&[SensorInput {
            value: 10.0,
            noise: 0.1,
        }]);
        let est = sensor.fuse(&[]);
        // 空入力は前回の推定値を返す
        assert!(est.is_finite());
    }

    #[test]
    fn fused_sensor_count() {
        let mut sensor = FusedSensor::new(FusionConfig::default());
        sensor.fuse(&[SensorInput {
            value: 1.0,
            noise: 0.1,
        }]);
        sensor.fuse(&[SensorInput {
            value: 2.0,
            noise: 0.1,
        }]);
        assert_eq!(sensor.fusion_count(), 2);
    }

    #[test]
    fn kalman_1d_predict_only() {
        let mut kf = KalmanFilter1D::new(5.0, 1.0, 0.1, 0.5);
        let before = kf.error();
        kf.predict();
        assert!(kf.error() > before);
        assert_eq!(kf.estimate(), 5.0);
    }
}
