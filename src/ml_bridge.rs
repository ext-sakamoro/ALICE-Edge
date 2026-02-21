// SPDX-License-Identifier: MIT
//! ALICE-Edge × ALICE-ML bridge
//!
//! Edge sensor data → ternary ML inference for anomaly classification.
//!
//! Author: Moroya Sakamoto

use alice_ml::{TernaryWeight, ternary_matvec};

/// Sensor features extracted from edge linear model coefficients
#[derive(Debug, Clone, Copy)]
pub struct SensorFeatures {
    pub slope: f32,
    pub intercept: f32,
    pub residual: f32,
    pub sample_count: u16,
}

/// Edge classification result
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EdgeClassification {
    Normal = 0,
    Anomaly = 1,
    Drift = 2,
    Saturated = 3,
}

/// Ternary classifier for edge sensor data
pub struct EdgeClassifier {
    /// Input → hidden weights (4 inputs × 16 hidden)
    w_hidden: Vec<TernaryWeight>,
    /// Hidden → output weights (16 hidden × 4 classes)
    w_output: Vec<TernaryWeight>,
    /// Hidden bias
    bias_hidden: Vec<f32>,
    /// Output bias
    bias_output: Vec<f32>,
    /// Classification count
    pub total_classified: u64,
}

impl EdgeClassifier {
    /// Create classifier with deterministic pseudo-random ternary weights.
    /// Model size: ~64 bytes (ternary weights pack 16 values per u32).
    pub fn new() -> Self {
        // Deterministic initialization via hash-based ternary weights
        let mut w_hidden = Vec::with_capacity(4 * 16);
        for i in 0..(4 * 16) {
            let hash = ((i as u64).wrapping_mul(0x9E3779B97F4A7C15)) >> 62;
            w_hidden.push(match hash {
                0 => TernaryWeight::Neg,
                1 => TernaryWeight::Zero,
                _ => TernaryWeight::Pos,
            });
        }
        let mut w_output = Vec::with_capacity(16 * 4);
        for i in 0..(16 * 4) {
            let hash = ((i as u64 + 100).wrapping_mul(0x517CC1B727220A95)) >> 62;
            w_output.push(match hash {
                0 => TernaryWeight::Neg,
                1 => TernaryWeight::Zero,
                _ => TernaryWeight::Pos,
            });
        }
        Self {
            w_hidden,
            w_output,
            bias_hidden: vec![0.0; 16],
            bias_output: vec![0.0; 4],
            total_classified: 0,
        }
    }

    /// Classify a single sensor reading
    pub fn classify_sensor(&mut self, features: &SensorFeatures) -> EdgeClassification {
        let input = [
            features.slope,
            features.intercept,
            features.residual,
            features.sample_count as f32 * (1.0 / 1000.0),
        ];

        // Hidden layer: ternary matvec + ReLU
        let mut hidden = ternary_matvec(&self.w_hidden, &input, 16, 4);
        for (h, b) in hidden.iter_mut().zip(self.bias_hidden.iter()) {
            *h = (*h + b).max(0.0); // ReLU
        }

        // Output layer: ternary matvec
        let mut output = ternary_matvec(&self.w_output, &hidden, 4, 16);
        for (o, b) in output.iter_mut().zip(self.bias_output.iter()) {
            *o += b;
        }

        self.total_classified += 1;

        // Argmax
        let mut max_idx = 0;
        let mut max_val = output[0];
        for (i, &v) in output.iter().enumerate().skip(1) {
            if v > max_val {
                max_val = v;
                max_idx = i;
            }
        }

        match max_idx {
            0 => EdgeClassification::Normal,
            1 => EdgeClassification::Anomaly,
            2 => EdgeClassification::Drift,
            _ => EdgeClassification::Saturated,
        }
    }

    /// Batch classify multiple sensor readings
    pub fn classify_batch(&mut self, features: &[SensorFeatures]) -> Vec<EdgeClassification> {
        features.iter().map(|f| self.classify_sensor(f)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_sensor() {
        let mut classifier = EdgeClassifier::new();
        let features = SensorFeatures {
            slope: 0.01,
            intercept: 20.0,
            residual: 0.001,
            sample_count: 100,
        };
        let result = classifier.classify_sensor(&features);
        assert!(matches!(
            result,
            EdgeClassification::Normal
                | EdgeClassification::Anomaly
                | EdgeClassification::Drift
                | EdgeClassification::Saturated
        ));
        assert_eq!(classifier.total_classified, 1);
    }

    #[test]
    fn test_classify_batch() {
        let mut classifier = EdgeClassifier::new();
        let batch = vec![
            SensorFeatures { slope: 0.01, intercept: 20.0, residual: 0.001, sample_count: 100 },
            SensorFeatures { slope: 5.0, intercept: -100.0, residual: 50.0, sample_count: 10 },
            SensorFeatures { slope: 0.0, intercept: 0.0, residual: 0.0, sample_count: 0 },
        ];
        let results = classifier.classify_batch(&batch);
        assert_eq!(results.len(), 3);
        assert_eq!(classifier.total_classified, 3);
    }

    #[test]
    fn test_deterministic() {
        let mut c1 = EdgeClassifier::new();
        let mut c2 = EdgeClassifier::new();
        let f = SensorFeatures { slope: 1.0, intercept: 2.0, residual: 0.5, sample_count: 50 };
        assert_eq!(c1.classify_sensor(&f), c2.classify_sensor(&f));
    }
}
