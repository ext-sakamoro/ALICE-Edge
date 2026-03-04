// SPDX-License-Identifier: MIT
//! ALICE-Edge × ALICE-ML bridge
//!
//! Edge sensor data → ternary ML inference for anomaly classification.
//!
//! Architecture: 4 → 16 → 4 (2-layer MLP, ~80 bytes)
//!
//! Author: Moroya Sakamoto

use alice_ml::{ternary_matvec_kernel, TernaryWeightKernel};

/// Input feature dimension (slope, intercept, residual, `sample_count`)
const INPUT_DIM: usize = 4;

/// Hidden layer dimension
const HIDDEN_DIM: usize = 16;

/// Number of output classes
const NUM_CLASSES: usize = 4;

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
///
/// 2-layer MLP using ALICE-ML bit-parallel ternary inference.
/// All computation is zero-allocation (stack buffers only).
pub struct EdgeClassifier {
    /// Input → hidden weights (4 → 16)
    w_hidden: TernaryWeightKernel,
    /// Hidden → output weights (16 → 4)
    w_output: TernaryWeightKernel,
    /// Hidden bias
    bias_hidden: [f32; HIDDEN_DIM],
    /// Output bias
    bias_output: [f32; NUM_CLASSES],
    /// Classification count
    pub total_classified: u64,
}

impl EdgeClassifier {
    /// Create classifier with deterministic pseudo-random ternary weights.
    ///
    /// Model size: ~80 bytes (ternary weights pack 32 values per u32).
    #[must_use]
    pub fn new() -> Self {
        // 決定論的初期化: ハッシュベースの三値重み
        let mut w_hidden_i8 = vec![0i8; HIDDEN_DIM * INPUT_DIM];
        for (i, w) in w_hidden_i8.iter_mut().enumerate() {
            let hash = ((i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)) >> 62;
            *w = match hash {
                0 => -1,
                1 => 0,
                _ => 1,
            };
        }
        let mut w_output_i8 = vec![0i8; NUM_CLASSES * HIDDEN_DIM];
        for (i, w) in w_output_i8.iter_mut().enumerate() {
            let hash = ((i as u64 + 100).wrapping_mul(0x517C_C1B7_2722_0A95)) >> 62;
            *w = match hash {
                0 => -1,
                1 => 0,
                _ => 1,
            };
        }
        Self {
            w_hidden: TernaryWeightKernel::from_ternary(&w_hidden_i8, HIDDEN_DIM, INPUT_DIM),
            w_output: TernaryWeightKernel::from_ternary(&w_output_i8, NUM_CLASSES, HIDDEN_DIM),
            bias_hidden: [0.0; HIDDEN_DIM],
            bias_output: [0.0; NUM_CLASSES],
            total_classified: 0,
        }
    }

    /// Classify a single sensor reading
    #[must_use]
    pub fn classify_sensor(&mut self, features: &SensorFeatures) -> EdgeClassification {
        let input = [
            features.slope,
            features.intercept,
            features.residual,
            features.sample_count as f32 * (1.0 / 1000.0),
        ];

        // Hidden layer: ternary matvec + bias + ReLU
        let mut hidden = [0.0f32; HIDDEN_DIM];
        ternary_matvec_kernel(&input, &self.w_hidden, &mut hidden);
        for (h, b) in hidden.iter_mut().zip(self.bias_hidden.iter()) {
            *h = (*h + b).max(0.0); // ReLU
        }

        // Output layer: ternary matvec + bias
        let mut output = [0.0f32; NUM_CLASSES];
        ternary_matvec_kernel(&hidden, &self.w_output, &mut output);
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

    /// Memory footprint of model weights in bytes
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.w_hidden.memory_bytes() + self.w_output.memory_bytes()
    }
}

impl Default for EdgeClassifier {
    fn default() -> Self {
        Self::new()
    }
}

impl crate::EdgeInference for EdgeClassifier {
    type Output = EdgeClassification;

    fn infer(&mut self, features: &[f32]) -> (usize, f32) {
        let sf = SensorFeatures {
            slope: features.first().copied().unwrap_or(0.0),
            intercept: features.get(1).copied().unwrap_or(0.0),
            residual: features.get(2).copied().unwrap_or(0.0),
            sample_count: features.get(3).copied().map_or(0, |v| v as u16),
        };
        let class = self.classify_sensor(&sf);
        (class as usize, 1.0) // EdgeClassifier は信頼度を返さないため 1.0
    }

    fn model_size_bytes(&self) -> usize {
        self.memory_bytes()
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
            SensorFeatures {
                slope: 0.01,
                intercept: 20.0,
                residual: 0.001,
                sample_count: 100,
            },
            SensorFeatures {
                slope: 5.0,
                intercept: -100.0,
                residual: 50.0,
                sample_count: 10,
            },
            SensorFeatures {
                slope: 0.0,
                intercept: 0.0,
                residual: 0.0,
                sample_count: 0,
            },
        ];
        let results = classifier.classify_batch(&batch);
        assert_eq!(results.len(), 3);
        assert_eq!(classifier.total_classified, 3);
    }

    #[test]
    fn test_deterministic() {
        let mut c1 = EdgeClassifier::new();
        let mut c2 = EdgeClassifier::new();
        let f = SensorFeatures {
            slope: 1.0,
            intercept: 2.0,
            residual: 0.5,
            sample_count: 50,
        };
        assert_eq!(c1.classify_sensor(&f), c2.classify_sensor(&f));
    }

    #[test]
    fn test_edge_inference_trait() {
        let mut classifier = EdgeClassifier::new();
        use crate::EdgeInference;
        let features = vec![0.01, 20.0, 0.001, 100.0];
        let (class_id, confidence) = classifier.infer(&features);
        assert!(class_id < 4);
        assert!((confidence - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_model_size_bytes() {
        let classifier = EdgeClassifier::new();
        let size = classifier.memory_bytes();
        assert!(size > 0);
        assert!(size < 1000); // 極めて小さいモデル
    }

    #[test]
    fn test_extreme_values_stability() {
        let mut classifier = EdgeClassifier::new();
        let extreme = SensorFeatures {
            slope: f32::MAX,
            intercept: f32::MIN,
            residual: f32::INFINITY,
            sample_count: u16::MAX,
        };
        // パニックしないことを確認
        let result = classifier.classify_sensor(&extreme);
        assert!(matches!(
            result,
            EdgeClassification::Normal
                | EdgeClassification::Anomaly
                | EdgeClassification::Drift
                | EdgeClassification::Saturated
        ));
    }

    #[test]
    fn test_zero_input_stability() {
        let mut classifier = EdgeClassifier::new();
        let zero = SensorFeatures {
            slope: 0.0,
            intercept: 0.0,
            residual: 0.0,
            sample_count: 0,
        };
        let result = classifier.classify_sensor(&zero);
        assert!(matches!(
            result,
            EdgeClassification::Normal
                | EdgeClassification::Anomaly
                | EdgeClassification::Drift
                | EdgeClassification::Saturated
        ));
    }

    #[test]
    fn test_classify_batch_empty() {
        let mut classifier = EdgeClassifier::new();
        let results = classifier.classify_batch(&[]);
        assert!(results.is_empty());
        assert_eq!(classifier.total_classified, 0);
    }

    #[test]
    fn test_infer_short_features() {
        // INPUT_DIM より短い入力でもパニックしないことを確認
        let mut classifier = EdgeClassifier::new();
        use crate::EdgeInference;
        let short = vec![1.0, 2.0]; // 4 要素未満
        let (class_id, _) = classifier.infer(&short);
        assert!(class_id < 4);
    }
}
