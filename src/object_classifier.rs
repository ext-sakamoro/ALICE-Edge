// SPDX-License-Identifier: MIT
//! 1.58-bit Ternary Object Classifier
//!
//! Ultra-lightweight MLP classifier using ALICE-ML ternary inference.
//! Input: 16-dimensional SDF feature vector
//! Architecture: 16 → 32 → 32 → N (3-layer MLP)
//! Model size: ~544 bytes (ternary weights)
//!
//! Author: Moroya Sakamoto

use alice_ml::{ternary_matvec_kernel, TernaryWeightKernel};

/// Default number of classification categories
pub const DEFAULT_NUM_CLASSES: usize = 8;

/// Input feature dimension (SDF descriptor)
pub const FEATURE_DIM: usize = 16;

/// Hidden layer dimension
pub const HIDDEN_DIM: usize = 32;

/// Object classification labels
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ObjectClass {
    Unknown = 0,
    Wall = 1,
    Floor = 2,
    Ceiling = 3,
    Furniture = 4,
    Person = 5,
    Vehicle = 6,
    Vegetation = 7,
}

impl ObjectClass {
    #[must_use]
    pub fn from_id(id: u8) -> Self {
        match id {
            1 => ObjectClass::Wall,
            2 => ObjectClass::Floor,
            3 => ObjectClass::Ceiling,
            4 => ObjectClass::Furniture,
            5 => ObjectClass::Person,
            6 => ObjectClass::Vehicle,
            7 => ObjectClass::Vegetation,
            _ => ObjectClass::Unknown,
        }
    }

    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            ObjectClass::Unknown => "unknown",
            ObjectClass::Wall => "wall",
            ObjectClass::Floor => "floor",
            ObjectClass::Ceiling => "ceiling",
            ObjectClass::Furniture => "furniture",
            ObjectClass::Person => "person",
            ObjectClass::Vehicle => "vehicle",
            ObjectClass::Vegetation => "vegetation",
        }
    }
}

/// SDF feature extractor — computes a 16-dimensional descriptor from SDF properties
#[derive(Debug, Clone)]
pub struct SdfFeatures {
    pub features: [f32; FEATURE_DIM],
}

impl SdfFeatures {
    /// Extract features from primitive fitting result
    #[inline(always)]
    #[must_use]
    pub fn from_primitive(
        kind: u8,
        params: &[f32],
        bounds_size: [f32; 3],
        point_count: usize,
    ) -> Self {
        let mut features = [0.0f32; FEATURE_DIM];

        // Feature 0-2: Bounding box dimensions (normalized)
        let max_dim = bounds_size[0]
            .max(bounds_size[1])
            .max(bounds_size[2])
            .max(1e-6);
        let inv_max = 1.0 / max_dim;
        features[0] = bounds_size[0] * inv_max;
        features[1] = bounds_size[1] * inv_max;
        features[2] = bounds_size[2] * inv_max;

        features[3] = bounds_size[1] / bounds_size[0].max(1e-6);

        if (kind as usize) < 4 {
            features[4 + kind as usize] = 1.0;
        }

        // Feature 8: Volume estimate
        features[8] = bounds_size[0] * bounds_size[1] * bounds_size[2];

        // Feature 9: Point density
        features[9] = point_count as f32 * (1.0 / features[8].max(1e-6));

        for (i, p) in params.iter().take(3).enumerate() {
            features[10 + i] = *p;
        }

        let min_dim = bounds_size[0].min(bounds_size[1]).min(bounds_size[2]);
        features[13] = min_dim * inv_max;

        // Feature 14: Elongation (max_dim / median_dim)
        let mut sorted = bounds_size;
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        features[14] = sorted[2] / sorted[1].max(1e-6);

        features[15] = (point_count as f32 + 1.0).ln();

        SdfFeatures { features }
    }

    #[inline(always)]
    #[must_use]
    pub fn from_svo_stats(
        node_count: u32,
        leaf_count: u32,
        depth: u32,
        bounds_size: [f32; 3],
    ) -> Self {
        let mut features = [0.0f32; FEATURE_DIM];

        let max_dim = bounds_size[0]
            .max(bounds_size[1])
            .max(bounds_size[2])
            .max(1e-6);
        let inv_max = 1.0 / max_dim;
        features[0] = bounds_size[0] * inv_max;
        features[1] = bounds_size[1] * inv_max;
        features[2] = bounds_size[2] * inv_max;

        features[3] = bounds_size[1] / bounds_size[0].max(1e-6);

        features[8] = node_count as f32;
        features[9] = leaf_count as f32 / node_count.max(1) as f32;
        features[10] = depth as f32;

        features[13] = bounds_size[0].min(bounds_size[1]).min(bounds_size[2]) * inv_max;

        let mut sorted = bounds_size;
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        features[14] = sorted[2] / sorted[1].max(1e-6);
        features[15] = (node_count as f32 + 1.0).ln();

        SdfFeatures { features }
    }
}

/// Ternary MLP classifier
///
/// Architecture: 16 → 32 → 32 → `num_classes`
/// Total weights: 16*32 + 32*32 + 32*8 = 1792 ternary values
/// Memory: 1792 / 4 = 448 bytes (packed) + ~96 bytes overhead = ~544 bytes
pub struct TernaryClassifier {
    layer1: TernaryWeightKernel, // 16 → 32
    layer2: TernaryWeightKernel, // 32 → 32
    layer3: TernaryWeightKernel, // 32 → num_classes
    num_classes: usize,
    /// save/load 用のオリジナル i8 `重み（from_weights` で構築しない場合は空）
    raw_w1: Vec<i8>,
    raw_w2: Vec<i8>,
    raw_w3: Vec<i8>,
}

impl TernaryClassifier {
    /// Maximum supported output classes (logits buffer size)
    pub const MAX_CLASSES: usize = 16;

    /// Create classifier with random weights (for initialization)
    /// In production, weights would be loaded from a trained model file.
    ///
    /// # Panics
    /// Panics if `num_classes` exceeds `MAX_CLASSES` (16).
    #[must_use]
    pub fn new(num_classes: usize) -> Self {
        assert!(
            num_classes <= Self::MAX_CLASSES,
            "num_classes ({}) exceeds MAX_CLASSES ({})",
            num_classes,
            Self::MAX_CLASSES
        );
        // Initialize with simple patterns (will be replaced by trained weights)
        let w1: Vec<i8> = (0..FEATURE_DIM * HIDDEN_DIM)
            .map(|i| (i % 3) as i8 - 1)
            .collect();
        let w2: Vec<i8> = (0..HIDDEN_DIM * HIDDEN_DIM)
            .map(|i| (i % 3) as i8 - 1)
            .collect();
        let w3: Vec<i8> = (0..HIDDEN_DIM * num_classes)
            .map(|i| (i % 3) as i8 - 1)
            .collect();

        Self {
            layer1: TernaryWeightKernel::from_ternary(&w1, HIDDEN_DIM, FEATURE_DIM),
            layer2: TernaryWeightKernel::from_ternary(&w2, HIDDEN_DIM, HIDDEN_DIM),
            layer3: TernaryWeightKernel::from_ternary(&w3, num_classes, HIDDEN_DIM),
            num_classes,
            raw_w1: w1,
            raw_w2: w2,
            raw_w3: w3,
        }
    }

    /// Load classifier from packed weight bytes
    ///
    /// # Panics
    /// Panics if `num_classes` exceeds `MAX_CLASSES` (16).
    #[must_use]
    pub fn from_weights(w1: &[i8], w2: &[i8], w3: &[i8], num_classes: usize) -> Self {
        assert!(
            num_classes <= Self::MAX_CLASSES,
            "num_classes ({}) exceeds MAX_CLASSES ({})",
            num_classes,
            Self::MAX_CLASSES
        );
        Self {
            layer1: TernaryWeightKernel::from_ternary(w1, HIDDEN_DIM, FEATURE_DIM),
            layer2: TernaryWeightKernel::from_ternary(w2, HIDDEN_DIM, HIDDEN_DIM),
            layer3: TernaryWeightKernel::from_ternary(w3, num_classes, HIDDEN_DIM),
            num_classes,
            raw_w1: w1.to_vec(),
            raw_w2: w2.to_vec(),
            raw_w3: w3.to_vec(),
        }
    }

    /// 重みをファイルに保存
    ///
    /// Format: [`num_classes:u32`][w1_len:u32][w1:bytes][w2_len:u32][w2:bytes][w3_len:u32][w3:bytes]
    ///
    /// # Errors
    ///
    /// Returns `Err(String)` if the file cannot be created or any weight data cannot be written.
    #[cfg(feature = "std")]
    pub fn save_weights(&self, path: &str) -> Result<(), String> {
        use std::io::Write;
        let mut f = std::fs::File::create(path).map_err(|e| format!("create: {e}"))?;
        f.write_all(&(self.num_classes as u32).to_le_bytes())
            .map_err(|e| format!("write: {e}"))?;
        for w in [&self.raw_w1, &self.raw_w2, &self.raw_w3] {
            f.write_all(&(w.len() as u32).to_le_bytes())
                .map_err(|e| format!("write: {e}"))?;
            // SAFETY: i8 と u8 は同サイズ・同アライメント。Vec<i8> のポインタと長さは有効。
            let bytes: &[u8] =
                unsafe { core::slice::from_raw_parts(w.as_ptr().cast::<u8>(), w.len()) };
            f.write_all(bytes).map_err(|e| format!("write: {e}"))?;
        }
        Ok(())
    }

    /// ファイルから重みを読み込んで分類器を構築
    ///
    /// # Errors
    ///
    /// Returns `Err(String)` if the file cannot be read, is too short, or contains truncated weight data.
    #[cfg(feature = "std")]
    pub fn load_weights(path: &str) -> Result<Self, String> {
        let data = std::fs::read(path).map_err(|e| format!("read: {e}"))?;
        if data.len() < 4 {
            return Err("file too short".into());
        }
        let mut offset = 0;
        let num_classes = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        offset += 4;

        let mut weights = Vec::new();
        for _ in 0..3 {
            if offset + 4 > data.len() {
                return Err("truncated weight header".into());
            }
            let len = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            offset += 4;
            if offset + len > data.len() {
                return Err("truncated weight data".into());
            }
            let w: Vec<i8> = data[offset..offset + len]
                .iter()
                .map(|&b| b as i8)
                .collect();
            weights.push(w);
            offset += len;
        }
        Ok(Self::from_weights(
            &weights[0],
            &weights[1],
            &weights[2],
            num_classes,
        ))
    }

    /// Classify an SDF feature vector
    ///
    /// Returns (`class_id`, confidence)
    /// All computation is zero-allocation (stack buffers only).
    #[inline(always)]
    #[must_use]
    pub fn classify(&self, features: &SdfFeatures) -> (ObjectClass, f32) {
        // Layer 1: 16 → 32 + ReLU
        let mut hidden1 = [0.0f32; HIDDEN_DIM];
        ternary_matvec_kernel(&features.features, &self.layer1, &mut hidden1);
        // Branchless ReLU
        for v in &mut hidden1 {
            *v = v.max(0.0);
        }

        // Layer 2: 32 → 32 + ReLU
        let mut hidden2 = [0.0f32; HIDDEN_DIM];
        ternary_matvec_kernel(&hidden1, &self.layer2, &mut hidden2);
        for v in &mut hidden2 {
            *v = v.max(0.0);
        }

        let mut logits_buf = [0.0f32; 16];
        let logits = &mut logits_buf[..self.num_classes];
        ternary_matvec_kernel(&hidden2, &self.layer3, logits);

        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_exp = 0.0f32;
        for v in logits.iter_mut() {
            *v = (*v - max_logit).exp();
            sum_exp += *v;
        }
        if sum_exp > 0.0 {
            let inv_sum = 1.0 / sum_exp;
            for v in logits.iter_mut() {
                *v *= inv_sum;
            }
        }

        let mut best_idx = 0usize;
        let mut best_conf = logits[0];
        for (i, &v) in logits.iter().enumerate().skip(1) {
            if v > best_conf {
                best_conf = v;
                best_idx = i;
            }
        }

        (ObjectClass::from_id(best_idx as u8), best_conf)
    }

    /// Memory footprint of all model weights in bytes
    #[must_use]
    pub fn memory_bytes(&self) -> usize {
        self.layer1.memory_bytes() + self.layer2.memory_bytes() + self.layer3.memory_bytes()
    }
}

impl crate::EdgeInference for TernaryClassifier {
    type Output = ObjectClass;

    fn infer(&mut self, features: &[f32]) -> (usize, f32) {
        let mut feat_arr = [0.0f32; FEATURE_DIM];
        let copy_len = features.len().min(FEATURE_DIM);
        feat_arr[..copy_len].copy_from_slice(&features[..copy_len]);
        let sdf_feat = SdfFeatures { features: feat_arr };
        let (class, confidence) = self.classify(&sdf_feat);
        (class as u8 as usize, confidence)
    }

    fn model_size_bytes(&self) -> usize {
        self.memory_bytes()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sdf_features_from_primitive() {
        let features = SdfFeatures::from_primitive(
            0, // sphere
            &[0.0, 0.0, 0.0, 1.0],
            [2.0, 2.0, 2.0],
            1000,
        );
        // Bounding box should be normalized to 1.0
        assert!((features.features[0] - 1.0).abs() < 1e-6);
        // Sphere one-hot should be set
        assert!((features.features[4] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_classifier_creation() {
        let classifier = TernaryClassifier::new(DEFAULT_NUM_CLASSES);
        // Model should be ~544 bytes
        assert!(classifier.memory_bytes() < 1000);
        assert!(classifier.memory_bytes() > 400);
    }

    #[test]
    fn test_classify_returns_valid() {
        let classifier = TernaryClassifier::new(DEFAULT_NUM_CLASSES);
        let features = SdfFeatures::from_primitive(
            1, // box
            &[0.0, 0.0, 0.0, 1.0, 2.0, 0.5],
            [2.0, 4.0, 1.0],
            5000,
        );

        let (class, confidence) = classifier.classify(&features);
        // Confidence should be valid probability
        assert!(confidence >= 0.0 && confidence <= 1.0);
        // Class should be a valid variant
        let _ = class.label();
    }

    #[test]
    fn test_object_class_labels() {
        assert_eq!(ObjectClass::Wall.label(), "wall");
        assert_eq!(ObjectClass::from_id(2).label(), "floor");
        assert_eq!(ObjectClass::from_id(255).label(), "unknown");
    }

    #[test]
    fn test_max_classes_boundary() {
        // MAX_CLASSES (16) はパニックしない
        let classifier = TernaryClassifier::new(TernaryClassifier::MAX_CLASSES);
        assert_eq!(classifier.num_classes, 16);
    }

    #[test]
    #[should_panic(expected = "exceeds MAX_CLASSES")]
    fn test_too_many_classes_panics() {
        let _ = TernaryClassifier::new(17);
    }

    #[test]
    fn test_save_load_weights_roundtrip() {
        let classifier = TernaryClassifier::new(DEFAULT_NUM_CLASSES);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("weights.bin");
        let path_str = path.to_str().unwrap();

        classifier.save_weights(path_str).expect("save failed");
        let loaded = TernaryClassifier::load_weights(path_str).expect("load failed");

        assert_eq!(loaded.num_classes, DEFAULT_NUM_CLASSES);
        assert_eq!(loaded.memory_bytes(), classifier.memory_bytes());

        // 同じ入力で同じ結果を返すことを確認
        let features = SdfFeatures::from_primitive(1, &[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], 500);
        let (c1, conf1) = classifier.classify(&features);
        let (c2, conf2) = loaded.classify(&features);
        assert_eq!(c1, c2);
        assert!((conf1 - conf2).abs() < 1e-6);
    }

    #[test]
    fn test_edge_inference_trait() {
        let mut classifier = TernaryClassifier::new(DEFAULT_NUM_CLASSES);
        use crate::EdgeInference;
        let features = vec![1.0f32; FEATURE_DIM];
        let (class_id, confidence) = classifier.infer(&features);
        assert!(class_id < DEFAULT_NUM_CLASSES);
        assert!(confidence >= 0.0 && confidence <= 1.0);
        assert!(classifier.model_size_bytes() > 0);
    }
}
