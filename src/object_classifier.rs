// SPDX-License-Identifier: LicenseRef-Proprietary
//! 1.58-bit Ternary Object Classifier
//!
//! Ultra-lightweight MLP classifier using ALICE-ML ternary inference.
//! Input: 16-dimensional SDF feature vector
//! Architecture: 16 → 32 → 32 → N (3-layer MLP)
//! Model size: ~544 bytes (ternary weights)
//!
//! Author: Moroya Sakamoto

pub use crate::semantics::IdStatus;
#[cfg(feature = "std")]
use crate::semantics::SemanticsMismatch;
use alice_ml::{ternary_matvec_kernel, TernaryWeightKernel};

/// The first four bytes of a weights file in the stamped format.
pub const WEIGHTS_MAGIC: [u8; 4] = *b"AEOW";

/// The weights format version this release writes and reads.
pub const WEIGHTS_FORMAT_VERSION: u16 = 1;

/// Why a weights file could not be loaded or saved.
#[cfg(feature = "std")]
#[derive(Debug)]
pub enum WeightsError {
    /// The file could not be read or written.
    Io(std::io::Error),
    /// The file is in the earlier format without a header, so it carries no
    /// identifier; see [`TernaryClassifier::load_weights_unverified_legacy`].
    LegacyNoId,
    /// [`TernaryClassifier::load_weights_unverified_legacy`] was given a file
    /// in the stamped format.
    HasHeader,
    /// A format version this release does not read.
    UnsupportedVersion(u16),
    /// The stored identifier is neither this build's nor a verified earlier one.
    Semantics(SemanticsMismatch),
    /// The file ends inside the named part, or the class count is out of range.
    Invalid(&'static str),
}

#[cfg(feature = "std")]
impl core::fmt::Display for WeightsError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "weights file: {e}"),
            Self::LegacyNoId => write!(f, "weights file has no header and no semantics identifier"),
            Self::HasHeader => write!(f, "weights file has a header; load it with load_weights"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported weights format version {v}"),
            Self::Semantics(m) => write!(f, "weights file: {m}"),
            Self::Invalid(part) => write!(f, "weights file truncated or invalid: {part}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for WeightsError {}

/// A value loaded from bytes that carry no semantics identifier.
///
/// Nothing ties it to the arithmetic it was produced under; take it out with
/// [`Unverified::into_unverified`] only after checking its provenance.
#[must_use = "an unverified value has to be taken out explicitly"]
#[derive(Debug)]
pub struct Unverified<T> {
    value: T,
}

impl<T> Unverified<T> {
    /// The value, accepted without an identifier check.
    pub fn into_unverified(self) -> T {
        self.value
    }
}

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
        let max_dim = alice_det_math::max(
            alice_det_math::max(
                alice_det_math::max(bounds_size[0], bounds_size[1]),
                bounds_size[2],
            ),
            1e-6,
        );
        let inv_max = 1.0 / max_dim;
        features[0] = bounds_size[0] * inv_max;
        features[1] = bounds_size[1] * inv_max;
        features[2] = bounds_size[2] * inv_max;

        features[3] = bounds_size[1] / alice_det_math::max(bounds_size[0], 1e-6);

        if (kind as usize) < 4 {
            features[4 + kind as usize] = 1.0;
        }

        // Feature 8: Volume estimate
        features[8] = bounds_size[0] * bounds_size[1] * bounds_size[2];

        // Feature 9: Point density
        features[9] = point_count as f32 * (1.0 / alice_det_math::max(features[8], 1e-6));

        for (i, p) in params.iter().take(3).enumerate() {
            features[10 + i] = *p;
        }

        let min_dim = alice_det_math::min(
            alice_det_math::min(bounds_size[0], bounds_size[1]),
            bounds_size[2],
        );
        features[13] = min_dim * inv_max;

        // Feature 14: Elongation (max_dim / median_dim)
        let mut sorted = bounds_size;
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        features[14] = sorted[2] / alice_det_math::max(sorted[1], 1e-6);

        features[15] = alice_det_math::ln(point_count as f32 + 1.0);

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

        let max_dim = alice_det_math::max(
            alice_det_math::max(
                alice_det_math::max(bounds_size[0], bounds_size[1]),
                bounds_size[2],
            ),
            1e-6,
        );
        let inv_max = 1.0 / max_dim;
        features[0] = bounds_size[0] * inv_max;
        features[1] = bounds_size[1] * inv_max;
        features[2] = bounds_size[2] * inv_max;

        features[3] = bounds_size[1] / alice_det_math::max(bounds_size[0], 1e-6);

        features[8] = node_count as f32;
        features[9] = leaf_count as f32 / node_count.max(1) as f32;
        features[10] = depth as f32;

        features[13] = alice_det_math::min(
            alice_det_math::min(bounds_size[0], bounds_size[1]),
            bounds_size[2],
        ) * inv_max;

        let mut sorted = bounds_size;
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        features[14] = sorted[2] / alice_det_math::max(sorted[1], 1e-6);
        features[15] = alice_det_math::ln(node_count as f32 + 1.0);

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
    ///
    /// The two bounds are checked separately so that the message says which
    /// one was broken:
    ///
    /// - `num_classes` is 0: a classifier with no output class has no logit
    ///   to report, so the construction is refused here rather than failing
    ///   inside [`TernaryClassifier::classify`]
    /// - `num_classes` exceeds [`TernaryClassifier::MAX_CLASSES`] (16): more
    ///   classes than the logits buffer holds
    #[must_use]
    pub fn new(num_classes: usize) -> Self {
        assert!(num_classes >= 1, "num_classes must be at least 1, got 0");
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
    /// The three layers are dense ternary matrices, so their lengths are
    /// fixed by the architecture: `w1` is `HIDDEN_DIM · FEATURE_DIM` (512),
    /// `w2` is `HIDDEN_DIM · HIDDEN_DIM` (1024) and `w3` is
    /// `num_classes · HIDDEN_DIM`. Values outside `-1 ..= 1` are quantised to
    /// the nearest ternary level.
    ///
    /// # Panics
    ///
    /// Each condition is checked separately so that the message says which
    /// one was broken:
    ///
    /// - `num_classes` is 0: a classifier with no output class has no logit
    ///   to report
    /// - `num_classes` exceeds [`TernaryClassifier::MAX_CLASSES`] (16)
    /// - a layer's length is not the product above (checked here, with the
    ///   layer named, rather than left to the kernel constructor)
    #[must_use]
    pub fn from_weights(w1: &[i8], w2: &[i8], w3: &[i8], num_classes: usize) -> Self {
        assert!(num_classes >= 1, "num_classes must be at least 1, got 0");
        assert!(
            num_classes <= Self::MAX_CLASSES,
            "num_classes ({}) exceeds MAX_CLASSES ({})",
            num_classes,
            Self::MAX_CLASSES
        );
        assert!(
            w1.len() == HIDDEN_DIM * FEATURE_DIM,
            "w1 has {} values, expected HIDDEN_DIM * FEATURE_DIM = {}",
            w1.len(),
            HIDDEN_DIM * FEATURE_DIM
        );
        assert!(
            w2.len() == HIDDEN_DIM * HIDDEN_DIM,
            "w2 has {} values, expected HIDDEN_DIM * HIDDEN_DIM = {}",
            w2.len(),
            HIDDEN_DIM * HIDDEN_DIM
        );
        assert!(
            w3.len() == num_classes * HIDDEN_DIM,
            "w3 has {} values, expected num_classes * HIDDEN_DIM = {}",
            w3.len(),
            num_classes * HIDDEN_DIM
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

    /// Saves the weights in the stamped format (ALICE-Law G8).
    ///
    /// Format version 1, integers little-endian: magic `AEOW`, `u16` version,
    /// this build's 32-byte [`EDGE_SEMANTICS_ID`](crate::EDGE_SEMANTICS_ID),
    /// then the body `[num_classes:u32][w1_len:u32][w1][w2_len:u32][w2][w3_len:u32][w3]`
    /// (each weight one `i8` byte).
    ///
    /// # Errors
    ///
    /// [`WeightsError::Io`] if the file cannot be created or written.
    #[cfg(feature = "std")]
    pub fn save_weights(&self, path: &str) -> Result<(), WeightsError> {
        std::fs::write(path, self.weights_bytes_with_id(&crate::EDGE_SEMANTICS_ID))
            .map_err(WeightsError::Io)
    }

    #[cfg(feature = "std")]
    fn weights_bytes_with_id(&self, id: &[u8; 32]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&WEIGHTS_MAGIC);
        out.extend_from_slice(&WEIGHTS_FORMAT_VERSION.to_le_bytes());
        out.extend_from_slice(id);
        out.extend_from_slice(&(self.num_classes as u32).to_le_bytes());
        for w in [&self.raw_w1, &self.raw_w2, &self.raw_w3] {
            out.extend_from_slice(&(w.len() as u32).to_le_bytes());
            out.extend(w.iter().map(|&x| x as u8));
        }
        out
    }

    /// Loads weights written by [`TernaryClassifier::save_weights`], checking the
    /// stored identifier: this build's gives [`IdStatus::Current`], one in
    /// [`PREVIOUS_EDGE_SEMANTICS_IDS`](crate::PREVIOUS_EDGE_SEMANTICS_IDS) gives
    /// [`IdStatus::VerifiedPrevious`].
    ///
    /// # Errors
    ///
    /// [`WeightsError::Semantics`] (both identifiers) for any other identifier,
    /// [`WeightsError::LegacyNoId`] for a file in the earlier format without a
    /// header, and the I/O, version and truncation errors.
    #[cfg(feature = "std")]
    pub fn load_weights(path: &str) -> Result<(Self, IdStatus), WeightsError> {
        let data = std::fs::read(path).map_err(WeightsError::Io)?;
        Self::from_weights_bytes(&data, crate::semantics::check_semantics)
    }

    #[cfg(feature = "std")]
    fn from_weights_bytes(
        data: &[u8],
        check: impl FnOnce(&[u8; 32]) -> Result<IdStatus, SemanticsMismatch>,
    ) -> Result<(Self, IdStatus), WeightsError> {
        if data.len() < 4 || data[..4] != WEIGHTS_MAGIC {
            return Err(WeightsError::LegacyNoId);
        }
        if data.len() < 6 + 32 {
            return Err(WeightsError::Invalid("header"));
        }
        let version = u16::from_le_bytes([data[4], data[5]]);
        if version != WEIGHTS_FORMAT_VERSION {
            return Err(WeightsError::UnsupportedVersion(version));
        }
        let mut id = [0u8; 32];
        id.copy_from_slice(&data[6..38]);
        let status = check(&id).map_err(WeightsError::Semantics)?;
        Ok((Self::parse_body(&data[38..])?, status))
    }

    /// Loads a weights file in the earlier, headerless format (no identifier).
    ///
    /// The file says nothing about the arithmetic its weights were trained and
    /// stored under, so the result is wrapped in [`Unverified`]: the caller has
    /// to take the value out explicitly, having checked its provenance some
    /// other way, and should re-save it with [`TernaryClassifier::save_weights`].
    ///
    /// # Errors
    ///
    /// [`WeightsError::HasHeader`] for a file in the stamped format (use
    /// [`TernaryClassifier::load_weights`]), and the I/O and truncation errors.
    #[cfg(feature = "std")]
    pub fn load_weights_unverified_legacy(path: &str) -> Result<Unverified<Self>, WeightsError> {
        let data = std::fs::read(path).map_err(WeightsError::Io)?;
        if data.len() >= 4 && data[..4] == WEIGHTS_MAGIC {
            return Err(WeightsError::HasHeader);
        }
        Ok(Unverified {
            value: Self::parse_body(&data)?,
        })
    }

    #[cfg(feature = "std")]
    fn parse_body(data: &[u8]) -> Result<Self, WeightsError> {
        if data.len() < 4 {
            return Err(WeightsError::Invalid("class count"));
        }
        let mut offset = 0;
        let num_classes = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
        offset += 4;
        if num_classes == 0 || num_classes > Self::MAX_CLASSES {
            return Err(WeightsError::Invalid("class count out of range"));
        }

        let mut weights = Vec::new();
        for _ in 0..3 {
            if offset + 4 > data.len() {
                return Err(WeightsError::Invalid("weight header"));
            }
            let len = u32::from_le_bytes([
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ]) as usize;
            offset += 4;
            if offset + len > data.len() {
                return Err(WeightsError::Invalid("weight data"));
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
            *v = alice_det_math::max(*v, 0.0);
        }

        // Layer 2: 32 → 32 + ReLU
        let mut hidden2 = [0.0f32; HIDDEN_DIM];
        ternary_matvec_kernel(&hidden1, &self.layer2, &mut hidden2);
        for v in &mut hidden2 {
            *v = alice_det_math::max(*v, 0.0);
        }

        let mut logits_buf = [0.0f32; 16];
        let logits = &mut logits_buf[..self.num_classes];
        ternary_matvec_kernel(&hidden2, &self.layer3, logits);

        let max_logit = logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, alice_det_math::max);
        let mut sum_exp = 0.0f32;
        for v in logits.iter_mut() {
            *v = alice_det_math::exp(*v - max_logit);
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
        assert!((0.0..=1.0).contains(&confidence));
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

    /// Weights stamped with another identifier are refused naming both; an
    /// identifier in the previous list is accepted with that status (ALICE-Law G8).
    #[cfg(feature = "std")]
    #[test]
    fn weights_with_another_identifier_are_refused_naming_both() {
        let c = TernaryClassifier::new(DEFAULT_NUM_CLASSES);
        let mut foreign = crate::EDGE_SEMANTICS_ID;
        foreign[3] ^= 0x40;
        let bytes = c.weights_bytes_with_id(&foreign);
        match TernaryClassifier::from_weights_bytes(&bytes, crate::semantics::check_semantics) {
            Err(WeightsError::Semantics(m)) => {
                assert_eq!(m.stored, foreign);
                assert_eq!(m.expected, crate::EDGE_SEMANTICS_ID);
                let hex = |b: &[u8]| b.iter().map(|x| format!("{x:02x}")).collect::<String>();
                let text = WeightsError::Semantics(m).to_string();
                assert!(
                    text.contains(&hex(&foreign)) && text.contains(&hex(&crate::EDGE_SEMANTICS_ID))
                );
            }
            Err(other) => panic!("expected a semantics mismatch, got {other:?}"),
            Ok(_) => panic!("a foreign identifier was accepted"),
        }
        let (_, status) = TernaryClassifier::from_weights_bytes(&bytes, |id| {
            crate::semantics::check_semantics_against(id, &crate::EDGE_SEMANTICS_ID, &[foreign])
        })
        .unwrap();
        assert_eq!(status, IdStatus::VerifiedPrevious(foreign));
    }

    /// A headerless file (the earlier format) is refused by `load_weights` and
    /// read only through the explicitly unverified loader, which in turn refuses
    /// a stamped file.
    #[cfg(feature = "std")]
    #[test]
    fn legacy_weights_need_the_unverified_loader() {
        let c = TernaryClassifier::new(DEFAULT_NUM_CLASSES);
        let dir = tempfile::tempdir().unwrap();
        let legacy = dir.path().join("legacy.bin");
        let stamped = dir.path().join("stamped.bin");
        std::fs::write(
            &legacy,
            &c.weights_bytes_with_id(&crate::EDGE_SEMANTICS_ID)[38..],
        )
        .unwrap();
        c.save_weights(stamped.to_str().unwrap()).unwrap();
        assert!(matches!(
            TernaryClassifier::load_weights(legacy.to_str().unwrap()),
            Err(WeightsError::LegacyNoId)
        ));
        let old = TernaryClassifier::load_weights_unverified_legacy(legacy.to_str().unwrap())
            .unwrap()
            .into_unverified();
        assert_eq!(old.num_classes, DEFAULT_NUM_CLASSES);
        assert!(matches!(
            TernaryClassifier::load_weights_unverified_legacy(stamped.to_str().unwrap()),
            Err(WeightsError::HasHeader)
        ));
    }

    #[cfg(feature = "std")]
    #[test]
    fn weights_header_and_body_errors() {
        let c = TernaryClassifier::new(DEFAULT_NUM_CLASSES);
        let bytes = c.weights_bytes_with_id(&crate::EDGE_SEMANTICS_ID);
        let check = crate::semantics::check_semantics;
        let mut v2 = bytes.clone();
        v2[4] = 2;
        assert!(matches!(
            TernaryClassifier::from_weights_bytes(&v2, check),
            Err(WeightsError::UnsupportedVersion(2))
        ));
        assert!(matches!(
            TernaryClassifier::from_weights_bytes(&bytes[..20], check),
            Err(WeightsError::Invalid("header"))
        ));
        assert!(matches!(
            TernaryClassifier::from_weights_bytes(&bytes[..bytes.len() - 1], check),
            Err(WeightsError::Invalid("weight data"))
        ));
        let mut zero = bytes.clone();
        zero[38..42].copy_from_slice(&0u32.to_le_bytes());
        assert!(matches!(
            TernaryClassifier::from_weights_bytes(&zero, check),
            Err(WeightsError::Invalid("class count out of range"))
        ));
    }

    #[test]
    fn test_save_load_weights_roundtrip() {
        let classifier = TernaryClassifier::new(DEFAULT_NUM_CLASSES);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("weights.bin");
        let path_str = path.to_str().unwrap();

        classifier.save_weights(path_str).expect("save failed");
        let (loaded, status) = TernaryClassifier::load_weights(path_str).expect("load failed");
        assert_eq!(status, IdStatus::Current);

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
        assert!((0.0..=1.0).contains(&confidence));
        assert!(classifier.model_size_bytes() > 0);
    }
}
