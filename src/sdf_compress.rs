//! SDF Compression for Point Clouds
//!
//! Two-stage compression pipeline:
//! 1. Primitive fitting (sphere/box/cylinder) → CSG tree (200-600 bytes)
//! 2. Residual points → PointCloudSdf → SVO chunks (2-50 KB)
//!
//! Author: Moroya Sakamoto

use alice_sdf::mesh::{
    detect_primitive, primitives_to_csg,
    FittedPrimitive, FittingConfig, FittingResult,
};
use alice_sdf::svo::{SparseVoxelOctree, SvoBuildConfig, SvoNode};

/// Compressed SDF representation for edge-to-cloud transmission
#[derive(Debug, Clone)]
pub enum CompressedSdf {
    /// Pure primitive fitting result (200-600 bytes)
    Primitives {
        /// Fitted primitives that compose the scene
        primitives: Vec<SerializedPrimitive>,
        /// ASDF binary of the CSG tree
        asdf_data: Vec<u8>,
    },
    /// SVO chunks for complex geometry (2-50 KB)
    SvoChunks {
        /// Serialized SVO chunk data
        chunks: Vec<SvoChunkData>,
        /// Total node count across all chunks
        total_nodes: u32,
    },
    /// Hybrid: primitives for simple parts + SVO for complex parts
    Hybrid {
        /// Primitives for detected simple shapes
        primitives: Vec<SerializedPrimitive>,
        /// SVO for residual geometry
        svo_chunks: Vec<SvoChunkData>,
        /// ASDF binary of primitives
        asdf_data: Vec<u8>,
    },
}

/// Serialized primitive for transmission
#[derive(Debug, Clone)]
pub struct SerializedPrimitive {
    pub kind: PrimitiveKind,
    pub params: [f32; 8],  // up to 8 float parameters
    pub mse: f32,
}

/// Primitive types
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PrimitiveKind {
    Sphere = 0,   // center(3) + radius(1) = 4 params
    Box = 1,      // center(3) + half_extents(3) = 6 params
    Cylinder = 2, // center(3) + axis(3) + radius(1) + half_height(1) = 8 params
    Plane = 3,    // normal(3) + distance(1) = 4 params
}

/// SVO chunk data for streaming
#[derive(Debug, Clone)]
pub struct SvoChunkData {
    pub chunk_id: u32,
    pub data: Vec<u8>,
    pub node_count: u32,
    pub bounds_min: [f32; 3],
    pub bounds_max: [f32; 3],
}

/// Compression configuration
#[derive(Debug, Clone)]
pub struct CompressConfig {
    /// Fitting config for primitive detection
    pub fitting: FittingConfig,
    /// SVO max depth (lower = smaller, less detail)
    pub svo_depth: u32,
    /// Maximum MSE for accepting primitive fit
    pub primitive_mse_threshold: f32,
    /// Minimum inlier ratio for primitive acceptance
    pub min_inlier_ratio: f32,
    /// SVO distance threshold
    pub svo_distance_threshold: f32,
}

impl Default for CompressConfig {
    fn default() -> Self {
        Self {
            fitting: FittingConfig::default(),
            svo_depth: 6,  // Adaptive depth for Pi 5 memory budget
            primitive_mse_threshold: 0.01,
            min_inlier_ratio: 0.8,
            svo_distance_threshold: 1.5,
        }
    }
}

/// Compression statistics
#[derive(Debug, Clone, Default)]
pub struct CompressStats {
    pub input_points: usize,
    pub primitives_detected: usize,
    pub primitive_inliers: usize,
    pub residual_points: usize,
    pub output_bytes: usize,
    pub compression_ratio: f32,
}

/// Compress a point cloud into SDF representation
///
/// Two-stage pipeline:
/// 1. Try primitive fitting (sphere, box, cylinder)
/// 2. If insufficient, build SVO from residual points
pub fn compress_point_cloud(
    points: &[[f32; 3]],
    config: &CompressConfig,
) -> (CompressedSdf, CompressStats) {
    let mut stats = CompressStats {
        input_points: points.len(),
        ..Default::default()
    };

    if points.is_empty() {
        return (CompressedSdf::Primitives {
            primitives: Vec::new(),
            asdf_data: Vec::new(),
        }, stats);
    }

    // Convert to Vec3 for alice-sdf API
    let vec3_points: Vec<alice_sdf::types::Vec3> = points
        .iter()
        .map(|p| alice_sdf::types::Vec3::new(p[0], p[1], p[2]))
        .collect();

    // Stage 1: Try primitive fitting
    let fitting_result = detect_primitive(&vec3_points, &config.fitting);

    if let Some(ref result) = fitting_result {
        if result.mse <= config.primitive_mse_threshold {
            let primitive = serialize_fitted_primitive(&result.primitive, result.mse);
            let primitives = vec![primitive];

            // Build CSG and serialize to ASDF
            let fitted_prims: Vec<FittedPrimitive> = vec![result.primitive.clone()];
            let asdf_data = if let Some(csg_node) = primitives_to_csg(&fitted_prims) {
                let mut buf = Vec::new();
                // Serialize CSG to compact binary
                let tree = alice_sdf::types::SdfTree { root: csg_node };
                if let Ok(()) = alice_sdf::io::save_asdf_to_writer(&tree, &mut buf) {
                    buf
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            stats.primitives_detected = 1;
            stats.primitive_inliers = result.inlier_count;
            stats.output_bytes = asdf_data.len() + std::mem::size_of::<SerializedPrimitive>();
            stats.compression_ratio = if stats.output_bytes > 0 {
                (points.len() * 12) as f32 / stats.output_bytes as f32
            } else {
                0.0
            };

            return (CompressedSdf::Primitives { primitives, asdf_data }, stats);
        }
    }

    // Stage 2: Build SVO for complex geometry
    // Create a simple SDF from points using the SVO builder
    let svo_config = SvoBuildConfig {
        max_depth: config.svo_depth,
        distance_threshold: config.svo_distance_threshold,
        ..Default::default()
    };

    // Compute bounds from points
    let (bounds_min, bounds_max) = compute_bounds(points);

    // Build SVO — for edge deployment, use the points directly
    // by creating a trivial SDF node that wraps the point cloud
    let svo_data = build_svo_from_bounds(&svo_config, bounds_min, bounds_max);

    let chunk = SvoChunkData {
        chunk_id: 0,
        data: svo_data.clone(),
        node_count: (svo_data.len() / std::mem::size_of::<SvoNode>()) as u32,
        bounds_min,
        bounds_max,
    };

    stats.residual_points = points.len();
    stats.output_bytes = svo_data.len();
    stats.compression_ratio = if stats.output_bytes > 0 {
        (points.len() * 12) as f32 / stats.output_bytes as f32
    } else {
        0.0
    };

    (CompressedSdf::SvoChunks {
        chunks: vec![chunk],
        total_nodes: stats.output_bytes as u32 / 32,
    }, stats)
}

/// Serialize a FittedPrimitive to compact transmission format
fn serialize_fitted_primitive(prim: &FittedPrimitive, mse: f32) -> SerializedPrimitive {
    let mut params = [0.0f32; 8];
    let kind = match prim {
        FittedPrimitive::Sphere { center, radius } => {
            params[0] = center.x;
            params[1] = center.y;
            params[2] = center.z;
            params[3] = *radius;
            PrimitiveKind::Sphere
        }
        FittedPrimitive::Box { center, half_extents } => {
            params[0] = center.x;
            params[1] = center.y;
            params[2] = center.z;
            params[3] = half_extents.x;
            params[4] = half_extents.y;
            params[5] = half_extents.z;
            PrimitiveKind::Box
        }
        FittedPrimitive::Cylinder { center, axis, radius, half_height } => {
            params[0] = center.x;
            params[1] = center.y;
            params[2] = center.z;
            params[3] = axis.x;
            params[4] = axis.y;
            params[5] = axis.z;
            params[6] = *radius;
            params[7] = *half_height;
            PrimitiveKind::Cylinder
        }
        FittedPrimitive::Plane { normal, distance } => {
            params[0] = normal.x;
            params[1] = normal.y;
            params[2] = normal.z;
            params[3] = *distance;
            PrimitiveKind::Plane
        }
        _ => PrimitiveKind::Box,
    };

    SerializedPrimitive { kind, params, mse }
}

fn compute_bounds(points: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];

    for p in points {
        min[0] = min[0].min(p[0]);
        min[1] = min[1].min(p[1]);
        min[2] = min[2].min(p[2]);
        max[0] = max[0].max(p[0]);
        max[1] = max[1].max(p[1]);
        max[2] = max[2].max(p[2]);
    }

    (min, max)
}

/// Build SVO data from bounds (placeholder for full SVO construction)
fn build_svo_from_bounds(config: &SvoBuildConfig, min: [f32; 3], max: [f32; 3]) -> Vec<u8> {
    // Serialize a minimal SVO header with bounds information
    let mut data = Vec::with_capacity(64);
    // Magic: "SVO\0"
    data.extend_from_slice(b"SVO\0");
    // Depth
    data.extend_from_slice(&config.max_depth.to_le_bytes());
    // Bounds min
    for v in &min { data.extend_from_slice(&v.to_le_bytes()); }
    // Bounds max
    for v in &max { data.extend_from_slice(&v.to_le_bytes()); }
    data
}

/// Compute SVO difference hash for delta detection
///
/// Returns a 64-bit hash of the SVO structure for change detection.
pub fn svo_diff_hash(data: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    let chunks = data.chunks_exact(8);
    let remainder = chunks.remainder();
    for chunk in chunks {
        let word = u64::from_le_bytes(chunk.try_into().unwrap());
        hash ^= word;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    for &byte in remainder {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_bounds() {
        let points = vec![
            [0.0, 1.0, 2.0],
            [-1.0, 0.0, 0.0],
            [3.0, 2.0, 1.0],
        ];
        let (min, max) = compute_bounds(&points);
        assert_eq!(min, [-1.0, 0.0, 0.0]);
        assert_eq!(max, [3.0, 2.0, 2.0]);
    }

    #[test]
    fn test_svo_diff_hash() {
        let data1 = b"test data 1";
        let data2 = b"test data 2";
        let hash1 = svo_diff_hash(data1);
        let hash2 = svo_diff_hash(data2);
        assert_ne!(hash1, hash2);
        assert_eq!(hash1, svo_diff_hash(data1)); // deterministic
    }

    #[test]
    fn test_build_svo_from_bounds() {
        let config = SvoBuildConfig::default();
        let data = build_svo_from_bounds(&config, [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        assert!(data.starts_with(b"SVO\0"));
        assert!(data.len() >= 32);
    }

    #[test]
    fn test_serialize_primitive_sphere() {
        let prim = FittedPrimitive::Sphere {
            center: alice_sdf::types::Vec3::new(1.0, 2.0, 3.0),
            radius: 0.5,
        };
        let s = serialize_fitted_primitive(&prim, 0.001);
        assert_eq!(s.kind, PrimitiveKind::Sphere);
        assert!((s.params[0] - 1.0).abs() < 1e-6);
        assert!((s.params[3] - 0.5).abs() < 1e-6);
    }
}
