// SPDX-License-Identifier: MIT
//! ASP Bridge — SDF to ALICE Streaming Protocol Packet Encoding
//!
//! Encodes compressed SDF data into ASP I-Packets (keyframes) and
//! D-Packets (deltas) for transmission from edge to cloud.
//!
//! Author: Moroya Sakamoto

use crate::object_classifier::ObjectClass;
use crate::sdf_compress::{svo_diff_hash, CompressedSdf, SvoChunkData};
use libasp::scene::{SdfDeltaType, SdfSceneDelta, SdfSceneDescriptor};

/// Edge stream encoder — manages keyframe/delta state for ASP transmission
pub struct EdgeStreamEncoder {
    /// Current scene version (monotonically increasing)
    scene_version: u32,
    /// Last transmitted SVO hash (for delta detection)
    last_svo_hash: u64,
    /// Frame counter
    frame_count: u64,
    /// Keyframe interval (frames between full I-Packets)
    keyframe_interval: u32,
}

/// Encoded ASP packet ready for transmission
#[derive(Debug)]
pub enum AspEdgePacket {
    /// I-Packet: Full scene keyframe
    Keyframe {
        scene: SdfSceneDescriptor,
        frame_id: u64,
    },
    /// D-Packet: Incremental SVO chunk delta
    Delta { delta: SdfSceneDelta, frame_id: u64 },
    /// No change detected — skip transmission
    Skip { frame_id: u64 },
}

impl EdgeStreamEncoder {
    /// Create a new encoder with given keyframe interval
    pub fn new(keyframe_interval: u32) -> Self {
        Self {
            scene_version: 0,
            last_svo_hash: 0,
            frame_count: 0,
            keyframe_interval,
        }
    }

    /// Encode a compressed SDF frame into an ASP packet
    ///
    /// Automatically decides between I-Packet (keyframe) and D-Packet (delta)
    /// based on keyframe interval and change detection.
    pub fn encode_frame(
        &mut self,
        compressed: &CompressedSdf,
        classifications: &[(u8, ObjectClass)],
    ) -> AspEdgePacket {
        self.frame_count += 1;

        let is_keyframe =
            self.frame_count == 1 || (self.frame_count % self.keyframe_interval as u64) == 0;

        if is_keyframe {
            self.encode_keyframe(compressed, classifications)
        } else {
            self.encode_delta(compressed)
        }
    }

    /// Encode a full I-Packet keyframe
    fn encode_keyframe(
        &mut self,
        compressed: &CompressedSdf,
        classifications: &[(u8, ObjectClass)],
    ) -> AspEdgePacket {
        self.scene_version += 1;

        let asdf_data = extract_asdf_data(compressed);
        let (bounds_min, bounds_max) = extract_bounds(compressed);

        // Build classification labels
        let labels: Option<Vec<(u8, String)>> = if !classifications.is_empty() {
            Some(
                classifications
                    .iter()
                    .map(|(id, class)| (*id, class.label().to_string()))
                    .collect(),
            )
        } else {
            None
        };

        let mut scene = SdfSceneDescriptor::new(asdf_data)
            .with_bounds(bounds_min, bounds_max)
            .with_render_resolution(0);
        scene.scene_version = self.scene_version;
        scene.classification_labels = labels;

        // Update hash state
        self.last_svo_hash = compute_compressed_hash(compressed);

        AspEdgePacket::Keyframe {
            scene,
            frame_id: self.frame_count,
        }
    }

    /// Encode a D-Packet delta (or skip if no change)
    fn encode_delta(&mut self, compressed: &CompressedSdf) -> AspEdgePacket {
        let new_hash = compute_compressed_hash(compressed);

        // Skip if no change detected
        if new_hash == self.last_svo_hash {
            return AspEdgePacket::Skip {
                frame_id: self.frame_count,
            };
        }

        self.last_svo_hash = new_hash;
        let ref_version = self.scene_version;
        self.scene_version += 1;

        let delta_data = extract_delta_data(compressed);

        let delta = SdfSceneDelta::svo_chunk_delta(ref_version, delta_data);

        AspEdgePacket::Delta {
            delta,
            frame_id: self.frame_count,
        }
    }

    /// Current scene version
    pub fn scene_version(&self) -> u32 {
        self.scene_version
    }

    /// Total frames processed
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }
}

fn extract_asdf_data(compressed: &CompressedSdf) -> Vec<u8> {
    match compressed {
        CompressedSdf::Primitives { asdf_data, .. } => asdf_data.clone(),
        CompressedSdf::SvoChunks { chunks, .. } => {
            let total_len: usize = chunks.iter().map(|c| c.data.len()).sum();
            let mut data = Vec::with_capacity(total_len);
            for chunk in chunks {
                data.extend_from_slice(&chunk.data);
            }
            data
        }
        CompressedSdf::Hybrid {
            asdf_data,
            svo_chunks,
            ..
        } => {
            let total_len =
                asdf_data.len() + svo_chunks.iter().map(|c| c.data.len()).sum::<usize>();
            let mut data = Vec::with_capacity(total_len);
            data.extend_from_slice(asdf_data);
            for chunk in svo_chunks {
                data.extend_from_slice(&chunk.data);
            }
            data
        }
    }
}

/// Extract bounds from compressed SDF
fn extract_bounds(compressed: &CompressedSdf) -> ([f32; 3], [f32; 3]) {
    match compressed {
        CompressedSdf::SvoChunks { chunks, .. }
        | CompressedSdf::Hybrid {
            svo_chunks: chunks, ..
        } => {
            if let Some(chunk) = chunks.first() {
                (chunk.bounds_min, chunk.bounds_max)
            } else {
                ([-10.0; 3], [10.0; 3])
            }
        }
        _ => ([-10.0; 3], [10.0; 3]),
    }
}

/// Compute hash of compressed SDF for change detection
fn compute_compressed_hash(compressed: &CompressedSdf) -> u64 {
    match compressed {
        CompressedSdf::Primitives { asdf_data, .. } => svo_diff_hash(asdf_data),
        CompressedSdf::SvoChunks { chunks, .. } => {
            let mut combined_hash = 0u64;
            for chunk in chunks {
                combined_hash ^= svo_diff_hash(&chunk.data);
            }
            combined_hash
        }
        CompressedSdf::Hybrid {
            asdf_data,
            svo_chunks,
            ..
        } => {
            let mut combined = svo_diff_hash(asdf_data);
            for chunk in svo_chunks {
                combined ^= svo_diff_hash(&chunk.data);
            }
            combined
        }
    }
}

/// Extract delta data from compressed SDF for D-Packet
fn extract_delta_data(compressed: &CompressedSdf) -> Vec<u8> {
    extract_asdf_data(compressed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_first_frame_is_keyframe() {
        let mut encoder = EdgeStreamEncoder::new(30);

        let compressed = CompressedSdf::SvoChunks {
            chunks: vec![SvoChunkData {
                chunk_id: 0,
                data: vec![1, 2, 3, 4],
                node_count: 1,
                bounds_min: [-1.0, -1.0, -1.0],
                bounds_max: [1.0, 1.0, 1.0],
            }],
            total_nodes: 1,
        };

        let packet = encoder.encode_frame(&compressed, &[]);
        assert!(matches!(packet, AspEdgePacket::Keyframe { .. }));
    }

    #[test]
    fn test_encoder_skip_on_no_change() {
        let mut encoder = EdgeStreamEncoder::new(100);

        let compressed = CompressedSdf::SvoChunks {
            chunks: vec![SvoChunkData {
                chunk_id: 0,
                data: vec![1, 2, 3, 4],
                node_count: 1,
                bounds_min: [-1.0; 3],
                bounds_max: [1.0; 3],
            }],
            total_nodes: 1,
        };

        // First frame = keyframe
        let _ = encoder.encode_frame(&compressed, &[]);

        // Same data = skip
        let packet = encoder.encode_frame(&compressed, &[]);
        assert!(matches!(packet, AspEdgePacket::Skip { .. }));
    }

    #[test]
    fn test_encoder_delta_on_change() {
        let mut encoder = EdgeStreamEncoder::new(100);

        let compressed1 = CompressedSdf::SvoChunks {
            chunks: vec![SvoChunkData {
                chunk_id: 0,
                data: vec![1, 2, 3, 4],
                node_count: 1,
                bounds_min: [-1.0; 3],
                bounds_max: [1.0; 3],
            }],
            total_nodes: 1,
        };

        let compressed2 = CompressedSdf::SvoChunks {
            chunks: vec![SvoChunkData {
                chunk_id: 0,
                data: vec![5, 6, 7, 8], // different data
                node_count: 1,
                bounds_min: [-1.0; 3],
                bounds_max: [1.0; 3],
            }],
            total_nodes: 1,
        };

        let _ = encoder.encode_frame(&compressed1, &[]);
        let packet = encoder.encode_frame(&compressed2, &[]);
        assert!(matches!(packet, AspEdgePacket::Delta { .. }));
    }

    #[test]
    fn test_encoder_periodic_keyframe() {
        let mut encoder = EdgeStreamEncoder::new(3);

        let compressed = CompressedSdf::Primitives {
            primitives: vec![],
            asdf_data: vec![1, 2, 3],
        };

        let _ = encoder.encode_frame(&compressed, &[]); // frame 1: keyframe
        let _ = encoder.encode_frame(&compressed, &[]); // frame 2: skip/delta
        let packet = encoder.encode_frame(&compressed, &[]); // frame 3: keyframe
        assert!(matches!(packet, AspEdgePacket::Keyframe { .. }));
    }
}
