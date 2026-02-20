//! Edge Pipeline Orchestrator
//!
//! Integrates capture → compress → classify → stream into a unified
//! processing loop running at 10Hz on Raspberry Pi 5.
//!
//! Author: Moroya Sakamoto

use std::time::{Duration, Instant};

use crate::asp_bridge::{AspEdgePacket, EdgeStreamEncoder};
use crate::depth_capture::{
    CameraConfig, DepthCameraDriver, DepthFrame, DolphinD5Driver, PointNormal,
};
use crate::object_classifier::{ObjectClass, SdfFeatures, TernaryClassifier, DEFAULT_NUM_CLASSES};
use crate::sdf_compress::{compress_point_cloud, CompressConfig, CompressStats, CompressedSdf};

/// Pipeline configuration
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Camera configuration
    pub camera: CameraConfig,
    /// SDF compression configuration
    pub compress: CompressConfig,
    /// Target frame rate (Hz)
    pub target_fps: f32,
    /// ASP keyframe interval (frames)
    pub keyframe_interval: u32,
    /// Number of classification categories
    pub num_classes: usize,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            camera: CameraConfig::default(),
            compress: CompressConfig::default(),
            target_fps: 10.0,
            keyframe_interval: 30,
            num_classes: DEFAULT_NUM_CLASSES,
        }
    }
}

/// Pipeline statistics for monitoring
#[derive(Debug, Clone, Default)]
pub struct PipelineStats {
    /// Total frames processed
    pub frames_processed: u64,
    /// Total keyframes sent
    pub keyframes_sent: u64,
    /// Total deltas sent
    pub deltas_sent: u64,
    /// Total frames skipped (no change)
    pub frames_skipped: u64,
    /// Average capture latency (ms)
    pub avg_capture_ms: f64,
    /// Average compress latency (ms)
    pub avg_compress_ms: f64,
    /// Average classify latency (ms)
    pub avg_classify_ms: f64,
    /// Average encode latency (ms)
    pub avg_encode_ms: f64,
    /// Average total pipeline latency (ms)
    pub avg_total_ms: f64,
    /// Total bytes transmitted
    pub total_bytes_sent: u64,
}

/// Callback for transmitting encoded packets
pub trait PacketSink: Send {
    fn send_keyframe(&mut self, scene_data: &[u8], frame_id: u64);
    fn send_delta(&mut self, delta_data: &[u8], frame_id: u64);
}

/// Edge processing pipeline
///
/// Orchestrates: capture → downsample → compress → classify → encode
pub struct EdgePipeline {
    config: PipelineConfig,
    driver: Box<dyn DepthCameraDriver>,
    classifier: TernaryClassifier,
    encoder: EdgeStreamEncoder,
    stats: PipelineStats,
    latency_sum_capture: f64,
    latency_sum_compress: f64,
    latency_sum_classify: f64,
    latency_sum_encode: f64,
    latency_sum_total: f64,
}

impl EdgePipeline {
    /// Create a new edge pipeline with Dolphin D5 Lite driver
    pub fn new(config: PipelineConfig) -> Self {
        let driver = Box::new(DolphinD5Driver::new(config.camera.clone()));
        let classifier = TernaryClassifier::new(config.num_classes);
        let encoder = EdgeStreamEncoder::new(config.keyframe_interval);

        Self {
            config,
            driver,
            classifier,
            encoder,
            stats: PipelineStats::default(),
            latency_sum_capture: 0.0,
            latency_sum_compress: 0.0,
            latency_sum_classify: 0.0,
            latency_sum_encode: 0.0,
            latency_sum_total: 0.0,
        }
    }

    /// Create pipeline with custom driver (for testing)
    pub fn with_driver(config: PipelineConfig, driver: Box<dyn DepthCameraDriver>) -> Self {
        let classifier = TernaryClassifier::new(config.num_classes);
        let encoder = EdgeStreamEncoder::new(config.keyframe_interval);

        Self {
            config,
            driver,
            classifier,
            encoder,
            stats: PipelineStats::default(),
            latency_sum_capture: 0.0,
            latency_sum_compress: 0.0,
            latency_sum_classify: 0.0,
            latency_sum_encode: 0.0,
            latency_sum_total: 0.0,
        }
    }

    /// Initialize the pipeline (connects to camera)
    pub fn init(&mut self) -> Result<(), crate::depth_capture::CaptureError> {
        self.driver.init()
    }

    /// Process a single frame through the pipeline
    ///
    /// Returns the encoded ASP packet and compression statistics.
    pub fn process_frame(
        &mut self,
    ) -> Result<(AspEdgePacket, CompressStats), crate::depth_capture::CaptureError> {
        let total_start = Instant::now();

        // Stage 1: Capture
        let capture_start = Instant::now();
        let frame = self.driver.capture_frame()?;
        let capture_ms = capture_start.elapsed().as_secs_f64() * 1000.0;

        // Downsample
        let mut points = frame.points;
        if !points.is_empty() {
            points = DolphinD5Driver::voxel_downsample(&points, self.config.camera.voxel_size);
            DolphinD5Driver::estimate_normals(&mut points, self.config.camera.normal_k);
        }

        // Convert to f32 array for compression
        let point_arrays: Vec<[f32; 3]> = points.iter().map(|p| [p.x, p.y, p.z]).collect();

        // Stage 2: Compress
        let compress_start = Instant::now();
        let (compressed, compress_stats) =
            compress_point_cloud(&point_arrays, &self.config.compress);
        let compress_ms = compress_start.elapsed().as_secs_f64() * 1000.0;

        // Stage 3: Classify
        let classify_start = Instant::now();
        let classifications = self.classify_compressed(&compressed, &point_arrays);
        let classify_ms = classify_start.elapsed().as_secs_f64() * 1000.0;

        // Stage 4: Encode to ASP
        let encode_start = Instant::now();
        let packet = self.encoder.encode_frame(&compressed, &classifications);
        let encode_ms = encode_start.elapsed().as_secs_f64() * 1000.0;

        let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;

        // Update stats
        self.stats.frames_processed += 1;
        self.latency_sum_capture += capture_ms;
        self.latency_sum_compress += compress_ms;
        self.latency_sum_classify += classify_ms;
        self.latency_sum_encode += encode_ms;
        self.latency_sum_total += total_ms;

        let n = self.stats.frames_processed as f64;
        self.stats.avg_capture_ms = self.latency_sum_capture / n;
        self.stats.avg_compress_ms = self.latency_sum_compress / n;
        self.stats.avg_classify_ms = self.latency_sum_classify / n;
        self.stats.avg_encode_ms = self.latency_sum_encode / n;
        self.stats.avg_total_ms = self.latency_sum_total / n;

        match &packet {
            AspEdgePacket::Keyframe { .. } => {
                self.stats.keyframes_sent += 1;
                self.stats.total_bytes_sent += compress_stats.output_bytes as u64;
            }
            AspEdgePacket::Delta { .. } => {
                self.stats.deltas_sent += 1;
                self.stats.total_bytes_sent += compress_stats.output_bytes as u64;
            }
            AspEdgePacket::Skip { .. } => {
                self.stats.frames_skipped += 1;
            }
        }

        Ok((packet, compress_stats))
    }

    /// Run the pipeline loop for a specified duration
    pub fn run_for(&mut self, duration: Duration) -> Vec<(AspEdgePacket, CompressStats)> {
        let frame_interval = Duration::from_secs_f64(1.0 / self.config.target_fps as f64);
        let start = Instant::now();
        let mut results = Vec::new();

        while start.elapsed() < duration {
            let frame_start = Instant::now();

            match self.process_frame() {
                Ok(result) => results.push(result),
                Err(_e) => {} // Skip failed frames
            }

            // Maintain target frame rate
            let frame_elapsed = frame_start.elapsed();
            if frame_elapsed < frame_interval {
                std::thread::sleep(frame_interval - frame_elapsed);
            }
        }

        results
    }

    /// Get current pipeline statistics
    pub fn stats(&self) -> &PipelineStats {
        &self.stats
    }

    fn classify_compressed(
        &self,
        compressed: &CompressedSdf,
        points: &[[f32; 3]],
    ) -> Vec<(u8, ObjectClass)> {
        match compressed {
            CompressedSdf::Primitives { primitives, .. } => {
                let bounds = compute_bounds_size(points);
                let point_len = points.len();
                primitives
                    .iter()
                    .enumerate()
                    .map(|(i, prim)| {
                        let features = SdfFeatures::from_primitive(
                            prim.kind as u8,
                            &prim.params,
                            bounds,
                            point_len,
                        );
                        let (class, _conf) = self.classifier.classify(&features);
                        (i as u8, class)
                    })
                    .collect()
            }
            CompressedSdf::SvoChunks {
                chunks,
                total_nodes,
            } => {
                chunks
                    .iter()
                    .map(|chunk| {
                        let bounds_size = [
                            chunk.bounds_max[0] - chunk.bounds_min[0],
                            chunk.bounds_max[1] - chunk.bounds_min[1],
                            chunk.bounds_max[2] - chunk.bounds_min[2],
                        ];
                        let features = SdfFeatures::from_svo_stats(
                            *total_nodes,
                            chunk.node_count,
                            6, // default depth
                            bounds_size,
                        );
                        let (class, _conf) = self.classifier.classify(&features);
                        (chunk.chunk_id as u8, class)
                    })
                    .collect()
            }
            CompressedSdf::Hybrid {
                primitives,
                svo_chunks,
                ..
            } => {
                let bounds = compute_bounds_size(points);
                let point_len = points.len();
                let mut results: Vec<(u8, ObjectClass)> = primitives
                    .iter()
                    .enumerate()
                    .map(|(i, prim)| {
                        let features = SdfFeatures::from_primitive(
                            prim.kind as u8,
                            &prim.params,
                            bounds,
                            point_len,
                        );
                        let (class, _) = self.classifier.classify(&features);
                        (i as u8, class)
                    })
                    .collect();

                for chunk in svo_chunks {
                    let bounds_size = [
                        chunk.bounds_max[0] - chunk.bounds_min[0],
                        chunk.bounds_max[1] - chunk.bounds_min[1],
                        chunk.bounds_max[2] - chunk.bounds_min[2],
                    ];
                    let features = SdfFeatures::from_svo_stats(
                        chunk.node_count,
                        chunk.node_count,
                        6,
                        bounds_size,
                    );
                    let (class, _) = self.classifier.classify(&features);
                    results.push((chunk.chunk_id as u8, class));
                }

                results
            }
        }
    }
}

fn compute_bounds_size(points: &[[f32; 3]]) -> [f32; 3] {
    if points.is_empty() {
        return [1.0, 1.0, 1.0];
    }
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
    [
        (max[0] - min[0]).max(0.001),
        (max[1] - min[1]).max(0.001),
        (max[2] - min[2]).max(0.001),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mock depth camera for testing
    struct MockCamera {
        frame_counter: u32,
    }

    impl DepthCameraDriver for MockCamera {
        fn init(&mut self) -> Result<(), crate::depth_capture::CaptureError> {
            Ok(())
        }

        fn capture_frame(&mut self) -> Result<DepthFrame, crate::depth_capture::CaptureError> {
            self.frame_counter += 1;
            Ok(DepthFrame {
                points: vec![
                    PointNormal {
                        x: 0.0,
                        y: 0.0,
                        z: 1.0,
                        ..Default::default()
                    },
                    PointNormal {
                        x: 0.1,
                        y: 0.0,
                        z: 1.0,
                        ..Default::default()
                    },
                    PointNormal {
                        x: 0.0,
                        y: 0.1,
                        z: 1.0,
                        ..Default::default()
                    },
                ],
                timestamp_ms: self.frame_counter as u64 * 100,
                frame_id: self.frame_counter,
            })
        }

        fn is_connected(&self) -> bool {
            true
        }
        fn info(&self) -> String {
            "MockCamera".to_string()
        }
    }

    #[test]
    fn test_pipeline_creation() {
        let config = PipelineConfig::default();
        let mock = Box::new(MockCamera { frame_counter: 0 });
        let pipeline = EdgePipeline::with_driver(config, mock);
        assert_eq!(pipeline.stats().frames_processed, 0);
    }

    #[test]
    fn test_pipeline_process_frame() {
        let config = PipelineConfig::default();
        let mock = Box::new(MockCamera { frame_counter: 0 });
        let mut pipeline = EdgePipeline::with_driver(config, mock);
        pipeline.init().unwrap();

        let (packet, stats) = pipeline.process_frame().unwrap();
        assert!(matches!(packet, AspEdgePacket::Keyframe { .. }));
        assert_eq!(pipeline.stats().frames_processed, 1);
    }

    #[test]
    fn test_pipeline_stats_update() {
        let config = PipelineConfig::default();
        let mock = Box::new(MockCamera { frame_counter: 0 });
        let mut pipeline = EdgePipeline::with_driver(config, mock);
        pipeline.init().unwrap();

        for _ in 0..5 {
            let _ = pipeline.process_frame();
        }

        let stats = pipeline.stats();
        assert_eq!(stats.frames_processed, 5);
        assert!(stats.avg_total_ms >= 0.0);
    }
}
