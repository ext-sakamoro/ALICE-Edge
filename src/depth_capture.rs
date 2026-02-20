//! Depth Camera Driver for Dolphin D5 Lite 3D Scanner
//!
//! USB 3.0 depth stream capture with voxel downsampling and normal estimation.
//! Designed for Raspberry Pi 5 edge deployment.
//!
//! Author: Moroya Sakamoto

use std::time::Instant;

/// Maximum points per frame (100K limit for Pi 5 memory budget)
pub const MAX_POINTS_PER_FRAME: usize = 100_000;

#[inline(always)]
fn fast_inv_sqrt(x: f32) -> f32 {
    let half = 0.5 * x;
    let i = f32::to_bits(x);
    let i = 0x5f3759df - (i >> 1);
    let y = f32::from_bits(i);
    y * (1.5 - half * y * y)
}

/// Dolphin D5 Lite USB identifiers
pub const DOLPHIN_D5_VID: u16 = 0x2BC5;
pub const DOLPHIN_D5_PID: u16 = 0x0615;

/// 3D point with normal
#[derive(Debug, Clone, Copy, Default)]
#[repr(C)]
pub struct PointNormal {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub nx: f32,
    pub ny: f32,
    pub nz: f32,
}

/// Depth frame from camera
#[derive(Debug, Clone)]
pub struct DepthFrame {
    pub points: Vec<PointNormal>,
    pub timestamp_ms: u64,
    pub frame_id: u32,
}

/// Camera configuration
#[derive(Debug, Clone)]
pub struct CameraConfig {
    /// Voxel grid size for downsampling (meters)
    pub voxel_size: f32,
    /// Maximum depth distance (meters)
    pub max_depth: f32,
    /// Minimum depth distance (meters)
    pub min_depth: f32,
    /// Target points per frame
    pub max_points: usize,
    /// KNN neighbors for normal estimation
    pub normal_k: usize,
}

impl Default for CameraConfig {
    fn default() -> Self {
        Self {
            voxel_size: 0.01, // 1cm voxels
            max_depth: 3.0,
            min_depth: 0.1,
            max_points: MAX_POINTS_PER_FRAME,
            normal_k: 8,
        }
    }
}

/// Trait for depth camera drivers (allows mock testing)
pub trait DepthCameraDriver: Send {
    /// Initialize the camera
    fn init(&mut self) -> Result<(), CaptureError>;
    /// Capture a single depth frame
    fn capture_frame(&mut self) -> Result<DepthFrame, CaptureError>;
    /// Check if camera is connected
    fn is_connected(&self) -> bool;
    /// Get camera info string
    fn info(&self) -> String;
}

/// Capture errors
#[derive(Debug)]
pub enum CaptureError {
    /// USB device not found
    DeviceNotFound,
    /// USB communication error
    UsbError(String),
    /// Frame capture timeout
    Timeout,
    /// Invalid frame data
    InvalidData(String),
    /// Camera not initialized
    NotInitialized,
}

impl std::fmt::Display for CaptureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CaptureError::DeviceNotFound => write!(f, "Dolphin D5 Lite not found"),
            CaptureError::UsbError(e) => write!(f, "USB error: {}", e),
            CaptureError::Timeout => write!(f, "Frame capture timeout"),
            CaptureError::InvalidData(e) => write!(f, "Invalid data: {}", e),
            CaptureError::NotInitialized => write!(f, "Camera not initialized"),
        }
    }
}

/// Dolphin D5 Lite driver implementation
pub struct DolphinD5Driver {
    config: CameraConfig,
    initialized: bool,
    frame_counter: u32,
    start_time: Option<Instant>,
}

impl DolphinD5Driver {
    pub fn new(config: CameraConfig) -> Self {
        Self {
            config,
            initialized: false,
            frame_counter: 0,
            start_time: None,
        }
    }

    /// Voxel grid downsampling — reduces point count while preserving structure
    ///
    /// Algorithm: hash each point to a voxel cell, keep one point per cell.
    /// O(N) time, O(N/voxel_ratio) memory.
    pub fn voxel_downsample(points: &[PointNormal], voxel_size: f32) -> Vec<PointNormal> {
        if voxel_size <= 0.0 || points.is_empty() {
            return points.to_vec();
        }

        let inv_voxel = 1.0 / voxel_size;
        let mut seen = std::collections::HashMap::with_capacity(points.len() / 4);
        let mut result = Vec::with_capacity(points.len() / 4);

        for p in points {
            let vx = (p.x * inv_voxel).floor() as i32;
            let vy = (p.y * inv_voxel).floor() as i32;
            let vz = (p.z * inv_voxel).floor() as i32;
            let key = (vx as i64) | ((vy as i64) << 21) | ((vz as i64) << 42);

            if seen.insert(key, ()).is_none() {
                result.push(*p);
            }
        }

        result
    }

    /// Estimate normals using cross-product of nearest neighbor displacement vectors.
    /// Simplified approach: uses axis-aligned neighbor pairs for speed on Pi 5.
    pub fn estimate_normals(points: &mut [PointNormal], k: usize) {
        if points.len() < 3 || k == 0 {
            return;
        }

        // Simple approach for edge: use local covariance from nearby points
        // For each point, find k nearest by scanning sorted neighbors
        let n = points.len();
        for i in 0..n {
            let p = points[i];

            // Use adjacent points in the array as neighbor approximation
            // (assumes spatially coherent input from scanner)
            let prev = if i > 0 { i - 1 } else { i };
            let next = if i + 1 < n { i + 1 } else { i };

            let dx1 = points[next].x - p.x;
            let dy1 = points[next].y - p.y;
            let dz1 = points[next].z - p.z;

            let dx2 = points[prev].x - p.x;
            let dy2 = points[prev].y - p.y;
            let dz2 = points[prev].z - p.z;

            let nx = dy1 * dz2 - dz1 * dy2;
            let ny = dz1 * dx2 - dx1 * dz2;
            let nz = dx1 * dy2 - dy1 * dx2;

            let len_sq = nx * nx + ny * ny + nz * nz;
            if len_sq > 1e-16 {
                let inv_len = fast_inv_sqrt(len_sq);
                points[i].nx = nx * inv_len;
                points[i].ny = ny * inv_len;
                points[i].nz = nz * inv_len;
            } else {
                points[i].nx = 0.0;
                points[i].ny = 1.0;
                points[i].nz = 0.0;
            }
        }
    }
}

impl DepthCameraDriver for DolphinD5Driver {
    fn init(&mut self) -> Result<(), CaptureError> {
        // Attempt to open Dolphin D5 Lite via rusb
        let context = rusb::Context::new()
            .map_err(|e| CaptureError::UsbError(format!("USB context: {}", e)))?;

        let device = context
            .devices()
            .map_err(|e| CaptureError::UsbError(format!("USB devices: {}", e)))?
            .iter()
            .find(|d| {
                d.device_descriptor()
                    .map(|desc| {
                        desc.vendor_id() == DOLPHIN_D5_VID && desc.product_id() == DOLPHIN_D5_PID
                    })
                    .unwrap_or(false)
            });

        if device.is_none() {
            return Err(CaptureError::DeviceNotFound);
        }

        self.initialized = true;
        self.start_time = Some(Instant::now());
        Ok(())
    }

    fn capture_frame(&mut self) -> Result<DepthFrame, CaptureError> {
        if !self.initialized {
            return Err(CaptureError::NotInitialized);
        }

        // In production: read USB bulk transfer from D5 Lite depth endpoint
        // The D5 Lite outputs structured light depth maps at up to 10 fps
        // Each frame is a 640x480 depth image (16-bit depth per pixel)
        //
        // For now, return an empty frame structure that the real USB driver
        // would fill with data from the device's depth endpoint.
        let timestamp_ms = self
            .start_time
            .map(|t| t.elapsed().as_millis() as u64)
            .unwrap_or(0);

        self.frame_counter += 1;

        Ok(DepthFrame {
            points: Vec::new(),
            timestamp_ms,
            frame_id: self.frame_counter,
        })
    }

    fn is_connected(&self) -> bool {
        self.initialized
    }

    fn info(&self) -> String {
        format!(
            "Dolphin D5 Lite (VID:{:04X} PID:{:04X}) voxel={}m max_depth={}m",
            DOLPHIN_D5_VID, DOLPHIN_D5_PID, self.config.voxel_size, self.config.max_depth
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_voxel_downsample() {
        let points = vec![
            PointNormal {
                x: 0.001,
                y: 0.001,
                z: 0.001,
                ..Default::default()
            },
            PointNormal {
                x: 0.002,
                y: 0.002,
                z: 0.002,
                ..Default::default()
            },
            PointNormal {
                x: 0.1,
                y: 0.1,
                z: 0.1,
                ..Default::default()
            },
            PointNormal {
                x: 0.101,
                y: 0.101,
                z: 0.101,
                ..Default::default()
            },
        ];

        let result = DolphinD5Driver::voxel_downsample(&points, 0.01);
        // Points 0,1 are in same voxel; points 2,3 are in same voxel
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_estimate_normals() {
        let mut points = vec![
            PointNormal {
                x: 0.0,
                y: 0.0,
                z: 0.0,
                ..Default::default()
            },
            PointNormal {
                x: 1.0,
                y: 0.0,
                z: 0.0,
                ..Default::default()
            },
            PointNormal {
                x: 0.0,
                y: 1.0,
                z: 0.0,
                ..Default::default()
            },
        ];

        DolphinD5Driver::estimate_normals(&mut points, 2);

        // Middle point normal should be approximately (0, 0, ±1)
        assert!(points[1].nz.abs() > 0.5);
    }

    #[test]
    fn test_config_default() {
        let config = CameraConfig::default();
        assert_eq!(config.max_points, MAX_POINTS_PER_FRAME);
        assert!(config.voxel_size > 0.0);
    }

    #[test]
    fn test_driver_info() {
        let driver = DolphinD5Driver::new(CameraConfig::default());
        let info = driver.info();
        assert!(info.contains("Dolphin D5 Lite"));
    }
}
