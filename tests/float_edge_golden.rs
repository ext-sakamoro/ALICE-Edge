//! Signed-zero and exact-cancellation cases of ALICE-Edge's float entry points,
//! pinned to fixed bit patterns (ALICE-Law G4).
//!
//! oracle: each expected value is worked out by hand from IEEE 754 binary32
//! round-to-nearest-even, written in the comment next to it; none is taken
//! from the implementation. The cases run on every CI leg in debug (ci.yml
//! test job) and release (release-profile.yml), x86_64 and aarch64, so a
//! contraction or reassociation that one target's optimiser makes shows up as
//! a changed bit.
//!
//! Signed zero through the bounding-box features and the point-cloud bounds is
//! pinned by tests/minmax_ties.rs (G2).

use alice_edge::q16_to_f32;
use alice_edge::sensor_fusion::{FusedSensor, FusionConfig, KalmanFilter1D, SensorInput};

/// Q16.16 to f32 is one exact multiplication by 2^-16 for every case here.
#[test]
fn q16_to_f32_exact_cases() {
    assert_eq!(q16_to_f32(0).to_bits(), 0x0000_0000); // +0
    assert_eq!(q16_to_f32(1).to_bits(), 0x3780_0000); // 2^-16
    assert_eq!(q16_to_f32(-65_536).to_bits(), 0xbf80_0000); // -1.0
    assert_eq!(q16_to_f32(i32::MIN).to_bits(), 0xc700_0000); // -32768.0
    assert_eq!(q16_to_f32(i32::MAX).to_bits(), 0x4700_0000); // (2^31 - 1) rounds to 2^31, times 2^-16 = 32768.0
}

/// Kalman 1D from x = -0.0 with a measurement of -0.0: p = 1 + 0 = 1,
/// k = 1 / (1 + 1) = 0.5, z - x = (-0) - (-0) = +0, x = -0 + 0.5 * +0 = -0 + +0
/// = +0 (an update written as x * (1 - k) + k * z would keep -0).
#[test]
fn kalman_update_of_negative_zero_gives_positive_zero() {
    let mut kf = KalmanFilter1D::new(-0.0, 1.0, 0.0, 1.0);
    assert_eq!(kf.filter(-0.0).to_bits(), 0x0000_0000);
    assert_eq!(kf.error().to_bits(), 0x3f00_0000); // p = 1 * (1 - 0.5) = 0.5
}

/// Kalman 1D at a large value: z - x = (1e7 + 2) - 1e7 = 2 exactly (both are
/// below 2^24, so representable), x = 1e7 + 0.5 * 2 = 10000001 exactly.
#[test]
fn kalman_update_cancels_a_large_offset_exactly() {
    let mut kf = KalmanFilter1D::new(10_000_000.0, 1.0, 0.0, 1.0);
    assert_eq!(kf.filter(10_000_002.0).to_bits(), 0x4b18_9681); // 10000001.0
    assert_eq!(kf.error().to_bits(), 0x3f00_0000); // 0.5
}

/// Fusion of +5 and -5 with equal noise from the initial estimate 0: both are
/// inliers (|5 - 0| < 3 * sqrt(100.01 + 1)), the weighted sum 1 * 5 + 1 * -5
/// cancels to +0, the fused measurement +0 / 2 = +0, and the update keeps x =
/// 0 + k * (+0 - 0) = +0.
#[test]
fn fusion_of_opposite_values_cancels_to_positive_zero() {
    let mut f = FusedSensor::new(FusionConfig::default());
    let inputs = [
        SensorInput {
            value: 5.0,
            noise: 1.0,
        },
        SensorInput {
            value: -5.0,
            noise: 1.0,
        },
    ];
    assert_eq!(f.fuse(&inputs).to_bits(), 0x0000_0000);
}
