//! The library f32 max / min sites follow maximumNumber / minimumNumber on
//! the inputs that reach them, checked by bit pattern through the public API
//! (ALICE-Law G2).
//!
//! oracle: IEEE 754-2019 maximumNumber / minimumNumber: a NaN operand yields
//! the other operand; of an equal pair of zeros, max gives +0 and min gives
//! -0, in either operand order. A comparison (`if a < b { a } else { b }`)
//! returns either zero by operand order and returns the NaN when it is the
//! second operand.
//!
//! Site classes and what reaches them:
//! - point cloud bounds (`sdf_compress::compress_point_cloud`, the SVO chunk
//!   `bounds_min` / `bounds_max`): coordinates are taken as given, so ±0 pairs
//!   and NaN coordinates reach the fold.
//! - bounding box features (`SdfFeatures::from_primitive` / `from_svo_stats`):
//!   `bounds_size` is taken as given; the min dimension (feature 13) keeps the
//!   sign of a -0.0 extent, and the max dimension skips a NaN extent.
//! - the extent floor of the edge pipeline bounds (`max(extent, 0.001)`) and
//!   the ReLU layers (`max(v, 0.0)`) have a positive constant second operand:
//!   a ±0 or NaN first operand gives the constant under maximumNumber and
//!   under `if v > c { v } else { c }` alike, so that rewrite is an equivalent
//!   mutant there. The edge pipeline's min / max fold over coordinates feeds
//!   only that floored extent, and the softmax max logit feeds `exp(v - max)`,
//!   whose result is NaN for a NaN logit either way; no public output
//!   distinguishes a comparison at those sites.

#[cfg(feature = "ml")]
use alice_edge::object_classifier::SdfFeatures;

fn bits(v: [f32; 3]) -> [u32; 3] {
    v.map(f32::to_bits)
}

#[cfg(feature = "sdf")]
fn svo_bounds(points: &[[f32; 3]]) -> ([f32; 3], [f32; 3]) {
    use alice_edge::sdf_compress::{compress_point_cloud, CompressConfig, CompressedSdf};
    match compress_point_cloud(points, &CompressConfig::default()).0 {
        CompressedSdf::SvoChunks { chunks, .. } => (chunks[0].bounds_min, chunks[0].bounds_max),
        other => panic!("expected the SVO path for this input, got {other:?}"),
    }
}

/// A +0 / -0 coordinate pair on every axis gives min -0 and max +0 on every
/// axis, in both orders of the pair (two points: three or more fit a plane
/// and take the primitive path instead of the SVO bounds).
#[cfg(feature = "sdf")]
#[test]
fn point_cloud_bounds_of_a_signed_zero_pair() {
    let pairs = [
        [[0.0, 0.0, 0.0], [-0.0, -0.0, -0.0]],
        [[-0.0, -0.0, -0.0], [0.0, 0.0, 0.0]],
        [[0.0, -0.0, 0.0], [-0.0, 0.0, -0.0]],
        [[-0.0, 0.0, -0.0], [0.0, -0.0, 0.0]],
    ];
    for points in pairs {
        let (min, max) = svo_bounds(&points);
        assert_eq!(bits(min), bits([-0.0; 3]), "{points:?}: min {min:?}");
        assert_eq!(bits(max), bits([0.0; 3]), "{points:?}: max {max:?}");
    }
}

/// A NaN coordinate is skipped by both folds, wherever it stands.
#[cfg(feature = "sdf")]
#[test]
fn point_cloud_bounds_skip_a_nan_coordinate() {
    let finite = [[1.0, 2.0, 3.0], [-1.0, 4.0, 0.0]];
    for at in 0..=finite.len() {
        let mut points = finite.to_vec();
        points.insert(at, [f32::NAN, f32::NAN, f32::NAN]);
        let (min, max) = svo_bounds(&points);
        assert_eq!(
            bits(min),
            bits([-1.0, 2.0, 0.0]),
            "NaN at {at}: min {min:?}"
        );
        assert_eq!(bits(max), bits([1.0, 4.0, 3.0]), "NaN at {at}: max {max:?}");
    }
}

/// Feature 13 (min dimension / max dimension) keeps the -0.0 of a ±0 pair of
/// extents in both orders, and the max dimension skips a NaN extent.
#[cfg(feature = "ml")]
#[test]
fn bounding_box_features_of_ties() {
    let primitive = |b: [f32; 3]| SdfFeatures::from_primitive(0, &[], b, 10).features;
    let svo = |b: [f32; 3]| SdfFeatures::from_svo_stats(10, 5, 3, b).features;
    for b in [[0.0, -0.0, 1.0], [-0.0, 0.0, 1.0]] {
        for f in [primitive(b), svo(b)] {
            assert_eq!(
                f[13].to_bits(),
                (-0.0_f32).to_bits(),
                "{b:?}: feature 13 {}",
                f[13]
            );
        }
    }
    for b in [
        [f32::NAN, 1.0, 2.0],
        [1.0, 2.0, f32::NAN],
        [2.0, f32::NAN, 1.0],
    ] {
        for f in [primitive(b), svo(b)] {
            // max dimension is 2.0, so the finite extents divide by it
            let expected: Vec<u32> = b
                .iter()
                .map(|&x| {
                    if x.is_nan() {
                        f32::NAN.to_bits()
                    } else {
                        (x * 0.5).to_bits()
                    }
                })
                .collect();
            let got: Vec<u32> = f[..3].iter().map(|x| x.to_bits()).collect();
            assert_eq!(got, expected, "{b:?}: features 0..3 {:?}", &f[..3]);
        }
    }
}

/// Reachability pin for the edge pipeline's bounds fold: `EdgePipeline` sizes
/// the cloud with that fold only on the primitive path, and a cloud holding a
/// NaN coordinate never takes it (the fit error is NaN, so `mse <= threshold`
/// is false and the cloud goes to the SVO path, whose bounds are the ones the
/// tests above pin). If a NaN cloud ever took the primitive path, this fails
/// and the fold needs a tie test through the public API. Positive control: the
/// same clouds without the NaN take the primitive path.
#[cfg(feature = "sdf")]
#[test]
fn a_nan_coordinate_takes_the_svo_path_not_the_primitive_path() {
    use alice_edge::sdf_compress::{compress_point_cloud, CompressConfig, CompressedSdf};
    let plane: Vec<[f32; 3]> = (0..400)
        .map(|k| [(k % 20) as f32 * 0.1, (k / 20) as f32 * 0.05, 0.0])
        .collect();
    let config = CompressConfig::default();
    assert!(
        matches!(
            compress_point_cloud(&plane, &config).0,
            CompressedSdf::Primitives { .. }
        ),
        "positive control: the finite plane must fit a primitive"
    );
    let mut compared = 0;
    for nan in [
        [f32::NAN, 0.5, 0.0],
        [0.5, f32::NAN, 0.0],
        [0.5, 0.5, f32::NAN],
        [f32::NAN; 3],
    ] {
        for at in [0, plane.len() / 2, plane.len()] {
            let mut cloud = plane.clone();
            cloud.insert(at, nan);
            let out = compress_point_cloud(&cloud, &config).0;
            assert!(
                matches!(out, CompressedSdf::SvoChunks { .. }),
                "NaN {nan:?} at {at}: expected the SVO path, got {}",
                match out {
                    CompressedSdf::Primitives { .. } => "Primitives",
                    CompressedSdf::Hybrid { .. } => "Hybrid",
                    CompressedSdf::SvoChunks { .. } => "SvoChunks",
                }
            );
            compared += 1;
        }
    }
    assert_eq!(compared, 12, "compared nothing");
}

#[test]
fn bits_helper_keeps_the_sign_of_zero() {
    assert_ne!(bits([0.0; 3]), bits([-0.0; 3]));
}
