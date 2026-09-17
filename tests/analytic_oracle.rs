//! Analytic oracles — closed-form checks for the fitting / filtering laws in
//! ALICE-Edge (CLAUDE.md § 解析解突合テスト規律, 2026-09-17).
//!
//! Expected values come from closed forms or f64 references written in this
//! file, never from the crate function under test.  `FusionConfig::default()`
//! and `CompressConfig::default()` are the paths a consumer takes first.
//!
//! Oracle sources:
//! - least squares on x = 0..n−1: an exact integer polynomial is recovered
//!   exactly (Q16 coefficients are dyadic-exact); the mean of integers
//! - robust fit: a ramp with one gross outlier recovers the true line within
//!   1 % (Hampel / MAD, k = 3)
//! - piecewise: two exact lines ⇒ two segments with the exact break
//! - delta coding: decode∘encode = id, savings = n − 1 stored deltas
//! - Kalman 1-D (random walk + noise): P⁺ = P⁻R/(P⁻+R), P⁻ = P⁺+Q, steady
//!   state P* = (−Q + √(Q² + 4QR))/2; constant measurement ⇒ x → z
//! - Kalman 2-D (constant velocity): the velocity estimate converges to the
//!   true slope, P stays symmetric positive definite
//! - inverse-variance fusion: x̂ = Σ zᵢ/σᵢ² / Σ 1/σᵢ²; a 10 σ outlier is rejected
//! - SVO / primitive compression: a sampled sphere is detected as a sphere
//!   with the sampled centre and radius

use alice_edge::adaptive_polyfit::{
    evaluate_cubic_fixed, evaluate_quadratic_fixed, fit_cubic_fixed, fit_quadratic_fixed,
};
use alice_edge::constant_fit::{compute_residual_error, fit_constant_fixed};
use alice_edge::piecewise::fit_piecewise_linear;
use alice_edge::q16_linear::{evaluate_linear_fixed, fit_linear_fixed, q16_to_f32, Q16_ONE};
use alice_edge::ring_buffer::RingBuffer;
use alice_edge::sensor_fusion::{
    FusedSensor, FusionConfig, KalmanFilter1D, KalmanFilter2D, SensorInput,
};
use alice_edge::simd_fit::fit_linear_simd;

fn q16(v: f64) -> f64 {
    v * f64::from(Q16_ONE)
}

// ───────────────────────── least squares fits ─────────────────────────────

#[test]
fn linear_fit_recovers_exact_integer_lines_and_matches_the_simd_path() {
    for (a, b) in [(0i64, 0i64), (1, 0), (3, -7), (-2, 1000), (17, 5)] {
        for n in [2usize, 3, 4, 5, 8, 33, 1000] {
            let data: Vec<i32> = (0..n as i64).map(|x| (a * x + b) as i32).collect();
            let (slope, intercept) = fit_linear_fixed(&data);
            // oracle: exact line ⇒ exact Q16 coefficients
            assert_eq!(f64::from(slope), q16(a as f64), "slope a={a} b={b} n={n}");
            assert_eq!(
                f64::from(intercept),
                q16(b as f64),
                "intercept a={a} b={b} n={n}"
            );
            assert_eq!(
                fit_linear_simd(&data),
                (slope, intercept),
                "SIMD path n={n}"
            );
            for x in 0..n as i32 {
                assert_eq!(
                    evaluate_linear_fixed(slope, intercept, x),
                    ((a * x as i64 + b) << 16) as i32
                );
            }
            assert_eq!(
                compute_residual_error(&data, slope, intercept),
                0,
                "exact fit has zero residual"
            );
        }
    }
    // single point / empty
    assert_eq!(fit_linear_fixed(&[42]), (0, 42 << 16));
    assert_eq!(fit_linear_fixed(&[]), (0, 0));
    // mean of integers, Q16, exact for divisible sums
    assert_eq!(fit_constant_fixed(&[1, 2, 3, 4]), (10 << 16) / 4);
    assert_eq!(q16_to_f32(fit_constant_fixed(&[7, 7, 7])), 7.0);
    // least squares on noisy data: the f64 normal equations agree to 1 Q16 ulp
    let data: Vec<i32> = (0..50)
        .map(|x| 3 * x - 40 + [1, -2, 0, 2, -1][x as usize % 5])
        .collect();
    let (slope, intercept) = fit_linear_fixed(&data);
    let n = data.len() as f64;
    let sx: f64 = (0..data.len()).map(|x| x as f64).sum();
    let sy: f64 = data.iter().map(|&y| f64::from(y)).sum();
    let sxy: f64 = data
        .iter()
        .enumerate()
        .map(|(x, &y)| x as f64 * f64::from(y))
        .sum();
    let sxx: f64 = (0..data.len()).map(|x| (x as f64).powi(2)).sum();
    let a = (n * sxy - sx * sy) / (n * sxx - sx * sx);
    let b = (sy - a * sx) / n;
    assert!(
        (f64::from(slope) - q16(a)).abs() <= 1.0,
        "noisy slope {} vs {}",
        f64::from(slope),
        q16(a)
    );
    assert!(
        (f64::from(intercept) - q16(b)).abs() <= 2.0,
        "noisy intercept"
    );
}

#[test]
fn quadratic_and_cubic_fits_recover_exact_integer_polynomials() {
    // capacities measured in the fuzz campaign (Backlog): cubic ≤ 64 samples,
    // quadratic ≤ 4096 — inside them the Bareiss solve is exact
    for (a, b, c) in [(1i64, 0i64, 0i64), (2, -3, 5), (-1, 4, -100), (0, 7, 1)] {
        // Q16 i32 holds |y| < 2¹⁵: keep a·n² inside that range
        for n in [3usize, 4, 7, 16, 100] {
            let data: Vec<i32> = (0..n as i64)
                .map(|x| (a * x * x + b * x + c) as i32)
                .collect();
            let (qa, qb, qc) = fit_quadratic_fixed(&data);
            assert_eq!(f64::from(qa), q16(a as f64), "quad a n={n} ({a},{b},{c})");
            assert_eq!(f64::from(qb), q16(b as f64), "quad b n={n}");
            assert_eq!(f64::from(qc), q16(c as f64), "quad c n={n}");
            for x in 0..n as i32 {
                let exact = ((a * (x as i64).pow(2) + b * x as i64 + c) << 16) as i32;
                assert_eq!(
                    evaluate_quadratic_fixed(qa, qb, qc, x),
                    exact,
                    "quad eval x={x}"
                );
            }
        }
    }
    for (a, b, c, d) in [(1i64, 0i64, 0i64, 0i64), (1, -2, 3, -4), (-2, 5, 0, 9)] {
        for n in [4usize, 5, 8, 16, 24] {
            let data: Vec<i32> = (0..n as i64)
                .map(|x| (a * x * x * x + b * x * x + c * x + d) as i32)
                .collect();
            let (ca, cb, cc, cd) = fit_cubic_fixed(&data);
            assert_eq!(f64::from(ca), q16(a as f64), "cubic a n={n}");
            assert_eq!(f64::from(cb), q16(b as f64), "cubic b n={n}");
            assert_eq!(f64::from(cc), q16(c as f64), "cubic c n={n}");
            assert_eq!(f64::from(cd), q16(d as f64), "cubic d n={n}");
            for x in 0..n as i32 {
                let xi = x as i64;
                let exact = ((a * xi * xi * xi + b * xi * xi + c * xi + d) << 16) as i32;
                assert_eq!(
                    evaluate_cubic_fixed(ca, cb, cc, cd, x),
                    exact,
                    "cubic eval x={x}"
                );
            }
        }
    }
    // degenerate lengths fall back to the lower-order exact fit
    assert_eq!(fit_quadratic_fixed(&[5, 8]), (0, 3 << 16, 5 << 16));
    assert_eq!(fit_cubic_fixed(&[5, 8, 13]).0, 0);
}

#[test]
fn piecewise_fit_splits_two_exact_lines_at_the_break() {
    let mut data: Vec<i32> = (0..40).map(|x| 2 * x + 10).collect(); // slope 2 up to x = 39 (y = 88)
    data.extend((40..80).map(|x| 200 - 3 * x)); // then slope −3 (y = 80 at x = 40: no shared point)
    let segs = fit_piecewise_linear(&data, 0, 4);
    assert_eq!(segs.len(), 2, "two exact lines ⇒ two segments: {segs:?}");
    assert_eq!((segs[0].start, segs[0].end), (0, 40));
    assert_eq!((segs[1].start, segs[1].end), (40, 80));
    assert_eq!(f64::from(segs[0].slope), q16(2.0));
    assert_eq!(f64::from(segs[0].intercept), q16(10.0));
    assert_eq!(f64::from(segs[1].slope), q16(-3.0));
    // one exact line ⇒ one segment regardless of the error budget
    let line: Vec<i32> = (0..100).map(|x| 5 * x - 3).collect();
    assert_eq!(fit_piecewise_linear(&line, 0, 4).len(), 1);
}

#[cfg(feature = "std")]
#[test]
fn robust_fit_ignores_a_gross_outlier_and_delta_coding_round_trips() {
    use alice_edge::delta::{
        delta_decode_coefficients, delta_encode_coefficients, delta_encoding_savings,
    };
    use alice_edge::robust::{filter_outliers_mad, fit_linear_robust};
    // Hampel identifier on a symmetric sample: median 5, MAD 1, k = 3 ⇒ 100 is out
    let filtered = filter_outliers_mad(&[3, 4, 5, 6, 7, 100], 3);
    assert_eq!(filtered.len(), 6);
    assert!(
        filtered.iter().all(|&v| v <= 7),
        "the outlier is neutralised: {filtered:?}"
    );
    assert_eq!(
        filter_outliers_mad(&[1, 2], 3),
        vec![1, 2],
        "n < 3 untouched"
    );
    assert_eq!(
        filter_outliers_mad(&[4, 4, 4, 4], 3),
        vec![4, 4, 4, 4],
        "MAD 0 untouched"
    );
    // oracle: a ramp y = x with one gross outlier at x = 10 refits to y = x
    // within 1 % (the outlier carries no information about the line)
    let mut ramp: Vec<i32> = (0..100).collect();
    ramp[10] = 1000;
    let (slope, intercept) = fit_linear_robust(&ramp, 3);
    assert!(
        (f64::from(slope) - q16(1.0)).abs() <= q16(0.01),
        "robust slope {} vs {}",
        f64::from(slope),
        q16(1.0)
    );
    assert!(
        f64::from(intercept).abs() <= q16(0.5),
        "robust intercept {}",
        f64::from(intercept)
    );

    let coeffs: Vec<(i32, i32)> = (0..20).map(|i| (100 + 3 * i, -5000 + 7 * i)).collect();
    let enc = delta_encode_coefficients(&coeffs);
    assert_eq!(
        delta_decode_coefficients(&enc),
        coeffs,
        "decode∘encode = id"
    );
    assert_eq!(enc[0], coeffs[0], "first entry is stored verbatim");
    for d in &enc[1..] {
        assert_eq!(*d, (3, 7), "constant deltas");
    }
    let (raw, packed) = delta_encoding_savings(&coeffs);
    assert!(raw >= packed && packed > 0);
}

// ───────────────────────── Kalman / fusion ────────────────────────────────

#[test]
fn kalman_1d_follows_the_scalar_riccati_recursion_and_converges() {
    let (q, r) = (0.01f64, 1.0f64);
    let mut kf = KalmanFilter1D::new(0.0, 100.0, q as f32, r as f32);
    let (mut x, mut p) = (0.0f64, 100.0f64);
    let z = 5.0f64;
    for i in 0..300 {
        let est = kf.filter(z as f32) as f64;
        // f64 reference of the same measurement
        p += q;
        let k = p / (p + r);
        x += k * (z - x);
        p *= 1.0 - k;
        assert!(
            (est - x).abs() < 1e-4 * x.abs().max(1.0),
            "step {i}: x {est} vs {x}"
        );
        assert!(
            (kf.error() as f64 - p).abs() < 1e-5 * p.max(1e-3),
            "step {i}: P {} vs {p}",
            kf.error()
        );
    }
    // oracle: constant measurement ⇒ estimate → z; P → steady state
    // P* solves P* = (P* + Q)R/(P* + Q + R) ⇒ P*² + QP* − QR = 0
    let p_star = (-q + (q * q + 4.0 * q * r).sqrt()) / 2.0;
    assert!(
        (kf.estimate() as f64 - z).abs() < 1e-3,
        "estimate {}",
        kf.estimate()
    );
    assert!(
        (kf.error() as f64 - p_star).abs() < 1e-4,
        "P {} vs P* {p_star}",
        kf.error()
    );
}

#[test]
fn kalman_2d_recovers_a_constant_velocity_and_keeps_p_symmetric_positive() {
    let dt = 0.1f32;
    let mut kf = KalmanFilter2D::new(dt, 1e-4, 0.05);
    let v_true = 2.5f32;
    let mut last = (0.0, 0.0);
    for i in 0..400 {
        let z = v_true * dt * i as f32;
        last = kf.filter(z);
    }
    let (pos, vel) = last;
    let expected_pos = v_true * dt * 399.0;
    assert!(
        (pos - expected_pos).abs() < 0.02,
        "position {pos} vs {expected_pos}"
    );
    assert!((vel - v_true).abs() < 0.02, "velocity {vel} vs {v_true}");
    // a fresh filter, exact measurements, dt independence of the velocity law:
    // the same physical motion sampled at 2× rate yields the same velocity
    let mut fine = KalmanFilter2D::new(dt / 2.0, 1e-4, 0.05);
    let mut v_fine = 0.0;
    for i in 0..800 {
        v_fine = fine.filter(v_true * dt / 2.0 * i as f32).1;
    }
    assert!(
        (v_fine - v_true).abs() < 0.02,
        "velocity at 2× rate {v_fine}"
    );
}

#[test]
fn fused_sensor_is_the_inverse_variance_mean_and_rejects_ten_sigma_outliers() {
    let cfg = FusionConfig::default();
    assert_eq!(
        (
            cfg.process_noise,
            cfg.default_measurement_noise,
            cfg.outlier_threshold
        ),
        (0.01, 1.0, 3.0)
    );
    let mut fused = FusedSensor::new(cfg);
    // prime the filter on the true value so the outlier gate has a reference
    for _ in 0..50 {
        fused.fuse(&[SensorInput {
            value: 10.0,
            noise: 0.1,
        }]);
    }
    let inputs = [
        SensorInput {
            value: 10.2,
            noise: 0.04,
        },
        SensorInput {
            value: 9.8,
            noise: 0.16,
        },
        SensorInput {
            value: 10.1,
            noise: 0.01,
        },
    ];
    // oracle: inverse-variance weighted mean of the (accepted) inputs
    let w: f64 = inputs.iter().map(|i| 1.0 / i.noise as f64).sum();
    let mean: f64 = inputs
        .iter()
        .map(|i| i.value as f64 / i.noise as f64)
        .sum::<f64>()
        / w;
    let out = fused.fuse(&inputs) as f64;
    // the Kalman blend pulls the fused measurement towards the prior (10.0)
    // by K = P/(P+R_fused); with the filter converged both are ≈ 10.1 — the
    // output must lie between the prior and the fused measurement
    assert!(
        (out - mean).abs() < (mean - 10.0).abs() + 1e-3,
        "fused {out} vs mean {mean}"
    );
    assert!(
        out >= 10.0 - 1e-6 && out <= mean + 1e-6,
        "between prior 10 and mean {mean}: {out}"
    );
    // a 10 σ outlier does not move the estimate: two identical filters, one
    // fed the outlier in addition, must agree
    let mut with = FusedSensor::new(FusionConfig::default());
    let mut without = FusedSensor::new(FusionConfig::default());
    for _ in 0..50 {
        with.fuse(&[SensorInput {
            value: 10.0,
            noise: 0.1,
        }]);
        without.fuse(&[SensorInput {
            value: 10.0,
            noise: 0.1,
        }]);
    }
    let a = without.fuse(&[SensorInput {
        value: 10.1,
        noise: 0.01,
    }]);
    let b = with.fuse(&[
        SensorInput {
            value: 10.1,
            noise: 0.01,
        },
        SensorInput {
            value: 60.0,
            noise: 0.01,
        },
    ]);
    assert_eq!(
        a.to_bits(),
        b.to_bits(),
        "outlier moved the estimate {a} → {b}"
    );
}

// ───────────────────────── ring buffer ────────────────────────────────────

#[test]
fn ring_buffer_is_a_fifo_window_of_the_last_n_pushes() {
    let mut rb: RingBuffer<i32, 4> = RingBuffer::new();
    assert!(rb.is_empty() && rb.capacity() == 4);
    for i in 0..10 {
        rb.push(i);
        let expected: Vec<i32> = ((i - 3).max(0)..=i).collect();
        let got: Vec<i32> = rb.iter().copied().collect();
        assert_eq!(got, expected, "after pushing {i}");
        assert_eq!(rb.len(), expected.len());
        assert_eq!(rb.first(), expected.first());
        assert_eq!(rb.last(), expected.last());
        assert_eq!(rb.is_full(), expected.len() == 4);
    }
    assert_eq!(rb.get(0), Some(&6));
    assert_eq!(rb.get(3), Some(&9));
    assert_eq!(rb.get(4), None);
    rb.clear();
    assert!(rb.is_empty());
}

// ───────────────────────── SDF compression (feature `sdf`) ────────────────

#[cfg(feature = "sdf")]
#[test]
fn a_sampled_sphere_compresses_to_a_sphere_primitive_with_its_centre_and_radius() {
    use alice_edge::sdf_compress::{
        compress_point_cloud, CompressConfig, CompressedSdf, PrimitiveKind,
    };
    let (cx, cy, cz, r) = (0.3f32, -0.2f32, 1.1f32, 0.5f32);
    let mut pts = Vec::new();
    for i in 0..48 {
        for j in 0..24 {
            let theta = std::f32::consts::TAU * i as f32 / 48.0;
            let phi = std::f32::consts::PI * (j as f32 + 0.5) / 24.0;
            pts.push([
                cx + r * phi.sin() * theta.cos(),
                cy + r * phi.sin() * theta.sin(),
                cz + r * phi.cos(),
            ]);
        }
    }
    let (sdf, stats) = compress_point_cloud(&pts, &CompressConfig::default());
    assert_eq!(stats.input_points, pts.len());
    match sdf {
        CompressedSdf::Primitives { primitives, .. } => {
            let sphere = primitives
                .iter()
                .find(|p| p.kind == PrimitiveKind::Sphere)
                .expect("a sphere primitive");
            let p = sphere.params;
            assert!(
                (p[0] - cx).abs() < 0.02 && (p[1] - cy).abs() < 0.02 && (p[2] - cz).abs() < 0.02,
                "centre {:?}",
                &p[..3]
            );
            assert!((p[3] - r).abs() < 0.02, "radius {}", p[3]);
            assert!(sphere.mse < 1e-3);
            assert!(
                stats.primitive_inliers as f32 >= 0.8 * pts.len() as f32,
                "inliers {}",
                stats.primitive_inliers
            );
        }
        other => panic!("expected a primitive fit for a clean sphere, got {other:?}"),
    }
}
