//! Oracles for `alice_edge::law` — an Edge linear fit as an
//! `alice_zip::law::SignalLaw`.
//!
//! Every expected value is a closed form written in this file (least squares
//! on `x = 0..n−1` solved by hand, the Q16 truncation of the intercept, the
//! verdict rules of `SignalLaw::ingest`), never a value computed by the
//! function under test.
//!
//! Conversion under test: Edge transmits `slope_q`, `intercept_q` (Q16.16) for
//! samples at `x = 0..n−1`; the law stores ascending coefficients in
//! `u = x / (n−1)`, so `c0 = intercept_q / 2¹⁶` and `c1 = slope_q·(n−1) / 2¹⁶`.
#![cfg(feature = "law")]

use alice_edge::law::{linear_law, sample_points, IngestPolicy, LawError, Provenance, Verdict};
use alice_edge::q16_linear::{evaluate_linear_fixed, fit_linear_fixed};

const Q16: f64 = 65_536.0;

fn prov() -> Provenance {
    Provenance::new("bench sensor A, window 1", "alice-edge fit_linear_fixed")
}

#[test]
fn exact_integer_line_is_recovered_with_zero_residual() {
    // y = 100 + 3x, x = 0..9  ⇒  c0 = 100, c1 = 3·9 = 27
    let samples: Vec<i32> = (0..10).map(|x| 100 + 3 * x).collect();
    let law = linear_law(&samples, prov()).unwrap();
    assert_eq!(law.coefficients(), &[100.0, 27.0]);
    assert_eq!(law.degree(), 1);
    let r = law.residual();
    assert_eq!((r.n, r.rms, r.max_abs), (10, 0.0, 0.0));
    // a condition between samples: x = 4.5 ⇒ 100 + 13.5
    assert_eq!(law.evaluate(4.5).unwrap(), 113.5);

    // falling line y = 500 − 7x, x = 0..6  ⇒  c0 = 500, c1 = −7·6
    let samples: Vec<i32> = (0..7).map(|x| 500 - 7 * x).collect();
    let law = linear_law(&samples, prov()).unwrap();
    assert_eq!(law.coefficients(), &[500.0, -42.0]);
    assert_eq!(law.residual().max_abs, 0.0);
}

#[test]
fn residual_of_a_parabola_equals_the_closed_form() {
    // y = x², x = 0..4: Σx = 10, Σy = 30, Σxy = 100, n = 5
    // slope = (5·100 − 10·30) / (5·30 − 10²) = 4, intercept = (30 − 4·10) / 5 = −2
    // residuals y − (4x − 2) = 2, −1, −2, −1, 2 ⇒ Σr² = 14, max |r| = 2
    let law = linear_law(&[0, 1, 4, 9, 16], prov()).unwrap();
    assert_eq!(law.coefficients(), &[-2.0, 16.0]); // c1 = 4·(5−1)
    let r = law.residual();
    assert_eq!(r.n, 5);
    assert_eq!(r.max_abs, 2.0);
    assert_eq!(r.rms, (14.0_f64 / 5.0).sqrt());
}

#[test]
fn residual_is_measured_against_the_samples_including_q16_truncation() {
    // y = 0, 0, 1: Σx = 3, Σy = 1, Σxy = 2, n = 3
    // slope = (3·2 − 3·1) / 6 = 1/2 → slope_q = 32768 (exact)
    // intercept = (2¹⁶ − 32768·3) / 3 = −32768/3 → truncated to −10922 (exact LS: −1/6)
    let (slope_q, intercept_q) = fit_linear_fixed(&[0, 0, 1]);
    assert_eq!((slope_q, intercept_q), (32_768, -10_922));

    let law = linear_law(&[0, 0, 1], prov()).unwrap();
    let c0 = -10_922.0 / Q16;
    let c1 = 32_768.0 * 2.0 / Q16;
    assert_eq!(law.coefficients(), &[c0, c1]);

    // residuals of the transmitted line: 10922, −21846, 10922 (in 2⁻¹⁶)
    let r = law.residual();
    let max_abs = 21_846.0 / Q16;
    let rms = ((2.0 * 10_922.0_f64.powi(2) + 21_846.0_f64.powi(2)) / 3.0).sqrt() / Q16;
    assert!((r.max_abs - max_abs).abs() <= 4.0 * f64::EPSILON * max_abs);
    assert!((r.rms - rms).abs() <= 4.0 * f64::EPSILON * rms);

    // the exact least-squares line has residuals 1/6, −1/3, 1/6 ⇒ rms = √(1/18);
    // the transmitted (truncated) line is measured, so it is strictly larger
    let ls_rms = (1.0_f64 / 18.0).sqrt();
    assert!(r.rms > ls_rms);
    assert!(r.rms - ls_rms < 1.0 / Q16);
}

/// Deterministic noisy integer samples (LCG, no external dependency)
fn noisy(n: usize, seed: u32) -> Vec<i32> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let noise = ((s >> 16) % 41) as i32 - 20;
            250 - 3 * i as i32 + noise
        })
        .collect()
}

#[test]
fn evaluate_reproduces_edge_reconstruction_at_every_sample() {
    for (n, seed) in [(2_usize, 1_u32), (5, 7), (37, 11), (256, 3)] {
        let samples = noisy(n, seed);
        let (slope_q, intercept_q) = fit_linear_fixed(&samples);
        let law = linear_law(&samples, prov()).unwrap();
        let [c0, c1] = [law.coefficients()[0], law.coefficients()[1]];

        // law evaluation is fl(fl(c1·fl(x/(n−1))) + c0): three roundings, so
        // |law − exact| ≤ ε|c0| + (3ε + 3ε² + ε³)|c1| ≤ 4ε(|c0| + |c1|)
        let bound = 4.0 * f64::EPSILON * (c0.abs() + c1.abs());
        // well below one Q16 step, the resolution Edge transmits at
        assert!(bound < 1.0 / Q16, "n = {n}: bound {bound}");

        for i in 0..n {
            // Edge's own receiver: y_q = slope_q·i + intercept_q (exact here, |y| ≪ 2¹⁵)
            let edge = f64::from(evaluate_linear_fixed(slope_q, intercept_q, i as i32)) / Q16;
            let got = law.evaluate(i as f64).unwrap();
            assert!(
                (got - edge).abs() <= bound,
                "n = {n}, i = {i}: law {got} vs edge {edge} (bound {bound})"
            );
        }
    }
}

#[test]
fn valid_range_is_the_sample_x_range_and_evidence_is_the_samples() {
    let samples = [12, 15, 11, 19, 14, 13];
    let law = linear_law(&samples, prov()).unwrap();
    let d = law.domain();
    assert_eq!((d.lo, d.hi), (0.0, 5.0));
    let expected: Vec<(f64, f64)> = vec![
        (0.0, 12.0),
        (1.0, 15.0),
        (2.0, 11.0),
        (3.0, 19.0),
        (4.0, 14.0),
        (5.0, 13.0),
    ];
    assert_eq!(law.evidence(), expected.as_slice());
    assert_eq!(sample_points(&samples), expected);
    assert_eq!(law.provenance(), &prov());
}

#[test]
fn evaluation_outside_the_sample_range_is_refused() {
    let samples: Vec<i32> = (0..10).map(|x| 100 + 3 * x).collect();
    let law = linear_law(&samples, prov()).unwrap();
    assert_eq!(law.evaluate(0.0).unwrap(), 100.0);
    assert_eq!(law.evaluate(9.0).unwrap(), 127.0);
    for x in [-1e-9, -1.0, 9.000_000_001, 10.0, f64::NAN, f64::INFINITY] {
        assert_eq!(law.evaluate(x), Err(LawError::OutOfRange), "x = {x}");
    }
}

#[test]
fn verdict_on_new_samples() {
    // law: y = 1 + 2x on x = 0..4 (residual 0)
    let law = linear_law(&[1, 3, 5, 7, 9], prov()).unwrap();
    let policy = IngestPolicy {
        abs_tolerance: 0.5,
        break_factor: 4.0,
    };

    // a second window on the same line
    assert_eq!(
        law.ingest(&sample_points(&[1, 3, 5, 7, 9]), &policy),
        Verdict::Supports { rms: 0.0 }
    );

    // a parabola y = x²: deviations from 1 + 2x are −1, −2, −1, 2, 7 ⇒ rms = √(59/5) ≈ 3.44,
    // above band·break_factor = 0.5·4 = 2, and no line fits old + new within 0.5
    match law.ingest(&sample_points(&[0, 1, 4, 9, 16]), &policy) {
        Verdict::Breaks { rms } => assert_eq!(rms, (59.0_f64 / 5.0).sqrt()),
        other => panic!("expected Breaks, got {other:?}"),
    }

    // a longer window reaches x = 5, outside [0, 4]
    assert_eq!(
        law.ingest(&sample_points(&[1, 3, 5, 7, 9, 11]), &policy),
        Verdict::OutOfRange { outside: 1 }
    );
    assert_eq!(
        law.ingest(&sample_points(&[]), &policy),
        Verdict::NoEvidence
    );
}

#[test]
fn degenerate_inputs_are_refused() {
    assert_eq!(linear_law(&[], prov()), Err(LawError::TooFewPoints));
    assert_eq!(linear_law(&[42], prov()), Err(LawError::TooFewPoints));
    assert!(sample_points(&[]).is_empty());

    // two samples are the smallest window with a range
    let law = linear_law(&[10, 14], prov()).unwrap();
    assert_eq!(law.coefficients(), &[10.0, 4.0]);
    assert_eq!((law.domain().lo, law.domain().hi), (0.0, 1.0));
}

#[test]
fn samples_outside_the_q16_range_show_up_in_the_residual() {
    // |y| ≥ 2¹⁵ is outside the Q16.16 contract of fit_linear_fixed: for
    // [40000, 40000] the intercept 40000·2¹⁶ wraps to −1673527296 = −25536·2¹⁶.
    // The law carries what Edge transmits, and the residual measured against
    // the samples exposes the wrap instead of hiding it: 40000 − (−25536) = 2¹⁶
    let law = linear_law(&[40_000, 40_000], prov()).unwrap();
    assert_eq!(law.coefficients(), &[-25_536.0, 0.0]);
    assert_eq!(law.residual().max_abs, 65_536.0);
    assert_eq!(law.residual().rms, 65_536.0);
}

#[test]
fn full_range_samples_do_not_panic_and_the_wrap_is_measured() {
    // M = i32::MAX, samples [M, −M−1, M]: Σy = M − 1, Σxy = M − 1, n = 3
    // slope_num = 3(M−1) − 3(M−1) = 0 ⇒ slope 0
    // intercept_q = (M−1)·2¹⁶/3 = 46912496074752, which is 2¹⁶·(−21846) mod 2³²
    // residual max = M − (−21846) = 2147505493
    const M: i32 = i32::MAX;
    let law = linear_law(&[M, i32::MIN, M], prov()).unwrap();
    assert_eq!(law.coefficients(), &[-21_846.0, 0.0]);
    assert_eq!(law.residual().n, 3);
    assert_eq!(law.residual().max_abs, 2_147_505_493.0);
}
