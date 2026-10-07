//! An Edge fit carried as a law: coefficients, evidence, measured residual,
//! valid range and provenance, then a verdict on later windows
//!
//! ```text
//! cargo run --example edge_law --features law
//! ```

use alice_edge::law::{linear_law, sample_points, IngestPolicy, LawError, OracleCase, Provenance};
use alice_edge::q16_linear::fit_linear_fixed;

fn main() -> Result<(), LawError> {
    // temperature × 100 over 16 samples: 25.00 °C rising 0.05 °C per sample,
    // with ±0.02 °C of alternating read noise
    let window: Vec<i32> = (0..16)
        .map(|i| 2500 + 5 * i + if i % 2 == 0 { 2 } else { -2 })
        .collect();

    let (slope_q, intercept_q) = fit_linear_fixed(&window);
    println!("transmitted (Q16.16): slope_q = {slope_q}, intercept_q = {intercept_q}");

    let law = linear_law(
        &window,
        Provenance::new("sensor 3, window 17", "alice-edge fit_linear_fixed"),
    )?
    // the reading at the start of the window, known from the reference probe
    .with_oracle(OracleCase::new(0.0, 2500.0, 3.0, "reference probe, t = 0"));

    let d = law.domain();
    let r = law.residual();
    println!(
        "law: coefficients in u = x/{} = {:?}",
        d.hi,
        law.coefficients()
    );
    println!("valid range: x in [{}, {}]", d.lo, d.hi);
    println!(
        "residual over {} samples: rms = {:.4}, max = {:.4}",
        r.n, r.rms, r.max_abs
    );
    println!("y(7.5) = {:.4}", law.evaluate(7.5)?);
    println!("y(16) = {:?} (outside the window)", law.evaluate(16.0));
    for o in law.check_oracles() {
        println!("oracle: value = {:?}, passed = {}", o.value, o.passed);
    }

    let policy = IngestPolicy {
        abs_tolerance: 3.0,
        break_factor: 4.0,
    };
    // the same trend measured again with the opposite noise phase
    let again: Vec<i32> = (0..16)
        .map(|i| 2500 + 5 * i + if i % 2 == 0 { -2 } else { 2 })
        .collect();
    // the sensor starts heating: a quadratic term appears
    let heating: Vec<i32> = (0..16).map(|i| 2500 + 5 * i + i * i).collect();
    println!(
        "same trend again: {:?}",
        law.ingest(&sample_points(&again), &policy)
    );
    println!(
        "heating window:   {:?}",
        law.ingest(&sample_points(&heating), &policy)
    );
    Ok(())
}
