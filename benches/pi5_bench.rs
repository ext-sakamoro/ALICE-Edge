//! ALICE-Edge benchmarks for Raspberry Pi 5 (Cortex-A76)
//!
//! Run on Pi 5:
//!   cargo bench
//!
//! Run on any host:
//!   cargo bench
//!
//! Author: Moroya Sakamoto

use alice_edge::{
    compute_residual_error, evaluate_linear_fixed, fit_constant_fixed, fit_linear_fixed,
    int_to_q16, q16_to_int, should_use_linear, Q16_SHIFT,
};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn generate_linear_data(n: usize) -> Vec<i32> {
    // y = 10x + 2500 + noise
    (0..n)
        .map(|i| 2500 + (i as i32) * 10 + (i as i32 % 7) - 3)
        .collect()
}

fn generate_constant_data(n: usize) -> Vec<i32> {
    // y = 2500 + noise
    (0..n).map(|i| 2500 + (i as i32 % 5) - 2).collect()
}

fn bench_fit_linear(c: &mut Criterion) {
    let mut group = c.benchmark_group("fit_linear_fixed");

    for &n in &[10, 100, 500, 1000, 4096] {
        let data = generate_linear_data(n);
        group.bench_with_input(BenchmarkId::new("samples", n), &data, |b, data| {
            b.iter(|| fit_linear_fixed(black_box(data)))
        });
    }

    group.finish();
}

fn bench_fit_constant(c: &mut Criterion) {
    let mut group = c.benchmark_group("fit_constant_fixed");

    for &n in &[10, 100, 500, 1000, 4096] {
        let data = generate_constant_data(n);
        group.bench_with_input(BenchmarkId::new("samples", n), &data, |b, data| {
            b.iter(|| fit_constant_fixed(black_box(data)))
        });
    }

    group.finish();
}

fn bench_evaluate(c: &mut Criterion) {
    let slope = 655360; // 10.0 in Q16.16
    let intercept = 163840000; // 2500.0 in Q16.16

    c.bench_function("evaluate_linear_fixed", |b| {
        b.iter(|| {
            for x in 0..1000 {
                black_box(evaluate_linear_fixed(
                    black_box(slope),
                    black_box(intercept),
                    black_box(x),
                ));
            }
        })
    });
}

fn bench_residual_error(c: &mut Criterion) {
    let data = generate_linear_data(1000);
    let (slope, intercept) = fit_linear_fixed(&data);

    c.bench_function("compute_residual_error/1000", |b| {
        b.iter(|| compute_residual_error(black_box(&data), black_box(slope), black_box(intercept)))
    });
}

fn bench_model_selection(c: &mut Criterion) {
    let mut group = c.benchmark_group("should_use_linear");

    let linear = generate_linear_data(1000);
    let constant = generate_constant_data(1000);

    group.bench_function("linear_data", |b| {
        b.iter(|| should_use_linear(black_box(&linear)))
    });
    group.bench_function("constant_data", |b| {
        b.iter(|| should_use_linear(black_box(&constant)))
    });

    group.finish();
}

fn bench_q16_conversion(c: &mut Criterion) {
    c.bench_function("int_to_q16/1000", |b| {
        b.iter(|| {
            for i in 0..1000 {
                black_box(int_to_q16(black_box(i)));
            }
        })
    });

    c.bench_function("q16_to_int/1000", |b| {
        b.iter(|| {
            for i in 0..1000 {
                black_box(q16_to_int(black_box(i * 65536)));
            }
        })
    });
}

fn bench_batch_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");

    for &n in &[100, 500, 1000] {
        let data = generate_linear_data(n);
        group.bench_with_input(BenchmarkId::new("samples", n), &data, |b, data| {
            b.iter(|| {
                // Full pipeline: fit → evaluate → error check
                let (slope, intercept) = fit_linear_fixed(black_box(data));
                let _error = compute_residual_error(data, slope, intercept);
                let _last = evaluate_linear_fixed(slope, intercept, (data.len() - 1) as i32);
                (slope, intercept)
            })
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_fit_linear,
    bench_fit_constant,
    bench_evaluate,
    bench_residual_error,
    bench_model_selection,
    bench_q16_conversion,
    bench_batch_pipeline,
);
criterion_main!(benches);
