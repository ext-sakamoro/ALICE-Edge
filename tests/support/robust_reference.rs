// Reference model of `alice_edge::fit_linear_robust`, written from the rule
// in its documentation and nothing else. Shared through `include!` by
// tests/robust_contract.rs and fuzz/fuzz_targets/fuzz_fit_linear.rs, so the
// unit tests and the fuzz target check the same contract.
//
// The rule, with the arithmetic the crate uses:
//
// - `k` is a plain integer multiplier of the MAD (`3` = three MADs, not
//   Q16.16); `k <= 0` is refused with `RobustError::NonPositiveK { k }`
//   before the window is looked at, so the rule below is for `k > 0` only
// - windows of fewer than 3 samples: the plain fit `fit_linear_fixed(data)`
// - otherwise start from `fit = fit_linear_fixed(data)` and repeat at most
//   `ROBUST_MAX_ROUNDS` times:
//   - residual `r_i = i64(int_to_q16(data[i])) - i64(evaluate_linear_fixed(
//     slope, intercept, i as i32))` (Q16.16, in `i64`)
//   - centre: `r_i -= median`, where `median` is `sorted(r)[n / 2]`
//   - `MAD = sorted(|r_i|)[n / 2]`; if `MAD == 0` the current fit is returned
//     (matches the current implementation)
//   - sample `i` is an outlier when `|r_i| > i64(k).saturating_mul(MAD)`
//     (strict `>`)
//   - the filled series takes `data[i]` for a kept sample and the line
//     prediction `q16_to_int(evaluate_linear_fixed(slope, intercept, i as
//     i32))` (Q16.16 floored by an arithmetic shift) for an outlier
//   - `fit = fit_linear_fixed(filled)`; stop when the outlier set equals
//     the previous round's (the set before the first round is empty)
//
// All values are integers, so there is no NaN to exclude.

use alice_edge::{evaluate_linear_fixed, fit_linear_fixed, int_to_q16, q16_to_int};

/// Rounds of fit → fill → refit before the loop gives up.
pub const ROBUST_MAX_ROUNDS: usize = 8;

/// The line prediction for sample `i` as the filled series stores it.
pub fn line_prediction(fit: (i32, i32), i: usize) -> i32 {
    q16_to_int(evaluate_linear_fixed(fit.0, fit.1, i as i32))
}

/// Median-centred Q16.16 residuals of `data` against `fit`, and their MAD.
pub fn centred_residuals(data: &[i32], fit: (i32, i32)) -> (Vec<i64>, i64) {
    let n = data.len();
    let mut res: Vec<i64> = (0..n)
        .map(|i| {
            i64::from(int_to_q16(data[i]))
                - i64::from(evaluate_linear_fixed(fit.0, fit.1, i as i32))
        })
        .collect();
    let mut sorted = res.clone();
    sorted.sort_unstable();
    let median = sorted[n / 2];
    for r in &mut res {
        *r -= median;
    }
    let mut abs: Vec<i64> = res.iter().map(|r| r.abs()).collect();
    abs.sort_unstable();
    (res, abs[n / 2])
}

/// `|r| > k · MAD`, with the product saturated in `i64`.
pub fn is_outlier(r: i64, k: i32, mad: i64) -> bool {
    r.abs() > i64::from(k).saturating_mul(mad)
}

/// How the reference loop ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RobustExit {
    /// fewer than 3 samples
    TooShort,
    /// the residual MAD of the current fit was 0
    MadZero,
    /// the outlier set did not change
    Converged,
    /// `ROBUST_MAX_ROUNDS` rounds without convergence
    RoundLimit,
}

/// The fit the rule produces, the last filled series and the fit whose
/// predictions filled it.
#[derive(Debug, Clone)]
pub struct RobustTrace {
    pub fit: (i32, i32),
    /// `None` when no round filled anything (`TooShort`, or `MadZero` in
    /// the first round)
    pub filled: Option<Vec<i32>>,
    pub predictor: (i32, i32),
    pub exit: RobustExit,
}

/// The rule for `k > 0`; a non-positive `k` has no rule, it is refused.
pub fn robust_reference(data: &[i32], k: i32) -> RobustTrace {
    assert!(k > 0, "the rule is defined for k > 0 only, got {k}");
    let n = data.len();
    let mut fit = fit_linear_fixed(data);
    if n < 3 {
        return RobustTrace {
            fit,
            filled: None,
            predictor: fit,
            exit: RobustExit::TooShort,
        };
    }
    let mut mask = vec![false; n];
    let mut filled = None;
    let mut predictor = fit;
    for _ in 0..ROBUST_MAX_ROUNDS {
        let (res, mad) = centred_residuals(data, fit);
        if mad == 0 {
            return RobustTrace {
                fit,
                filled,
                predictor,
                exit: RobustExit::MadZero,
            };
        }
        let new_mask: Vec<bool> = res.iter().map(|&r| is_outlier(r, k, mad)).collect();
        let series: Vec<i32> = (0..n)
            .map(|i| {
                if new_mask[i] {
                    line_prediction(fit, i)
                } else {
                    data[i]
                }
            })
            .collect();
        predictor = fit;
        fit = fit_linear_fixed(&series);
        filled = Some(series);
        let converged = new_mask == mask;
        mask = new_mask;
        if converged {
            return RobustTrace {
                fit,
                filled,
                predictor,
                exit: RobustExit::Converged,
            };
        }
    }
    RobustTrace {
        fit,
        filled,
        predictor,
        exit: RobustExit::RoundLimit,
    }
}

/// Checks `fit_linear_robust(data, k)` against the contract and panics with
/// the failing clause. Used by the unit tests and the fuzz target.
pub fn check_robust_contract(data: &[i32], k: i32) {
    // 0. k <= 0 is refused, whatever the window
    if k <= 0 {
        assert_eq!(
            alice_edge::fit_linear_robust(data, k),
            Err(alice_edge::RobustError::NonPositiveK { k }),
            "k = {k} must be refused"
        );
        return;
    }
    let got = alice_edge::fit_linear_robust(data, k).expect("k > 0 is accepted");
    let plain = fit_linear_fixed(data);

    // 1. no outlier in the first round (or MAD 0, or fewer than 3 samples)
    //    → the plain fit, computed from the first-round residuals only
    if data.len() < 3 {
        assert_eq!(got, plain, "fewer than 3 samples must give the plain fit");
    } else {
        let (res, mad) = centred_residuals(data, plain);
        if mad == 0 || res.iter().all(|&r| !is_outlier(r, k, mad)) {
            assert_eq!(
                got, plain,
                "no residual exceeds k·MAD (k = {k}, MAD = {mad}): the plain fit"
            );
        }
    }

    // 2. the fit is the one the documented rule produces
    let trace = robust_reference(data, k);
    assert_eq!(
        got, trace.fit,
        "fit differs from the documented rule (k = {k}, exit {:?})",
        trace.exit
    );

    // 3. the fit is the plain fit of the filled series, whose every element
    //    is the input sample or the line prediction at that index
    if let Some(filled) = &trace.filled {
        assert_eq!(filled.len(), data.len(), "filled series length");
        for (i, (&f, &x)) in filled.iter().zip(data).enumerate() {
            assert!(
                f == x || f == line_prediction(trace.predictor, i),
                "filled[{i}] = {f} is neither the sample {x} nor the line prediction"
            );
        }
        assert_eq!(
            got,
            fit_linear_fixed(filled),
            "fit is not the plain fit of the filled series"
        );
    }
}
