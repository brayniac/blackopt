//! Truncated normal distribution.
//!
//! Provides CDF, log-CDF, PPF (inverse CDF), log-PDF, and sampling for
//! the standard truncated normal distribution on \[a, b\].
//!
//! Uses numerically stable log-space computations throughout.

use std::f64::consts::{FRAC_1_SQRT_2, PI};

const LOG_SQRT_2PI: f64 = 0.9189385332046727; // ln(sqrt(2*pi))

const EPS: f64 = 1e-12;

/// Standard normal log-PDF: log phi(x) = -x^2/2 - ln(sqrt(2*pi))
fn norm_logpdf(x: f64) -> f64 {
    -0.5 * x * x - LOG_SQRT_2PI
}

/// Standard normal CDF: Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))
fn ndtr(x: f64) -> f64 {
    0.5 + 0.5 * libm::erf(x * FRAC_1_SQRT_2)
}

/// Log of standard normal CDF, computed stably.
fn log_ndtr(a: f64) -> f64 {
    if a > 6.0 {
        // Phi(a) ~ 1, use: log(1 - Phi(-a))
        let v = ndtr(-a);
        if v <= 0.0 {
            0.0
        } else {
            (-v).ln_1p()
        }
    } else if a > -20.0 {
        let v = ndtr(a);
        if v <= 0.0 {
            f64::NEG_INFINITY
        } else {
            v.ln()
        }
    } else {
        // Asymptotic expansion for very negative a:
        // log Phi(a) ≈ -a^2/2 - ln(sqrt(2pi)) - ln(-a) + log(1 - 1/a^2)
        let a2 = a * a;
        -0.5 * a2 - LOG_SQRT_2PI - (-a).ln() + (-1.0 / a2).ln_1p()
    }
}

/// log(exp(log_p) - exp(log_q)) computed stably, assuming log_p >= log_q.
fn log_diff(log_p: f64, log_q: f64) -> f64 {
    if log_p == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    log_p + (-(log_q - log_p).exp()).ln_1p()
}

/// log(Phi(b) - Phi(a)), the log of Gaussian mass in [a, b].
pub fn log_gauss_mass(a: f64, b: f64) -> f64 {
    if a >= b {
        return f64::NEG_INFINITY;
    }
    if b <= 0.0 {
        // Both in left tail
        log_diff(log_ndtr(b), log_ndtr(a))
    } else if a >= 0.0 {
        // Both in right tail — use symmetry: Phi(b)-Phi(a) = Phi(-a)-Phi(-b)
        log_diff(log_ndtr(-a), log_ndtr(-b))
    } else {
        // Spans zero: Phi(b) - Phi(a) = 1 - Phi(-b) - (1 - Phi(b_negative_part))
        // = 1 - ndtr(a) - ndtr(-b)
        let tail_mass = ndtr(a) + ndtr(-b);
        if tail_mass >= 1.0 {
            f64::NEG_INFINITY
        } else {
            (-tail_mass).ln_1p()
        }
    }
}

/// Inverse of log_ndtr: find x such that log(Phi(x)) = log_p.
/// Uses Newton's method.
fn ndtri_exp(log_p: f64) -> f64 {
    if log_p > -EPS {
        return 8.2; // saturated near log(1) = 0
    }
    if log_p < -1e15 {
        return -100.0;
    }

    // Initial guess using inverse normal approximation.
    // For log_p close to 0 (p ~ 1), use right-tail approximation.
    // For very negative log_p, use: x ≈ -sqrt(-2 * log_p)
    let mut x = if log_p < -5.0 {
        // Deep left tail: Phi(x) ~ exp(-x^2/2) / (|x| * sqrt(2pi))
        // log(Phi(x)) ~ -x^2/2 - ln(|x|) - ln(sqrt(2pi))
        // Approximate: x ~ -sqrt(-2 * log_p)
        -(-2.0 * log_p - LOG_SQRT_2PI).sqrt()
    } else {
        // Moderate region: use logistic approximation
        // Phi(x) ≈ 1/(1 + exp(-x * sqrt(pi/8)))
        let c = (PI / 8.0).sqrt();
        let p = log_p.exp();
        if p > 0.0 && p < 1.0 {
            (p / (1.0 - p)).ln() / c
        } else {
            -6.0
        }
    };

    // Newton iterations: f(x) = log_ndtr(x) - log_p
    // f'(x) = phi(x) / Phi(x) = exp(norm_logpdf(x) - log_ndtr(x))
    for _ in 0..100 {
        let f = log_ndtr(x) - log_p;
        if f.abs() < 1e-12 {
            break;
        }
        let log_deriv = norm_logpdf(x) - log_ndtr(x);
        let deriv = log_deriv.exp();
        if deriv < 1e-300 {
            break;
        }
        let step = f / deriv;
        x -= step.clamp(-2.0, 2.0);
    }
    x
}

/// PPF (quantile function) of the standard truncated normal on [a, b].
///
/// Returns x such that P(a <= Z <= x) / P(a <= Z <= b) = q.
pub fn ppf(q: f64, a: f64, b: f64) -> f64 {
    if q <= 0.0 {
        return a;
    }
    if q >= 1.0 {
        return b;
    }
    if (a - b).abs() < EPS {
        return f64::NAN;
    }

    let lm = log_gauss_mass(a, b);

    if a < 0.0 {
        // Left case:
        // We want x such that Phi(x) = Phi(a) + q * (Phi(b) - Phi(a))
        // In log: log(Phi(x)) = logaddexp(log(Phi(a)), log(q) + lm)
        let log_phi_a = log_ndtr(a);
        let log_q_mass = q.ln() + lm;
        let log_phi_x = logaddexp(log_phi_a, log_q_mass);
        ndtri_exp(log_phi_x)
    } else {
        // Right case: work with upper tail for numerical stability.
        // Phi(-x) = Phi(-b) + (1-q) * (Phi(b) - Phi(a))
        let log_phi_neg_b = log_ndtr(-b);
        let log_1mq_mass = (1.0 - q).ln() + lm;
        let log_phi_neg_x = logaddexp(log_phi_neg_b, log_1mq_mass);
        -ndtri_exp(log_phi_neg_x)
    }
}

/// Sample from truncated normal: TN(a, b, loc, scale).
///
/// Uses inverse-CDF method: draw uniform q, then ppf(q, a_std, b_std) * scale + loc.
pub fn rvs(
    a: &[f64],
    b: &[f64],
    loc: &[f64],
    scale: &[f64],
    rng: &mut impl rand::Rng,
) -> Vec<f64> {
    let n = a.len();
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let a_std = (a[i] - loc[i]) / scale[i];
        let b_std = (b[i] - loc[i]) / scale[i];
        let q: f64 = rng.r#gen();
        let x = ppf(q, a_std, b_std) * scale[i] + loc[i];
        result.push(x.clamp(a[i], b[i]));
    }
    result
}

/// Log-PDF of truncated normal TN(a, b, loc, scale).
pub fn logpdf(x: f64, a: f64, b: f64, loc: f64, scale: f64) -> f64 {
    if (a - b).abs() < EPS {
        return f64::NAN;
    }
    let z = (x - loc) / scale;
    let a_std = (a - loc) / scale;
    let b_std = (b - loc) / scale;
    if z < a_std - EPS || z > b_std + EPS {
        return f64::NEG_INFINITY;
    }
    norm_logpdf(z) - log_gauss_mass(a_std, b_std) - scale.ln()
}

fn logaddexp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ndtr_basic() {
        assert!((ndtr(0.0) - 0.5).abs() < 1e-10);
        assert!(ndtr(-40.0) < 1e-100);
        assert!((ndtr(40.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_log_gauss_mass_full_range() {
        // log(Phi(10) - Phi(-10)) ≈ 0
        let m = log_gauss_mass(-10.0, 10.0);
        assert!(m.abs() < 1e-6, "got {m}");
    }

    #[test]
    fn test_log_gauss_mass_half() {
        // log(Phi(0) - Phi(-10)) ≈ log(0.5)
        let m = log_gauss_mass(-10.0, 0.0);
        assert!((m - (-std::f64::consts::LN_2)).abs() < 1e-4, "got {m}");
    }

    #[test]
    fn test_ppf_midpoint() {
        // Median of standard normal truncated to [-10, 10] ≈ 0
        let x = ppf(0.5, -10.0, 10.0);
        assert!(x.abs() < 0.05, "got {x}");
    }

    #[test]
    fn test_ppf_boundaries() {
        let a = -2.0;
        let b = 2.0;
        assert!((ppf(0.0, a, b) - a).abs() < 1e-10);
        assert!((ppf(1.0, a, b) - b).abs() < 1e-10);
    }

    #[test]
    fn test_ppf_monotone() {
        let a = -1.0;
        let b = 1.0;
        let mut prev = a;
        for i in 1..10 {
            let q = i as f64 / 10.0;
            let x = ppf(q, a, b);
            assert!(x >= prev - 1e-10, "ppf not monotone at q={q}: {x} < {prev}");
            prev = x;
        }
    }

    #[test]
    fn test_rvs_within_bounds() {
        let mut rng = rand::thread_rng();
        let n = 100;
        let a = vec![-1.0; n];
        let b = vec![1.0; n];
        let loc = vec![0.0; n];
        let scale = vec![1.0; n];
        let samples = rvs(&a, &b, &loc, &scale, &mut rng);
        for &s in &samples {
            assert!(
                s >= -1.0 - 1e-10 && s <= 1.0 + 1e-10,
                "sample {s} out of bounds"
            );
        }
    }

    #[test]
    fn test_logpdf_in_range() {
        let lp = logpdf(0.0, -1.0, 1.0, 0.0, 1.0);
        assert!(lp.is_finite());
        assert!(lp > -10.0);
    }

    #[test]
    fn test_logpdf_out_of_range() {
        let lp = logpdf(2.0, -1.0, 1.0, 0.0, 1.0);
        assert!(lp == f64::NEG_INFINITY);
    }

    #[test]
    fn test_logpdf_consistency_with_ppf() {
        // Numerical integration of PDF over [a, b] should ≈ 1.
        let a = -2.0;
        let b = 2.0;
        let n = 1000;
        let dx = (b - a) / n as f64;
        let mut integral = 0.0;
        for i in 0..n {
            let x = a + (i as f64 + 0.5) * dx;
            integral += logpdf(x, a, b, 0.0, 1.0).exp() * dx;
        }
        assert!(
            (integral - 1.0).abs() < 0.01,
            "integral = {integral}, expected ~1.0"
        );
    }

    #[test]
    fn test_ndtri_exp_roundtrip() {
        for &y in &[-0.1, -1.0, -5.0, -10.0, -20.0] {
            let x = ndtri_exp(y);
            let back = log_ndtr(x);
            assert!(
                (back - y).abs() < 1e-6,
                "roundtrip failed: y={y}, x={x}, back={back}"
            );
        }
    }

    #[test]
    fn test_ppf_various_quantiles() {
        // Check PPF for a narrower range.
        let a = -1.0;
        let b = 1.0;
        for &q in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            let x = ppf(q, a, b);
            assert!(
                x >= a - 1e-10 && x <= b + 1e-10,
                "ppf({q}, {a}, {b}) = {x} out of range"
            );
        }
    }
}
