//! Gaussian Process with Matern 5/2 kernel, Cholesky inference, and Thompson sampling.

use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

/// A Gaussian Process with Matern 5/2 ARD kernel.
pub struct GaussianProcess {
    /// Training inputs, each row is a data point in \[0,1\]^d.
    x_train: Vec<Vec<f64>>,
    /// Per-dimension lengthscales.
    lengthscales: Vec<f64>,
    /// Output scale (signal variance).
    output_scale: f64,
    /// Cholesky factor L of (K + noise*I).
    chol_l: Vec<Vec<f64>>,
    /// Alpha = L^{-T} L^{-1} y  (precomputed for predictions).
    alpha: Vec<f64>,
}

impl GaussianProcess {
    /// Fit a GP to training data with automatic hyperparameter selection (expensive).
    ///
    /// `x_train`: N x D matrix (each inner Vec is one D-dimensional point).
    /// `y_train`: N target values (will be standardized internally).
    pub fn fit(x_train: Vec<Vec<f64>>, y_train: Vec<f64>) -> Option<Self> {
        let n = x_train.len();
        if n == 0 || y_train.len() != n {
            return None;
        }
        let n_dims = x_train[0].len();
        if n_dims == 0 {
            return None;
        }

        let y_std_train = standardize(&y_train);

        // Optimize hyperparameters via coordinate-wise grid search on marginal likelihood.
        let (lengthscales, output_scale, noise_var) =
            optimize_hyperparams(&x_train, &y_std_train, n_dims);

        Self::fit_inner(x_train, y_std_train, lengthscales, output_scale, noise_var)
    }

    fn fit_inner(
        x_train: Vec<Vec<f64>>,
        y_std_train: Vec<f64>,
        lengthscales: Vec<f64>,
        output_scale: f64,
        noise_var: f64,
    ) -> Option<Self> {
        let k = kernel_matrix(&x_train, &x_train, &lengthscales, output_scale);
        let chol_l = cholesky_with_jitter(&k, noise_var)?;
        let alpha = cholesky_solve(&chol_l, &y_std_train);

        Some(Self {
            x_train,
            lengthscales,
            output_scale,
            chol_l,
            alpha,
        })
    }

    /// Predict mean and variance at a set of test points.
    pub fn predict(&self, x_test: &[Vec<f64>]) -> (Vec<f64>, Vec<f64>) {
        let n_test = x_test.len();
        let k_star = kernel_matrix(x_test, &self.x_train, &self.lengthscales, self.output_scale);

        let mut means = Vec::with_capacity(n_test);
        let mut variances = Vec::with_capacity(n_test);

        for i in 0..n_test {
            // mean = k_star[i] . alpha
            let mu: f64 = k_star[i]
                .iter()
                .zip(self.alpha.iter())
                .map(|(k, a)| k * a)
                .sum();
            means.push(mu);

            // v = L^{-1} k_star[i]
            let v = forward_solve(&self.chol_l, &k_star[i]);
            let k_ss = self.output_scale; // k(x*, x*) for Matern 5/2 with r=0
            let var = (k_ss - v.iter().map(|vi| vi * vi).sum::<f64>()).max(1e-10);
            variances.push(var);
        }

        (means, variances)
    }

    /// Draw a Thompson sample: evaluate a single draw from the GP posterior at each test point.
    pub fn thompson_sample(&self, x_test: &[Vec<f64>], rng: &mut ChaCha8Rng) -> Vec<f64> {
        let (means, variances) = self.predict(x_test);
        means
            .into_iter()
            .zip(variances)
            .map(|(mu, var)| {
                let z: f64 = rng.sample(StandardNormal);
                mu + z * var.sqrt()
            })
            .collect()
    }
}

/// Standardize targets to zero mean, unit variance.
fn standardize(y: &[f64]) -> Vec<f64> {
    let n = y.len() as f64;
    let mean = y.iter().sum::<f64>() / n;
    let std = {
        let var = y.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
        var.sqrt().max(1e-8)
    };
    y.iter().map(|v| (v - mean) / std).collect()
}

// ── Matern 5/2 kernel ───────────────────────────────────────────────────────

/// Matern 5/2 kernel between two points with ARD lengthscales.
fn matern52(x: &[f64], y: &[f64], lengthscales: &[f64], output_scale: f64) -> f64 {
    let r2: f64 = x
        .iter()
        .zip(y.iter())
        .zip(lengthscales.iter())
        .map(|((xi, yi), l)| {
            let d = (xi - yi) / l;
            d * d
        })
        .sum();
    let r = r2.sqrt();
    let sqrt5_r = 5.0_f64.sqrt() * r;
    output_scale * (1.0 + sqrt5_r + 5.0 / 3.0 * r2) * (-sqrt5_r).exp()
}

/// Build the kernel matrix K[i][j] = k(X[i], X[j]).
fn kernel_matrix(
    x1: &[Vec<f64>],
    x2: &[Vec<f64>],
    lengthscales: &[f64],
    output_scale: f64,
) -> Vec<Vec<f64>> {
    let n1 = x1.len();
    let n2 = x2.len();
    let mut k = vec![vec![0.0; n2]; n1];
    for i in 0..n1 {
        for j in 0..n2 {
            k[i][j] = matern52(&x1[i], &x2[j], lengthscales, output_scale);
        }
    }
    k
}

// ── Cholesky factorization ──────────────────────────────────────────────────

/// Cholesky decomposition of (K + noise_var * I) with jitter escalation.
fn cholesky_with_jitter(k: &[Vec<f64>], noise_var: f64) -> Option<Vec<Vec<f64>>> {
    let n = k.len();
    let mut a: Vec<Vec<f64>> = k.to_vec();
    for i in 0..n {
        a[i][i] += noise_var;
    }

    // Try increasing jitter levels on failure.
    for &jitter in &[0.0, 1e-8, 1e-6, 1e-4, 1e-3, 1e-2] {
        let mut m = a.clone();
        if jitter > 0.0 {
            for i in 0..n {
                m[i][i] += jitter;
            }
        }
        if let Some(l) = cholesky(&m) {
            return Some(l);
        }
    }
    None
}

/// Standard Cholesky decomposition (returns L such that A = L L^T).
fn cholesky(a: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = a.len();
    let mut l = vec![vec![0.0; n]; n];
    for j in 0..n {
        let mut sum = 0.0;
        for k in 0..j {
            sum += l[j][k] * l[j][k];
        }
        let diag = a[j][j] - sum;
        if diag <= 0.0 {
            return None;
        }
        l[j][j] = diag.sqrt();
        for i in (j + 1)..n {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i][k] * l[j][k];
            }
            l[i][j] = (a[i][j] - sum) / l[j][j];
        }
    }
    Some(l)
}

/// Forward substitution: solve L x = b for x.
fn forward_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += l[i][j] * x[j];
        }
        x[i] = (b[i] - sum) / l[i][i];
    }
    x
}

/// Backward substitution: solve L^T x = b for x.
fn backward_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += l[j][i] * x[j];
        }
        x[i] = (b[i] - sum) / l[i][i];
    }
    x
}

/// Solve (K + noise*I) x = b via Cholesky: alpha = L^{-T} L^{-1} b.
fn cholesky_solve(l: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let z = forward_solve(l, b);
    backward_solve(l, &z)
}

// ── Hyperparameter optimization ─────────────────────────────────────────────

/// Log marginal likelihood: -0.5 * (y^T alpha + sum(log(diag(L))) * 2 + n*log(2pi))
fn log_marginal_likelihood(l: &[Vec<f64>], alpha: &[f64], y: &[f64]) -> f64 {
    let n = y.len() as f64;
    let data_fit: f64 = y.iter().zip(alpha.iter()).map(|(yi, ai)| yi * ai).sum();
    let log_det: f64 = (0..l.len()).map(|i| l[i][i].ln()).sum();
    -0.5 * (data_fit + 2.0 * log_det + n * (2.0 * std::f64::consts::PI).ln())
}

/// Coordinate-wise grid search over hyperparameters.
fn optimize_hyperparams(
    x_train: &[Vec<f64>],
    y_train: &[f64],
    n_dims: usize,
) -> (Vec<f64>, f64, f64) {
    // Log-spaced grids.
    let lengthscale_grid: Vec<f64> = (-3..=2).map(|i| 10.0_f64.powf(i as f64 * 0.5)).collect();
    let output_scale_grid: Vec<f64> = vec![0.1, 0.5, 1.0, 2.0, 5.0];
    let noise_grid: Vec<f64> = vec![1e-4, 1e-3, 1e-2, 0.1];

    let mut best_lml = f64::NEG_INFINITY;
    let mut best_ls = vec![0.2; n_dims];
    let mut best_os = 1.0;
    let mut best_nv = 1e-3;

    // Coarse search: shared lengthscale.
    for &ls_val in &lengthscale_grid {
        let ls = vec![ls_val; n_dims];
        for &os in &output_scale_grid {
            for &nv in &noise_grid {
                let k = kernel_matrix(x_train, x_train, &ls, os);
                if let Some(l) = cholesky_with_jitter(&k, nv) {
                    let alpha = cholesky_solve(&l, y_train);
                    let lml = log_marginal_likelihood(&l, &alpha, y_train);
                    if lml > best_lml {
                        best_lml = lml;
                        best_ls = ls.clone();
                        best_os = os;
                        best_nv = nv;
                    }
                }
            }
        }
    }

    // Fine-tune: per-dimension lengthscale sweep (one dim at a time).
    if x_train.len() >= 2 * n_dims {
        for d in 0..n_dims {
            let mut local_best_lml = best_lml;
            let mut local_best_ls = best_ls[d];
            for &ls_val in &lengthscale_grid {
                let mut ls = best_ls.clone();
                ls[d] = ls_val;
                let k = kernel_matrix(x_train, x_train, &ls, best_os);
                if let Some(l) = cholesky_with_jitter(&k, best_nv) {
                    let alpha = cholesky_solve(&l, y_train);
                    let lml = log_marginal_likelihood(&l, &alpha, y_train);
                    if lml > local_best_lml {
                        local_best_lml = lml;
                        local_best_ls = ls_val;
                    }
                }
            }
            best_ls[d] = local_best_ls;
            best_lml = local_best_lml;
        }
    }

    (best_ls, best_os, best_nv)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_gp_fit_predict_sin() {
        // Fit GP to sin(x) on [0, 1].
        let n = 20;
        let x_train: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64 / (n - 1) as f64]).collect();
        let y_train: Vec<f64> = x_train
            .iter()
            .map(|x| (x[0] * std::f64::consts::TAU).sin())
            .collect();

        let gp = GaussianProcess::fit(x_train, y_train).expect("GP fit should succeed");

        // Predict at training points — should be close.
        let x_test: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64 / 4.0]).collect();
        let (means, variances) = gp.predict(&x_test);

        // Variance near training data should be small.
        for &v in &variances {
            assert!(v < 1.0, "variance should be small near data, got {v}");
        }

        // Mean should approximate sin.
        for (i, x) in x_test.iter().enumerate() {
            let expected = (x[0] * std::f64::consts::TAU).sin();
            let err = (means[i] - expected).abs();
            assert!(
                err < 1.0,
                "prediction error too large at x={}: expected {expected}, got {}",
                x[0],
                means[i]
            );
        }
    }

    #[test]
    fn test_gp_thompson_sample() {
        let x_train = vec![vec![0.0], vec![0.5], vec![1.0]];
        let y_train = vec![0.0, 1.0, 0.0];
        let gp = GaussianProcess::fit(x_train, y_train).expect("GP fit should succeed");

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let x_test = vec![vec![0.25], vec![0.5], vec![0.75]];
        let samples = gp.thompson_sample(&x_test, &mut rng);
        assert_eq!(samples.len(), 3);
        // Near x=0.5 (training point), sample should be close to 1.0.
        assert!(
            (samples[1] - 1.0).abs() < 1.5,
            "Thompson sample at training point should be near target"
        );
    }

    #[test]
    fn test_variance_decreases_near_data() {
        let x_train = vec![vec![0.5]];
        let y_train = vec![1.0];
        let gp = GaussianProcess::fit(x_train, y_train).expect("GP fit should succeed");

        let x_near = vec![vec![0.5]];
        let x_far = vec![vec![0.0]];
        let (_, var_near) = gp.predict(&x_near);
        let (_, var_far) = gp.predict(&x_far);
        assert!(
            var_near[0] < var_far[0],
            "variance should be smaller near data: near={}, far={}",
            var_near[0],
            var_far[0]
        );
    }
}
