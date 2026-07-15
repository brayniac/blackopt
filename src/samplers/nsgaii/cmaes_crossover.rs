use parking_lot::Mutex;
use rand::Rng;
use rand_distr::StandardNormal;

use super::crossover::{Crossover, RngCore};

/// CMA-ES-inspired crossover operator for NSGA-II.
///
/// Instead of pointwise crossover between two parents, maintains a running
/// covariance estimate of the Pareto-optimal parent distribution and samples
/// children from a Gaussian centered on the parents' mean.
///
/// This learns parameter correlations (e.g., gate_proj and up_proj should
/// have similar settings) and generates children that respect these correlations.
pub struct CmaEsCrossover {
    state: Mutex<Option<CovState>>,
    /// Exponential decay factor for covariance updates (0-1, higher = more memory).
    alpha: f64,
    /// Initial step size.
    sigma: f64,
}

struct CovState {
    n: usize,
    mean: Vec<f64>,
    /// Covariance matrix (n × n), stored flattened.
    cov: Vec<f64>,
    sigma: f64,
    n_updates: usize,
}

impl CovState {
    fn new(n: usize, sigma: f64) -> Self {
        // Initialize with identity covariance.
        let mut cov = vec![0.0; n * n];
        for i in 0..n {
            cov[i * n + i] = sigma * sigma;
        }
        Self {
            n,
            mean: vec![0.5; n],
            cov,
            sigma,
            n_updates: 0,
        }
    }

    /// Update the covariance estimate from a set of parent vectors.
    fn update(&mut self, parents: &[Vec<f64>], alpha: f64) {
        let k = parents.len();
        if k == 0 {
            return;
        }
        let n = self.n;

        // Compute parents' mean.
        let mut new_mean = vec![0.0; n];
        for p in parents {
            for i in 0..n {
                new_mean[i] += p[i];
            }
        }
        for i in 0..n {
            new_mean[i] /= k as f64;
        }

        // Compute sample covariance.
        let mut new_cov = vec![0.0; n * n];
        for p in parents {
            for i in 0..n {
                for j in i..n {
                    let v = (p[i] - new_mean[i]) * (p[j] - new_mean[j]);
                    new_cov[i * n + j] += v;
                    if i != j {
                        new_cov[j * n + i] += v;
                    }
                }
            }
        }
        let scale = 1.0 / (k as f64).max(1.0);
        for v in &mut new_cov {
            *v *= scale;
        }

        // Add regularization to prevent degenerate covariance.
        let reg = self.sigma * self.sigma * 0.01;
        for i in 0..n {
            new_cov[i * n + i] += reg;
        }

        // Exponential moving average update.
        if self.n_updates == 0 {
            self.mean = new_mean;
            self.cov = new_cov;
        } else {
            for i in 0..n {
                self.mean[i] = alpha * self.mean[i] + (1.0 - alpha) * new_mean[i];
            }
            for i in 0..n * n {
                self.cov[i] = alpha * self.cov[i] + (1.0 - alpha) * new_cov[i];
            }
        }
        self.n_updates += 1;
    }

    /// Sample a child from the current distribution using Cholesky decomposition.
    fn sample(&self, rng: &mut dyn RngCore) -> Vec<f64> {
        let n = self.n;

        // Cholesky decomposition of covariance: C = L @ L^T
        let mut l = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let mut sum = self.cov[i * n + j];
                for k in 0..j {
                    sum -= l[i * n + k] * l[j * n + k];
                }
                if i == j {
                    l[i * n + j] = sum.max(1e-20).sqrt();
                } else {
                    let div = l[j * n + j];
                    l[i * n + j] = if div.abs() > 1e-20 { sum / div } else { 0.0 };
                }
            }
        }

        // Sample z ~ N(0, I), then x = mean + L @ z
        let z: Vec<f64> = (0..n)
            .map(|_| {
                let v: f64 = rng.sample(StandardNormal);
                v
            })
            .collect();

        let mut child = vec![0.0; n];
        for i in 0..n {
            child[i] = self.mean[i];
            for j in 0..=i {
                child[i] += l[i * n + j] * z[j];
            }
            child[i] = child[i].clamp(0.0, 1.0);
        }
        child
    }
}

impl CmaEsCrossover {
    /// Create a new CMA-ES crossover.
    ///
    /// * `alpha` — EMA decay (0.5-0.9 typical). Higher = more memory of past distributions.
    /// * `sigma` — initial step size in \[0,1\] space (0.1-0.3 typical).
    pub fn new(alpha: Option<f64>, sigma: Option<f64>) -> Self {
        Self {
            state: Mutex::new(None),
            alpha: alpha.unwrap_or(0.7),
            sigma: sigma.unwrap_or(0.2),
        }
    }
}

impl Crossover for CmaEsCrossover {
    fn n_parents(&self) -> usize {
        // We accept 2 parents from tournament selection, but the crossover
        // uses the accumulated covariance from all elite parents over time.
        2
    }

    fn crossover(&self, parents: &[Vec<f64>], rng: &mut dyn RngCore) -> Vec<f64> {
        let n = parents[0].len();

        let mut state = self.state.lock();
        if state.is_none() {
            *state = Some(CovState::new(n, self.sigma));
        }
        let cov_state = state.as_mut().unwrap();

        // Update covariance with the selected parents.
        cov_state.update(parents, self.alpha);

        // Sample a child from the learned distribution.
        cov_state.sample(rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_cmaes_crossover_basic() {
        let cx = CmaEsCrossover::new(Some(0.7), Some(0.2));
        assert_eq!(cx.n_parents(), 2);

        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let p0 = vec![0.3, 0.3, 0.7, 0.7];
        let p1 = vec![0.4, 0.4, 0.6, 0.6];

        // First call initializes state.
        let child = cx.crossover(&[p0.clone(), p1.clone()], &mut rng);
        assert_eq!(child.len(), 4);
        for &v in &child {
            assert!((0.0..=1.0).contains(&v));
        }

        // Subsequent calls should refine the distribution.
        for _ in 0..10 {
            let child = cx.crossover(&[p0.clone(), p1.clone()], &mut rng);
            assert_eq!(child.len(), 4);
        }
    }

    #[test]
    fn test_cmaes_crossover_correlation() {
        // Parents have correlated parameters (dims 0,1 move together).
        let cx = CmaEsCrossover::new(Some(0.5), Some(0.15));
        let mut rng = ChaCha8Rng::seed_from_u64(42);

        // Feed many correlated parents.
        for i in 0..20 {
            let v = 0.3 + 0.02 * i as f64;
            let p0 = vec![v, v, 0.5, 0.5];
            let p1 = vec![v + 0.01, v + 0.01, 0.5, 0.5];
            let _ = cx.crossover(&[p0, p1], &mut rng);
        }

        // Sample children while continuing to feed correlated parents,
        // since crossover() updates the covariance on every call.
        let mut children = Vec::new();
        for i in 0..100 {
            let v = 0.3 + 0.004 * i as f64;
            let p0 = vec![v, v, 0.5, 0.5];
            let p1 = vec![v + 0.01, v + 0.01, 0.5, 0.5];
            children.push(cx.crossover(&[p0, p1], &mut rng));
        }

        // Compute correlation between dim 0 and dim 1.
        let mean0: f64 = children.iter().map(|c| c[0]).sum::<f64>() / children.len() as f64;
        let mean1: f64 = children.iter().map(|c| c[1]).sum::<f64>() / children.len() as f64;
        let cov01: f64 = children
            .iter()
            .map(|c| (c[0] - mean0) * (c[1] - mean1))
            .sum::<f64>()
            / children.len() as f64;
        let var0: f64 =
            children.iter().map(|c| (c[0] - mean0).powi(2)).sum::<f64>() / children.len() as f64;
        let var1: f64 =
            children.iter().map(|c| (c[1] - mean1).powi(2)).sum::<f64>() / children.len() as f64;
        let corr = cov01 / (var0.sqrt() * var1.sqrt());

        // Dims 0,1 should be positively correlated.
        assert!(corr > 0.3, "Expected positive correlation, got {corr}");
    }
}
