use std::collections::HashMap;
use std::sync::Arc;

use indexmap::IndexMap;
use parking_lot::Mutex;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::StandardNormal;

use crate::distributions::Distribution;
use crate::error::Result;
use crate::samplers::random::RandomSampler;
use crate::samplers::Sampler;
use crate::search_space::{IntersectionSearchSpace, SearchSpaceTransform};
use crate::study::StudyDirection;
use crate::trial::{FrozenTrial, TrialState};

/// CMA-ES (Covariance Matrix Adaptation Evolution Strategy) sampler.
///
/// A (mu/mu_w, lambda)-CMA-ES implementation for single-objective optimization.
pub struct CmaEsSampler {
    direction: StudyDirection,
    sigma0: Option<f64>,
    n_startup_trials: usize,
    popsize: Option<usize>,
    independent_sampler: Arc<dyn Sampler>,
    state: Mutex<Option<CmaState>>,
    rng: Mutex<ChaCha8Rng>,
    search_space: Mutex<IntersectionSearchSpace>,
}

impl std::fmt::Debug for CmaEsSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CmaEsSampler")
            .field("n_startup_trials", &self.n_startup_trials)
            .finish()
    }
}

/// Internal CMA-ES algorithm state.
struct CmaState {
    mean: Vec<f64>,
    sigma: f64,
    n: usize,
    // Covariance matrix (n x n)
    c: Vec<Vec<f64>>,
    // Evolution paths
    p_sigma: Vec<f64>,
    p_c: Vec<f64>,
    // Eigen decomposition cache
    eigenvalues: Vec<f64>,
    b: Vec<Vec<f64>>,
    // Strategy parameters
    lambda: usize,
    mu: usize,
    weights: Vec<f64>,
    mu_eff: f64,
    c_sigma: f64,
    d_sigma: f64,
    c_c: f64,
    c1: f64,
    c_mu: f64,
    chi_n: f64,
    // Generation tracking
    generation: usize,
    // Pending candidates from current ask batch
    pending: Vec<Vec<f64>>,
    pending_idx: usize,
    // Collected (params, value) pairs for current generation
    results: Vec<(Vec<f64>, f64)>,
    // Param names in order
    param_names: Vec<String>,
}

#[allow(clippy::needless_range_loop)]
impl CmaState {
    fn new(mean: Vec<f64>, sigma: f64, lambda: usize, param_names: Vec<String>) -> Self {
        let n = mean.len();
        let mu = lambda / 2;

        // Compute recombination weights
        let mut weights: Vec<f64> = (0..mu)
            .map(|i| ((lambda as f64 + 1.0) / 2.0).ln() - ((i + 1) as f64).ln())
            .collect();
        let w_sum: f64 = weights.iter().sum();
        for w in &mut weights {
            *w /= w_sum;
        }

        let mu_eff: f64 = 1.0 / weights.iter().map(|w| w * w).sum::<f64>();

        // Adaptation parameters
        let c_sigma = (mu_eff + 2.0) / (n as f64 + mu_eff + 5.0);
        let d_sigma = 1.0
            + 2.0 * (((mu_eff - 1.0) / (n as f64 + 1.0)).sqrt() - 1.0).max(0.0)
            + c_sigma;
        let c_c = (4.0 + mu_eff / n as f64) / (n as f64 + 4.0 + 2.0 * mu_eff / n as f64);
        let c1 = 2.0 / ((n as f64 + 1.3).powi(2) + mu_eff);
        let c_mu = (2.0 * (mu_eff - 2.0 + 1.0 / mu_eff)
            / ((n as f64 + 2.0).powi(2) + mu_eff))
            .min(1.0 - c1);
        let chi_n =
            (n as f64).sqrt() * (1.0 - 1.0 / (4.0 * n as f64) + 1.0 / (21.0 * (n as f64).powi(2)));

        // Identity covariance matrix
        let mut c = vec![vec![0.0; n]; n];
        for i in 0..n {
            c[i][i] = 1.0;
        }

        let eigenvalues = vec![1.0; n];
        let mut b = vec![vec![0.0; n]; n];
        for i in 0..n {
            b[i][i] = 1.0;
        }

        Self {
            mean,
            sigma,
            n,
            c,
            p_sigma: vec![0.0; n],
            p_c: vec![0.0; n],
            eigenvalues,
            b,
            lambda,
            mu,
            weights,
            mu_eff,
            c_sigma,
            d_sigma,
            c_c,
            c1,
            c_mu,
            chi_n,
            generation: 0,
            pending: Vec::new(),
            pending_idx: 0,
            results: Vec::new(),
            param_names,
        }
    }

    /// Sample a new candidate from the distribution.
    fn ask(&mut self, rng: &mut ChaCha8Rng) -> Vec<f64> {
        if self.pending_idx < self.pending.len() {
            let result = self.pending[self.pending_idx].clone();
            self.pending_idx += 1;
            return result;
        }

        // Generate a full batch of lambda candidates
        self.pending.clear();
        self.pending_idx = 0;

        for _ in 0..self.lambda {
            let z: Vec<f64> = (0..self.n)
                .map(|_| {
                    rng.sample(StandardNormal)
                })
                .collect();

            // y = B * D * z
            let mut y = vec![0.0; self.n];
            for i in 0..self.n {
                for j in 0..self.n {
                    y[i] += self.b[i][j] * self.eigenvalues[j].sqrt() * z[j];
                }
            }

            // x = mean + sigma * y
            let x: Vec<f64> = (0..self.n)
                .map(|i| (self.mean[i] + self.sigma * y[i]).clamp(0.0, 1.0))
                .collect();
            self.pending.push(x);
        }

        let result = self.pending[self.pending_idx].clone();
        self.pending_idx += 1;
        result
    }

    /// Record a result and update if generation is complete.
    fn tell(&mut self, params: Vec<f64>, value: f64) {
        self.results.push((params, value));

        if self.results.len() >= self.lambda {
            self.update();
        }
    }

    /// Perform the CMA-ES update step.
    fn update(&mut self) {
        // Sort by objective value (ascending = minimize)
        self.results
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let old_mean = self.mean.clone();

        // Update mean
        self.mean = vec![0.0; self.n];
        for i in 0..self.mu {
            for j in 0..self.n {
                self.mean[j] += self.weights[i] * self.results[i].0[j];
            }
        }

        // Compute mean displacement
        let mut mean_diff = vec![0.0; self.n];
        for i in 0..self.n {
            mean_diff[i] = (self.mean[i] - old_mean[i]) / self.sigma;
        }

        // Compute C^(-1/2) * mean_diff for p_sigma update
        let invsqrt_c_diff = self.invsqrt_c_times(&mean_diff);

        // Update evolution path p_sigma
        let cs_comp = (self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff).sqrt();
        for i in 0..self.n {
            self.p_sigma[i] =
                (1.0 - self.c_sigma) * self.p_sigma[i] + cs_comp * invsqrt_c_diff[i];
        }

        // h_sigma: indicator for p_sigma length
        let ps_norm: f64 = self.p_sigma.iter().map(|v| v * v).sum::<f64>().sqrt();
        let threshold = (1.0 - (1.0 - self.c_sigma).powi(2 * (self.generation as i32 + 1)))
            .sqrt()
            * (1.4 + 2.0 / (self.n as f64 + 1.0))
            * self.chi_n;
        let h_sigma = if ps_norm < threshold { 1.0 } else { 0.0 };

        // Update evolution path p_c
        let cc_comp = (self.c_c * (2.0 - self.c_c) * self.mu_eff).sqrt();
        for i in 0..self.n {
            self.p_c[i] =
                (1.0 - self.c_c) * self.p_c[i] + h_sigma * cc_comp * mean_diff[i];
        }

        // Update covariance matrix
        let delta_h = (1.0 - h_sigma) * self.c_c * (2.0 - self.c_c);
        for i in 0..self.n {
            for j in 0..self.n {
                let rank_one = self.p_c[i] * self.p_c[j];
                let mut rank_mu = 0.0;
                for k in 0..self.mu {
                    let yi = (self.results[k].0[i] - old_mean[i]) / self.sigma;
                    let yj = (self.results[k].0[j] - old_mean[j]) / self.sigma;
                    rank_mu += self.weights[k] * yi * yj;
                }

                self.c[i][j] = (1.0 - self.c1 - self.c_mu + delta_h * self.c1)
                    * self.c[i][j]
                    + self.c1 * rank_one
                    + self.c_mu * rank_mu;
            }
        }

        // Update sigma
        let sigma_factor = (ps_norm / self.chi_n - 1.0) * self.c_sigma / self.d_sigma;
        self.sigma *= (sigma_factor).exp();
        // Clamp sigma to reasonable range
        self.sigma = self.sigma.clamp(1e-20, 1e10);

        // Update eigen decomposition
        self.update_eigen();

        self.generation += 1;
        self.results.clear();
        self.pending.clear();
        self.pending_idx = 0;
    }

    /// Compute C^(-1/2) * v using eigen decomposition.
    fn invsqrt_c_times(&self, v: &[f64]) -> Vec<f64> {
        // C^(-1/2) = B * D^(-1) * B^T
        // First: B^T * v
        let mut bt_v = vec![0.0; self.n];
        for i in 0..self.n {
            for j in 0..self.n {
                bt_v[i] += self.b[j][i] * v[j];
            }
        }
        // D^(-1) * (B^T * v)
        for i in 0..self.n {
            let ev = self.eigenvalues[i].max(1e-20);
            bt_v[i] /= ev.sqrt();
        }
        // B * result
        let mut result = vec![0.0; self.n];
        for i in 0..self.n {
            for j in 0..self.n {
                result[i] += self.b[i][j] * bt_v[j];
            }
        }
        result
    }

    /// Update eigen decomposition of C using Jacobi iteration.
    fn update_eigen(&mut self) {
        let n = self.n;

        // Force symmetry
        for i in 0..n {
            for j in (i + 1)..n {
                let avg = (self.c[i][j] + self.c[j][i]) / 2.0;
                self.c[i][j] = avg;
                self.c[j][i] = avg;
            }
        }

        // Simple Jacobi eigenvalue algorithm for small matrices
        let mut a = self.c.clone();
        let mut v = vec![vec![0.0; n]; n];
        for i in 0..n {
            v[i][i] = 1.0;
        }

        for _ in 0..100 {
            // Find largest off-diagonal element
            let mut max_val = 0.0_f64;
            let mut p = 0;
            let mut q = 1;
            for i in 0..n {
                for j in (i + 1)..n {
                    if a[i][j].abs() > max_val {
                        max_val = a[i][j].abs();
                        p = i;
                        q = j;
                    }
                }
            }

            if max_val < 1e-15 {
                break;
            }

            // Compute rotation
            let theta = if (a[p][p] - a[q][q]).abs() < 1e-30 {
                std::f64::consts::FRAC_PI_4
            } else {
                0.5 * (2.0 * a[p][q] / (a[p][p] - a[q][q])).atan()
            };

            let cos_t = theta.cos();
            let sin_t = theta.sin();

            // Apply rotation to a
            let mut new_a = a.clone();
            for i in 0..n {
                if i != p && i != q {
                    new_a[i][p] = cos_t * a[i][p] + sin_t * a[i][q];
                    new_a[p][i] = new_a[i][p];
                    new_a[i][q] = -sin_t * a[i][p] + cos_t * a[i][q];
                    new_a[q][i] = new_a[i][q];
                }
            }
            new_a[p][p] = cos_t * cos_t * a[p][p]
                + 2.0 * sin_t * cos_t * a[p][q]
                + sin_t * sin_t * a[q][q];
            new_a[q][q] = sin_t * sin_t * a[p][p]
                - 2.0 * sin_t * cos_t * a[p][q]
                + cos_t * cos_t * a[q][q];
            new_a[p][q] = 0.0;
            new_a[q][p] = 0.0;
            a = new_a;

            // Apply rotation to eigenvectors
            for i in 0..n {
                let vip = v[i][p];
                let viq = v[i][q];
                v[i][p] = cos_t * vip + sin_t * viq;
                v[i][q] = -sin_t * vip + cos_t * viq;
            }
        }

        for i in 0..n {
            self.eigenvalues[i] = a[i][i].max(1e-20);
        }
        self.b = v;
    }
}

impl CmaEsSampler {
    pub fn new(
        direction: StudyDirection,
        sigma0: Option<f64>,
        n_startup_trials: Option<usize>,
        popsize: Option<usize>,
        independent_sampler: Option<Arc<dyn Sampler>>,
        seed: Option<u64>,
    ) -> Self {
        let rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };

        Self {
            direction,
            sigma0,
            n_startup_trials: n_startup_trials.unwrap_or(25),
            popsize,
            independent_sampler: independent_sampler
                .unwrap_or_else(|| Arc::new(RandomSampler::new(seed))),
            state: Mutex::new(None),
            rng: Mutex::new(rng),
            search_space: Mutex::new(IntersectionSearchSpace::new(false)),
        }
    }

    fn default_popsize(n: usize) -> usize {
        (4 + (3.0 * (n as f64).ln()).floor() as usize).max(5)
    }
}

impl Sampler for CmaEsSampler {
    fn infer_relative_search_space(
        &self,
        trials: &[FrozenTrial],
    ) -> HashMap<String, Distribution> {
        let n_complete = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .count();

        if n_complete < self.n_startup_trials {
            return HashMap::new();
        }

        self.search_space.lock().calculate(trials)
    }

    fn sample_relative(
        &self,
        trials: &[FrozenTrial],
        search_space: &HashMap<String, Distribution>,
    ) -> Result<HashMap<String, f64>> {
        if search_space.is_empty() {
            return Ok(HashMap::new());
        }

        let complete: Vec<&FrozenTrial> = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete && t.values.is_some())
            .collect();

        if complete.len() < self.n_startup_trials {
            return Ok(HashMap::new());
        }

        // Build ordered search space
        let mut ordered_space = IndexMap::new();
        let mut param_names: Vec<String> = search_space.keys().cloned().collect();
        param_names.sort();
        for name in &param_names {
            ordered_space.insert(name.clone(), search_space[name].clone());
        }

        let transform = SearchSpaceTransform::new(ordered_space.clone(), true, true, true);
        let n_dims = transform.n_encoded();

        let mut state_guard = self.state.lock();
        let mut rng = self.rng.lock();

        // Initialize state if needed
        if state_guard.is_none() {
            let sigma = self.sigma0.unwrap_or(1.0 / 6.0);
            let lambda = self.popsize.unwrap_or_else(|| Self::default_popsize(n_dims));

            // Initialize mean from the best trial
            let best_idx = complete
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    let va = a.values.as_ref().unwrap()[0];
                    let vb = b.values.as_ref().unwrap()[0];
                    let va = if self.direction == StudyDirection::Maximize {
                        -va
                    } else {
                        va
                    };
                    let vb = if self.direction == StudyDirection::Maximize {
                        -vb
                    } else {
                        vb
                    };
                    va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0);

            let best_trial = complete[best_idx];
            let mut best_params = IndexMap::new();
            for name in &param_names {
                if let Some(pv) = best_trial.params.get(name) {
                    best_params.insert(name.clone(), pv.clone());
                }
            }

            let mean = if best_params.len() == ordered_space.len() {
                transform.transform(&best_params)
            } else {
                vec![0.5; n_dims]
            };

            *state_guard = Some(CmaState::new(mean, sigma, lambda, param_names.clone()));
        }

        let state = state_guard.as_mut().unwrap();
        let candidate = state.ask(&mut rng);

        drop(rng);
        drop(state_guard);

        // Untransform
        let decoded = transform.untransform(&candidate)?;
        let mut result = HashMap::new();
        for (name, dist) in &ordered_space {
            if let Some(pv) = decoded.get(name) {
                let internal = dist.to_internal_repr(pv)?;
                result.insert(name.clone(), internal);
            }
        }

        Ok(result)
    }

    fn sample_independent(
        &self,
        trial: &FrozenTrial,
        param_name: &str,
        distribution: &Distribution,
    ) -> Result<f64> {
        self.independent_sampler
            .sample_independent(trial, param_name, distribution)
    }

    fn after_trial(
        &self,
        _trials: &[FrozenTrial],
        trial: &FrozenTrial,
        state: TrialState,
        values: Option<&[f64]>,
    ) {
        if state != TrialState::Complete {
            return;
        }
        let values = match values {
            Some(v) if !v.is_empty() => v,
            _ => return,
        };

        let mut state_guard = self.state.lock();
        if let Some(cma_state) = state_guard.as_mut() {
            let param_names = &cma_state.param_names;
            if param_names.is_empty() {
                return;
            }

            // Reconstruct the search space from trial distributions
            let mut ordered_space = IndexMap::new();
            for name in param_names {
                if let Some(dist) = trial.distributions.get(name) {
                    ordered_space.insert(name.clone(), dist.clone());
                }
            }

            if ordered_space.len() != param_names.len() {
                return;
            }

            let transform = SearchSpaceTransform::new(ordered_space, true, true, true);
            let mut trial_params = IndexMap::new();
            for name in param_names {
                if let Some(pv) = trial.params.get(name) {
                    trial_params.insert(name.clone(), pv.clone());
                }
            }

            if trial_params.len() == param_names.len() {
                let encoded = transform.transform(&trial_params);
                let value = if self.direction == StudyDirection::Maximize {
                    -values[0]
                } else {
                    values[0]
                };
                cma_state.tell(encoded, value);
            }
        }
    }
}

/// Builder for constructing a [`CmaEsSampler`] with custom parameters.
///
/// # Example
///
/// ```
/// use blackopt::{CmaEsSamplerBuilder, StudyDirection};
///
/// let sampler = CmaEsSamplerBuilder::new(StudyDirection::Minimize)
///     .sigma0(0.5)
///     .n_startup_trials(10)
///     .seed(42)
///     .build();
/// ```
pub struct CmaEsSamplerBuilder {
    direction: StudyDirection,
    sigma0: Option<f64>,
    n_startup_trials: Option<usize>,
    popsize: Option<usize>,
    independent_sampler: Option<Arc<dyn Sampler>>,
    seed: Option<u64>,
}

impl CmaEsSamplerBuilder {
    /// Create a new builder with the given optimization direction.
    pub fn new(direction: StudyDirection) -> Self {
        Self {
            direction,
            sigma0: None,
            n_startup_trials: None,
            popsize: None,
            independent_sampler: None,
            seed: None,
        }
    }

    /// Set the initial step size (sigma).
    pub fn sigma0(mut self, sigma: f64) -> Self {
        self.sigma0 = Some(sigma);
        self
    }

    /// Set the number of random startup trials before CMA-ES kicks in.
    pub fn n_startup_trials(mut self, n: usize) -> Self {
        self.n_startup_trials = Some(n);
        self
    }

    /// Set the population size (lambda).
    pub fn popsize(mut self, popsize: usize) -> Self {
        self.popsize = Some(popsize);
        self
    }

    /// Set the independent sampler used for parameters outside the search space.
    pub fn independent_sampler(mut self, sampler: Arc<dyn Sampler>) -> Self {
        self.independent_sampler = Some(sampler);
        self
    }

    /// Set the random seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Build the [`CmaEsSampler`].
    pub fn build(self) -> CmaEsSampler {
        CmaEsSampler::new(
            self.direction,
            self.sigma0,
            self.n_startup_trials,
            self.popsize,
            self.independent_sampler,
            self.seed,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::study::{create_study, StudyDirection};

    #[test]
    fn test_cmaes_creation() {
        let sampler = CmaEsSampler::new(
            StudyDirection::Minimize,
            None,
            Some(10),
            None,
            None,
            Some(42),
        );
        assert_eq!(sampler.n_startup_trials, 10);
    }

    #[test]
    fn test_cmaes_startup_random() {
        let sampler: Arc<dyn Sampler> = Arc::new(CmaEsSampler::new(
            StudyDirection::Minimize,
            None,
            Some(10),
            None,
            None,
            Some(42),
        ));

        let study = create_study(
            None,
            Some(sampler),
            None,
            None,
            Some(StudyDirection::Minimize),
            None,
            false,
        )
        .unwrap();

        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                    Ok(x * x)
                },
                Some(5),
                None,
                None,
            )
            .unwrap();

        assert_eq!(study.trials().unwrap().len(), 5);
    }

    #[test]
    fn test_cmaes_full_run() {
        let sampler: Arc<dyn Sampler> = Arc::new(CmaEsSampler::new(
            StudyDirection::Minimize,
            Some(0.5),
            Some(10),
            Some(8),
            None,
            Some(42),
        ));

        let study = create_study(
            None,
            Some(sampler),
            None,
            None,
            Some(StudyDirection::Minimize),
            None,
            false,
        )
        .unwrap();

        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                    let y = trial.suggest_float("y", -5.0, 5.0, false, None)?;
                    Ok(x * x + y * y)
                },
                Some(50),
                None,
                None,
            )
            .unwrap();

        let trials = study.trials().unwrap();
        assert_eq!(trials.len(), 50);

        // CMA-ES should find a decent solution
        let best = study.best_value().unwrap();
        assert!(
            best < 25.0,
            "CMA-ES should find a reasonable solution, got {best}"
        );
    }
}
