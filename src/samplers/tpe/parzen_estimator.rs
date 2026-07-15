//! Parzen (KDE) estimator for TPE.
//!
//! Fits a mixture of kernels to observations, then can sample and compute
//! log-PDF. Used as the density model in the TPE sampler.
//!
//! Reference: Bergstra et al., "Algorithms for Hyper-Parameter Optimization" (NIPS 2011).

use std::collections::HashMap;

use indexmap::IndexMap;
use rand::Rng;

use crate::distributions::Distribution;

use super::truncnorm;

const EPS: f64 = 1e-12;

/// Parameters for the Parzen estimator.
#[derive(Debug, Clone)]
pub struct ParzenEstimatorParameters {
    pub prior_weight: f64,
    pub consider_endpoints: bool,
    pub multivariate: bool,
}

impl Default for ParzenEstimatorParameters {
    fn default() -> Self {
        Self {
            prior_weight: 1.0,
            consider_endpoints: false,
            multivariate: false,
        }
    }
}

/// A fitted kernel density estimator as a mixture model.
///
/// Supports numerical (truncated normal) and categorical parameters.
pub struct ParzenEstimator {
    /// Mixture component weights, length = n_kernels (n_obs + 1 for numerical).
    weights: Vec<f64>,
    /// Per-parameter distribution info.
    param_dists: Vec<ParamKernels>,
    /// Ordered parameter names matching param_dists.
    param_names: Vec<String>,
}

/// Kernel info for a single parameter dimension.
enum ParamKernels {
    /// Truncated normal kernels for continuous/discrete numerical params.
    Numerical {
        mus: Vec<f64>,    // kernel centers, len = n_kernels
        sigmas: Vec<f64>, // kernel widths, len = n_kernels
        low: f64,
        high: f64,
        log: bool,
        step: Option<f64>, // if Some, discrete rounding
    },
    /// Categorical kernels: weights[kernel][choice].
    Categorical {
        cat_weights: Vec<Vec<f64>>, // n_kernels × n_choices, row-normalized
        n_choices: usize,
    },
}

impl ParzenEstimator {
    /// Build a Parzen estimator from observations and search space.
    ///
    /// `observations`: param_name → Vec<f64> of internal repr values.
    /// `search_space`: param_name → Distribution, same order (use IndexMap).
    /// `params`: estimator configuration.
    /// `predetermined_weights`: if Some, used instead of default_weights.
    pub fn new(
        observations: &HashMap<String, Vec<f64>>,
        search_space: &IndexMap<String, Distribution>,
        params: &ParzenEstimatorParameters,
        predetermined_weights: Option<&[f64]>,
    ) -> Self {
        let n_obs = search_space
            .keys()
            .next()
            .and_then(|k| observations.get(k))
            .map(|v| v.len())
            .unwrap_or(0);

        // Compute mixture weights.
        let mut weights = if let Some(pw) = predetermined_weights {
            pw.to_vec()
        } else {
            default_weights(n_obs)
        };

        if weights.is_empty() {
            weights = vec![1.0];
        } else {
            weights.push(params.prior_weight);
        }

        // Normalize weights.
        let sum: f64 = weights.iter().sum();
        if sum > 0.0 {
            for w in &mut weights {
                *w /= sum;
            }
        }

        let n_kernels = weights.len();

        // Build per-parameter kernel distributions.
        let mut param_dists = Vec::with_capacity(search_space.len());
        let mut param_names = Vec::with_capacity(search_space.len());

        for (name, dist) in search_space {
            param_names.push(name.clone());
            let obs = observations.get(name).cloned().unwrap_or_default();
            let kernels = match dist {
                Distribution::CategoricalDistribution(cd) => {
                    Self::build_categorical_kernels(&obs, cd.choices.len(), n_kernels, params)
                }
                Distribution::FloatDistribution(fd) => Self::build_numerical_kernels(
                    &obs,
                    fd.low,
                    fd.high,
                    fd.log,
                    fd.step,
                    n_kernels,
                    search_space,
                    params,
                ),
                Distribution::IntDistribution(id) => Self::build_numerical_kernels(
                    &obs,
                    id.low as f64,
                    id.high as f64,
                    id.log,
                    Some(id.step as f64),
                    n_kernels,
                    search_space,
                    params,
                ),
            };
            param_dists.push(kernels);
        }

        Self {
            weights,
            param_dists,
            param_names,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn build_numerical_kernels(
        obs: &[f64],
        mut low: f64,
        mut high: f64,
        log: bool,
        step: Option<f64>,
        n_kernels: usize,
        search_space: &IndexMap<String, Distribution>,
        params: &ParzenEstimatorParameters,
    ) -> ParamKernels {
        // Extend bounds for discrete distributions.
        if let Some(s) = step {
            low -= s / 2.0;
            high += s / 2.0;
        }

        let mut mus: Vec<f64> = obs.to_vec();

        // Apply log transform.
        if log {
            mus = mus.iter().map(|&v| v.max(EPS).ln()).collect();
            low = low.max(EPS).ln();
            high = high.max(EPS).ln();
        }

        // Compute sigmas.
        let sigmas = if params.multivariate {
            let n_params = search_space.len();
            let sigma0_mag = 0.2;
            let sigma = sigma0_mag
                * (mus.len().max(1) as f64).powf(-1.0 / (n_params as f64 + 4.0))
                * (high - low);
            vec![sigma; mus.len()]
        } else {
            Self::compute_univariate_sigmas(&mus, low, high, params)
        };

        // Clip sigmas. The minimum bandwidth scales as range / n, ensuring
        // kernels don't collapse while still narrowing with more observations.
        let max_sigma = high - low;
        let min_sigma = (high - low) / (n_kernels.max(1) as f64);

        let mut clipped_sigmas: Vec<f64> = sigmas
            .iter()
            .map(|&s| s.clamp(min_sigma, max_sigma))
            .collect();

        // Append prior kernel: center at midpoint, sigma = full range.
        let prior_mu = 0.5 * (low + high);
        let prior_sigma = high - low;
        mus.push(prior_mu);
        clipped_sigmas.push(prior_sigma);

        ParamKernels::Numerical {
            mus,
            sigmas: clipped_sigmas,
            low,
            high,
            log,
            step,
        }
    }

    /// Compute univariate kernel bandwidths from sorted neighbor distances.
    fn compute_univariate_sigmas(
        mus: &[f64],
        low: f64,
        high: f64,
        params: &ParzenEstimatorParameters,
    ) -> Vec<f64> {
        if mus.is_empty() {
            return vec![];
        }

        // Include prior center in sorting.
        let prior_mu = 0.5 * (low + high);
        let mut mus_with_prior: Vec<f64> = mus.to_vec();
        mus_with_prior.push(prior_mu);

        // Sort and compute index mapping.
        let mut indices: Vec<usize> = (0..mus_with_prior.len()).collect();
        indices.sort_by(|&a, &b| {
            mus_with_prior[a]
                .partial_cmp(&mus_with_prior[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut sorted_mus = vec![0.0; mus_with_prior.len()];
        for (rank, &orig_idx) in indices.iter().enumerate() {
            sorted_mus[rank] = mus_with_prior[orig_idx];
        }

        // Pad with low and high.
        let n = sorted_mus.len();
        let mut padded = Vec::with_capacity(n + 2);
        padded.push(low);
        padded.extend_from_slice(&sorted_mus);
        padded.push(high);

        // sigma[i] = max(gap_left, gap_right)
        let mut sorted_sigmas = Vec::with_capacity(n);
        for i in 0..n {
            let gap_left = padded[i + 1] - padded[i];
            let gap_right = padded[i + 2] - padded[i + 1];
            sorted_sigmas.push(gap_left.max(gap_right));
        }

        // Consider endpoints: clip boundary sigmas to nearest interior neighbor.
        if !params.consider_endpoints && padded.len() >= 4 {
            sorted_sigmas[0] = padded[2] - padded[1];
            let last = sorted_sigmas.len() - 1;
            sorted_sigmas[last] = padded[padded.len() - 2] - padded[padded.len() - 3];
        }

        // Unsort to original order, excluding the prior (last in mus_with_prior).
        let mut inv_indices = vec![0usize; mus_with_prior.len()];
        for (rank, &orig_idx) in indices.iter().enumerate() {
            inv_indices[orig_idx] = rank;
        }

        let mut sigmas = Vec::with_capacity(mus.len());
        for i in 0..mus.len() {
            sigmas.push(sorted_sigmas[inv_indices[i]]);
        }
        sigmas
    }

    fn build_categorical_kernels(
        obs: &[f64],
        n_choices: usize,
        n_kernels: usize,
        params: &ParzenEstimatorParameters,
    ) -> ParamKernels {
        if obs.is_empty() {
            // Single uniform prior kernel.
            let uniform = vec![1.0 / n_choices as f64; n_choices];
            return ParamKernels::Categorical {
                cat_weights: vec![uniform],
                n_choices,
            };
        }

        let base_weight = params.prior_weight / n_kernels as f64;
        let mut cat_weights = vec![vec![base_weight; n_choices]; n_kernels];

        // Each observation kernel gets +1 for its observed category.
        for (i, &v) in obs.iter().enumerate() {
            let idx = v as usize;
            if idx < n_choices {
                cat_weights[i][idx] += 1.0;
            }
        }

        // Normalize each row.
        for row in &mut cat_weights {
            let sum: f64 = row.iter().sum();
            if sum > 0.0 {
                for w in row.iter_mut() {
                    *w /= sum;
                }
            }
        }

        ParamKernels::Categorical {
            cat_weights,
            n_choices,
        }
    }

    /// Sample `size` parameter vectors from the mixture.
    ///
    /// Returns: param_name → Vec<f64> of internal-repr values, each of length `size`.
    pub fn sample(&self, rng: &mut impl Rng, size: usize) -> HashMap<String, Vec<f64>> {
        // Choose which mixture component each sample comes from.
        let active_indices = self.sample_component_indices(rng, size);

        let mut result = HashMap::new();
        for (d, name) in self.param_dists.iter().zip(self.param_names.iter()) {
            let samples = match d {
                ParamKernels::Numerical {
                    mus,
                    sigmas,
                    low,
                    high,
                    log,
                    step,
                } => {
                    let mut a_vec = Vec::with_capacity(size);
                    let mut b_vec = Vec::with_capacity(size);
                    let mut loc_vec = Vec::with_capacity(size);
                    let mut scale_vec = Vec::with_capacity(size);

                    for &k in &active_indices {
                        loc_vec.push(mus[k]);
                        scale_vec.push(sigmas[k]);
                        a_vec.push(*low);
                        b_vec.push(*high);
                    }

                    let mut samples = truncnorm::rvs(&a_vec, &b_vec, &loc_vec, &scale_vec, rng);

                    // Undo log transform.
                    if *log {
                        for s in &mut samples {
                            *s = s.exp();
                        }
                    }

                    // Round discrete.
                    if let Some(st) = step {
                        // Recover original low/high before step extension.
                        let orig_low = if *log {
                            (low + st / 2.0).exp()
                        } else {
                            low + st / 2.0
                        };
                        let orig_high = if *log {
                            (high - st / 2.0).exp()
                        } else {
                            high - st / 2.0
                        };
                        for s in &mut samples {
                            *s = ((*s - orig_low) / st).round() * st + orig_low;
                            *s = s.clamp(orig_low, orig_high);
                        }
                    }

                    samples
                }
                ParamKernels::Categorical {
                    cat_weights,
                    n_choices,
                } => {
                    let mut samples = Vec::with_capacity(size);
                    for &k in &active_indices {
                        let weights = &cat_weights[k];
                        let q: f64 = rng.r#gen();
                        let mut cum = 0.0;
                        let mut choice = *n_choices - 1;
                        for (i, &w) in weights.iter().enumerate() {
                            cum += w;
                            if q < cum {
                                choice = i;
                                break;
                            }
                        }
                        samples.push(choice as f64);
                    }
                    samples
                }
            };
            result.insert(name.clone(), samples);
        }
        result
    }

    /// Compute log-PDF of the mixture for given samples.
    ///
    /// `samples`: param_name → Vec<f64> of internal-repr values, all same length.
    /// Returns: Vec<f64> of log-pdf values, one per sample.
    pub fn log_pdf(&self, samples: &HashMap<String, Vec<f64>>) -> Vec<f64> {
        let n_samples = samples.values().next().map(|v| v.len()).unwrap_or(0);
        let n_kernels = self.weights.len();

        // weighted_log_pdf[sample][kernel] = log(w_k) + sum_d log p_k^d(x_d)
        let mut weighted_log_pdf = vec![vec![0.0; n_kernels]; n_samples];

        for (d, name) in self.param_dists.iter().zip(self.param_names.iter()) {
            let xs = match samples.get(name) {
                Some(v) => v,
                None => continue,
            };

            match d {
                ParamKernels::Numerical {
                    mus,
                    sigmas,
                    low,
                    high,
                    log,
                    step,
                } => {
                    for (si, &x) in xs.iter().enumerate() {
                        for k in 0..n_kernels {
                            let lp = if let Some(st) = step {
                                // Discrete: log probability mass in [x - step/2, x + step/2]
                                let x_val = if *log { x.max(EPS).ln() } else { x };
                                let st_t = if *log {
                                    // In log space, step is approximate
                                    st.max(EPS)
                                } else {
                                    *st
                                };
                                let a_norm = (x_val - st_t / 2.0 - mus[k]) / sigmas[k];
                                let b_norm = (x_val + st_t / 2.0 - mus[k]) / sigmas[k];
                                let mass = truncnorm::log_gauss_mass(a_norm, b_norm);
                                let total_a = (*low - mus[k]) / sigmas[k];
                                let total_b = (*high - mus[k]) / sigmas[k];
                                let total = truncnorm::log_gauss_mass(total_a, total_b);
                                mass - total
                            } else {
                                let x_val = if *log { x.max(EPS).ln() } else { x };
                                truncnorm::logpdf(x_val, *low, *high, mus[k], sigmas[k])
                            };
                            weighted_log_pdf[si][k] += lp;
                        }
                    }
                }
                ParamKernels::Categorical { cat_weights, .. } => {
                    for (si, &x) in xs.iter().enumerate() {
                        let idx = x as usize;
                        for k in 0..n_kernels {
                            let w = cat_weights[k].get(idx).copied().unwrap_or(EPS);
                            weighted_log_pdf[si][k] += w.max(EPS).ln();
                        }
                    }
                }
            }
        }

        // Add log mixture weights and logsumexp over kernels.
        let mut result = Vec::with_capacity(n_samples);
        for row in &mut weighted_log_pdf {
            for (k, cell) in row.iter_mut().enumerate() {
                *cell += self.weights[k].max(EPS).ln();
            }
            result.push(logsumexp(row));
        }
        result
    }

    /// Sample mixture component indices.
    fn sample_component_indices(&self, rng: &mut impl Rng, size: usize) -> Vec<usize> {
        let mut indices = Vec::with_capacity(size);
        for _ in 0..size {
            let q: f64 = rng.r#gen();
            let mut cum = 0.0;
            let mut idx = self.weights.len() - 1;
            for (i, &w) in self.weights.iter().enumerate() {
                cum += w;
                if q < cum {
                    idx = i;
                    break;
                }
            }
            indices.push(idx);
        }
        indices
    }
}

/// Uniform observation weights.
///
/// All observations are weighted equally, as described in
/// Bergstra et al. (2011).
pub fn default_weights(n: usize) -> Vec<f64> {
    vec![1.0; n]
}

/// Split quantile for TPE: number of trials in the "good" group.
///
/// Uses 15% quantile following the suggestion in
/// Bergstra et al. (2011) §3.
pub fn default_gamma(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    ((n as f64 * 0.15).ceil() as usize).max(1)
}

fn logsumexp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let sum: f64 = values.iter().map(|&v| (v - max).exp()).sum();
    max + sum.ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::*;
    use rand::SeedableRng;

    fn make_search_space() -> IndexMap<String, Distribution> {
        let mut ss = IndexMap::new();
        ss.insert(
            "x".to_string(),
            Distribution::FloatDistribution(
                FloatDistribution::new(0.0, 10.0, false, None).unwrap(),
            ),
        );
        ss
    }

    #[test]
    fn test_default_weights() {
        let w = default_weights(5);
        assert_eq!(w.len(), 5);
        assert!(w.iter().all(|&v| (v - 1.0).abs() < 1e-10));

        let w = default_weights(30);
        assert_eq!(w.len(), 30);
        assert!(w.iter().all(|&v| (v - 1.0).abs() < 1e-10));

        assert!(default_weights(0).is_empty());
    }

    #[test]
    fn test_default_gamma() {
        assert_eq!(default_gamma(0), 0);
        assert_eq!(default_gamma(1), 1);
        assert_eq!(default_gamma(10), 2); // ceil(1.5)
        assert_eq!(default_gamma(20), 3); // ceil(3.0)
        assert_eq!(default_gamma(100), 15); // ceil(15.0)
        assert_eq!(default_gamma(300), 45); // ceil(45.0)
    }

    #[test]
    fn test_parzen_estimator_no_observations() {
        let ss = make_search_space();
        let obs: HashMap<String, Vec<f64>> = HashMap::new();
        let pe = ParzenEstimator::new(&obs, &ss, &ParzenEstimatorParameters::default(), None);

        assert_eq!(pe.weights.len(), 1); // Just prior
        assert!((pe.weights[0] - 1.0).abs() < 1e-10);

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let samples = pe.sample(&mut rng, 10);
        assert_eq!(samples["x"].len(), 10);
        for &v in &samples["x"] {
            assert!(v >= 0.0 && v <= 10.0, "sample {v} out of bounds");
        }
    }

    #[test]
    fn test_parzen_estimator_with_observations() {
        let ss = make_search_space();
        let mut obs = HashMap::new();
        obs.insert("x".to_string(), vec![2.0, 5.0, 8.0]);
        let pe = ParzenEstimator::new(&obs, &ss, &ParzenEstimatorParameters::default(), None);

        // 3 obs + 1 prior = 4 kernels
        assert_eq!(pe.weights.len(), 4);

        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let samples = pe.sample(&mut rng, 100);
        for &v in &samples["x"] {
            assert!(v >= 0.0 && v <= 10.0, "sample {v} out of bounds");
        }

        // log_pdf should return finite values for in-range samples
        let lp = pe.log_pdf(&samples);
        assert_eq!(lp.len(), 100);
        for &v in &lp {
            assert!(v.is_finite(), "log_pdf returned non-finite: {v}");
        }
    }

    #[test]
    fn test_parzen_estimator_categorical() {
        let mut ss = IndexMap::new();
        ss.insert(
            "opt".to_string(),
            Distribution::CategoricalDistribution(
                CategoricalDistribution::new(vec![
                    CategoricalChoice::Str("a".into()),
                    CategoricalChoice::Str("b".into()),
                    CategoricalChoice::Str("c".into()),
                ])
                .unwrap(),
            ),
        );

        let mut obs = HashMap::new();
        obs.insert("opt".to_string(), vec![0.0, 0.0, 1.0]); // mostly "a"

        let pe = ParzenEstimator::new(&obs, &ss, &ParzenEstimatorParameters::default(), None);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let samples = pe.sample(&mut rng, 100);
        for &v in &samples["opt"] {
            assert!(
                v >= 0.0 && v < 3.0 && (v - v.round()).abs() < 1e-10,
                "categorical sample {v} invalid"
            );
        }
    }

    #[test]
    fn test_parzen_estimator_log_scale() {
        let mut ss = IndexMap::new();
        ss.insert(
            "lr".to_string(),
            Distribution::FloatDistribution(
                FloatDistribution::new(0.001, 1.0, true, None).unwrap(),
            ),
        );

        let mut obs = HashMap::new();
        obs.insert("lr".to_string(), vec![0.01, 0.1]);

        let pe = ParzenEstimator::new(&obs, &ss, &ParzenEstimatorParameters::default(), None);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let samples = pe.sample(&mut rng, 100);
        for &v in &samples["lr"] {
            assert!(
                v >= 0.001 - 1e-10 && v <= 1.0 + 1e-10,
                "log-scale sample {v} out of bounds"
            );
        }
    }

    #[test]
    fn test_parzen_estimator_int_step() {
        let mut ss = IndexMap::new();
        ss.insert(
            "n".to_string(),
            Distribution::IntDistribution(IntDistribution::new(0, 10, false, 2).unwrap()),
        );

        let mut obs = HashMap::new();
        obs.insert("n".to_string(), vec![2.0, 4.0, 6.0]);

        let pe = ParzenEstimator::new(&obs, &ss, &ParzenEstimatorParameters::default(), None);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(42);
        let samples = pe.sample(&mut rng, 100);
        for &v in &samples["n"] {
            let iv = v as i64;
            assert!(
                iv >= 0 && iv <= 10 && iv % 2 == 0,
                "int step sample {v} invalid"
            );
        }
    }

    #[test]
    fn test_logsumexp() {
        assert!((logsumexp(&[0.0, 0.0]) - 2.0_f64.ln()).abs() < 1e-10);
        assert!((logsumexp(&[-1000.0, -1000.0]) - (-1000.0 + 2.0_f64.ln())).abs() < 1e-10);
    }
}
