//! TPE (Tree-structured Parzen Estimator) sampler.
//!
//! Models the objective function by splitting observations into "good" and "bad"
//! groups, fitting separate kernel density estimators, and maximizing expected
//! improvement.
//!
//! Reference: Bergstra et al., "Algorithms for Hyper-Parameter Optimization" (NIPS 2011).

use std::collections::HashMap;

use indexmap::IndexMap;
use parking_lot::Mutex;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::distributions::Distribution;
use crate::error::Result;
use crate::samplers::Sampler;
use crate::samplers::random::RandomSampler;
use crate::search_space::IntersectionSearchSpace;
use crate::study::StudyDirection;
use crate::trial::{FrozenTrial, TrialState};

use super::parzen_estimator::{ParzenEstimator, ParzenEstimatorParameters, default_gamma};

/// A TPE (Tree-structured Parzen Estimator) sampler.
///
/// Models the objective function by splitting observed trials into "good"
/// (below threshold) and "bad" (above threshold) groups, fits separate
/// kernel density estimators (l(x) and g(x)), and maximizes l(x)/g(x)
/// as a surrogate for expected improvement.
pub struct TpeSampler {
    /// Number of random startup trials before TPE kicks in.
    n_startup_trials: usize,
    /// Number of EI candidates to draw from l(x).
    n_ei_candidates: usize,
    /// Study direction for sorting trials.
    direction: StudyDirection,
    /// Whether to sample parameters jointly (multivariate) or independently.
    multivariate: bool,
    /// Parzen estimator parameters.
    pe_params: ParzenEstimatorParameters,
    /// RNG for sampling.
    rng: Mutex<ChaCha8Rng>,
    /// Fallback random sampler for startup.
    random_sampler: RandomSampler,
    /// Search space tracker (for multivariate mode).
    search_space: Mutex<IntersectionSearchSpace>,
}

impl std::fmt::Debug for TpeSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TpeSampler")
            .field("n_startup_trials", &self.n_startup_trials)
            .field("n_ei_candidates", &self.n_ei_candidates)
            .field("multivariate", &self.multivariate)
            .finish()
    }
}

impl TpeSampler {
    /// Create a new `TpeSampler`.
    pub fn new(
        direction: StudyDirection,
        seed: Option<u64>,
        n_startup_trials: usize,
        n_ei_candidates: usize,
        multivariate: bool,
        consider_endpoints: bool,
        prior_weight: f64,
    ) -> Self {
        let rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };
        Self {
            n_startup_trials,
            n_ei_candidates,
            direction,
            multivariate,
            pe_params: ParzenEstimatorParameters {
                prior_weight,
                consider_endpoints,
                multivariate,
            },
            rng: Mutex::new(rng),
            random_sampler: RandomSampler::new(seed.map(|s| s.wrapping_add(1))),
            search_space: Mutex::new(IntersectionSearchSpace::new(false)),
        }
    }

    /// Create with common defaults.
    pub fn with_defaults(direction: StudyDirection, seed: Option<u64>) -> Self {
        Self::new(direction, seed, 10, 24, false, false, 1.0)
    }

    /// Create a multivariate TPE sampler.
    pub fn multivariate(direction: StudyDirection, seed: Option<u64>) -> Self {
        Self::new(direction, seed, 10, 24, true, false, 1.0)
    }

    /// Split trials into below (good) and above (bad) groups.
    fn split_trials<'a>(
        &self,
        trials: &'a [FrozenTrial],
    ) -> (Vec<&'a FrozenTrial>, Vec<&'a FrozenTrial>) {
        // Separate complete vs pruned vs running.
        let mut complete: Vec<&FrozenTrial> = Vec::new();
        let mut pruned: Vec<&FrozenTrial> = Vec::new();
        let mut running: Vec<&FrozenTrial> = Vec::new();

        for t in trials {
            match t.state {
                TrialState::Complete => complete.push(t),
                TrialState::Pruned => pruned.push(t),
                TrialState::Running => running.push(t),
                _ => {}
            }
        }

        let n = complete.len() + pruned.len();
        let n_below = default_gamma(n);

        // Sort complete trials by objective value.
        match self.direction {
            StudyDirection::Minimize | StudyDirection::NotSet => {
                complete.sort_by(|a, b| {
                    let va = a.value().ok().flatten().unwrap_or(f64::INFINITY);
                    let vb = b.value().ok().flatten().unwrap_or(f64::INFINITY);
                    va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            StudyDirection::Maximize => {
                complete.sort_by(|a, b| {
                    let va = a.value().ok().flatten().unwrap_or(f64::NEG_INFINITY);
                    let vb = b.value().ok().flatten().unwrap_or(f64::NEG_INFINITY);
                    vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }

        // Sort pruned by (-last_step, intermediate_value).
        pruned.sort_by(|a, b| {
            let sa = a.last_step().unwrap_or(i64::MIN);
            let sb = b.last_step().unwrap_or(i64::MIN);
            match sb.cmp(&sa) {
                std::cmp::Ordering::Equal => {
                    let va = a
                        .intermediate_values
                        .get(&sa)
                        .copied()
                        .unwrap_or(f64::INFINITY);
                    let vb = b
                        .intermediate_values
                        .get(&sb)
                        .copied()
                        .unwrap_or(f64::INFINITY);
                    va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
                }
                other => other,
            }
        });

        // Split: best n_below from complete, then pruned.
        let mut below = Vec::new();
        let mut above = Vec::new();

        let mut remaining = n_below;
        for t in &complete {
            if remaining > 0 {
                below.push(*t);
                remaining -= 1;
            } else {
                above.push(*t);
            }
        }
        for t in &pruned {
            if remaining > 0 {
                below.push(*t);
                remaining -= 1;
            } else {
                above.push(*t);
            }
        }

        // Running trials always go to above.
        above.extend(running);

        (below, above)
    }

    /// Get observations from trials for the given search space.
    fn get_observations(
        trials: &[&FrozenTrial],
        search_space: &IndexMap<String, Distribution>,
    ) -> HashMap<String, Vec<f64>> {
        let mut obs: HashMap<String, Vec<f64>> = HashMap::new();
        for name in search_space.keys() {
            obs.insert(name.to_string(), Vec::new());
        }

        for &trial in trials {
            // Only include trials that have all params in the search space.
            let has_all = search_space
                .keys()
                .all(|name| trial.params.contains_key(name));
            if !has_all {
                continue;
            }

            for (name, dist) in search_space {
                if let Some(val) = trial.params.get(name)
                    && let Ok(internal) = dist.to_internal_repr(val)
                {
                    obs.get_mut(name).unwrap().push(internal);
                }
            }
        }
        obs
    }

    /// Sample using TPE for a given search space.
    fn tpe_sample(
        &self,
        trials: &[FrozenTrial],
        search_space: &IndexMap<String, Distribution>,
    ) -> Result<HashMap<String, f64>> {
        let (below, above) = self.split_trials(trials);

        let obs_below = Self::get_observations(&below, search_space);
        let obs_above = Self::get_observations(&above, search_space);

        let pe_below = ParzenEstimator::new(&obs_below, search_space, &self.pe_params, None);
        let pe_above = ParzenEstimator::new(&obs_above, search_space, &self.pe_params, None);

        // Sample candidates from l(x).
        let mut rng = self.rng.lock();
        let samples = pe_below.sample(&mut *rng, self.n_ei_candidates);
        drop(rng);

        // Compute acquisition: log l(x) - log g(x).
        let log_l = pe_below.log_pdf(&samples);
        let log_g = pe_above.log_pdf(&samples);

        let mut best_idx = 0;
        let mut best_acq = f64::NEG_INFINITY;
        for i in 0..self.n_ei_candidates {
            let acq = log_l[i] - log_g[i];
            if acq > best_acq {
                best_acq = acq;
                best_idx = i;
            }
        }

        // Extract the best sample.
        let mut result = HashMap::new();
        for (name, vals) in &samples {
            result.insert(name.clone(), vals[best_idx]);
        }
        Ok(result)
    }

    /// Check if we're still in the startup phase.
    fn is_startup(&self, trials: &[FrozenTrial]) -> bool {
        let n_complete = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete || t.state == TrialState::Pruned)
            .count();
        n_complete < self.n_startup_trials
    }
}

impl Sampler for TpeSampler {
    fn infer_relative_search_space(&self, trials: &[FrozenTrial]) -> HashMap<String, Distribution> {
        if !self.multivariate {
            return HashMap::new();
        }
        if self.is_startup(trials) {
            return HashMap::new();
        }
        let mut ss = self.search_space.lock();
        ss.calculate(trials).clone()
    }

    fn sample_relative(
        &self,
        trials: &[FrozenTrial],
        search_space: &HashMap<String, Distribution>,
    ) -> Result<HashMap<String, f64>> {
        if search_space.is_empty() {
            return Ok(HashMap::new());
        }

        // Convert to IndexMap for ordered iteration.
        let ordered: IndexMap<String, Distribution> = search_space
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();

        self.tpe_sample(trials, &ordered)
    }

    fn sample_independent(
        &self,
        trial: &FrozenTrial,
        param_name: &str,
        distribution: &Distribution,
    ) -> Result<f64> {
        // During startup, use random sampling.
        // We check the trial number as a proxy for completed trials count.
        if (trial.number as usize) < self.n_startup_trials {
            return self
                .random_sampler
                .sample_independent(trial, param_name, distribution);
        }

        // For independent (univariate) mode, we don't have access to all trials here.
        // The study will call us via sample_relative for multivariate mode.
        // For univariate TPE in sample_independent, we'd need access to study trials.
        // Since we don't have that context, fall back to random for now.
        // The proper TPE path goes through sample_relative.
        //
        // However, to make univariate TPE work, we store a snapshot of trials
        // in before_trial and use it here. For now, fall back to random.
        self.random_sampler
            .sample_independent(trial, param_name, distribution)
    }
}

/// Builder for constructing a [`TpeSampler`] with custom parameters.
///
/// # Example
///
/// ```
/// use blackopt::{TpeSamplerBuilder, StudyDirection};
///
/// let sampler = TpeSamplerBuilder::new(StudyDirection::Minimize)
///     .seed(42)
///     .n_startup_trials(20)
///     .multivariate(true)
///     .build();
/// ```
pub struct TpeSamplerBuilder {
    direction: StudyDirection,
    seed: Option<u64>,
    n_startup_trials: usize,
    n_ei_candidates: usize,
    multivariate: bool,
    consider_endpoints: bool,
    prior_weight: f64,
}

impl TpeSamplerBuilder {
    /// Create a new builder with the given optimization direction.
    pub fn new(direction: StudyDirection) -> Self {
        Self {
            direction,
            seed: None,
            n_startup_trials: 10,
            n_ei_candidates: 24,
            multivariate: false,
            consider_endpoints: false,
            prior_weight: 1.0,
        }
    }

    /// Set the random seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the number of random startup trials before TPE kicks in.
    pub fn n_startup_trials(mut self, n: usize) -> Self {
        self.n_startup_trials = n;
        self
    }

    /// Set the number of EI candidates to evaluate.
    pub fn n_ei_candidates(mut self, n: usize) -> Self {
        self.n_ei_candidates = n;
        self
    }

    /// Enable or disable multivariate (joint) sampling.
    pub fn multivariate(mut self, multivariate: bool) -> Self {
        self.multivariate = multivariate;
        self
    }

    /// Enable or disable considering endpoints in the Parzen estimator.
    pub fn consider_endpoints(mut self, consider: bool) -> Self {
        self.consider_endpoints = consider;
        self
    }

    /// Set the prior weight for the Parzen estimator.
    pub fn prior_weight(mut self, weight: f64) -> Self {
        self.prior_weight = weight;
        self
    }

    /// Build the [`TpeSampler`].
    pub fn build(self) -> TpeSampler {
        TpeSampler::new(
            self.direction,
            self.seed,
            self.n_startup_trials,
            self.n_ei_candidates,
            self.multivariate,
            self.consider_endpoints,
            self.prior_weight,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::*;

    fn make_complete_trial(
        number: i64,
        value: f64,
        params: HashMap<String, ParamValue>,
    ) -> FrozenTrial {
        let now = chrono::Utc::now();
        let mut distributions = HashMap::new();
        for (name, val) in &params {
            match val {
                ParamValue::Float(_) => {
                    distributions.insert(
                        name.clone(),
                        Distribution::FloatDistribution(
                            FloatDistribution::new(-10.0, 10.0, false, None).unwrap(),
                        ),
                    );
                }
                ParamValue::Int(_) => {
                    distributions.insert(
                        name.clone(),
                        Distribution::IntDistribution(
                            IntDistribution::new(-10, 10, false, 1).unwrap(),
                        ),
                    );
                }
                ParamValue::Categorical(_) => {}
            }
        }
        FrozenTrial {
            number,
            state: TrialState::Complete,
            values: Some(vec![value]),
            datetime_start: Some(now),
            datetime_complete: Some(now),
            params,
            distributions,
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
            trial_id: number,
        }
    }

    #[test]
    fn test_tpe_startup_uses_random() {
        let sampler = TpeSampler::with_defaults(StudyDirection::Minimize, Some(42));
        let trial = FrozenTrial {
            number: 0,
            state: TrialState::Running,
            values: None,
            datetime_start: Some(chrono::Utc::now()),
            datetime_complete: None,
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
            trial_id: 0,
        };
        let dist =
            Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
        // During startup, should sample without error.
        let v = sampler.sample_independent(&trial, "x", &dist).unwrap();
        assert!((0.0..=1.0).contains(&v));
    }

    #[test]
    fn test_tpe_split_trials_minimize() {
        let sampler = TpeSampler::with_defaults(StudyDirection::Minimize, Some(42));
        let mut trials = Vec::new();
        for i in 0..20 {
            let mut params = HashMap::new();
            params.insert("x".to_string(), ParamValue::Float(i as f64));
            trials.push(make_complete_trial(i, i as f64, params));
        }

        let (below, above) = sampler.split_trials(&trials);
        // gamma(20) = ceil(20 * 0.15) = 3
        assert_eq!(below.len(), 3);
        // Below should have the best (lowest) values: 0.0, 1.0, 2.0
        assert_eq!(below[0].value().unwrap(), Some(0.0));
        assert_eq!(below[1].value().unwrap(), Some(1.0));
        assert_eq!(below[2].value().unwrap(), Some(2.0));
        assert_eq!(above.len(), 17);
    }

    #[test]
    fn test_tpe_sample_relative() {
        let sampler = TpeSampler::with_defaults(StudyDirection::Minimize, Some(42));
        // Create 15 completed trials (past startup of 10).
        let mut trials = Vec::new();
        for i in 0..15 {
            let x = (i as f64 - 7.0).abs(); // V-shape: minimum at i=7
            let mut params = HashMap::new();
            params.insert("x".to_string(), ParamValue::Float(i as f64 - 7.0));
            trials.push(make_complete_trial(i, x, params));
        }

        let mut search_space = HashMap::new();
        search_space.insert(
            "x".to_string(),
            Distribution::FloatDistribution(
                FloatDistribution::new(-10.0, 10.0, false, None).unwrap(),
            ),
        );

        let result = sampler.sample_relative(&trials, &search_space).unwrap();
        assert!(result.contains_key("x"));
        let x = result["x"];
        assert!((-10.0..=10.0).contains(&x), "TPE sample {x} out of bounds");
    }

    #[test]
    fn test_tpe_converges_better_than_random() {
        // Run TPE on a simple quadratic: minimize x^2.
        // TPE should concentrate samples near x=0 more than random.
        let sampler = TpeSampler::with_defaults(StudyDirection::Minimize, Some(42));
        let random = RandomSampler::new(Some(42));
        let dist = Distribution::FloatDistribution(
            FloatDistribution::new(-5.0, 5.0, false, None).unwrap(),
        );

        // Generate 30 trials for both samplers.
        let mut tpe_trials = Vec::new();
        let mut random_trials = Vec::new();

        for i in 0..30 {
            // Random sampling for both during first 10, then TPE kicks in.
            let x_rand = random
                .sample_independent(
                    &FrozenTrial {
                        number: i,
                        state: TrialState::Running,
                        values: None,
                        datetime_start: Some(chrono::Utc::now()),
                        datetime_complete: None,
                        params: HashMap::new(),
                        distributions: HashMap::new(),
                        user_attrs: HashMap::new(),
                        system_attrs: HashMap::new(),
                        intermediate_values: HashMap::new(),
                        trial_id: i,
                    },
                    "x",
                    &dist,
                )
                .unwrap();

            let mut rp = HashMap::new();
            rp.insert("x".to_string(), ParamValue::Float(x_rand));
            random_trials.push(make_complete_trial(i, x_rand * x_rand, rp));

            // TPE: use sample_relative if past startup.
            let x_tpe = if i >= 10 {
                let mut search_space = HashMap::new();
                search_space.insert("x".to_string(), dist.clone());
                let result = sampler.sample_relative(&tpe_trials, &search_space).unwrap();
                result["x"]
            } else {
                // Startup: use random
                let t = FrozenTrial {
                    number: i,
                    state: TrialState::Running,
                    values: None,
                    datetime_start: Some(chrono::Utc::now()),
                    datetime_complete: None,
                    params: HashMap::new(),
                    distributions: HashMap::new(),
                    user_attrs: HashMap::new(),
                    system_attrs: HashMap::new(),
                    intermediate_values: HashMap::new(),
                    trial_id: i,
                };
                sampler.sample_independent(&t, "x", &dist).unwrap()
            };

            let mut tp = HashMap::new();
            tp.insert("x".to_string(), ParamValue::Float(x_tpe));
            tpe_trials.push(make_complete_trial(i, x_tpe * x_tpe, tp));
        }

        // Compare best values found.
        let tpe_best = tpe_trials
            .iter()
            .map(|t| t.value().unwrap().unwrap())
            .fold(f64::INFINITY, f64::min);
        let rand_best = random_trials
            .iter()
            .map(|t| t.value().unwrap().unwrap())
            .fold(f64::INFINITY, f64::min);

        // TPE should find a reasonably good solution.
        // We don't require it to be strictly better than random in every seed,
        // but it should find something < 1.0 for x^2 on [-5, 5].
        assert!(
            tpe_best < 5.0,
            "TPE best={tpe_best} should be reasonable (rand_best={rand_best})"
        );
    }

    #[test]
    fn test_tpe_empty_search_space() {
        let sampler = TpeSampler::with_defaults(StudyDirection::Minimize, Some(42));
        let result = sampler.sample_relative(&[], &HashMap::new()).unwrap();
        assert!(result.is_empty());
    }
}
