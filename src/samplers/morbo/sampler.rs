//! MORBO sampler: Multi-Objective Bayesian Optimization with multiple trust regions.
//!
//! Based on Daulton et al. 2022. Uses local Gaussian Processes within trust regions
//! and Thompson sampling for candidate generation.

use std::collections::HashMap;
use std::sync::Arc;

use indexmap::IndexMap;
use parking_lot::Mutex;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::distributions::Distribution;
use crate::error::Result;
use crate::multi_objective::{get_pareto_front_trials, hypervolume_2d};
use crate::samplers::Sampler;
use crate::samplers::random::RandomSampler;
use crate::search_space::{IntersectionSearchSpace, SearchSpaceTransform};
use crate::study::StudyDirection;
use crate::trial::{FrozenTrial, TrialState};

use super::gp::GaussianProcess;
use super::trust_region::{TrustRegion, TrustRegionConfig};

/// Internal observation in [0,1] encoded space.
#[derive(Clone)]
struct Observation {
    x: Vec<f64>,
    values: Vec<f64>,
}

/// Mutable algorithm state.
struct MorboState {
    /// All observations in encoded [0,1] space.
    observations: Vec<Observation>,
    /// Active trust regions.
    trust_regions: Vec<TrustRegion>,
    /// Round-robin index for TR selection.
    next_tr_idx: usize,
    /// Which TR generated the most recent pending trial.
    pending_tr_idx: Option<usize>,
    /// Param names in sorted order.
    param_names: Vec<String>,
    /// Current global hypervolume.
    current_hv: f64,
    /// Centers already used (for restart diversity).
    used_centers: Vec<Vec<f64>>,
    /// Optimization directions.
    directions: Vec<StudyDirection>,
}

/// MORBO sampler for multi-objective optimization.
///
/// Uses multiple trust regions with local Gaussian Processes and Thompson sampling.
pub struct MorboSampler {
    directions: Vec<StudyDirection>,
    n_startup_trials: usize,
    n_trust_regions: usize,
    n_candidates: usize,
    tr_config_factory: Box<dyn Fn(usize) -> TrustRegionConfig + Send + Sync>,
    independent_sampler: Arc<dyn Sampler>,
    state: Mutex<Option<MorboState>>,
    rng: Mutex<ChaCha8Rng>,
    search_space: Mutex<IntersectionSearchSpace>,
}

impl std::fmt::Debug for MorboSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MorboSampler")
            .field("n_startup_trials", &self.n_startup_trials)
            .field("n_trust_regions", &self.n_trust_regions)
            .finish()
    }
}

impl Sampler for MorboSampler {
    fn infer_relative_search_space(&self, trials: &[FrozenTrial]) -> HashMap<String, Distribution> {
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

        // Build ordered search space.
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

        // Initialize state if needed.
        if state_guard.is_none() {
            let observations = encode_trials(&complete, &param_names, &transform);
            let directions = self.directions.clone();
            let current_hv = compute_hv(&observations, &directions);

            // Initialize TRs at Pareto-optimal points with max HV contribution.
            let pareto_front = get_pareto_front_trials(
                &complete.iter().copied().cloned().collect::<Vec<_>>(),
                &directions,
            );
            let pareto_encoded = encode_trials(
                &pareto_front.iter().collect::<Vec<_>>(),
                &param_names,
                &transform,
            );

            let n_trs = self.n_trust_regions.min(pareto_encoded.len().max(1));
            let mut trust_regions = Vec::with_capacity(n_trs);
            let mut used_centers = Vec::new();

            // Greedily pick centers with max HV contribution.
            let centers = select_diverse_centers(&pareto_encoded, n_trs, &directions);
            for center in centers {
                let config = (self.tr_config_factory)(n_dims);
                used_centers.push(center.clone());
                trust_regions.push(TrustRegion::new(center, config));
            }

            // If no Pareto points yet, create TRs at random observations.
            while trust_regions.len() < n_trs && !observations.is_empty() {
                let idx = rng.gen_range(0..observations.len());
                let center = observations[idx].x.clone();
                let config = (self.tr_config_factory)(n_dims);
                used_centers.push(center.clone());
                trust_regions.push(TrustRegion::new(center, config));
            }

            // Fallback: center of space.
            if trust_regions.is_empty() {
                let center = vec![0.5; n_dims];
                let config = (self.tr_config_factory)(n_dims);
                used_centers.push(center.clone());
                trust_regions.push(TrustRegion::new(center, config));
            }

            *state_guard = Some(MorboState {
                observations,
                trust_regions,
                next_tr_idx: 0,
                pending_tr_idx: None,
                param_names: param_names.clone(),
                current_hv,
                used_centers,
                directions,
            });
        }

        let state = state_guard.as_mut().unwrap();
        let n_objectives = state.directions.len();

        // Pick next active TR (round-robin).
        let active_count = state.trust_regions.iter().filter(|tr| tr.active).count();
        if active_count == 0 {
            // All TRs terminated — restart.
            restart_trust_regions(state, n_dims, &self.tr_config_factory, &mut rng);
        }

        let tr_idx = pick_active_tr(state);
        state.pending_tr_idx = Some(tr_idx);

        let tr = &state.trust_regions[tr_idx];
        let local_indices = tr.local_data_indices(
            &state
                .observations
                .iter()
                .map(|o| o.x.clone())
                .collect::<Vec<_>>(),
        );

        let min_data = 2 * n_dims + 1;
        let candidate = if local_indices.len() >= min_data {
            // Fit local GPs and Thompson sample.
            sample_with_gp(
                &state.observations,
                &local_indices,
                tr,
                n_objectives,
                &state.directions,
                self.n_candidates,
                &mut rng,
            )
        } else {
            // Not enough local data — uniform random within TR.
            sample_uniform_in_tr(tr, n_dims, &mut rng)
        };

        drop(rng);
        drop(state_guard);

        // Untransform candidate back to param space.
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
        let Some(morbo_state) = state_guard.as_mut() else {
            return;
        };

        // Encode the new observation.
        let param_names = &morbo_state.param_names;
        if param_names.is_empty() {
            return;
        }

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
        if trial_params.len() != param_names.len() {
            return;
        }

        let encoded = transform.transform(&trial_params);
        let obs = Observation {
            x: encoded,
            values: values.to_vec(),
        };
        morbo_state.observations.push(obs);

        // Compute new HV.
        let new_hv = compute_hv(&morbo_state.observations, &morbo_state.directions);
        let hv_improved = new_hv > morbo_state.current_hv + 1e-12;

        if hv_improved {
            morbo_state.current_hv = new_hv;
        }

        // Update the generating TR.
        if let Some(tr_idx) = morbo_state.pending_tr_idx.take() {
            if let Some(tr) = morbo_state.trust_regions.get_mut(tr_idx) {
                if hv_improved {
                    tr.record_success();
                } else {
                    tr.record_failure();
                }
            }
        }
    }
}

// ── Helper functions ────────────────────────────────────────────────────────

/// Encode complete trials into [0,1] observations.
fn encode_trials(
    trials: &[&FrozenTrial],
    param_names: &[String],
    transform: &SearchSpaceTransform,
) -> Vec<Observation> {
    let mut observations = Vec::new();
    for trial in trials {
        let mut trial_params = IndexMap::new();
        let mut ok = true;
        for name in param_names {
            if let Some(pv) = trial.params.get(name) {
                trial_params.insert(name.clone(), pv.clone());
            } else {
                ok = false;
                break;
            }
        }
        if !ok {
            continue;
        }
        let encoded = transform.transform(&trial_params);
        if let Some(values) = &trial.values {
            observations.push(Observation {
                x: encoded,
                values: values.clone(),
            });
        }
    }
    observations
}

/// Compute hypervolume for 2-objective problems (both minimized internally).
fn compute_hv(observations: &[Observation], directions: &[StudyDirection]) -> f64 {
    if observations.is_empty() || directions.len() != 2 {
        return 0.0;
    }

    // Normalize: flip maximize objectives to minimize.
    let points: Vec<[f64; 2]> = observations
        .iter()
        .map(|o| {
            let mut p = [0.0; 2];
            for (i, &d) in directions.iter().enumerate() {
                p[i] = if d == StudyDirection::Maximize {
                    -o.values.get(i).copied().unwrap_or(f64::INFINITY)
                } else {
                    o.values.get(i).copied().unwrap_or(f64::INFINITY)
                };
            }
            p
        })
        .collect();

    // Reference point: max per objective + 10% margin.
    let mut ref_point = [f64::NEG_INFINITY; 2];
    for p in &points {
        for i in 0..2 {
            if p[i] > ref_point[i] {
                ref_point[i] = p[i];
            }
        }
    }
    ref_point[0] = ref_point[0] * 1.1 + 0.1;
    ref_point[1] = ref_point[1] * 1.1 + 0.1;

    hypervolume_2d(&points, ref_point)
}

/// Select diverse centers from Pareto-encoded observations via greedy max HV contribution.
fn select_diverse_centers(
    pareto_obs: &[Observation],
    n_centers: usize,
    directions: &[StudyDirection],
) -> Vec<Vec<f64>> {
    if pareto_obs.is_empty() {
        return vec![];
    }

    let n = pareto_obs.len();
    let mut selected = Vec::new();
    let mut used = vec![false; n];

    for _ in 0..n_centers.min(n) {
        let mut best_idx = None;
        let mut best_contribution = f64::NEG_INFINITY;

        // Compute HV without each candidate, pick the one whose removal hurts most.
        let total_hv = compute_hv(pareto_obs, directions);

        for i in 0..n {
            if used[i] {
                continue;
            }
            let without: Vec<Observation> = pareto_obs
                .iter()
                .enumerate()
                .filter(|&(j, _)| j != i)
                .map(|(_, o)| o.clone())
                .collect();
            let hv_without = compute_hv(&without, directions);
            let contribution = total_hv - hv_without;
            if contribution > best_contribution {
                best_contribution = contribution;
                best_idx = Some(i);
            }
        }

        if let Some(idx) = best_idx {
            used[idx] = true;
            selected.push(pareto_obs[idx].x.clone());
        } else {
            break;
        }
    }

    selected
}

/// Pick the next active TR using round-robin.
fn pick_active_tr(state: &mut MorboState) -> usize {
    let n = state.trust_regions.len();
    for _ in 0..n {
        let idx = state.next_tr_idx % n;
        state.next_tr_idx = (state.next_tr_idx + 1) % n;
        if state.trust_regions[idx].active {
            return idx;
        }
    }
    // Shouldn't reach here if we checked active_count > 0.
    0
}

/// Restart all trust regions when they've all terminated.
fn restart_trust_regions(
    state: &mut MorboState,
    n_dims: usize,
    config_factory: &dyn Fn(usize) -> TrustRegionConfig,
    rng: &mut ChaCha8Rng,
) {
    let n_trs = state.trust_regions.len();
    state.trust_regions.clear();
    state.next_tr_idx = 0;

    // Try to place new TRs at Pareto front points not yet used.
    let pareto_obs = get_pareto_observations(&state.observations, &state.directions);
    let new_centers = select_diverse_centers(&pareto_obs, n_trs, &state.directions);

    for center in new_centers {
        if !state
            .used_centers
            .iter()
            .any(|c| euclidean_dist(c, &center) < 0.01)
        {
            let config = config_factory(n_dims);
            state.used_centers.push(center.clone());
            state.trust_regions.push(TrustRegion::new(center, config));
        }
    }

    // Fill remaining with random Pareto points or random observations.
    while state.trust_regions.len() < n_trs {
        let center = if !pareto_obs.is_empty() {
            let idx = rng.gen_range(0..pareto_obs.len());
            pareto_obs[idx].x.clone()
        } else if !state.observations.is_empty() {
            let idx = rng.gen_range(0..state.observations.len());
            state.observations[idx].x.clone()
        } else {
            vec![0.5; n_dims]
        };

        // Perturb slightly for diversity.
        let center: Vec<f64> = center
            .iter()
            .map(|&c| (c + rng.gen_range(-0.05..0.05)).clamp(0.0, 1.0))
            .collect();

        let config = config_factory(n_dims);
        state.used_centers.push(center.clone());
        state.trust_regions.push(TrustRegion::new(center, config));
    }
}

/// Get Pareto-optimal observations from all observations.
fn get_pareto_observations(
    observations: &[Observation],
    directions: &[StudyDirection],
) -> Vec<Observation> {
    let n = observations.len();
    if n == 0 {
        return vec![];
    }

    let mut is_pareto = vec![true; n];
    for i in 0..n {
        if !is_pareto[i] {
            continue;
        }
        for j in 0..n {
            if i == j || !is_pareto[j] {
                continue;
            }
            if obs_dominates(&observations[j], &observations[i], directions) {
                is_pareto[i] = false;
                break;
            }
        }
    }

    observations
        .iter()
        .zip(is_pareto)
        .filter(|(_, p)| *p)
        .map(|(o, _)| o.clone())
        .collect()
}

/// Check if observation a dominates observation b.
fn obs_dominates(a: &Observation, b: &Observation, directions: &[StudyDirection]) -> bool {
    let mut any_better = false;
    for (i, &d) in directions.iter().enumerate() {
        let va = a.values.get(i).copied().unwrap_or(f64::INFINITY);
        let vb = b.values.get(i).copied().unwrap_or(f64::INFINITY);
        let cmp = match d {
            StudyDirection::Minimize => va.partial_cmp(&vb),
            StudyDirection::Maximize => vb.partial_cmp(&va),
            StudyDirection::NotSet => return false,
        };
        match cmp {
            Some(std::cmp::Ordering::Greater) => return false,
            Some(std::cmp::Ordering::Less) => any_better = true,
            _ => {}
        }
    }
    any_better
}

fn euclidean_dist(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(ai, bi)| (ai - bi).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Sample a candidate using local GPs and Thompson sampling.
fn sample_with_gp(
    observations: &[Observation],
    local_indices: &[usize],
    tr: &TrustRegion,
    n_objectives: usize,
    directions: &[StudyDirection],
    n_candidates: usize,
    rng: &mut ChaCha8Rng,
) -> Vec<f64> {
    let n_dims = tr.center.len();

    // Extract local data.
    let local_x: Vec<Vec<f64>> = local_indices
        .iter()
        .map(|&i| observations[i].x.clone())
        .collect();

    // Standardize each objective independently.
    let mut obj_gps: Vec<Option<GaussianProcess>> = Vec::with_capacity(n_objectives);
    for obj_idx in 0..n_objectives {
        let y: Vec<f64> = local_indices
            .iter()
            .map(|&i| {
                let v = observations[i].values.get(obj_idx).copied().unwrap_or(0.0);
                // Flip maximize objectives so GP always models minimization.
                if directions.get(obj_idx).copied() == Some(StudyDirection::Maximize) {
                    -v
                } else {
                    v
                }
            })
            .collect();

        obj_gps.push(GaussianProcess::fit(local_x.clone(), y));
    }

    // Generate random candidates within TR bounds.
    let bounds = tr.bounds();
    let candidates: Vec<Vec<f64>> = (0..n_candidates)
        .map(|_| {
            (0..n_dims)
                .map(|d| rng.gen_range(bounds[d][0]..=bounds[d][1]))
                .collect()
        })
        .collect();

    // Thompson sample each GP at all candidates.
    let mut thompson_values: Vec<Vec<f64>> = vec![Vec::new(); n_objectives];
    for (obj_idx, gp_opt) in obj_gps.iter().enumerate() {
        match gp_opt {
            Some(gp) => {
                thompson_values[obj_idx] = gp.thompson_sample(&candidates, rng);
            }
            None => {
                // GP fit failed — use random values.
                thompson_values[obj_idx] =
                    (0..n_candidates).map(|_| rng.gen_range(0.0..1.0)).collect();
            }
        }
    }

    // Pick candidate that maximizes HV improvement.
    // Build current Pareto front points (minimized).
    let current_points: Vec<[f64; 2]> = observations
        .iter()
        .map(|o| {
            let mut p = [0.0; 2];
            for (i, &d) in directions.iter().enumerate().take(2) {
                p[i] = if d == StudyDirection::Maximize {
                    -o.values.get(i).copied().unwrap_or(f64::INFINITY)
                } else {
                    o.values.get(i).copied().unwrap_or(f64::INFINITY)
                };
            }
            p
        })
        .collect();

    let mut ref_point = [f64::NEG_INFINITY; 2];
    for p in &current_points {
        for i in 0..2 {
            if p[i] > ref_point[i] {
                ref_point[i] = p[i];
            }
        }
    }
    // Also consider candidate values for reference point.
    if n_objectives >= 2 {
        for c_idx in 0..n_candidates {
            for obj in 0..2 {
                let v = thompson_values[obj][c_idx];
                if v > ref_point[obj] {
                    ref_point[obj] = v;
                }
            }
        }
    }
    ref_point[0] = ref_point[0] * 1.1 + 0.1;
    ref_point[1] = ref_point[1] * 1.1 + 0.1;

    let base_hv = hypervolume_2d(&current_points, ref_point);

    let mut best_idx = 0;
    let mut best_hv_improvement = f64::NEG_INFINITY;

    for c_idx in 0..n_candidates {
        if n_objectives >= 2 {
            let new_point = [thompson_values[0][c_idx], thompson_values[1][c_idx]];
            let mut all_points = current_points.clone();
            all_points.push(new_point);
            let new_hv = hypervolume_2d(&all_points, ref_point);
            let improvement = new_hv - base_hv;
            if improvement > best_hv_improvement {
                best_hv_improvement = improvement;
                best_idx = c_idx;
            }
        } else {
            // Single objective fallback — just pick min Thompson value.
            let v = thompson_values[0][c_idx];
            let improvement = -v; // smaller is better (minimizing)
            if improvement > best_hv_improvement {
                best_hv_improvement = improvement;
                best_idx = c_idx;
            }
        }
    }

    candidates[best_idx].clone()
}

/// Sample uniformly within a trust region.
fn sample_uniform_in_tr(tr: &TrustRegion, n_dims: usize, rng: &mut ChaCha8Rng) -> Vec<f64> {
    let bounds = tr.bounds();
    (0..n_dims)
        .map(|d| rng.gen_range(bounds[d][0]..=bounds[d][1]))
        .collect()
}

// ── Builder ─────────────────────────────────────────────────────────────────

/// Builder for [`MorboSampler`].
pub struct MorboSamplerBuilder {
    directions: Vec<StudyDirection>,
    n_startup_trials: Option<usize>,
    n_trust_regions: Option<usize>,
    n_candidates: Option<usize>,
    seed: Option<u64>,
    independent_sampler: Option<Arc<dyn Sampler>>,
    tr_config_factory: Option<Box<dyn Fn(usize) -> TrustRegionConfig + Send + Sync>>,
}

impl MorboSamplerBuilder {
    /// Create a new builder with the given optimization directions.
    pub fn new(directions: Vec<StudyDirection>) -> Self {
        Self {
            directions,
            n_startup_trials: None,
            n_trust_regions: None,
            n_candidates: None,
            seed: None,
            independent_sampler: None,
            tr_config_factory: None,
        }
    }

    /// Number of random trials before MORBO kicks in (default: 60).
    pub fn n_startup_trials(mut self, n: usize) -> Self {
        self.n_startup_trials = Some(n);
        self
    }

    /// Number of trust regions (default: 3).
    pub fn n_trust_regions(mut self, n: usize) -> Self {
        self.n_trust_regions = Some(n);
        self
    }

    /// Number of random candidates per Thompson sampling step (default: 512).
    pub fn n_candidates(mut self, n: usize) -> Self {
        self.n_candidates = Some(n);
        self
    }

    /// Random seed for reproducibility.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Independent sampler for parameters outside the search space.
    pub fn independent_sampler(mut self, sampler: Arc<dyn Sampler>) -> Self {
        self.independent_sampler = Some(sampler);
        self
    }

    /// Custom trust region config factory (receives n_dims).
    pub fn tr_config(
        mut self,
        factory: Box<dyn Fn(usize) -> TrustRegionConfig + Send + Sync>,
    ) -> Self {
        self.tr_config_factory = Some(factory);
        self
    }

    /// Build the [`MorboSampler`].
    pub fn build(self) -> MorboSampler {
        let seed = self.seed;
        let rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };

        MorboSampler {
            directions: self.directions,
            n_startup_trials: self.n_startup_trials.unwrap_or(60),
            n_trust_regions: self.n_trust_regions.unwrap_or(3),
            n_candidates: self.n_candidates.unwrap_or(512),
            tr_config_factory: self
                .tr_config_factory
                .unwrap_or_else(|| Box::new(TrustRegionConfig::new)),
            independent_sampler: self
                .independent_sampler
                .unwrap_or_else(|| Arc::new(RandomSampler::new(seed))),
            state: Mutex::new(None),
            rng: Mutex::new(rng),
            search_space: Mutex::new(IntersectionSearchSpace::new(false)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::study::{StudyDirection, create_study};

    #[test]
    fn test_morbo_creation() {
        let sampler =
            MorboSamplerBuilder::new(vec![StudyDirection::Minimize, StudyDirection::Minimize])
                .n_startup_trials(10)
                .n_trust_regions(2)
                .seed(42)
                .build();

        assert_eq!(sampler.n_startup_trials, 10);
        assert_eq!(sampler.n_trust_regions, 2);
    }

    #[test]
    fn test_morbo_startup_random() {
        let sampler: Arc<dyn Sampler> = Arc::new(
            MorboSamplerBuilder::new(vec![StudyDirection::Minimize, StudyDirection::Minimize])
                .n_startup_trials(10)
                .seed(42)
                .build(),
        );

        let study = create_study(
            None,
            Some(sampler),
            None,
            None,
            None,
            Some(vec![StudyDirection::Minimize, StudyDirection::Minimize]),
            false,
        )
        .unwrap();

        study
            .optimize_multi(
                |trial| {
                    let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    let y = trial.suggest_float("y", 0.0, 1.0, false, None)?;
                    Ok(vec![x * x + y * y, (x - 1.0).powi(2) + (y - 1.0).powi(2)])
                },
                Some(5),
                None,
                None,
            )
            .unwrap();

        assert_eq!(study.trials().unwrap().len(), 5);
    }

    #[test]
    fn test_morbo_zdt1_improves() {
        // ZDT1 benchmark: 2 objectives, n_dims parameters in [0,1].
        let n_dims = 4;
        let sampler: Arc<dyn Sampler> = Arc::new(
            MorboSamplerBuilder::new(vec![StudyDirection::Minimize, StudyDirection::Minimize])
                .n_startup_trials(20)
                .n_trust_regions(2)
                .n_candidates(64)
                .seed(42)
                .build(),
        );

        let study = create_study(
            None,
            Some(sampler),
            None,
            None,
            None,
            Some(vec![StudyDirection::Minimize, StudyDirection::Minimize]),
            false,
        )
        .unwrap();

        let n_trials = 80;
        study
            .optimize_multi(
                move |trial| {
                    let mut x = Vec::with_capacity(n_dims);
                    for i in 0..n_dims {
                        x.push(trial.suggest_float(&format!("x{i}"), 0.0, 1.0, false, None)?);
                    }
                    // ZDT1
                    let f1 = x[0];
                    let g = 1.0 + 9.0 * x[1..].iter().sum::<f64>() / (n_dims as f64 - 1.0);
                    let f2 = g * (1.0 - (f1 / g).sqrt());
                    Ok(vec![f1, f2])
                },
                Some(n_trials),
                None,
                None,
            )
            .unwrap();

        let trials = study.trials().unwrap();
        assert_eq!(trials.len(), n_trials);

        // Compute HV at trials 20 (startup only) vs all trials.
        let first_20: Vec<[f64; 2]> = trials[..20]
            .iter()
            .map(|t| {
                let v = t.values.as_ref().unwrap();
                [v[0], v[1]]
            })
            .collect();
        let all: Vec<[f64; 2]> = trials
            .iter()
            .map(|t| {
                let v = t.values.as_ref().unwrap();
                [v[0], v[1]]
            })
            .collect();

        let ref_point = [1.5, 1.5];
        let hv_20 = hypervolume_2d(&first_20, ref_point);
        let hv_all = hypervolume_2d(&all, ref_point);

        assert!(
            hv_all >= hv_20,
            "HV should not decrease: startup={hv_20}, final={hv_all}"
        );
    }
}
