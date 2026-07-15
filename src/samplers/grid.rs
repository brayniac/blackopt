use std::collections::HashMap;

use parking_lot::Mutex;
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;

use crate::distributions::Distribution;
use crate::error::{Error, Result};
use crate::samplers::Sampler;
use crate::trial::{FrozenTrial, TrialState};

/// A sampler that exhaustively searches over a given parameter grid.
///
///
/// Generates a cartesian product of all parameter values, shuffles them,
/// and assigns grid points to trials. Once all grid points are exhausted,
/// raises an error.
pub struct GridSampler {
    /// The parameter grid: param name → list of internal-repr f64 values.
    search_space: HashMap<String, Vec<f64>>,
    /// All grid points as cartesian product, shuffled at construction.
    /// Each entry: Vec of (param_name, value) pairs in stable order.
    all_grids: Vec<Vec<(String, f64)>>,
    /// RNG for random selection from unvisited grids.
    rng: Mutex<ChaCha8Rng>,
}

impl std::fmt::Debug for GridSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GridSampler")
            .field("n_grids", &self.all_grids.len())
            .finish()
    }
}

impl GridSampler {
    /// Create a new `GridSampler`.
    ///
    /// # Arguments
    /// * `search_space` - Map from param name to list of internal-repr values.
    /// * `seed` - Optional seed for shuffling and random selection.
    pub fn new(search_space: HashMap<String, Vec<f64>>, seed: Option<u64>) -> Self {
        let mut rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };

        // Sort param names for deterministic ordering.
        let mut param_names: Vec<String> = search_space.keys().cloned().collect();
        param_names.sort();

        // Build cartesian product.
        let mut grids: Vec<Vec<(String, f64)>> = vec![vec![]];
        for name in &param_names {
            let values = &search_space[name];
            let mut new_grids = Vec::with_capacity(grids.len() * values.len());
            for grid in &grids {
                for &val in values {
                    let mut entry = grid.clone();
                    entry.push((name.clone(), val));
                    new_grids.push(entry);
                }
            }
            grids = new_grids;
        }

        // Shuffle the grid for randomized traversal order.
        grids.shuffle(&mut rng);

        Self {
            search_space,
            all_grids: grids,
            rng: Mutex::new(rng),
        }
    }

    /// Convenience: create a GridSampler from distributions, using all discrete values.
    ///
    /// For `IntDistribution(low, high, step)`, enumerates all steps.
    /// For `FloatDistribution` with step, enumerates all steps.
    /// For `CategoricalDistribution`, enumerates all indices.
    /// For continuous `FloatDistribution`, returns an error.
    pub fn from_distributions(
        distributions: HashMap<String, Distribution>,
        seed: Option<u64>,
    ) -> Result<Self> {
        let mut search_space = HashMap::new();
        for (name, dist) in &distributions {
            let values = Self::enumerate_distribution(dist).ok_or_else(|| {
                Error::ValueError(format!(
                    "GridSampler: cannot enumerate continuous distribution for param '{name}'"
                ))
            })?;
            search_space.insert(name.clone(), values);
        }
        Ok(Self::new(search_space, seed))
    }

    /// Enumerate all values in a distribution if it's discrete, returns None for continuous.
    fn enumerate_distribution(dist: &Distribution) -> Option<Vec<f64>> {
        match dist {
            Distribution::IntDistribution(d) => {
                let mut vals = Vec::new();
                let mut v = d.low;
                while v <= d.high {
                    vals.push(v as f64);
                    v += d.step;
                }
                Some(vals)
            }
            Distribution::FloatDistribution(d) => {
                if let Some(step) = d.step {
                    let mut vals = Vec::new();
                    let n_steps = ((d.high - d.low) / step).round() as i64;
                    for i in 0..=n_steps {
                        let v = d.low + step * i as f64;
                        if v <= d.high + 1e-8 {
                            vals.push(v);
                        }
                    }
                    Some(vals)
                } else if d.single() {
                    Some(vec![d.low])
                } else {
                    None // continuous, can't enumerate
                }
            }
            Distribution::CategoricalDistribution(d) => {
                Some((0..d.choices.len()).map(|i| i as f64).collect())
            }
        }
    }

    /// Get the grid_id for a trial from its system_attrs.
    fn get_grid_id(trial: &FrozenTrial) -> Option<usize> {
        trial.system_attrs.get("grid_id").and_then(|v| match v {
            serde_json::Value::Number(n) => n.as_u64().map(|n| n as usize),
            serde_json::Value::String(s) => s.parse::<usize>().ok(),
            _ => None,
        })
    }

    /// Get all grid_ids already assigned to trials.
    fn get_visited_grid_ids(&self, trials: &[FrozenTrial]) -> Vec<usize> {
        trials.iter().filter_map(Self::get_grid_id).collect()
    }

    /// Get unvisited grid indices.
    fn get_unvisited_grid_ids(&self, trials: &[FrozenTrial]) -> Vec<usize> {
        let visited: std::collections::HashSet<usize> =
            self.get_visited_grid_ids(trials).into_iter().collect();
        (0..self.all_grids.len())
            .filter(|i| !visited.contains(i))
            .collect()
    }

    /// Pick a grid_id for a trial. Returns the index into all_grids.
    fn pick_grid_id(&self, trials: &[FrozenTrial]) -> Result<usize> {
        let unvisited = self.get_unvisited_grid_ids(trials);
        if unvisited.is_empty() {
            return Err(Error::ValueError(
                "GridSampler: all grid points have been visited".to_string(),
            ));
        }

        // Pick randomly from unvisited.
        let mut rng = self.rng.lock();
        let &grid_id = unvisited.choose(&mut *rng).unwrap();
        Ok(grid_id)
    }
}

impl Sampler for GridSampler {
    fn infer_relative_search_space(
        &self,
        _trials: &[FrozenTrial],
    ) -> HashMap<String, Distribution> {
        // GridSampler doesn't use relative sampling; everything goes through sample_independent.
        HashMap::new()
    }

    fn sample_independent(
        &self,
        trial: &FrozenTrial,
        param_name: &str,
        _distribution: &Distribution,
    ) -> Result<f64> {
        // Check if this param is in our search space.
        if !self.search_space.contains_key(param_name) {
            return Err(Error::ValueError(format!(
                "GridSampler: unknown param '{param_name}'"
            )));
        }

        // Get or assign grid_id for this trial.
        let grid_id = match Self::get_grid_id(trial) {
            Some(id) => id,
            None => {
                // This shouldn't happen normally — grid_id should be set in before_trial.
                // Fall back to picking one now.
                return Err(Error::ValueError(
                    "GridSampler: trial has no grid_id; call before_trial first".to_string(),
                ));
            }
        };

        if grid_id >= self.all_grids.len() {
            return Err(Error::ValueError(format!(
                "GridSampler: grid_id {grid_id} out of range"
            )));
        }

        // Find the value for this param in the grid point.
        let grid_point = &self.all_grids[grid_id];
        for (name, value) in grid_point {
            if name == param_name {
                return Ok(*value);
            }
        }

        Err(Error::ValueError(format!(
            "GridSampler: param '{param_name}' not found in grid point"
        )))
    }

    fn before_trial(&self, trials: &[FrozenTrial]) {
        // This is called before a trial starts. We need to assign a grid_id.
        // However, we can't modify the trial from here — the study will need to
        // set the system_attr. We'll store the suggested grid_id so the study
        // can retrieve it.
        //
        // Note: grid_id is set as a system_attr in before_trial
        // on the trial via study._storage. Since our trait doesn't give us storage
        // access, the Study::ask() method handles this by calling
        // `suggest_grid_id()` and setting the system_attr.
        let _ = trials;
    }

    fn after_trial(
        &self,
        _trials: &[FrozenTrial],
        _trial: &FrozenTrial,
        _state: TrialState,
        _values: Option<&[f64]>,
    ) {
    }
}

impl GridSampler {
    /// Suggest a grid_id for a new trial. Called by Study::ask().
    ///
    /// Returns the grid_id that should be set as system_attr "grid_id".
    pub fn suggest_grid_id(&self, trials: &[FrozenTrial]) -> Result<usize> {
        self.pick_grid_id(trials)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::*;
    use crate::trial::TrialState;

    fn make_trial_with_grid_id(number: i64, grid_id: usize, state: TrialState) -> FrozenTrial {
        let now = chrono::Utc::now();
        let mut system_attrs = HashMap::new();
        system_attrs.insert(
            "grid_id".to_string(),
            serde_json::Value::Number(serde_json::Number::from(grid_id)),
        );
        FrozenTrial {
            number,
            state,
            values: if state == TrialState::Complete {
                Some(vec![0.0])
            } else {
                None
            },
            datetime_start: Some(now),
            datetime_complete: if state.is_finished() { Some(now) } else { None },
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs,
            intermediate_values: HashMap::new(),
            trial_id: number,
        }
    }

    #[test]
    fn test_grid_sampler_exhausts_all_points() {
        let mut space = HashMap::new();
        space.insert("x".to_string(), vec![1.0, 2.0]);
        space.insert("y".to_string(), vec![10.0, 20.0]);
        let sampler = GridSampler::new(space, Some(42));

        assert_eq!(sampler.all_grids.len(), 4);

        // Assign all 4 grid points.
        let mut trials = Vec::new();
        let mut assigned_ids = Vec::new();
        for i in 0..4 {
            let grid_id = sampler.suggest_grid_id(&trials).unwrap();
            assigned_ids.push(grid_id);
            trials.push(make_trial_with_grid_id(i, grid_id, TrialState::Complete));
        }

        // All 4 grid_ids should be unique.
        assigned_ids.sort();
        assigned_ids.dedup();
        assert_eq!(assigned_ids.len(), 4);

        // 5th trial should error.
        assert!(sampler.suggest_grid_id(&trials).is_err());
    }

    #[test]
    fn test_grid_sampler_samples_correct_values() {
        let mut space = HashMap::new();
        space.insert("x".to_string(), vec![1.0, 2.0, 3.0]);
        let sampler = GridSampler::new(space, Some(42));

        let grid_id = sampler.suggest_grid_id(&[]).unwrap();
        let trial = make_trial_with_grid_id(0, grid_id, TrialState::Running);
        let dist = Distribution::IntDistribution(IntDistribution::new(1, 3, false, 1).unwrap());
        let val = sampler.sample_independent(&trial, "x", &dist).unwrap();
        assert!([1.0, 2.0, 3.0].contains(&val));
    }

    #[test]
    fn test_grid_sampler_from_distributions() {
        let mut dists = HashMap::new();
        dists.insert(
            "x".to_string(),
            Distribution::IntDistribution(IntDistribution::new(0, 4, false, 2).unwrap()),
        );
        dists.insert(
            "opt".to_string(),
            Distribution::CategoricalDistribution(
                CategoricalDistribution::new(vec![
                    CategoricalChoice::Str("a".into()),
                    CategoricalChoice::Str("b".into()),
                ])
                .unwrap(),
            ),
        );
        let sampler = GridSampler::from_distributions(dists, Some(0)).unwrap();
        // x: [0, 2, 4] = 3 values, opt: [0, 1] = 2 values → 6 grid points
        assert_eq!(sampler.all_grids.len(), 6);
    }

    #[test]
    fn test_grid_sampler_continuous_float_rejected() {
        let mut dists = HashMap::new();
        dists.insert(
            "x".to_string(),
            Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap()),
        );
        assert!(GridSampler::from_distributions(dists, None).is_err());
    }

    #[test]
    fn test_grid_sampler_deterministic_with_seed() {
        let mk = || {
            let mut space = HashMap::new();
            space.insert("x".to_string(), vec![1.0, 2.0, 3.0]);
            space.insert("y".to_string(), vec![10.0, 20.0]);
            GridSampler::new(space, Some(99))
        };
        let s1 = mk();
        let s2 = mk();

        // Same seed should produce same grid order.
        assert_eq!(s1.all_grids, s2.all_grids);

        // Same sequence of grid_id suggestions.
        let id1 = s1.suggest_grid_id(&[]).unwrap();
        let id2 = s2.suggest_grid_id(&[]).unwrap();
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_grid_sampler_unknown_param_error() {
        let mut space = HashMap::new();
        space.insert("x".to_string(), vec![1.0]);
        let sampler = GridSampler::new(space, Some(42));
        let trial = make_trial_with_grid_id(0, 0, TrialState::Running);
        let dist =
            Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
        assert!(
            sampler
                .sample_independent(&trial, "unknown", &dist)
                .is_err()
        );
    }

    #[test]
    fn test_grid_sampler_float_step() {
        let mut dists = HashMap::new();
        dists.insert(
            "lr".to_string(),
            Distribution::FloatDistribution(
                FloatDistribution::new(0.0, 1.0, false, Some(0.25)).unwrap(),
            ),
        );
        let sampler = GridSampler::from_distributions(dists, Some(0)).unwrap();
        // 0.0, 0.25, 0.5, 0.75, 1.0 → 5 values
        assert_eq!(sampler.all_grids.len(), 5);
    }
}
