use std::collections::{HashMap, HashSet};

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
/// and assigns grid points to trials in that (seeded) order. Once all grid
/// points are exhausted, new trials fail with an error.
///
/// Grid points are assigned by the `trial_created` hook, which the study
/// invokes right after a trial is started; the assigned id is recorded on
/// the trial and served by `sample_independent`.
///
/// Each grid point is *attempted* exactly once. A trial that fails or is
/// pruned still consumes its point — the sampler makes no attempt to retry,
/// because a retry loop on a deterministic grid point would repeat the same
/// evaluation forever.
pub struct GridSampler {
    /// The parameter grid: param name → list of internal-repr f64 values.
    search_space: HashMap<String, Vec<f64>>,
    /// All grid points as cartesian product, shuffled at construction.
    /// Each entry: Vec of (param_name, value) pairs in stable order.
    all_grids: Vec<Vec<(String, f64)>>,
    /// Which grid ids have been handed out, and to what.
    allocation: Mutex<GridAllocation>,
}

/// Ownership of grid ids.
///
/// This is the sampler's own state rather than something re-derived from a
/// storage snapshot on each call: a snapshot read outside the lock races with
/// concurrent `ask` calls (two trials would be handed the same point), and
/// inferring consumption from a trial's recorded params makes the answer
/// depend on how far the objective got before it threw.
///
/// Both sets only ever grow (`pending` → `consumed` is the one transition),
/// so reconciling against a stale snapshot can never release a live
/// reservation.
#[derive(Default)]
struct GridAllocation {
    /// Ids belonging to trials that have finished, in any terminal state.
    consumed: HashSet<usize>,
    /// Ids belonging to trials that are still in flight.
    pending: HashSet<usize>,
}

impl GridAllocation {
    /// Fold a trial history into the allocation.
    ///
    /// Used to recover state when a fresh sampler is pointed at a study that
    /// already has trials, and to catch up on any trial that finished without
    /// `after_trial` running (a storage error in `tell`, say).
    fn reconcile(&mut self, trials: &[FrozenTrial]) {
        for t in trials {
            let Some(id) = GridSampler::get_grid_id(t).and_then(|id| usize::try_from(id).ok())
            else {
                continue;
            };
            if is_finished(t.state) {
                self.pending.remove(&id);
                self.consumed.insert(id);
            } else if !self.consumed.contains(&id) {
                self.pending.insert(id);
            }
        }
    }
}

fn is_finished(state: TrialState) -> bool {
    matches!(
        state,
        TrialState::Complete | TrialState::Fail | TrialState::Pruned
    )
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
            allocation: Mutex::new(GridAllocation::default()),
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

    /// Read the grid_id recorded on a trial's system attributes.
    ///
    /// Returns `None` if not assigned, `Some(-1)` if the sampler has run
    /// out of grid points.
    fn get_grid_id(trial: &FrozenTrial) -> Option<i64> {
        trial.system_attrs.get("grid_id").and_then(|v| match v {
            serde_json::Value::Number(n) => n.as_i64(),
            serde_json::Value::String(s) => s.parse::<i64>().ok(),
            _ => None,
        })
    }

    /// Reserve the next unassigned grid id, in shuffled grid order.
    ///
    /// `trials` is folded into the allocation first so a fresh sampler
    /// recovers the state of an existing study. The read and the reserve
    /// happen under one lock, so concurrent `ask` calls cannot be handed the
    /// same point.
    fn reserve_grid_id(&self, trials: &[FrozenTrial]) -> std::result::Result<usize, GridExhausted> {
        let mut alloc = self.allocation.lock();
        alloc.reconcile(trials);
        match (0..self.all_grids.len())
            .find(|i| !alloc.consumed.contains(i) && !alloc.pending.contains(i))
        {
            Some(id) => {
                alloc.pending.insert(id);
                Ok(id)
            }
            // Distinguish "the search really is finished" from "every
            // remaining point is held by a trial that was never told" — the
            // two need very different action from the caller.
            None if alloc.pending.is_empty() => Err(GridExhausted::AllVisited),
            None => Err(GridExhausted::AllInFlight(alloc.pending.len())),
        }
    }
}

/// Why no grid id could be reserved.
#[derive(Debug, Clone, Copy)]
enum GridExhausted {
    /// Every grid point has been attempted.
    AllVisited,
    /// Every remaining grid point is reserved by an unfinished trial.
    AllInFlight(usize),
}

impl GridExhausted {
    fn message(self) -> String {
        match self {
            Self::AllVisited => "GridSampler: all grid points have been visited".to_string(),
            Self::AllInFlight(n) => format!(
                concat!(
                    "GridSampler: all remaining grid points are assigned to {} ",
                    "unfinished trial(s); finish them (call `Study::tell`) ",
                    "before asking for another"
                ),
                n
            ),
        }
    }
}

impl Sampler for GridSampler {
    fn infer_relative_search_space(
        &self,
        _trials: &[FrozenTrial],
    ) -> HashMap<String, Distribution> {
        // GridSampler doesn't use relative sampling; every parameter is
        // served from the trial's assigned grid point.
        HashMap::new()
    }

    fn sample_independent(
        &self,
        _trials: &[FrozenTrial],
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

        let grid_id = match Self::get_grid_id(trial) {
            Some(id) if id >= 0 => id as usize,
            // Negative ids encode the exhaustion reason recorded by
            // `trial_created`; the message is stored alongside them.
            Some(_) => {
                let reason = trial
                    .system_attrs
                    .get("grid_exhausted")
                    .and_then(|v| v.as_str())
                    .unwrap_or("GridSampler: all grid points have been visited");
                return Err(Error::ValueError(reason.to_string()));
            }
            None => {
                return Err(Error::ValueError(format!(
                    concat!(
                        "GridSampler: trial {} has no grid point assigned, so it cannot ",
                        "supply '{}'. Parameters of a trial enqueued with ",
                        "`enqueue_trial` must all be pre-specified."
                    ),
                    trial.number, param_name
                )));
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

    fn trial_created(&self, trials: &[FrozenTrial], trial: &crate::trial::Trial) -> Result<()> {
        // Assign this trial's grid point before any parameter is suggested.
        match self.reserve_grid_id(trials) {
            Ok(id) => {
                // Record the assignment before it is relied on. If this write
                // fails the reservation must not stick, or the point would be
                // lost to a trial that never used it.
                if let Err(e) = trial.set_system_attr("grid_id", serde_json::json!(id)) {
                    self.allocation.lock().pending.remove(&id);
                    return Err(e);
                }
                Ok(())
            }
            Err(reason) => {
                // Mark the trial so suggest calls fail with a clear message.
                trial.set_system_attr("grid_id", serde_json::json!(-1))?;
                trial.set_system_attr("grid_exhausted", serde_json::json!(reason.message()))?;
                Ok(())
            }
        }
    }

    fn after_trial(
        &self,
        _trials: &[FrozenTrial],
        trial: &FrozenTrial,
        _state: TrialState,
        _values: Option<&[f64]>,
    ) {
        // The point has now been attempted; retiring it here (rather than
        // inferring it from the trial's recorded params) makes the outcome
        // independent of how far the objective got.
        if let Some(id) = Self::get_grid_id(trial).and_then(|id| usize::try_from(id).ok()) {
            let mut alloc = self.allocation.lock();
            alloc.pending.remove(&id);
            alloc.consumed.insert(id);
        }
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

        // Assign all 4 grid points (each stays pending until after_trial).
        let mut assigned_ids = Vec::new();
        for _ in 0..4 {
            let grid_id = sampler.reserve_grid_id(&[]).unwrap();
            assigned_ids.push(grid_id);
        }

        // All 4 grid_ids should be unique.
        let mut deduped = assigned_ids.clone();
        deduped.sort();
        deduped.dedup();
        assert_eq!(deduped.len(), 4);

        // 5th reservation fails, and says why: the points are in flight, not
        // exhausted.
        let err = sampler.reserve_grid_id(&[]).unwrap_err();
        assert!(matches!(err, GridExhausted::AllInFlight(4)), "{err:?}");
    }

    #[test]
    fn test_grid_sampler_samples_correct_values() {
        let mut space = HashMap::new();
        space.insert("x".to_string(), vec![1.0, 2.0, 3.0]);
        let sampler = GridSampler::new(space, Some(42));

        let grid_id = sampler.reserve_grid_id(&[]).unwrap();
        let trial = make_trial_with_grid_id(0, grid_id, TrialState::Running);
        let dist = Distribution::IntDistribution(IntDistribution::new(1, 3, false, 1).unwrap());
        let val = sampler.sample_independent(&[], &trial, "x", &dist).unwrap();
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

        // Same sequence of grid_id assignments.
        let id1 = s1.reserve_grid_id(&[]).unwrap();
        let id2 = s2.reserve_grid_id(&[]).unwrap();
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
                .sample_independent(&[], &trial, "unknown", &dist)
                .is_err()
        );
    }

    #[test]
    fn test_grid_sampler_recovers_allocation_from_history() {
        // A fresh sampler pointed at an existing study must not re-hand-out
        // points that history shows are already taken.
        let mut space = HashMap::new();
        space.insert("x".to_string(), vec![1.0, 2.0, 3.0]);
        let sampler = GridSampler::new(space, Some(42));

        let history = vec![
            make_trial_with_grid_id(0, 0, TrialState::Complete),
            make_trial_with_grid_id(1, 1, TrialState::Running),
        ];
        // 0 is consumed, 1 is in flight, so only 2 is free.
        assert_eq!(sampler.reserve_grid_id(&history).unwrap(), 2);
        assert!(matches!(
            sampler.reserve_grid_id(&history).unwrap_err(),
            GridExhausted::AllInFlight(_)
        ));
    }

    #[test]
    fn test_grid_sampler_failed_trial_consumes_its_point() {
        // Retrying a failed grid point would repeat the same deterministic
        // evaluation forever; a point is attempted exactly once.
        let mut space = HashMap::new();
        space.insert("x".to_string(), vec![1.0, 2.0]);
        let sampler = GridSampler::new(space, Some(42));

        let id = sampler.reserve_grid_id(&[]).unwrap();
        let failed = make_trial_with_grid_id(0, id, TrialState::Fail);
        sampler.after_trial(&[], &failed, TrialState::Fail, None);

        let next = sampler.reserve_grid_id(&[]).unwrap();
        assert_ne!(next, id, "a failed point must not be handed out again");
    }

    #[test]
    fn test_grid_sampler_end_to_end_evaluates_every_combination() {
        use crate::samplers::Sampler;
        use crate::study::{StudyDirection, create_study};
        use std::collections::HashSet;
        use std::sync::Arc;

        let mut space = HashMap::new();
        space.insert("x".to_string(), vec![1.0, 2.0]);
        space.insert("y".to_string(), vec![10.0, 20.0]);
        let sampler: Arc<dyn Sampler> = Arc::new(GridSampler::new(space, Some(42)));

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
                    let x = trial.suggest_int("x", 1, 2, false, 1)?;
                    let y = trial.suggest_int("y", 10, 20, false, 10)?;
                    Ok(x as f64 + y as f64)
                },
                Some(4),
                None,
                None,
            )
            .unwrap();

        let trials = study.trials().unwrap();
        let completed: Vec<&FrozenTrial> = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .collect();
        assert_eq!(completed.len(), 4, "all 4 grid points should complete");

        let combos: HashSet<(i64, i64)> = completed
            .iter()
            .map(|t| {
                let x = match t.params.get("x").unwrap() {
                    crate::distributions::ParamValue::Int(v) => *v,
                    crate::distributions::ParamValue::Float(v) => *v as i64,
                    _ => panic!("x should be numeric"),
                };
                let y = match t.params.get("y").unwrap() {
                    crate::distributions::ParamValue::Int(v) => *v,
                    crate::distributions::ParamValue::Float(v) => *v as i64,
                    _ => panic!("y should be numeric"),
                };
                (x, y)
            })
            .collect();
        assert_eq!(combos, HashSet::from([(1, 10), (1, 20), (2, 10), (2, 20)]));
    }

    #[test]
    fn test_grid_sampler_end_to_end_exhaustion_fails_clearly() {
        use crate::samplers::Sampler;
        use crate::study::{StudyDirection, create_study};
        use std::sync::Arc;

        let mut space = HashMap::new();
        space.insert("x".to_string(), vec![1.0, 2.0]);
        let sampler: Arc<dyn Sampler> = Arc::new(GridSampler::new(space, Some(42)));

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
                    let x = trial.suggest_int("x", 1, 2, false, 1)?;
                    Ok(x as f64)
                },
                Some(3),
                None,
                None,
            )
            .unwrap();

        let trials = study.trials().unwrap();
        let states: Vec<TrialState> = trials.iter().map(|t| t.state).collect();
        assert_eq!(
            states,
            vec![TrialState::Complete, TrialState::Complete, TrialState::Fail]
        );
        let failed = trials.iter().find(|t| t.state == TrialState::Fail).unwrap();
        assert_eq!(
            failed
                .system_attrs
                .get("failure_reason")
                .and_then(|v| v.as_str()),
            Some("GridSampler: all grid points have been visited")
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

    #[test]
    fn test_grid_sampler_enqueued_trial_does_not_consume_grid_point() {
        // Regression: an enqueued trial used to reserve a grid point it never
        // evaluated, so that point was silently lost and the grid reported
        // itself exhausted having missed a combination.
        use crate::distributions::ParamValue;
        use crate::samplers::Sampler;
        use crate::study::{StudyDirection, create_study};
        use std::collections::HashSet;
        use std::sync::Arc;

        let mut space = HashMap::new();
        space.insert("x".to_string(), vec![1.0, 2.0]);
        let sampler: Arc<dyn Sampler> = Arc::new(GridSampler::new(space, Some(42)));

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

        let mut enqueued = HashMap::new();
        enqueued.insert("x".to_string(), ParamValue::Int(1));
        study.enqueue_trial(enqueued, None).unwrap();

        // One enqueued trial plus both grid points.
        study
            .optimize(
                |trial| {
                    let x = trial.suggest_int("x", 1, 2, false, 1)?;
                    Ok(x as f64)
                },
                Some(3),
                None,
                None,
            )
            .unwrap();

        let trials = study.trials().unwrap();
        assert!(
            trials.iter().all(|t| t.state == TrialState::Complete),
            "no trial should fail: {:?}",
            trials.iter().map(|t| t.state).collect::<Vec<_>>()
        );
        // The enqueued trial holds no grid point...
        assert!(GridSampler::get_grid_id(&trials[0]).is_none());
        // ...and the grid still covered every combination.
        let grid_values: HashSet<i64> = trials[1..]
            .iter()
            .map(|t| match t.params.get("x").unwrap() {
                ParamValue::Int(v) => *v,
                other => panic!("expected an int, got {other:?}"),
            })
            .collect();
        assert_eq!(grid_values, HashSet::from([1, 2]));
    }

    #[test]
    fn test_grid_sampler_works_through_ask_with_fixed_distributions() {
        // Regression: `trial_created` used to run *after* the independent
        // sampling block, so this entry point never had a grid point assigned
        // and GridSampler was unusable through it.
        use crate::samplers::Sampler;
        use crate::study::{StudyDirection, create_study};
        use std::sync::Arc;

        let mut space = HashMap::new();
        space.insert("x".to_string(), vec![1.0, 2.0]);
        let sampler: Arc<dyn Sampler> = Arc::new(GridSampler::new(space, Some(42)));

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

        let mut fixed = HashMap::new();
        fixed.insert(
            "x".to_string(),
            Distribution::IntDistribution(IntDistribution::new(1, 2, false, 1).unwrap()),
        );

        let mut trial = study.ask(Some(&fixed)).unwrap();
        let x = trial.suggest_int("x", 1, 2, false, 1).unwrap();
        assert!([1, 2].contains(&x), "got {x}");
    }
}
