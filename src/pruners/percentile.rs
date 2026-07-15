use crate::error::{Error, Result};
use crate::pruners::Pruner;
use crate::study::StudyDirection;
use crate::trial::{FrozenTrial, TrialState};

/// A pruner that prunes trials whose best intermediate value is worse than
/// a given percentile of completed trials at the same step.
///
#[derive(Debug, Clone)]
pub struct PercentilePruner {
    percentile: f64,
    n_startup_trials: usize,
    n_warmup_steps: i64,
    interval_steps: i64,
    n_min_trials: usize,
    direction: StudyDirection,
}

impl PercentilePruner {
    /// Create a new `PercentilePruner`.
    ///
    /// # Arguments
    /// * `percentile` - Percentile threshold (0.0–100.0). E.g., 25.0 keeps top 25%.
    /// * `n_startup_trials` - Disable pruning until this many trials complete.
    /// * `n_warmup_steps` - Disable pruning until step >= this.
    /// * `interval_steps` - Only check every N steps (>= 1).
    /// * `n_min_trials` - Need at least this many values at a step before pruning.
    /// * `direction` - Study direction (Minimize or Maximize).
    pub fn new(
        percentile: f64,
        n_startup_trials: usize,
        n_warmup_steps: i64,
        interval_steps: i64,
        n_min_trials: usize,
        direction: StudyDirection,
    ) -> Result<Self> {
        if !(0.0..=100.0).contains(&percentile) {
            return Err(Error::ValueError(
                "percentile must be in [0, 100]".to_string(),
            ));
        }
        if interval_steps < 1 {
            return Err(Error::ValueError("interval_steps must be >= 1".to_string()));
        }
        if n_min_trials < 1 {
            return Err(Error::ValueError("n_min_trials must be >= 1".to_string()));
        }
        Ok(Self {
            percentile,
            n_startup_trials,
            n_warmup_steps,
            interval_steps,
            n_min_trials,
            direction,
        })
    }
}

impl Pruner for PercentilePruner {
    fn prune(&self, study_trials: &[FrozenTrial], trial: &FrozenTrial) -> Result<bool> {
        let completed: Vec<&FrozenTrial> = study_trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .collect();
        let n_trials = completed.len();

        if n_trials == 0 {
            return Ok(false);
        }
        if n_trials < self.n_startup_trials {
            return Ok(false);
        }

        let step = match trial.last_step() {
            Some(s) => s,
            None => return Ok(false),
        };

        if step < self.n_warmup_steps {
            return Ok(false);
        }

        if !is_first_in_interval_step(
            step,
            &trial.intermediate_values,
            self.n_warmup_steps,
            self.interval_steps,
        ) {
            return Ok(false);
        }

        let best = get_best_intermediate_result_over_steps(trial, self.direction);
        if best.is_nan() {
            return Ok(true); // all NaN → prune
        }

        let p = get_percentile_intermediate_result_over_trials(
            &completed,
            self.direction,
            step,
            self.percentile,
            self.n_min_trials,
        );
        if p.is_nan() {
            return Ok(false); // not enough data at step → keep
        }

        Ok(match self.direction {
            StudyDirection::Maximize => best < p,
            _ => best > p,
        })
    }
}

/// Get the best (min or max) intermediate value so far in a trial.
fn get_best_intermediate_result_over_steps(trial: &FrozenTrial, direction: StudyDirection) -> f64 {
    let values: Vec<f64> = trial
        .intermediate_values
        .values()
        .copied()
        .filter(|v| !v.is_nan())
        .collect();

    if values.is_empty() {
        return f64::NAN;
    }

    match direction {
        StudyDirection::Maximize => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        _ => values.iter().copied().fold(f64::INFINITY, f64::min),
    }
}

/// Get the percentile of intermediate values at a given step across completed trials.
fn get_percentile_intermediate_result_over_trials(
    completed_trials: &[&FrozenTrial],
    direction: StudyDirection,
    step: i64,
    percentile: f64,
    n_min_trials: usize,
) -> f64 {
    let mut values: Vec<f64> = completed_trials
        .iter()
        .filter_map(|t| t.intermediate_values.get(&step))
        .copied()
        .filter(|v| !v.is_nan())
        .collect();

    if values.len() < n_min_trials {
        return f64::NAN;
    }

    values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let effective_percentile = match direction {
        StudyDirection::Maximize => 100.0 - percentile,
        _ => percentile,
    };

    nan_percentile(&values, effective_percentile)
}

/// Compute the p-th percentile of a sorted slice (linear interpolation).
fn nan_percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }

    let n = sorted.len() as f64;
    let idx = p / 100.0 * (n - 1.0);
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;

    if hi >= sorted.len() {
        sorted[sorted.len() - 1]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

/// Check if this step is the first reported step at or after a pruning checkpoint.
fn is_first_in_interval_step(
    step: i64,
    intermediate_values: &std::collections::HashMap<i64, f64>,
    n_warmup_steps: i64,
    interval_steps: i64,
) -> bool {
    let nearest_lower_pruning_step =
        (step - n_warmup_steps) / interval_steps * interval_steps + n_warmup_steps;

    let second_last_step = intermediate_values
        .keys()
        .filter(|&&s| s < step)
        .max()
        .copied()
        .unwrap_or(-1);

    second_last_step < nearest_lower_pruning_step
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_complete_trial(number: i64, intermediate_values: Vec<(i64, f64)>) -> FrozenTrial {
        let now = chrono::Utc::now();
        FrozenTrial {
            number,
            state: TrialState::Complete,
            values: Some(vec![0.0]),
            datetime_start: Some(now),
            datetime_complete: Some(now),
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: intermediate_values.into_iter().collect(),
            trial_id: number,
        }
    }

    fn make_running_trial(number: i64, intermediate_values: Vec<(i64, f64)>) -> FrozenTrial {
        let now = chrono::Utc::now();
        FrozenTrial {
            number,
            state: TrialState::Running,
            values: None,
            datetime_start: Some(now),
            datetime_complete: None,
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: intermediate_values.into_iter().collect(),
            trial_id: number,
        }
    }

    #[test]
    fn test_no_pruning_before_startup() {
        let pruner = PercentilePruner::new(50.0, 5, 0, 1, 1, StudyDirection::Minimize).unwrap();
        let completed = vec![
            make_complete_trial(0, vec![(0, 1.0)]),
            make_complete_trial(1, vec![(0, 2.0)]),
        ];
        let trial = make_running_trial(2, vec![(0, 100.0)]);
        assert!(!pruner.prune(&completed, &trial).unwrap());
    }

    #[test]
    fn test_no_pruning_before_warmup() {
        let pruner = PercentilePruner::new(50.0, 0, 5, 1, 1, StudyDirection::Minimize).unwrap();
        let completed = vec![
            make_complete_trial(0, vec![(0, 1.0)]),
            make_complete_trial(1, vec![(0, 2.0)]),
        ];
        let trial = make_running_trial(2, vec![(2, 100.0)]);
        assert!(!pruner.prune(&completed, &trial).unwrap());
    }

    #[test]
    fn test_prune_minimize_bad_trial() {
        let pruner = PercentilePruner::new(50.0, 0, 0, 1, 1, StudyDirection::Minimize).unwrap();
        let completed = vec![
            make_complete_trial(0, vec![(0, 1.0)]),
            make_complete_trial(1, vec![(0, 2.0)]),
            make_complete_trial(2, vec![(0, 3.0)]),
        ];
        // Trial with very high value should be pruned (minimize)
        let trial = make_running_trial(3, vec![(0, 100.0)]);
        assert!(pruner.prune(&completed, &trial).unwrap());
    }

    #[test]
    fn test_keep_minimize_good_trial() {
        let pruner = PercentilePruner::new(50.0, 0, 0, 1, 1, StudyDirection::Minimize).unwrap();
        let completed = vec![
            make_complete_trial(0, vec![(0, 1.0)]),
            make_complete_trial(1, vec![(0, 2.0)]),
            make_complete_trial(2, vec![(0, 3.0)]),
        ];
        // Trial with low value should be kept (minimize)
        let trial = make_running_trial(3, vec![(0, 0.5)]);
        assert!(!pruner.prune(&completed, &trial).unwrap());
    }

    #[test]
    fn test_prune_maximize_bad_trial() {
        let pruner = PercentilePruner::new(50.0, 0, 0, 1, 1, StudyDirection::Maximize).unwrap();
        let completed = vec![
            make_complete_trial(0, vec![(0, 10.0)]),
            make_complete_trial(1, vec![(0, 20.0)]),
            make_complete_trial(2, vec![(0, 30.0)]),
        ];
        // Trial with low value should be pruned (maximize)
        let trial = make_running_trial(3, vec![(0, 1.0)]);
        assert!(pruner.prune(&completed, &trial).unwrap());
    }

    #[test]
    fn test_no_intermediate_values() {
        let pruner = PercentilePruner::new(50.0, 0, 0, 1, 1, StudyDirection::Minimize).unwrap();
        let completed = vec![make_complete_trial(0, vec![(0, 1.0)])];
        let trial = make_running_trial(1, vec![]);
        assert!(!pruner.prune(&completed, &trial).unwrap());
    }

    #[test]
    fn test_n_min_trials() {
        let pruner = PercentilePruner::new(50.0, 0, 0, 1, 5, StudyDirection::Minimize).unwrap();
        // Only 2 completed trials at step 0, but n_min_trials=5
        let completed = vec![
            make_complete_trial(0, vec![(0, 1.0)]),
            make_complete_trial(1, vec![(0, 2.0)]),
        ];
        let trial = make_running_trial(2, vec![(0, 100.0)]);
        assert!(!pruner.prune(&completed, &trial).unwrap());
    }

    #[test]
    fn test_interval_steps() {
        let pruner = PercentilePruner::new(50.0, 0, 0, 3, 1, StudyDirection::Minimize).unwrap();
        let completed = vec![make_complete_trial(
            0,
            vec![(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)],
        )];
        // Step 1 is not at an interval boundary (interval=3, warmup=0)
        // The first pruning steps are at 0, 3, 6, ...
        let trial_at_step_1 = make_running_trial(1, vec![(0, 100.0), (1, 100.0)]);
        assert!(!pruner.prune(&completed, &trial_at_step_1).unwrap());

        // Step 3 IS at an interval boundary
        let trial_at_step_3 =
            make_running_trial(1, vec![(0, 100.0), (1, 100.0), (2, 100.0), (3, 100.0)]);
        assert!(pruner.prune(&completed, &trial_at_step_3).unwrap());
    }

    #[test]
    fn test_nan_percentile_fn() {
        assert_eq!(nan_percentile(&[1.0, 2.0, 3.0, 4.0, 5.0], 50.0), 3.0);
        assert_eq!(nan_percentile(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.0), 1.0);
        assert_eq!(nan_percentile(&[1.0, 2.0, 3.0, 4.0, 5.0], 100.0), 5.0);
        assert_eq!(nan_percentile(&[10.0], 50.0), 10.0);
        assert!(nan_percentile(&[], 50.0).is_nan());
    }
}
