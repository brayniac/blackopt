use crate::pruners::PercentilePruner;
use crate::study::StudyDirection;

/// A pruner that prunes trials whose best intermediate value is worse than
/// the median of completed trials at the same step.
///
///
/// This is simply a `PercentilePruner` with `percentile = 50.0`.
pub struct MedianPruner {
    inner: PercentilePruner,
}

impl MedianPruner {
    /// Create a new `MedianPruner`.
    ///
    /// # Arguments
    /// * `n_startup_trials` - Disable pruning until this many trials complete. Default: 5.
    /// * `n_warmup_steps` - Disable pruning until step >= this. Default: 0.
    /// * `interval_steps` - Only check every N steps. Default: 1.
    /// * `n_min_trials` - Need at least this many values at step before pruning. Default: 1.
    /// * `direction` - Study direction.
    pub fn new(
        n_startup_trials: usize,
        n_warmup_steps: i64,
        interval_steps: i64,
        n_min_trials: usize,
        direction: StudyDirection,
    ) -> Self {
        Self {
            // MedianPruner always uses valid parameters, so unwrap is safe.
            inner: PercentilePruner::new(
                50.0,
                n_startup_trials,
                n_warmup_steps,
                interval_steps,
                n_min_trials,
                direction,
            )
            .expect("MedianPruner uses valid PercentilePruner parameters"),
        }
    }

    /// Create with default parameters (n_startup=5, n_warmup=0, interval=1, n_min=1).
    pub fn with_defaults(direction: StudyDirection) -> Self {
        Self::new(5, 0, 1, 1, direction)
    }
}

impl crate::pruners::Pruner for MedianPruner {
    fn prune(
        &self,
        study_trials: &[crate::trial::FrozenTrial],
        trial: &crate::trial::FrozenTrial,
    ) -> crate::error::Result<bool> {
        self.inner.prune(study_trials, trial)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pruners::Pruner;
    use crate::trial::{FrozenTrial, TrialState};
    use std::collections::HashMap;

    fn make_trial(number: i64, state: TrialState, iv: Vec<(i64, f64)>) -> FrozenTrial {
        let now = chrono::Utc::now();
        FrozenTrial {
            number,
            state,
            values: if state == TrialState::Complete {
                Some(vec![0.0])
            } else {
                None
            },
            datetime_start: Some(now),
            datetime_complete: if state.is_finished() {
                Some(now)
            } else {
                None
            },
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: iv.into_iter().collect(),
            trial_id: number,
        }
    }

    #[test]
    fn test_median_pruner_prunes_correctly() {
        let pruner = MedianPruner::new(0, 0, 1, 1, StudyDirection::Minimize);
        let completed = vec![
            make_trial(0, TrialState::Complete, vec![(0, 1.0)]),
            make_trial(1, TrialState::Complete, vec![(0, 2.0)]),
            make_trial(2, TrialState::Complete, vec![(0, 3.0)]),
        ];
        // Trial worse than median (2.0): should prune
        let bad = make_trial(3, TrialState::Running, vec![(0, 10.0)]);
        assert!(pruner.prune(&completed, &bad).unwrap());

        // Trial better than median: should keep
        let good = make_trial(3, TrialState::Running, vec![(0, 0.5)]);
        assert!(!pruner.prune(&completed, &good).unwrap());
    }

    #[test]
    fn test_median_pruner_respects_startup() {
        let pruner = MedianPruner::with_defaults(StudyDirection::Minimize);
        // with_defaults has n_startup_trials=5
        let completed = vec![
            make_trial(0, TrialState::Complete, vec![(0, 1.0)]),
            make_trial(1, TrialState::Complete, vec![(0, 2.0)]),
        ];
        let trial = make_trial(2, TrialState::Running, vec![(0, 100.0)]);
        // Only 2 completed trials < 5 startup, so no pruning
        assert!(!pruner.prune(&completed, &trial).unwrap());
    }
}
