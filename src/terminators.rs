//! Early stopping terminators for optimization loops.
//!
//! Terminators provide conditions under which optimization should stop early,
//! beyond the basic `n_trials` and `timeout` parameters.

use crate::study::Study;
use crate::study::StudyDirection;
use crate::trial::TrialState;

/// A terminator decides whether optimization should stop early.
pub trait Terminator: Send + Sync {
    /// Returns `true` if optimization should stop.
    fn should_terminate(&self, study: &Study) -> bool;
}

/// Stops optimization after a fixed number of completed trials.
pub struct MaxTrialsTerminator {
    max_trials: usize,
}

impl MaxTrialsTerminator {
    /// Create a new terminator that stops after `max_trials` completed trials.
    pub fn new(max_trials: usize) -> Self {
        Self { max_trials }
    }
}

impl Terminator for MaxTrialsTerminator {
    fn should_terminate(&self, study: &Study) -> bool {
        let n = study
            .get_trials(Some(&[TrialState::Complete]))
            .map(|t| t.len())
            .unwrap_or(0);
        n >= self.max_trials
    }
}

/// Stops optimization when no improvement has been observed for `patience`
/// trials.
///
/// "Trials" here means *finished* trials (complete, failed, or pruned): a
/// run of failures or prunings counts as a run without improvement, so a
/// sampler that keeps failing will stop the loop instead of spinning forever.
///
/// Improvement itself is measured only over *completed* trials. A pruned
/// trial's recorded value is its last intermediate report — a partial-epoch
/// score, not an objective value — and letting those compete would both
/// reset patience on trials that never finished and let an early partial
/// score mask real progress.
pub struct NoImprovementTerminator {
    patience: usize,
}

impl NoImprovementTerminator {
    /// Create a new terminator that stops after `patience` consecutive trials
    /// without improvement.
    pub fn new(patience: usize) -> Self {
        Self { patience }
    }
}

impl Terminator for NoImprovementTerminator {
    fn should_terminate(&self, study: &Study) -> bool {
        // Count all finished trials: failures and prunings count against
        // patience as much as completed trials do.
        let Ok(trials) = study.get_trials(Some(&[
            TrialState::Complete,
            TrialState::Fail,
            TrialState::Pruned,
        ])) else {
            return false;
        };

        if trials.len() < self.patience + 1 {
            return false;
        }

        let Ok(direction) = study.direction() else {
            return false;
        };

        // Find the best value and when it was last achieved
        let mut best_value = match direction {
            StudyDirection::Minimize | StudyDirection::NotSet => f64::INFINITY,
            StudyDirection::Maximize => f64::NEG_INFINITY,
        };
        let mut best_idx = 0;

        for (i, trial) in trials.iter().enumerate() {
            // Only completed trials carry a real objective value.
            if trial.state != TrialState::Complete {
                continue;
            }
            if let Some(values) = &trial.values
                && !values.is_empty()
            {
                let v = values[0];
                let is_better = match direction {
                    StudyDirection::Minimize | StudyDirection::NotSet => v < best_value,
                    StudyDirection::Maximize => v > best_value,
                };
                if is_better {
                    best_value = v;
                    best_idx = i;
                }
            }
        }

        // Check if patience is exceeded
        trials.len() - best_idx > self.patience
    }
}

/// Stops optimization when the best value reaches or exceeds a target.
pub struct TargetValueTerminator {
    target: f64,
    direction: StudyDirection,
}

impl TargetValueTerminator {
    /// Create a new terminator that stops when the best value reaches `target`.
    ///
    /// For minimize: stops when best <= target.
    /// For maximize: stops when best >= target.
    pub fn new(target: f64, direction: StudyDirection) -> Self {
        Self { target, direction }
    }
}

impl Terminator for TargetValueTerminator {
    fn should_terminate(&self, study: &Study) -> bool {
        let Ok(best) = study.best_value() else {
            return false;
        };

        match self.direction {
            StudyDirection::Minimize | StudyDirection::NotSet => best <= self.target,
            StudyDirection::Maximize => best >= self.target,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::samplers::RandomSampler;
    use crate::study::create_study;
    use std::sync::Arc;

    #[test]
    fn test_max_trials_terminator() {
        let term = MaxTrialsTerminator::new(5);
        let sampler: Arc<dyn crate::samplers::Sampler> = Arc::new(RandomSampler::new(Some(42)));
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

        assert!(!term.should_terminate(&study));

        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    Ok(x)
                },
                Some(5),
                None,
                None,
            )
            .unwrap();

        assert!(term.should_terminate(&study));
    }

    #[test]
    fn test_no_improvement_terminator() {
        let term = NoImprovementTerminator::new(3);
        let sampler: Arc<dyn crate::samplers::Sampler> = Arc::new(RandomSampler::new(Some(42)));
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

        assert!(!term.should_terminate(&study));

        // Run a few trials — random optimization may or may not trigger patience
        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    Ok(x)
                },
                Some(20),
                None,
                None,
            )
            .unwrap();

        // With 20 random trials, the terminator should at least be callable
        let _ = term.should_terminate(&study);
    }

    #[test]
    fn test_no_improvement_terminator_counts_failures() {
        // A stream of failing trials must count against patience — otherwise
        // a perpetually-failing sampler would never stop the loop.
        let term = NoImprovementTerminator::new(3);
        let sampler: Arc<dyn crate::samplers::Sampler> = Arc::new(RandomSampler::new(Some(42)));
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
                    let _ = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    Err(crate::error::Error::ValueError("always fails".into()))
                },
                Some(10),
                None,
                None,
            )
            .unwrap();

        assert!(
            term.should_terminate(&study),
            "4+ failures with patience 3 should terminate"
        );
    }

    #[test]
    fn test_target_value_terminator_minimize() {
        let term = TargetValueTerminator::new(0.5, StudyDirection::Minimize);
        let sampler: Arc<dyn crate::samplers::Sampler> = Arc::new(RandomSampler::new(Some(42)));
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

        assert!(!term.should_terminate(&study));

        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    Ok(x * x) // min possible is ~0.0
                },
                Some(50),
                None,
                None,
            )
            .unwrap();

        // With 50 random trials in [0,1], we should find x^2 < 0.5
        assert!(
            term.should_terminate(&study),
            "should have found a value <= 0.5"
        );
    }

    #[test]
    fn test_target_value_terminator_maximize() {
        let term = TargetValueTerminator::new(0.8, StudyDirection::Maximize);
        let sampler: Arc<dyn crate::samplers::Sampler> = Arc::new(RandomSampler::new(Some(42)));
        let study = create_study(
            None,
            Some(sampler),
            None,
            None,
            Some(StudyDirection::Maximize),
            None,
            false,
        )
        .unwrap();

        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    Ok(x)
                },
                Some(50),
                None,
                None,
            )
            .unwrap();

        assert!(
            term.should_terminate(&study),
            "should have found a value >= 0.8"
        );
    }

    #[test]
    fn test_terminators_in_optimize() {
        let sampler: Arc<dyn crate::samplers::Sampler> = Arc::new(RandomSampler::new(Some(42)));
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

        let terminators: Vec<Arc<dyn Terminator>> = vec![Arc::new(MaxTrialsTerminator::new(10))];

        study
            .optimize_with_terminators(
                |trial| {
                    let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    Ok(x * x)
                },
                Some(1000),
                None,
                None,
                Some(&terminators),
            )
            .unwrap();

        let n = study.trials().unwrap().len();
        assert!(
            n <= 11, // allow a tiny overshoot since check is after trial
            "expected ~10 trials with MaxTrialsTerminator, got {n}"
        );
    }

    #[test]
    fn test_no_improvement_terminator_ignores_pruned_intermediate_values() {
        // A pruned trial's recorded value is its last intermediate report, not
        // an objective value. Letting those compete broke the terminator in
        // both directions; this pins the "never stops" direction.
        let term = NoImprovementTerminator::new(3);
        let sampler: Arc<dyn crate::samplers::Sampler> = Arc::new(RandomSampler::new(Some(42)));
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

        // Every completed trial scores exactly 100.0 — zero real improvement —
        // while pruned trials report an ever-better partial score.
        let i = std::sync::atomic::AtomicUsize::new(0);
        study
            .optimize(
                |trial| {
                    let _x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    let n = i.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if n % 2 == 1 {
                        trial.report(1.0 / (n as f64 + 1.0), 0)?;
                        Err(crate::error::Error::TrialPruned)
                    } else {
                        Ok(100.0)
                    }
                },
                Some(40),
                None,
                None,
            )
            .unwrap();

        assert!(
            term.should_terminate(&study),
            "improving partial scores must not reset patience"
        );
    }

    #[test]
    fn test_no_improvement_terminator_not_masked_by_early_prune() {
        // The other direction: an early pruned trial reporting a very good
        // partial score must not look like a best-so-far that never improves.
        let sampler: Arc<dyn crate::samplers::Sampler> = Arc::new(RandomSampler::new(Some(42)));
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

        let i = std::sync::atomic::AtomicUsize::new(0);
        let terminators: Vec<Arc<dyn Terminator>> = vec![Arc::new(NoImprovementTerminator::new(3))];
        study
            .optimize_with_terminators(
                |trial| {
                    let _x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    let n = i.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                    if n == 0 {
                        trial.report(0.0, 0)?;
                        Err(crate::error::Error::TrialPruned)
                    } else {
                        Ok(100.0 - n as f64)
                    }
                },
                Some(30),
                None,
                None,
                Some(&terminators),
            )
            .unwrap();

        assert_eq!(
            study.trials().unwrap().len(),
            30,
            "steadily improving completed trials must keep the study running"
        );
    }
}
