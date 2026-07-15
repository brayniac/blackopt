use std::collections::HashMap;

use crate::distributions::Distribution;
use crate::trial::{FrozenTrial, TrialState};

/// Incrementally computes the intersection of trial distributions.
///
///
/// The intersection is the set of (param_name, distribution) pairs that
/// are identical across all completed (and optionally pruned) trials.
#[derive(Debug, Clone)]
pub struct IntersectionSearchSpace {
    cached_trial_number: i64,
    search_space: Option<HashMap<String, Distribution>>,
    include_pruned: bool,
}

impl IntersectionSearchSpace {
    pub fn new(include_pruned: bool) -> Self {
        Self {
            cached_trial_number: -1,
            search_space: None,
            include_pruned,
        }
    }

    /// Calculate the intersection search space from a study's trials.
    ///
    /// Returns a sorted map of param name → distribution for parameters
    /// that have identical distributions across all relevant trials.
    pub fn calculate(&mut self, trials: &[FrozenTrial]) -> HashMap<String, Distribution> {
        let (space, cached) = calculate_inner(
            trials,
            self.include_pruned,
            self.search_space.take(),
            self.cached_trial_number,
        );
        self.search_space = space;
        self.cached_trial_number = cached;

        let result = self.search_space.clone().unwrap_or_default();
        // Return sorted by key
        let mut sorted: Vec<_> = result.into_iter().collect();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));
        sorted.into_iter().collect()
    }
}

/// States of interest for intersection calculation.
fn is_state_of_interest(state: TrialState, include_pruned: bool) -> bool {
    matches!(
        state,
        TrialState::Complete | TrialState::Waiting | TrialState::Running
    ) || (include_pruned && state == TrialState::Pruned)
}

/// Core intersection calculation.
///
/// Iterates trials in reverse order (newest first) for cache efficiency.
fn calculate_inner(
    trials: &[FrozenTrial],
    include_pruned: bool,
    mut search_space: Option<HashMap<String, Distribution>>,
    cached_trial_number: i64,
) -> (Option<HashMap<String, Distribution>>, i64) {
    let mut next_cached_trial_number = cached_trial_number;

    // Iterate in reverse (newest first)
    for trial in trials.iter().rev() {
        if !is_state_of_interest(trial.state, include_pruned) {
            continue;
        }

        // First valid trial sets the next cache point
        if next_cached_trial_number == cached_trial_number {
            next_cached_trial_number = trial.number + 1;
        }

        // Already processed this trial
        if cached_trial_number > trial.number {
            break;
        }

        // Non-finished trials: don't use for intersection, but don't
        // advance cache past them (they need processing when they finish)
        if !trial.state.is_finished() {
            next_cached_trial_number = trial.number;
            continue;
        }

        // Skip Pruned if not included
        if trial.state == TrialState::Pruned && !include_pruned {
            continue;
        }

        match &mut search_space {
            None => {
                // First finished trial: initialize with its distributions
                search_space = Some(trial.distributions.clone());
            }
            Some(space) => {
                // Intersect: keep only params with matching distributions
                space.retain(|name, dist| trial.distributions.get(name).is_some_and(|d| d == dist));
            }
        }
    }

    (search_space, next_cached_trial_number)
}

/// Stateless convenience function (no caching).
pub fn intersection_search_space(
    trials: &[FrozenTrial],
    include_pruned: bool,
) -> HashMap<String, Distribution> {
    let (space, _) = calculate_inner(trials, include_pruned, None, -1);
    let result = space.unwrap_or_default();
    let mut sorted: Vec<_> = result.into_iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    sorted.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::{FloatDistribution, IntDistribution, ParamValue};
    use chrono::Utc;

    fn make_trial(
        number: i64,
        state: TrialState,
        params: Vec<(&str, ParamValue, Distribution)>,
    ) -> FrozenTrial {
        let now = Utc::now();
        let mut p = HashMap::new();
        let mut d = HashMap::new();
        for (name, val, dist) in params {
            p.insert(name.to_string(), val);
            d.insert(name.to_string(), dist);
        }

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
            params: p,
            distributions: d,
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
            trial_id: number,
        }
    }

    fn float_dist() -> Distribution {
        Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap())
    }

    fn int_dist() -> Distribution {
        Distribution::IntDistribution(IntDistribution::new(1, 10, false, 1).unwrap())
    }

    #[test]
    fn test_empty_trials() {
        let result = intersection_search_space(&[], false);
        assert!(result.is_empty());
    }

    #[test]
    fn test_single_trial() {
        let trials = vec![make_trial(
            0,
            TrialState::Complete,
            vec![("x", ParamValue::Float(0.5), float_dist())],
        )];
        let result = intersection_search_space(&trials, false);
        assert_eq!(result.len(), 1);
        assert!(result.contains_key("x"));
    }

    #[test]
    fn test_intersection_same_params() {
        let trials = vec![
            make_trial(
                0,
                TrialState::Complete,
                vec![
                    ("x", ParamValue::Float(0.5), float_dist()),
                    ("n", ParamValue::Int(5), int_dist()),
                ],
            ),
            make_trial(
                1,
                TrialState::Complete,
                vec![
                    ("x", ParamValue::Float(0.3), float_dist()),
                    ("n", ParamValue::Int(3), int_dist()),
                ],
            ),
        ];
        let result = intersection_search_space(&trials, false);
        assert_eq!(result.len(), 2);
        assert!(result.contains_key("x"));
        assert!(result.contains_key("n"));
    }

    #[test]
    fn test_intersection_different_params() {
        let trials = vec![
            make_trial(
                0,
                TrialState::Complete,
                vec![("x", ParamValue::Float(0.5), float_dist())],
            ),
            make_trial(
                1,
                TrialState::Complete,
                vec![("y", ParamValue::Float(0.3), float_dist())],
            ),
        ];
        let result = intersection_search_space(&trials, false);
        assert!(result.is_empty());
    }

    #[test]
    fn test_intersection_different_distributions() {
        let dist2 = Distribution::FloatDistribution(
            FloatDistribution::new(0.0, 10.0, false, None).unwrap(),
        );
        let trials = vec![
            make_trial(
                0,
                TrialState::Complete,
                vec![("x", ParamValue::Float(0.5), float_dist())],
            ),
            make_trial(
                1,
                TrialState::Complete,
                vec![("x", ParamValue::Float(0.3), dist2)],
            ),
        ];
        let result = intersection_search_space(&trials, false);
        assert!(result.is_empty());
    }

    #[test]
    fn test_incremental_caching() {
        let mut iss = IntersectionSearchSpace::new(false);

        let trials = vec![make_trial(
            0,
            TrialState::Complete,
            vec![("x", ParamValue::Float(0.5), float_dist())],
        )];
        let r1 = iss.calculate(&trials);
        assert_eq!(r1.len(), 1);

        // Add another trial with same param
        let trials2 = vec![
            make_trial(
                0,
                TrialState::Complete,
                vec![("x", ParamValue::Float(0.5), float_dist())],
            ),
            make_trial(
                1,
                TrialState::Complete,
                vec![("x", ParamValue::Float(0.3), float_dist())],
            ),
        ];
        let r2 = iss.calculate(&trials2);
        assert_eq!(r2.len(), 1);
    }

    #[test]
    fn test_running_trials_dont_affect_intersection() {
        let trials = vec![
            make_trial(
                0,
                TrialState::Complete,
                vec![("x", ParamValue::Float(0.5), float_dist())],
            ),
            make_trial(1, TrialState::Running, vec![]),
        ];
        let result = intersection_search_space(&trials, false);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_include_pruned() {
        let trials = vec![
            make_trial(
                0,
                TrialState::Complete,
                vec![
                    ("x", ParamValue::Float(0.5), float_dist()),
                    ("y", ParamValue::Float(0.5), float_dist()),
                ],
            ),
            make_trial(
                1,
                TrialState::Pruned,
                vec![("x", ParamValue::Float(0.3), float_dist())],
            ),
        ];

        // Without pruned: only complete trials
        let r1 = intersection_search_space(&trials, false);
        assert_eq!(r1.len(), 2);

        // With pruned: intersection reduces
        let r2 = intersection_search_space(&trials, true);
        assert_eq!(r2.len(), 1);
        assert!(r2.contains_key("x"));
    }
}
