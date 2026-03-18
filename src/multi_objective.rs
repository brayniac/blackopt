use crate::study::StudyDirection;
use crate::trial::{FrozenTrial, TrialState};

/// Returns true if solution `a` Pareto-dominates solution `b`.
///
/// A dominates B when A is at least as good in all objectives and strictly
/// better in at least one.
pub fn dominates(a: &[f64], b: &[f64], directions: &[StudyDirection]) -> bool {
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(a.len(), directions.len());

    let mut dominated_in_any = false;
    for i in 0..a.len() {
        let cmp = match directions[i] {
            StudyDirection::Minimize => a[i].partial_cmp(&b[i]),
            StudyDirection::Maximize => b[i].partial_cmp(&a[i]),
            StudyDirection::NotSet => return false,
        };
        match cmp {
            Some(std::cmp::Ordering::Greater) => return false, // a is worse in this objective
            Some(std::cmp::Ordering::Less) => dominated_in_any = true,
            _ => {}
        }
    }
    dominated_in_any
}

/// Fast non-dominated sorting (NSGA-II style).
///
/// Returns rank-ordered fronts as vectors of indices into the input slice.
/// Front 0 is the Pareto front, front 1 is the second-best, etc.
pub fn fast_non_dominated_sort(
    trials: &[&FrozenTrial],
    directions: &[StudyDirection],
) -> Vec<Vec<usize>> {
    let n = trials.len();
    if n == 0 {
        return vec![];
    }

    let values: Vec<&[f64]> = trials
        .iter()
        .map(|t| t.values.as_deref().unwrap_or(&[]))
        .collect();

    // domination_count[i] = how many solutions dominate i
    let mut domination_count = vec![0usize; n];
    // dominated_set[i] = set of solutions that i dominates
    let mut dominated_set: Vec<Vec<usize>> = vec![vec![]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            if dominates(values[i], values[j], directions) {
                dominated_set[i].push(j);
                domination_count[j] += 1;
            } else if dominates(values[j], values[i], directions) {
                dominated_set[j].push(i);
                domination_count[i] += 1;
            }
        }
    }

    let mut fronts = Vec::new();
    let mut current_front: Vec<usize> = (0..n).filter(|&i| domination_count[i] == 0).collect();

    while !current_front.is_empty() {
        let mut next_front = Vec::new();
        for &i in &current_front {
            for &j in &dominated_set[i] {
                domination_count[j] -= 1;
                if domination_count[j] == 0 {
                    next_front.push(j);
                }
            }
        }
        fronts.push(current_front);
        current_front = next_front;
    }

    fronts
}

/// Compute crowding distance for each trial.
///
/// Returns a vector of crowding distances, one per input trial.
/// Boundary solutions get `f64::INFINITY`.
pub fn crowding_distance(trials: &[&FrozenTrial], directions: &[StudyDirection]) -> Vec<f64> {
    let n = trials.len();
    if n == 0 {
        return vec![];
    }
    if n <= 2 {
        return vec![f64::INFINITY; n];
    }

    let n_objectives = directions.len();
    let mut distances = vec![0.0_f64; n];

    for m in 0..n_objectives {
        // Sort indices by objective m
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            let va = trials[a].values.as_ref().map(|v| v[m]).unwrap_or(f64::NAN);
            let vb = trials[b].values.as_ref().map(|v| v[m]).unwrap_or(f64::NAN);
            va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Boundary solutions get infinity
        distances[indices[0]] = f64::INFINITY;
        distances[indices[n - 1]] = f64::INFINITY;

        let f_min = trials[indices[0]]
            .values
            .as_ref()
            .map(|v| v[m])
            .unwrap_or(0.0);
        let f_max = trials[indices[n - 1]]
            .values
            .as_ref()
            .map(|v| v[m])
            .unwrap_or(0.0);
        let range = f_max - f_min;

        if range < f64::EPSILON {
            continue;
        }

        for i in 1..(n - 1) {
            let prev_val = trials[indices[i - 1]]
                .values
                .as_ref()
                .map(|v| v[m])
                .unwrap_or(0.0);
            let next_val = trials[indices[i + 1]]
                .values
                .as_ref()
                .map(|v| v[m])
                .unwrap_or(0.0);
            distances[indices[i]] += (next_val - prev_val) / range;
        }
    }

    distances
}

/// Mark which trials are on the Pareto front.
pub fn is_pareto_front(trials: &[&FrozenTrial], directions: &[StudyDirection]) -> Vec<bool> {
    let n = trials.len();
    if n == 0 {
        return vec![];
    }

    let fronts = fast_non_dominated_sort(trials, directions);
    let mut result = vec![false; n];
    if let Some(front_0) = fronts.first() {
        for &i in front_0 {
            result[i] = true;
        }
    }
    result
}

/// Compute 2D hypervolume indicator.
///
/// `points` are objective value pairs; `reference` is the reference point.
/// Points that are dominated by the reference point contribute positive volume.
pub fn hypervolume_2d(points: &[[f64; 2]], reference: [f64; 2]) -> f64 {
    if points.is_empty() {
        return 0.0;
    }

    // Filter to points that are not dominated by the reference point
    let mut pts: Vec<[f64; 2]> = points
        .iter()
        .filter(|p| p[0] < reference[0] && p[1] < reference[1])
        .copied()
        .collect();

    if pts.is_empty() {
        return 0.0;
    }

    // Sort by first objective ascending
    pts.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap_or(std::cmp::Ordering::Equal));

    let mut volume = 0.0;
    let mut y_bound = reference[1];

    for p in &pts {
        if p[1] < y_bound {
            volume += (reference[0] - p[0]) * (y_bound - p[1]);
            y_bound = p[1];
        }
    }

    volume
}

/// Get the Pareto-optimal trials from a list of complete trials.
pub fn get_pareto_front_trials(
    trials: &[FrozenTrial],
    directions: &[StudyDirection],
) -> Vec<FrozenTrial> {
    let complete: Vec<&FrozenTrial> = trials
        .iter()
        .filter(|t| t.state == TrialState::Complete && t.values.is_some())
        .collect();

    if complete.is_empty() {
        return vec![];
    }

    let on_front = is_pareto_front(&complete, directions);
    complete
        .into_iter()
        .zip(on_front)
        .filter(|(_, is_front)| *is_front)
        .map(|(t, _)| t.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trial::TrialState;
    use std::collections::HashMap;

    fn make_trial(number: i64, values: Vec<f64>) -> FrozenTrial {
        let now = chrono::Utc::now();
        FrozenTrial {
            number,
            state: TrialState::Complete,
            values: Some(values),
            datetime_start: Some(now),
            datetime_complete: Some(now),
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
            trial_id: number,
        }
    }

    #[test]
    fn test_dominates_minimize() {
        let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
        assert!(dominates(&[1.0, 1.0], &[2.0, 2.0], &dirs));
        assert!(!dominates(&[2.0, 2.0], &[1.0, 1.0], &dirs));
        // Not dominating if equal
        assert!(!dominates(&[1.0, 1.0], &[1.0, 1.0], &dirs));
        // Not dominating if better in one but worse in other
        assert!(!dominates(&[1.0, 3.0], &[2.0, 2.0], &dirs));
    }

    #[test]
    fn test_dominates_maximize() {
        let dirs = vec![StudyDirection::Maximize, StudyDirection::Maximize];
        assert!(dominates(&[2.0, 2.0], &[1.0, 1.0], &dirs));
        assert!(!dominates(&[1.0, 1.0], &[2.0, 2.0], &dirs));
    }

    #[test]
    fn test_dominates_mixed() {
        let dirs = vec![StudyDirection::Minimize, StudyDirection::Maximize];
        // a=[1, 3] vs b=[2, 2]: a is better (lower) in obj0, better (higher) in obj1
        assert!(dominates(&[1.0, 3.0], &[2.0, 2.0], &dirs));
        assert!(!dominates(&[2.0, 2.0], &[1.0, 3.0], &dirs));
    }

    #[test]
    fn test_fast_non_dominated_sort_simple() {
        let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
        let t0 = make_trial(0, vec![1.0, 4.0]);
        let t1 = make_trial(1, vec![2.0, 3.0]);
        let t2 = make_trial(2, vec![3.0, 2.0]);
        let t3 = make_trial(3, vec![4.0, 4.0]); // dominated by all front-0

        let trials: Vec<&FrozenTrial> = vec![&t0, &t1, &t2, &t3];
        let fronts = fast_non_dominated_sort(&trials, &dirs);

        assert_eq!(fronts.len(), 2);
        // Front 0 should have t0, t1, t2
        let mut front0 = fronts[0].clone();
        front0.sort();
        assert_eq!(front0, vec![0, 1, 2]);
        // Front 1 should have t3
        assert_eq!(fronts[1], vec![3]);
    }

    #[test]
    fn test_fast_non_dominated_sort_empty() {
        let dirs = vec![StudyDirection::Minimize];
        let fronts = fast_non_dominated_sort(&[], &dirs);
        assert!(fronts.is_empty());
    }

    #[test]
    fn test_crowding_distance_basic() {
        let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
        let t0 = make_trial(0, vec![1.0, 4.0]);
        let t1 = make_trial(1, vec![2.0, 3.0]);
        let t2 = make_trial(2, vec![3.0, 2.0]);
        let t3 = make_trial(3, vec![4.0, 1.0]);

        let trials: Vec<&FrozenTrial> = vec![&t0, &t1, &t2, &t3];
        let dists = crowding_distance(&trials, &dirs);

        // Boundary solutions (min/max in each objective) get infinity
        assert!(dists[0].is_infinite());
        assert!(dists[3].is_infinite());
        // Interior solutions get finite positive distances
        assert!(dists[1] > 0.0 && dists[1].is_finite());
        assert!(dists[2] > 0.0 && dists[2].is_finite());
    }

    #[test]
    fn test_crowding_distance_two_points() {
        let dirs = vec![StudyDirection::Minimize];
        let t0 = make_trial(0, vec![1.0]);
        let t1 = make_trial(1, vec![2.0]);
        let trials: Vec<&FrozenTrial> = vec![&t0, &t1];
        let dists = crowding_distance(&trials, &dirs);
        assert!(dists[0].is_infinite());
        assert!(dists[1].is_infinite());
    }

    #[test]
    fn test_is_pareto_front() {
        let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
        let t0 = make_trial(0, vec![1.0, 4.0]);
        let t1 = make_trial(1, vec![2.0, 3.0]);
        let t2 = make_trial(2, vec![5.0, 5.0]); // dominated

        let trials: Vec<&FrozenTrial> = vec![&t0, &t1, &t2];
        let front = is_pareto_front(&trials, &dirs);
        assert_eq!(front, vec![true, true, false]);
    }

    #[test]
    fn test_hypervolume_2d_simple() {
        // Simple case: one point at (1,1), reference at (3,3)
        let vol = hypervolume_2d(&[[1.0, 1.0]], [3.0, 3.0]);
        assert!((vol - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_hypervolume_2d_two_points() {
        let vol = hypervolume_2d(&[[1.0, 3.0], [3.0, 1.0]], [4.0, 4.0]);
        // Area: (4-1)*(4-3) + (4-3)*(3-1) = 3 + 2 = 5
        assert!((vol - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_hypervolume_2d_empty() {
        assert_eq!(hypervolume_2d(&[], [1.0, 1.0]), 0.0);
    }

    #[test]
    fn test_hypervolume_2d_dominated_by_reference() {
        // Point is worse than reference
        let vol = hypervolume_2d(&[[5.0, 5.0]], [3.0, 3.0]);
        assert_eq!(vol, 0.0);
    }

    #[test]
    fn test_get_pareto_front_trials() {
        let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
        let t0 = make_trial(0, vec![1.0, 4.0]);
        let t1 = make_trial(1, vec![2.0, 3.0]);
        let t2 = make_trial(2, vec![5.0, 5.0]);
        let trials = vec![t0, t1, t2];

        let front = get_pareto_front_trials(&trials, &dirs);
        assert_eq!(front.len(), 2);
        let numbers: Vec<i64> = front.iter().map(|t| t.number).collect();
        assert!(numbers.contains(&0));
        assert!(numbers.contains(&1));
    }
}
