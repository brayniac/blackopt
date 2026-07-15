//! Parameter importance analysis.
//!
//! Provides tools for evaluating which hyperparameters have the most
//! impact on objective values. This is useful for understanding search
//! spaces and pruning unimportant parameters.

use indexmap::IndexMap;

use crate::distributions::ParamValue;
use crate::error::{Error, Result};
use crate::study::Study;
use crate::trial::{FrozenTrial, TrialState};

/// Evaluator for computing parameter importance scores.
pub trait ImportanceEvaluator: Send + Sync {
    /// Evaluate the importance of each parameter.
    ///
    /// Returns a map from parameter name to importance score (0.0 to 1.0),
    /// ordered by decreasing importance. Scores are normalized to sum to 1.0.
    fn evaluate(
        &self,
        trials: &[FrozenTrial],
        params: &[String],
        target_values: &[f64],
    ) -> Result<IndexMap<String, f64>>;
}

/// Functional ANOVA (fANOVA) importance evaluator.
///
/// Estimates parameter importance by computing between-group variance
/// of objective values when trials are grouped by discretized parameter
/// values. Parameters that produce large variance between groups are
/// considered more important.
pub struct FanovaEvaluator {
    /// Number of bins for discretizing continuous parameters.
    n_bins: usize,
}

impl Default for FanovaEvaluator {
    fn default() -> Self {
        Self { n_bins: 16 }
    }
}

impl FanovaEvaluator {
    /// Create a new evaluator with the given number of bins.
    pub fn new(n_bins: usize) -> Self {
        Self { n_bins }
    }
}

impl ImportanceEvaluator for FanovaEvaluator {
    fn evaluate(
        &self,
        trials: &[FrozenTrial],
        params: &[String],
        target_values: &[f64],
    ) -> Result<IndexMap<String, f64>> {
        if trials.is_empty() || params.is_empty() {
            return Ok(IndexMap::new());
        }

        let global_mean: f64 = target_values.iter().sum::<f64>() / target_values.len() as f64;

        let mut raw_importances: Vec<(String, f64)> = Vec::new();

        for param_name in params {
            // Collect (param_value, objective_value) pairs
            let mut pairs: Vec<(f64, f64)> = Vec::new();
            for (i, trial) in trials.iter().enumerate() {
                if let Some(pv) = trial.params.get(param_name) {
                    let internal = param_value_to_f64(pv);
                    pairs.push((internal, target_values[i]));
                }
            }

            if pairs.is_empty() {
                raw_importances.push((param_name.clone(), 0.0));
                continue;
            }

            // Discretize into bins and compute between-group variance
            let importance = between_group_variance(&pairs, self.n_bins, global_mean);
            raw_importances.push((param_name.clone(), importance));
        }

        // Normalize importances to sum to 1.0
        let total: f64 = raw_importances.iter().map(|(_, v)| *v).sum();
        let mut result = IndexMap::new();

        if total > 0.0 {
            // Sort by importance descending
            raw_importances
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (name, imp) in raw_importances {
                result.insert(name, imp / total);
            }
        } else {
            // All importances are zero — assign equal weight
            let uniform = 1.0 / params.len() as f64;
            for name in params {
                result.insert(name.clone(), uniform);
            }
        }

        Ok(result)
    }
}

/// Convert a ParamValue to f64 for importance computation.
fn param_value_to_f64(pv: &ParamValue) -> f64 {
    match pv {
        ParamValue::Float(v) => *v,
        ParamValue::Int(v) => *v as f64,
        ParamValue::Categorical(c) => {
            // Use a hash-based approach for categorical values
            use std::hash::{Hash, Hasher};
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            format!("{c:?}").hash(&mut hasher);
            (hasher.finish() % 1000) as f64
        }
    }
}

/// Compute between-group variance for a set of (param_value, objective_value) pairs.
///
/// Groups values into `n_bins` equally-spaced bins based on param_value,
/// then computes weighted variance of group means around the global mean.
fn between_group_variance(pairs: &[(f64, f64)], n_bins: usize, global_mean: f64) -> f64 {
    if pairs.len() <= 1 {
        return 0.0;
    }

    let min_val = pairs.iter().map(|(v, _)| *v).fold(f64::INFINITY, f64::min);
    let max_val = pairs
        .iter()
        .map(|(v, _)| *v)
        .fold(f64::NEG_INFINITY, f64::max);

    // If all param values are the same, this parameter has no importance
    let range = max_val - min_val;
    if range < 1e-14 {
        return 0.0;
    }

    // Group into bins
    let mut bin_sums = vec![0.0_f64; n_bins];
    let mut bin_counts = vec![0_usize; n_bins];

    for &(param_val, obj_val) in pairs {
        let bin = ((param_val - min_val) / range * (n_bins as f64 - 1.0)).round() as usize;
        let bin = bin.min(n_bins - 1);
        bin_sums[bin] += obj_val;
        bin_counts[bin] += 1;
    }

    // Compute between-group variance: sum of n_k * (mean_k - global_mean)^2
    let n_total = pairs.len() as f64;
    let mut variance = 0.0;
    for k in 0..n_bins {
        if bin_counts[k] > 0 {
            let group_mean = bin_sums[k] / bin_counts[k] as f64;
            let diff = group_mean - global_mean;
            variance += (bin_counts[k] as f64 / n_total) * diff * diff;
        }
    }

    variance
}

/// Compute parameter importances for a study.
///
/// Returns a map from parameter name to importance score, ordered by
/// decreasing importance. Scores are normalized to sum to 1.0.
///
/// # Arguments
///
/// * `study` - The study to analyze.
/// * `evaluator` - The importance evaluator to use. Defaults to [`FanovaEvaluator`].
/// * `params` - Optional subset of parameter names to evaluate. If `None`,
///   all parameters from completed trials are used.
pub fn get_param_importances(
    study: &Study,
    evaluator: Option<&dyn ImportanceEvaluator>,
    params: Option<&[&str]>,
) -> Result<IndexMap<String, f64>> {
    let default_evaluator = FanovaEvaluator::default();
    let evaluator = evaluator.unwrap_or(&default_evaluator);

    let trials: Vec<FrozenTrial> = study
        .get_trials(Some(&[TrialState::Complete]))?
        .into_iter()
        .filter(|t| t.values.is_some())
        .collect();

    if trials.is_empty() {
        return Err(Error::ValueError("study has no completed trials".into()));
    }

    // Collect all parameter names from completed trials
    let param_names: Vec<String> = if let Some(names) = params {
        names.iter().map(|s| s.to_string()).collect()
    } else {
        let mut all_params: Vec<String> = trials
            .iter()
            .flat_map(|t| t.params.keys().cloned())
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        all_params.sort();
        all_params
    };

    if param_names.is_empty() {
        return Ok(IndexMap::new());
    }

    // Extract target values (first objective for single-objective)
    let target_values: Vec<f64> = trials
        .iter()
        .map(|t| t.values.as_ref().unwrap()[0])
        .collect();

    evaluator.evaluate(&trials, &param_names, &target_values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::samplers::RandomSampler;
    use crate::study::{StudyDirection, create_study};
    use std::sync::Arc;

    #[test]
    fn test_fanova_evaluator_basic() {
        let evaluator = FanovaEvaluator::default();
        assert_eq!(evaluator.n_bins, 16);
    }

    #[test]
    fn test_fanova_evaluator_custom_bins() {
        let evaluator = FanovaEvaluator::new(8);
        assert_eq!(evaluator.n_bins, 8);
    }

    #[test]
    fn test_fanova_empty_trials() {
        let evaluator = FanovaEvaluator::default();
        let result = evaluator.evaluate(&[], &[], &[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_get_param_importances_quadratic() {
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

        // f(x, y) = x^2 + 0.01*y: x is much more important than y
        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", -10.0, 10.0, false, None)?;
                    let y = trial.suggest_float("y", -10.0, 10.0, false, None)?;
                    Ok(x * x + 0.01 * y)
                },
                Some(100),
                None,
                None,
            )
            .unwrap();

        let importances = get_param_importances(&study, None, None).unwrap();
        assert_eq!(importances.len(), 2);

        // Importances should sum to ~1.0
        let total: f64 = importances.values().sum();
        assert!(
            (total - 1.0).abs() < 1e-10,
            "importances should sum to 1.0, got {total}"
        );

        // x should be more important than y
        let x_imp = importances["x"];
        let y_imp = importances["y"];
        assert!(
            x_imp > y_imp,
            "x importance ({x_imp}) should be > y importance ({y_imp})"
        );
    }

    #[test]
    fn test_get_param_importances_with_subset() {
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
                    let x = trial.suggest_float("x", -10.0, 10.0, false, None)?;
                    let _y = trial.suggest_float("y", -10.0, 10.0, false, None)?;
                    Ok(x * x)
                },
                Some(50),
                None,
                None,
            )
            .unwrap();

        // Only evaluate importance for "x"
        let importances = get_param_importances(&study, None, Some(&["x"])).unwrap();
        assert_eq!(importances.len(), 1);
        assert!(importances.contains_key("x"));
        assert!((importances["x"] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_get_param_importances_no_completed_trials() {
        let study = create_study(None, None, None, None, None, None, false).unwrap();
        let result = get_param_importances(&study, None, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_between_group_variance_identical() {
        // All same param value => zero importance
        let pairs = vec![(1.0, 2.0), (1.0, 3.0), (1.0, 4.0)];
        let v = between_group_variance(&pairs, 8, 3.0);
        assert!((v - 0.0).abs() < 1e-14);
    }

    #[test]
    fn test_between_group_variance_distinct() {
        // Two clearly separated groups
        let mut pairs = Vec::new();
        for _ in 0..10 {
            pairs.push((0.0, 1.0)); // group 1: low param, low obj
        }
        for _ in 0..10 {
            pairs.push((10.0, 100.0)); // group 2: high param, high obj
        }
        let global_mean = 50.5;
        let v = between_group_variance(&pairs, 8, global_mean);
        assert!(v > 0.0, "variance should be positive for distinct groups");
    }

    #[test]
    fn test_importance_three_params() {
        let sampler: Arc<dyn crate::samplers::Sampler> = Arc::new(RandomSampler::new(Some(123)));
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

        // f(x, y, z) = 10*x^2 + y^2 + 0.001*z
        // Importance: x >> y >> z
        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                    let y = trial.suggest_float("y", -5.0, 5.0, false, None)?;
                    let z = trial.suggest_float("z", -5.0, 5.0, false, None)?;
                    Ok(10.0 * x * x + y * y + 0.001 * z)
                },
                Some(200),
                None,
                None,
            )
            .unwrap();

        let importances = get_param_importances(&study, None, None).unwrap();
        assert_eq!(importances.len(), 3);

        // First key should be x (most important)
        let first_key = importances.keys().next().unwrap();
        assert_eq!(first_key, "x", "x should be most important");
    }
}
