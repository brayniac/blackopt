//! Parameter importance analysis.
//!
//! Provides tools for evaluating which hyperparameters have the most
//! impact on objective values. This is useful for understanding search
//! spaces and pruning unimportant parameters.

use indexmap::IndexMap;

use crate::distributions::{CategoricalChoice, Distribution, ParamValue};
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

/// Between-group-variance importance evaluator.
///
/// Estimates parameter importance by grouping trials into equally-spaced
/// bins of the (discretized) parameter value and computing the variance of
/// the bin means around the global mean. Parameters whose values separate
/// the objective into distinct groups are considered more important.
///
/// Categorical parameters are grouped by choice index.
///
/// This is a fast, model-free approximation of parameter importance; it is
/// not a true functional ANOVA decomposition.
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
            // Determine this param's categorical choices (if any) from the
            // first trial that records a distribution for it. This gives a
            // stable, deterministic numeric code for categorical values
            // (the choice index) instead of a run-dependent hash.
            let cat_choices: Option<Vec<CategoricalChoice>> = trials
                .iter()
                .find(|t| t.params.contains_key(param_name))
                .and_then(|t| t.distributions.get(param_name))
                .and_then(|d| match d {
                    Distribution::CategoricalDistribution(cd) => Some(cd.choices.clone()),
                    _ => None,
                });

            // Collect (param_value, objective_value) pairs
            let mut pairs: Vec<(f64, f64)> = Vec::new();
            for (i, trial) in trials.iter().enumerate() {
                if let Some(pv) = trial.params.get(param_name) {
                    let internal = param_value_to_f64(pv, &cat_choices);
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
///
/// For categorical values, the stable index of the choice within
/// `cat_choices` is used. If the choice is not found (should not happen when
/// `cat_choices` comes from the trial's own distribution), 0.0 is returned.
fn param_value_to_f64(pv: &ParamValue, cat_choices: &Option<Vec<CategoricalChoice>>) -> f64 {
    match pv {
        ParamValue::Float(v) => *v,
        ParamValue::Int(v) => *v as f64,
        ParamValue::Categorical(c) => match cat_choices {
            Some(choices) => choices.iter().position(|x| x == c).unwrap_or(0) as f64,
            None => 0.0,
        },
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
    fn test_param_value_to_f64_categorical_is_deterministic() {
        use crate::distributions::{CategoricalChoice, CategoricalDistribution};
        let choices = vec![
            CategoricalChoice::Str("a".into()),
            CategoricalChoice::Str("b".into()),
            CategoricalChoice::Str("c".into()),
        ];
        let dist = CategoricalDistribution::new(choices.clone()).unwrap();
        let cat = Some(dist.choices.clone());

        // Same input must always give the same code (no run-to-run jitter).
        let a1 = param_value_to_f64(
            &ParamValue::Categorical(CategoricalChoice::Str("a".into())),
            &cat,
        );
        let a2 = param_value_to_f64(
            &ParamValue::Categorical(CategoricalChoice::Str("a".into())),
            &cat,
        );
        assert_eq!(a1, a2);
        assert_eq!(a1, 0.0);

        let b = param_value_to_f64(
            &ParamValue::Categorical(CategoricalChoice::Str("b".into())),
            &cat,
        );
        assert_eq!(b, 1.0);

        // Distinct choices get distinct, contiguous codes.
        let c = param_value_to_f64(
            &ParamValue::Categorical(CategoricalChoice::Str("c".into())),
            &cat,
        );
        assert_eq!(c, 2.0);
        assert!(a1 < b && b < c);
    }

    #[test]
    fn test_importance_categorical_param() {
        // The categorical param drives the objective; a continuous param is
        // noise. Importance should rank the categorical param first and be
        // deterministic across repeated evaluations.
        let sampler: Arc<dyn crate::samplers::Sampler> = Arc::new(RandomSampler::new(Some(7)));
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
                    let mode = trial.suggest_categorical(
                        "mode",
                        vec![
                            CategoricalChoice::Str("lo".to_string()),
                            CategoricalChoice::Str("mid".to_string()),
                            CategoricalChoice::Str("hi".to_string()),
                        ],
                    )?;
                    let noise = trial.suggest_float("noise", 0.0, 1.0, false, None)?;
                    let base = match &mode {
                        CategoricalChoice::Str(s) if s == "lo" => 0.0,
                        CategoricalChoice::Str(s) if s == "mid" => 5.0,
                        _ => 20.0,
                    };
                    Ok(base + noise)
                },
                Some(120),
                None,
                None,
            )
            .unwrap();

        let a = get_param_importances(&study, None, None).unwrap();
        let b = get_param_importances(&study, None, None).unwrap();

        // Deterministic: two evaluations agree exactly.
        for k in a.keys() {
            assert!(
                (a[k] - b[k]).abs() < 1e-15,
                "importance of {k} must be deterministic"
            );
        }

        // The categorical param should dominate the noise param.
        assert!(
            a["mode"] > a["noise"],
            "mode ({}) should be more important than noise ({})",
            a["mode"],
            a["noise"]
        );
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
