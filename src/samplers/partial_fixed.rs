use std::collections::HashMap;
use std::sync::Arc;

use crate::distributions::{Distribution, ParamValue};
use crate::error::Result;
use crate::samplers::Sampler;
use crate::trial::{FrozenTrial, TrialState};

/// A sampler that fixes some parameters to constant values and delegates
/// the rest to a base sampler.
pub struct PartialFixedSampler {
    fixed_params: HashMap<String, f64>,
    base_sampler: Arc<dyn Sampler>,
}

impl std::fmt::Debug for PartialFixedSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PartialFixedSampler")
            .field("n_fixed", &self.fixed_params.len())
            .finish()
    }
}

impl PartialFixedSampler {
    /// Create a new `PartialFixedSampler`.
    ///
    /// # Arguments
    /// * `fixed_params` - Map from param name to fixed internal value.
    /// * `base_sampler` - The sampler to use for non-fixed parameters.
    pub fn new(fixed_params: HashMap<String, f64>, base_sampler: Arc<dyn Sampler>) -> Self {
        Self {
            fixed_params,
            base_sampler,
        }
    }

    /// Create from external ParamValue representations.
    ///
    /// Converts each `ParamValue` to its internal `f64` representation using
    /// the provided distributions.
    pub fn from_param_values(
        fixed_params: HashMap<String, ParamValue>,
        distributions: &HashMap<String, Distribution>,
        base_sampler: Arc<dyn Sampler>,
    ) -> Result<Self> {
        let mut internal_params = HashMap::new();
        for (name, value) in &fixed_params {
            if let Some(dist) = distributions.get(name) {
                let internal = dist.to_internal_repr(value)?;
                internal_params.insert(name.clone(), internal);
            } else {
                // No distribution found; store Float/Int directly
                let internal = match value {
                    ParamValue::Float(v) => *v,
                    ParamValue::Int(v) => *v as f64,
                    ParamValue::Categorical(_) => {
                        return Err(crate::error::Error::ValueError(format!(
                            "cannot fix categorical param '{name}' without a distribution"
                        )));
                    }
                };
                internal_params.insert(name.clone(), internal);
            }
        }
        Ok(Self::new(internal_params, base_sampler))
    }
}

impl Sampler for PartialFixedSampler {
    fn infer_relative_search_space(&self, trials: &[FrozenTrial]) -> HashMap<String, Distribution> {
        let mut space = self.base_sampler.infer_relative_search_space(trials);
        // Remove fixed params from the relative search space
        for name in self.fixed_params.keys() {
            space.remove(name);
        }
        space
    }

    fn sample_relative(
        &self,
        trials: &[FrozenTrial],
        search_space: &HashMap<String, Distribution>,
    ) -> Result<HashMap<String, f64>> {
        let mut result = self.base_sampler.sample_relative(trials, search_space)?;
        // Inject fixed params
        for (name, &value) in &self.fixed_params {
            result.insert(name.clone(), value);
        }
        Ok(result)
    }

    fn sample_independent(
        &self,
        trial: &FrozenTrial,
        param_name: &str,
        distribution: &Distribution,
    ) -> Result<f64> {
        if let Some(&value) = self.fixed_params.get(param_name) {
            return Ok(value);
        }
        self.base_sampler
            .sample_independent(trial, param_name, distribution)
    }

    fn before_trial(&self, trials: &[FrozenTrial]) {
        self.base_sampler.before_trial(trials);
    }

    fn after_trial(
        &self,
        trials: &[FrozenTrial],
        trial: &FrozenTrial,
        state: TrialState,
        values: Option<&[f64]>,
    ) {
        self.base_sampler.after_trial(trials, trial, state, values);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::samplers::RandomSampler;
    use crate::study::{StudyDirection, create_study};

    #[test]
    fn test_partial_fixed_basic() {
        let base: Arc<dyn Sampler> = Arc::new(RandomSampler::new(Some(42)));
        let mut fixed = HashMap::new();
        fixed.insert("x".to_string(), 0.5);

        let sampler: Arc<dyn Sampler> = Arc::new(PartialFixedSampler::new(fixed, base));

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
                    let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    let y = trial.suggest_float("y", 0.0, 1.0, false, None)?;
                    Ok(x * x + y * y)
                },
                Some(10),
                None,
                None,
            )
            .unwrap();

        let trials = study.trials().unwrap();
        assert_eq!(trials.len(), 10);
        // x should always be 0.5
        for trial in &trials {
            let x = match trial.params.get("x") {
                Some(ParamValue::Float(v)) => *v,
                _ => panic!("expected float param x"),
            };
            assert!((x - 0.5).abs() < 1e-10, "x should be fixed at 0.5, got {x}");
        }
    }

    #[test]
    fn test_partial_fixed_non_fixed_varies() {
        let base: Arc<dyn Sampler> = Arc::new(RandomSampler::new(Some(42)));
        let mut fixed = HashMap::new();
        fixed.insert("x".to_string(), 0.5);

        let sampler: Arc<dyn Sampler> = Arc::new(PartialFixedSampler::new(fixed, base));

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
                    let _x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    let y = trial.suggest_float("y", 0.0, 1.0, false, None)?;
                    Ok(y * y)
                },
                Some(20),
                None,
                None,
            )
            .unwrap();

        let trials = study.trials().unwrap();
        let y_values: Vec<f64> = trials
            .iter()
            .map(|t| match t.params.get("y") {
                Some(ParamValue::Float(v)) => *v,
                _ => panic!("expected float param y"),
            })
            .collect();

        // y should vary (not all the same)
        let first = y_values[0];
        assert!(
            y_values.iter().any(|&v| (v - first).abs() > 1e-10),
            "y values should vary"
        );
    }
}
