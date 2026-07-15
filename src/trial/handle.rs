use std::collections::HashMap;
use std::sync::Arc;

use crate::distributions::{
    CategoricalChoice, CategoricalDistribution, Distribution, FloatDistribution, IntDistribution,
};
use crate::error::{Error, Result};
use crate::pruners::Pruner;
use crate::samplers::Sampler;
use crate::storage::Storage;
use crate::trial::TrialState;

/// A mutable handle to a running trial.
///
///
/// Provides `suggest_*` methods that record sampled parameter values into
/// storage, `report` for intermediate values, and `should_prune` to query
/// the pruner.
pub struct Trial {
    trial_id: i64,
    study_id: i64,
    storage: Arc<dyn Storage>,
    sampler: Arc<dyn Sampler>,
    pruner: Arc<dyn Pruner>,
    number: i64,
    /// Relative param values pre-sampled by the sampler (internal repr).
    relative_params: HashMap<String, f64>,
}

impl Trial {
    /// Create a new `Trial`. Called internally by `Study::ask()`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        trial_id: i64,
        study_id: i64,
        number: i64,
        storage: Arc<dyn Storage>,
        sampler: Arc<dyn Sampler>,
        pruner: Arc<dyn Pruner>,
        relative_params: HashMap<String, f64>,
    ) -> Self {
        Self {
            trial_id,
            study_id,
            storage,
            sampler,
            pruner,
            number,
            relative_params,
        }
    }

    /// The trial's unique id.
    pub fn trial_id(&self) -> i64 {
        self.trial_id
    }

    /// The trial's number within the study (0-indexed).
    pub fn number(&self) -> i64 {
        self.number
    }

    /// Suggest a floating-point parameter.
    pub fn suggest_float(
        &mut self,
        name: &str,
        low: f64,
        high: f64,
        log: bool,
        step: Option<f64>,
    ) -> Result<f64> {
        let dist = Distribution::FloatDistribution(FloatDistribution::new(low, high, log, step)?);
        let internal = self.suggest(name, &dist)?;
        dist.to_external_repr(internal).map(|v| match v {
            crate::distributions::ParamValue::Float(f) => f,
            _ => unreachable!(),
        })
    }

    /// Suggest an integer parameter.
    pub fn suggest_int(
        &mut self,
        name: &str,
        low: i64,
        high: i64,
        log: bool,
        step: i64,
    ) -> Result<i64> {
        let dist = Distribution::IntDistribution(IntDistribution::new(low, high, log, step)?);
        let internal = self.suggest(name, &dist)?;
        dist.to_external_repr(internal).map(|v| match v {
            crate::distributions::ParamValue::Int(i) => i,
            _ => unreachable!(),
        })
    }

    /// Suggest a categorical parameter.
    pub fn suggest_categorical(
        &mut self,
        name: &str,
        choices: Vec<CategoricalChoice>,
    ) -> Result<CategoricalChoice> {
        let dist = Distribution::CategoricalDistribution(CategoricalDistribution::new(choices)?);
        let internal = self.suggest(name, &dist)?;
        dist.to_external_repr(internal).map(|v| match v {
            crate::distributions::ParamValue::Categorical(c) => c,
            _ => unreachable!(),
        })
    }

    /// Core suggest logic: check if already suggested or in relative params,
    /// otherwise fall back to independent sampling.
    fn suggest(&mut self, name: &str, dist: &Distribution) -> Result<f64> {
        // Check if this param was already set (re-suggest returns same value)
        let existing = self.storage.get_trial(self.trial_id)?;
        if let Some(existing_dist) = existing.distributions.get(name) {
            if existing_dist != dist {
                return Err(Error::ValueError(format!(
                    "cannot use different distribution for param '{name}'"
                )));
            }
            let val = existing.params.get(name).unwrap();
            return dist.to_internal_repr(val);
        }

        // Check if we have a pre-sampled relative param
        let internal = if let Some(&v) = self.relative_params.get(name) {
            v
        } else {
            // Fall back to independent sampling
            self.sampler.sample_independent(&existing, name, dist)?
        };

        // Record the param in storage
        self.storage
            .set_trial_param(self.trial_id, name, internal, dist)?;
        Ok(internal)
    }

    /// Report an intermediate objective value at a given step.
    pub fn report(&self, value: f64, step: i64) -> Result<()> {
        self.storage
            .set_trial_intermediate_value(self.trial_id, step, value)
    }

    /// Check if the trial should be pruned.
    pub fn should_prune(&self) -> Result<bool> {
        let all_trials = self.storage.get_all_trials(
            self.study_id,
            Some(&[TrialState::Complete, TrialState::Pruned]),
        )?;
        let trial = self.storage.get_trial(self.trial_id)?;
        self.pruner.prune(&all_trials, &trial)
    }

    /// Set a user attribute on the trial.
    pub fn set_user_attr(&self, key: &str, value: serde_json::Value) -> Result<()> {
        self.storage.set_trial_user_attr(self.trial_id, key, value)
    }

    /// Set a system attribute on the trial.
    pub fn set_system_attr(&self, key: &str, value: serde_json::Value) -> Result<()> {
        self.storage
            .set_trial_system_attr(self.trial_id, key, value)
    }
}
