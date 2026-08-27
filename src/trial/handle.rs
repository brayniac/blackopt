use std::collections::HashMap;
use std::sync::Arc;

use crate::distributions::{
    CategoricalChoice, CategoricalDistribution, Distribution, FloatDistribution, IntDistribution,
};
use crate::error::{Error, Result};
use crate::pruners::Pruner;
use crate::samplers::Sampler;
use crate::storage::Storage;
use crate::trial::{FrozenTrial, TrialState};

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
    /// Pre-specified (enqueued) param values; take precedence over the sampler.
    fixed_params: HashMap<String, crate::distributions::ParamValue>,
    /// Trial history, fetched at most once per trial for independent
    /// sampling. Samplers that need history need the same history for every
    /// parameter of a trial, and fetching it per-parameter deep-clones the
    /// whole study once per suggest.
    history: Option<Vec<FrozenTrial>>,
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
        fixed_params: HashMap<String, crate::distributions::ParamValue>,
    ) -> Self {
        Self {
            trial_id,
            study_id,
            storage,
            sampler,
            pruner,
            number,
            relative_params,
            fixed_params,
            history: None,
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

    /// Record the sampler's pre-sampled relative parameters.
    pub(crate) fn set_relative_params(&mut self, params: HashMap<String, f64>) {
        self.relative_params = params;
    }

    /// Names of enqueued parameters the objective never asked for.
    ///
    /// An enqueued trial is a request to evaluate one specific configuration.
    /// If a name was never suggested — a typo, or a parameter behind a branch
    /// the objective did not take — that configuration was not the one
    /// evaluated, and saying so beats silently running an ordinary trial.
    pub(crate) fn unconsumed_fixed_params(&self) -> Result<Vec<String>> {
        if self.fixed_params.is_empty() {
            return Ok(Vec::new());
        }
        let frozen = self.storage.get_trial(self.trial_id)?;
        let mut unconsumed: Vec<String> = self
            .fixed_params
            .keys()
            .filter(|name| !frozen.params.contains_key(*name))
            .cloned()
            .collect();
        unconsumed.sort();
        Ok(unconsumed)
    }

    /// Core suggest logic: check if already suggested or in relative params,
    /// otherwise fall back to independent sampling.
    fn suggest(&mut self, name: &str, dist: &Distribution) -> Result<f64> {
        let existing = self.storage.get_trial(self.trial_id)?;

        // Check if this param was already set (re-suggest returns same value)
        if let Some(existing_dist) = existing.distributions.get(name) {
            if existing_dist != dist {
                return Err(Error::ValueError(format!(
                    "cannot use different distribution for param '{name}'"
                )));
            }
            let val = existing.params.get(name).unwrap();
            return dist.to_internal_repr(val);
        }

        // Enqueued (fixed) values take precedence, then relative params, then
        // independent sampling.
        let internal = if let Some(pv) = self.fixed_params.get(name) {
            dist.to_internal_repr(pv).map_err(|e| {
                Error::ValueError(format!(
                    "enqueued value for parameter '{name}' cannot be used: {e}"
                ))
            })?
        } else if let Some(&v) = self.relative_params.get(name) {
            v
        } else {
            // Fall back to independent sampling. Only this branch needs the
            // trial history, so it is fetched lazily and then reused for the
            // rest of the trial: fetching per-parameter would deep-clone the
            // whole study N*P times over a study of N trials with P
            // parameters. Within one trial the only history that changes is
            // this trial's own, which independent samplers do not model.
            if self.history.is_none() {
                self.history = Some(self.storage.get_all_trials(self.study_id, None)?);
            }
            let all_trials = self.history.as_deref().unwrap_or(&[]);
            self.sampler
                .sample_independent(all_trials, &existing, name, dist)?
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
