mod in_memory;

pub use in_memory::InMemoryStorage;

use std::collections::HashMap;

use crate::distributions::Distribution;
use crate::error::{Error, Result};
use crate::study::{FrozenStudy, StudyDirection};
use crate::trial::{FrozenTrial, TrialState};

/// The storage trait: full CRUD for studies and trials.
///
///
/// All implementations must be `Send + Sync` for thread-safe study access.
pub trait Storage: Send + Sync {
    // ── Study CRUD ──────────────────────────────────────────────────────

    /// Create a new study. Returns the study_id.
    ///
    /// If `study_name` is `None`, a unique name is generated.
    fn create_new_study(
        &self,
        directions: &[StudyDirection],
        study_name: Option<&str>,
    ) -> Result<i64>;

    /// Delete a study by id.
    fn delete_study(&self, study_id: i64) -> Result<()>;

    /// Set a user attribute on a study.
    fn set_study_user_attr(
        &self,
        study_id: i64,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()>;

    /// Set a system attribute on a study.
    fn set_study_system_attr(
        &self,
        study_id: i64,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()>;

    /// Look up a study_id by name.
    fn get_study_id_from_name(&self, study_name: &str) -> Result<i64>;

    /// Look up a study name by id.
    fn get_study_name_from_id(&self, study_id: i64) -> Result<String>;

    /// Get the optimization directions for a study.
    fn get_study_directions(&self, study_id: i64) -> Result<Vec<StudyDirection>>;

    /// Get user attributes for a study.
    fn get_study_user_attrs(&self, study_id: i64) -> Result<HashMap<String, serde_json::Value>>;

    /// Get system attributes for a study.
    fn get_study_system_attrs(&self, study_id: i64) -> Result<HashMap<String, serde_json::Value>>;

    /// Get all studies.
    fn get_all_studies(&self) -> Result<Vec<FrozenStudy>>;

    // ── Trial CRUD ──────────────────────────────────────────────────────

    /// Create a new trial. Returns the trial_id.
    ///
    /// If `template_trial` is `None`, a fresh RUNNING trial is created.
    fn create_new_trial(
        &self,
        study_id: i64,
        template_trial: Option<&FrozenTrial>,
    ) -> Result<i64>;

    /// Set a parameter on a trial (internal representation).
    fn set_trial_param(
        &self,
        trial_id: i64,
        param_name: &str,
        param_value_internal: f64,
        distribution: &Distribution,
    ) -> Result<()>;

    /// Set the state and optional values of a trial.
    ///
    /// Returns `false` if the transition was silently rejected (e.g. RUNNING on
    /// a non-WAITING trial).
    fn set_trial_state_values(
        &self,
        trial_id: i64,
        state: TrialState,
        values: Option<&[f64]>,
    ) -> Result<bool>;

    /// Set an intermediate value at a given step.
    fn set_trial_intermediate_value(
        &self,
        trial_id: i64,
        step: i64,
        intermediate_value: f64,
    ) -> Result<()>;

    /// Set a user attribute on a trial.
    fn set_trial_user_attr(
        &self,
        trial_id: i64,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()>;

    /// Set a system attribute on a trial.
    fn set_trial_system_attr(
        &self,
        trial_id: i64,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()>;

    /// Get a trial by id.
    fn get_trial(&self, trial_id: i64) -> Result<FrozenTrial>;

    /// Get all trials for a study, optionally filtered by state.
    fn get_all_trials(
        &self,
        study_id: i64,
        states: Option<&[TrialState]>,
    ) -> Result<Vec<FrozenTrial>>;

    // ── Default implementations ─────────────────────────────────────────

    /// Look up a trial_id given (study_id, trial_number).
    fn get_trial_id_from_study_id_trial_number(
        &self,
        study_id: i64,
        trial_number: i64,
    ) -> Result<i64> {
        let trials = self.get_all_trials(study_id, None)?;
        trials
            .iter()
            .find(|t| t.number == trial_number)
            .map(|t| t.trial_id)
            .ok_or_else(|| {
                Error::ValueError(format!(
                    "trial number {trial_number} not found in study {study_id}"
                ))
            })
    }

    /// Get a trial's number from its id.
    fn get_trial_number_from_id(&self, trial_id: i64) -> Result<i64> {
        let trial = self.get_trial(trial_id)?;
        Ok(trial.number)
    }

    /// Get a single parameter's internal value.
    fn get_trial_param(&self, trial_id: i64, param_name: &str) -> Result<f64> {
        let trial = self.get_trial(trial_id)?;
        let dist = trial
            .distributions
            .get(param_name)
            .ok_or_else(|| Error::ValueError(format!("param '{param_name}' not found")))?;
        let val = trial
            .params
            .get(param_name)
            .ok_or_else(|| Error::ValueError(format!("param '{param_name}' not found")))?;
        dist.to_internal_repr(val)
    }

    /// Count trials, optionally filtered by state.
    fn get_n_trials(&self, study_id: i64, states: Option<&[TrialState]>) -> Result<usize> {
        Ok(self.get_all_trials(study_id, states)?.len())
    }

    /// Get the best trial for a single-objective study.
    fn get_best_trial(&self, study_id: i64) -> Result<FrozenTrial> {
        let trials =
            self.get_all_trials(study_id, Some(&[TrialState::Complete]))?;
        if trials.is_empty() {
            return Err(Error::ValueError(
                "no trials are completed yet".into(),
            ));
        }
        let directions = self.get_study_directions(study_id)?;
        if directions.len() > 1 {
            return Err(Error::ValueError(
                "best trial can be obtained only for single-objective optimization".into(),
            ));
        }
        let direction = directions[0];
        let best = trials
            .into_iter()
            .min_by(|a, b| {
                let va = a.value().unwrap().unwrap();
                let vb = b.value().unwrap().unwrap();
                match direction {
                    StudyDirection::Minimize | StudyDirection::NotSet => {
                        va.partial_cmp(&vb).unwrap()
                    }
                    StudyDirection::Maximize => vb.partial_cmp(&va).unwrap(),
                }
            })
            .unwrap();
        Ok(best)
    }

    /// Check if a trial is still updatable (not finished).
    fn check_trial_is_updatable(&self, trial_id: i64, trial_state: TrialState) -> Result<()> {
        if trial_state.is_finished() {
            let trial = self.get_trial(trial_id)?;
            return Err(Error::UpdateFinishedTrialError(format!(
                "trial #{} has already finished and cannot be updated",
                trial.number
            )));
        }
        Ok(())
    }

    /// No-op session cleanup hook.
    fn remove_session(&self) {}
}
