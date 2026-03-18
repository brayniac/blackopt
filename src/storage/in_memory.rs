use std::collections::HashMap;

use chrono::Utc;
use parking_lot::Mutex;

use crate::distributions::Distribution;
use crate::error::{Error, Result};
use crate::storage::Storage;
use crate::study::{FrozenStudy, StudyDirection};
use crate::trial::{FrozenTrial, TrialState};

/// Per-study metadata and trial storage.
#[derive(Debug, Clone)]
struct StudyInfo {
    name: String,
    directions: Vec<StudyDirection>,
    trials: Vec<FrozenTrial>,
    param_distribution: HashMap<String, Distribution>,
    user_attrs: HashMap<String, serde_json::Value>,
    system_attrs: HashMap<String, serde_json::Value>,
    best_trial_id: Option<i64>,
}

/// The mutable inner state behind the lock.
#[derive(Debug)]
struct Inner {
    studies: HashMap<i64, StudyInfo>,
    study_name_to_id: HashMap<String, i64>,
    trial_id_to_study_id_and_number: HashMap<i64, (i64, i64)>,
    max_study_id: i64,
    max_trial_id: i64,
}

/// Thread-safe in-memory storage.
///
///
/// All state is behind a single `Mutex`. Reads and writes are serialized.
pub struct InMemoryStorage {
    inner: Mutex<Inner>,
}

impl std::fmt::Debug for InMemoryStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InMemoryStorage").finish()
    }
}

impl Default for InMemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryStorage {
    /// Create a new empty in-memory storage.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                studies: HashMap::new(),
                study_name_to_id: HashMap::new(),
                trial_id_to_study_id_and_number: HashMap::new(),
                max_study_id: -1,
                max_trial_id: -1,
            }),
        }
    }

    /// Get the study info, or error if not found.
    fn get_study(inner: &Inner, study_id: i64) -> Result<&StudyInfo> {
        inner.studies.get(&study_id).ok_or_else(|| {
            Error::ValueError(format!("study {study_id} not found"))
        })
    }

    /// Get the study info mutably, or error if not found.
    fn get_study_mut(inner: &mut Inner, study_id: i64) -> Result<&mut StudyInfo> {
        inner.studies.get_mut(&study_id).ok_or_else(|| {
            Error::ValueError(format!("study {study_id} not found"))
        })
    }

    /// Resolve trial_id to (study_id, trial_number).
    fn resolve_trial(inner: &Inner, trial_id: i64) -> Result<(i64, i64)> {
        inner
            .trial_id_to_study_id_and_number
            .get(&trial_id)
            .copied()
            .ok_or_else(|| Error::ValueError(format!("trial {trial_id} not found")))
    }

    /// Check updatability inline (avoids re-acquiring the lock).
    fn check_updatable(trial: &FrozenTrial) -> Result<()> {
        if trial.state.is_finished() {
            return Err(Error::UpdateFinishedTrialError(format!(
                "trial #{} has already finished and cannot be updated",
                trial.number
            )));
        }
        Ok(())
    }

    /// Update the best_trial_id cache for single-objective studies.
    fn update_cache(study: &mut StudyInfo, trial_id: i64) {
        // Only cache for single-objective COMPLETE trials
        if study.directions.len() != 1 {
            return;
        }
        let trial_number = study
            .trials
            .iter()
            .position(|t| t.trial_id == trial_id);
        let trial_number = match trial_number {
            Some(n) => n,
            None => return,
        };
        let trial = &study.trials[trial_number];
        if trial.state != TrialState::Complete {
            return;
        }

        let new_value = match trial.value() {
            Ok(Some(v)) => v,
            _ => return,
        };

        let direction = study.directions[0];
        let is_better = if let Some(best_id) = study.best_trial_id {
            let best_trial = study.trials.iter().find(|t| t.trial_id == best_id);
            match best_trial.and_then(|t| t.value().ok().flatten()) {
                Some(best_value) => match direction {
                    StudyDirection::Minimize | StudyDirection::NotSet => new_value < best_value,
                    StudyDirection::Maximize => new_value > best_value,
                },
                None => true,
            }
        } else {
            true
        };

        if is_better {
            study.best_trial_id = Some(trial_id);
        }
    }
}

impl Storage for InMemoryStorage {
    fn create_new_study(
        &self,
        directions: &[StudyDirection],
        study_name: Option<&str>,
    ) -> Result<i64> {
        let mut inner = self.inner.lock();
        inner.max_study_id += 1;
        let study_id = inner.max_study_id;

        let name = match study_name {
            Some(n) => {
                if inner.study_name_to_id.contains_key(n) {
                    return Err(Error::DuplicatedStudyError(n.to_string()));
                }
                n.to_string()
            }
            None => format!("no-name-{study_id}"),
        };

        inner.study_name_to_id.insert(name.clone(), study_id);
        inner.studies.insert(
            study_id,
            StudyInfo {
                name,
                directions: directions.to_vec(),
                trials: Vec::new(),
                param_distribution: HashMap::new(),
                user_attrs: HashMap::new(),
                system_attrs: HashMap::new(),
                best_trial_id: None,
            },
        );
        Ok(study_id)
    }

    fn delete_study(&self, study_id: i64) -> Result<()> {
        let mut inner = self.inner.lock();
        let study = inner
            .studies
            .remove(&study_id)
            .ok_or_else(|| Error::ValueError(format!("study {study_id} not found")))?;
        inner.study_name_to_id.remove(&study.name);

        // Remove trial mappings
        let trial_ids: Vec<i64> = inner
            .trial_id_to_study_id_and_number
            .iter()
            .filter(|(_, (sid, _))| *sid == study_id)
            .map(|(tid, _)| *tid)
            .collect();
        for tid in trial_ids {
            inner.trial_id_to_study_id_and_number.remove(&tid);
        }
        Ok(())
    }

    fn set_study_user_attr(
        &self,
        study_id: i64,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()> {
        let mut inner = self.inner.lock();
        let study = Self::get_study_mut(&mut inner, study_id)?;
        study.user_attrs.insert(key.to_string(), value);
        Ok(())
    }

    fn set_study_system_attr(
        &self,
        study_id: i64,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()> {
        let mut inner = self.inner.lock();
        let study = Self::get_study_mut(&mut inner, study_id)?;
        study.system_attrs.insert(key.to_string(), value);
        Ok(())
    }

    fn get_study_id_from_name(&self, study_name: &str) -> Result<i64> {
        let inner = self.inner.lock();
        inner
            .study_name_to_id
            .get(study_name)
            .copied()
            .ok_or_else(|| {
                Error::ValueError(format!("study '{study_name}' not found"))
            })
    }

    fn get_study_name_from_id(&self, study_id: i64) -> Result<String> {
        let inner = self.inner.lock();
        let study = Self::get_study(&inner, study_id)?;
        Ok(study.name.clone())
    }

    fn get_study_directions(&self, study_id: i64) -> Result<Vec<StudyDirection>> {
        let inner = self.inner.lock();
        let study = Self::get_study(&inner, study_id)?;
        Ok(study.directions.clone())
    }

    fn get_study_user_attrs(&self, study_id: i64) -> Result<HashMap<String, serde_json::Value>> {
        let inner = self.inner.lock();
        let study = Self::get_study(&inner, study_id)?;
        Ok(study.user_attrs.clone())
    }

    fn get_study_system_attrs(&self, study_id: i64) -> Result<HashMap<String, serde_json::Value>> {
        let inner = self.inner.lock();
        let study = Self::get_study(&inner, study_id)?;
        Ok(study.system_attrs.clone())
    }

    fn get_all_studies(&self) -> Result<Vec<FrozenStudy>> {
        let inner = self.inner.lock();
        Ok(inner
            .studies
            .iter()
            .map(|(study_id, info)| FrozenStudy {
                study_id: *study_id,
                study_name: info.name.clone(),
                directions: info.directions.clone(),
                user_attrs: info.user_attrs.clone(),
                system_attrs: info.system_attrs.clone(),
            })
            .collect())
    }

    fn create_new_trial(
        &self,
        study_id: i64,
        template_trial: Option<&FrozenTrial>,
    ) -> Result<i64> {
        let mut inner = self.inner.lock();

        // Get study and compute trial number
        let study = Self::get_study(&inner, study_id)?;
        let trial_number = study.trials.len() as i64;

        inner.max_trial_id += 1;
        let trial_id = inner.max_trial_id;

        let trial = if let Some(template) = template_trial {
            let mut t = template.clone();
            t.trial_id = trial_id;
            t.number = trial_number;
            t
        } else {
            // Create a fresh RUNNING trial
            FrozenTrial {
                number: trial_number,
                state: TrialState::Running,
                values: None,
                datetime_start: Some(Utc::now()),
                datetime_complete: None,
                params: HashMap::new(),
                distributions: HashMap::new(),
                user_attrs: HashMap::new(),
                system_attrs: HashMap::new(),
                intermediate_values: HashMap::new(),
                trial_id,
            }
        };

        inner
            .trial_id_to_study_id_and_number
            .insert(trial_id, (study_id, trial_number));

        let study = Self::get_study_mut(&mut inner, study_id)?;
        study.trials.push(trial);
        Self::update_cache(study, trial_id);

        Ok(trial_id)
    }

    fn set_trial_param(
        &self,
        trial_id: i64,
        param_name: &str,
        param_value_internal: f64,
        distribution: &Distribution,
    ) -> Result<()> {
        let mut inner = self.inner.lock();
        let (study_id, trial_number) = Self::resolve_trial(&inner, trial_id)?;

        let study = Self::get_study(&inner, study_id)?;
        let trial = &study.trials[trial_number as usize];
        Self::check_updatable(trial)?;

        // Validate distribution compatibility
        if let Some(existing) = study.param_distribution.get(param_name)
            && existing != distribution
        {
            return Err(Error::ValueError(format!(
                "cannot set different distribution for param '{param_name}'"
            )));
        }

        let external_value = distribution.to_external_repr(param_value_internal)?;

        let study = Self::get_study_mut(&mut inner, study_id)?;
        study
            .param_distribution
            .insert(param_name.to_string(), distribution.clone());

        let trial = &mut study.trials[trial_number as usize];
        trial.params.insert(param_name.to_string(), external_value);
        trial
            .distributions
            .insert(param_name.to_string(), distribution.clone());

        Ok(())
    }

    fn set_trial_state_values(
        &self,
        trial_id: i64,
        state: TrialState,
        values: Option<&[f64]>,
    ) -> Result<bool> {
        let mut inner = self.inner.lock();
        let (study_id, trial_number) = Self::resolve_trial(&inner, trial_id)?;

        let study = Self::get_study(&inner, study_id)?;
        let trial = &study.trials[trial_number as usize];

        // Silently reject Running on non-Waiting trials
        if state == TrialState::Running && trial.state != TrialState::Waiting {
            return Ok(false);
        }

        Self::check_updatable(trial)?;

        let study = Self::get_study_mut(&mut inner, study_id)?;
        let trial = &mut study.trials[trial_number as usize];

        // Set timestamps
        if state == TrialState::Running {
            trial.datetime_start = Some(Utc::now());
        }
        if state.is_finished() {
            trial.datetime_complete = Some(Utc::now());
        }

        trial.state = state;
        trial.values = values.map(|v| v.to_vec());

        Self::update_cache(study, trial_id);
        Ok(true)
    }

    fn set_trial_intermediate_value(
        &self,
        trial_id: i64,
        step: i64,
        intermediate_value: f64,
    ) -> Result<()> {
        let mut inner = self.inner.lock();
        let (study_id, trial_number) = Self::resolve_trial(&inner, trial_id)?;

        let study = Self::get_study(&inner, study_id)?;
        let trial = &study.trials[trial_number as usize];
        Self::check_updatable(trial)?;

        let study = Self::get_study_mut(&mut inner, study_id)?;
        let trial = &mut study.trials[trial_number as usize];
        trial
            .intermediate_values
            .insert(step, intermediate_value);
        Ok(())
    }

    fn set_trial_user_attr(
        &self,
        trial_id: i64,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()> {
        let mut inner = self.inner.lock();
        let (study_id, trial_number) = Self::resolve_trial(&inner, trial_id)?;

        let study = Self::get_study(&inner, study_id)?;
        let trial = &study.trials[trial_number as usize];
        Self::check_updatable(trial)?;

        let study = Self::get_study_mut(&mut inner, study_id)?;
        let trial = &mut study.trials[trial_number as usize];
        trial.user_attrs.insert(key.to_string(), value);
        Ok(())
    }

    fn set_trial_system_attr(
        &self,
        trial_id: i64,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()> {
        let mut inner = self.inner.lock();
        let (study_id, trial_number) = Self::resolve_trial(&inner, trial_id)?;

        let study = Self::get_study(&inner, study_id)?;
        let trial = &study.trials[trial_number as usize];
        Self::check_updatable(trial)?;

        let study = Self::get_study_mut(&mut inner, study_id)?;
        let trial = &mut study.trials[trial_number as usize];
        trial.system_attrs.insert(key.to_string(), value);
        Ok(())
    }

    fn get_trial(&self, trial_id: i64) -> Result<FrozenTrial> {
        let inner = self.inner.lock();
        let (study_id, trial_number) = Self::resolve_trial(&inner, trial_id)?;
        let study = Self::get_study(&inner, study_id)?;
        Ok(study.trials[trial_number as usize].clone())
    }

    fn get_all_trials(
        &self,
        study_id: i64,
        states: Option<&[TrialState]>,
    ) -> Result<Vec<FrozenTrial>> {
        let inner = self.inner.lock();
        let study = Self::get_study(&inner, study_id)?;

        let trials: Vec<FrozenTrial> = match states {
            None => study.trials.clone(),
            Some(s) => study
                .trials
                .iter()
                .filter(|t| s.contains(&t.state))
                .cloned()
                .collect(),
        };
        Ok(trials)
    }

    fn get_trial_id_from_study_id_trial_number(
        &self,
        study_id: i64,
        trial_number: i64,
    ) -> Result<i64> {
        let inner = self.inner.lock();
        let study = Self::get_study(&inner, study_id)?;
        study
            .trials
            .get(trial_number as usize)
            .map(|t| t.trial_id)
            .ok_or_else(|| {
                Error::ValueError(format!(
                    "trial number {trial_number} not found in study {study_id}"
                ))
            })
    }

    fn get_trial_number_from_id(&self, trial_id: i64) -> Result<i64> {
        let inner = self.inner.lock();
        let (_, trial_number) = Self::resolve_trial(&inner, trial_id)?;
        Ok(trial_number)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::ParamValue;

    #[test]
    fn test_create_study() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], Some("test"))
            .unwrap();
        assert_eq!(sid, 0);
        assert_eq!(storage.get_study_name_from_id(sid).unwrap(), "test");
        assert_eq!(
            storage.get_study_directions(sid).unwrap(),
            vec![StudyDirection::Minimize]
        );
    }

    #[test]
    fn test_create_study_auto_name() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], None)
            .unwrap();
        let name = storage.get_study_name_from_id(sid).unwrap();
        assert!(name.starts_with("no-name-"));
    }

    #[test]
    fn test_duplicate_study_name() {
        let storage = InMemoryStorage::new();
        storage
            .create_new_study(&[StudyDirection::Minimize], Some("dup"))
            .unwrap();
        assert!(storage
            .create_new_study(&[StudyDirection::Minimize], Some("dup"))
            .is_err());
    }

    #[test]
    fn test_delete_study() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], Some("del"))
            .unwrap();
        storage.delete_study(sid).unwrap();
        assert!(storage.get_study_name_from_id(sid).is_err());
    }

    #[test]
    fn test_create_trial() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], None)
            .unwrap();
        let tid = storage.create_new_trial(sid, None).unwrap();
        let trial = storage.get_trial(tid).unwrap();
        assert_eq!(trial.number, 0);
        assert_eq!(trial.state, TrialState::Running);
        assert!(trial.datetime_start.is_some());
    }

    #[test]
    fn test_set_trial_param() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], None)
            .unwrap();
        let tid = storage.create_new_trial(sid, None).unwrap();

        let dist = Distribution::FloatDistribution(
            crate::distributions::FloatDistribution::new(0.0, 1.0, false, None).unwrap(),
        );
        storage
            .set_trial_param(tid, "x", 0.5, &dist)
            .unwrap();

        let trial = storage.get_trial(tid).unwrap();
        assert_eq!(trial.params.get("x"), Some(&ParamValue::Float(0.5)));
    }

    #[test]
    fn test_set_trial_state_values() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], None)
            .unwrap();
        let tid = storage.create_new_trial(sid, None).unwrap();

        let ok = storage
            .set_trial_state_values(tid, TrialState::Complete, Some(&[1.0]))
            .unwrap();
        assert!(ok);

        let trial = storage.get_trial(tid).unwrap();
        assert_eq!(trial.state, TrialState::Complete);
        assert_eq!(trial.values, Some(vec![1.0]));
        assert!(trial.datetime_complete.is_some());
    }

    #[test]
    fn test_set_trial_state_running_on_running_rejected() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], None)
            .unwrap();
        let tid = storage.create_new_trial(sid, None).unwrap();

        // Trial is already RUNNING; trying to set RUNNING again should return false
        let ok = storage
            .set_trial_state_values(tid, TrialState::Running, None)
            .unwrap();
        assert!(!ok);
    }

    #[test]
    fn test_update_finished_trial_rejected() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], None)
            .unwrap();
        let tid = storage.create_new_trial(sid, None).unwrap();
        storage
            .set_trial_state_values(tid, TrialState::Complete, Some(&[1.0]))
            .unwrap();

        // Trying to update a finished trial should error
        assert!(storage
            .set_trial_state_values(tid, TrialState::Complete, Some(&[2.0]))
            .is_err());
    }

    #[test]
    fn test_intermediate_values() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], None)
            .unwrap();
        let tid = storage.create_new_trial(sid, None).unwrap();
        storage.set_trial_intermediate_value(tid, 0, 0.5).unwrap();
        storage.set_trial_intermediate_value(tid, 1, 0.3).unwrap();

        let trial = storage.get_trial(tid).unwrap();
        assert_eq!(trial.intermediate_values.len(), 2);
        assert_eq!(trial.intermediate_values[&0], 0.5);
        assert_eq!(trial.intermediate_values[&1], 0.3);
    }

    #[test]
    fn test_get_all_trials_filtered() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], None)
            .unwrap();
        let t0 = storage.create_new_trial(sid, None).unwrap();
        let _t1 = storage.create_new_trial(sid, None).unwrap();

        storage
            .set_trial_state_values(t0, TrialState::Complete, Some(&[1.0]))
            .unwrap();

        let all = storage.get_all_trials(sid, None).unwrap();
        assert_eq!(all.len(), 2);

        let complete = storage
            .get_all_trials(sid, Some(&[TrialState::Complete]))
            .unwrap();
        assert_eq!(complete.len(), 1);

        let running = storage
            .get_all_trials(sid, Some(&[TrialState::Running]))
            .unwrap();
        assert_eq!(running.len(), 1);
    }

    #[test]
    fn test_get_best_trial() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], None)
            .unwrap();

        let t0 = storage.create_new_trial(sid, None).unwrap();
        storage
            .set_trial_state_values(t0, TrialState::Complete, Some(&[3.0]))
            .unwrap();

        let t1 = storage.create_new_trial(sid, None).unwrap();
        storage
            .set_trial_state_values(t1, TrialState::Complete, Some(&[1.0]))
            .unwrap();

        let t2 = storage.create_new_trial(sid, None).unwrap();
        storage
            .set_trial_state_values(t2, TrialState::Complete, Some(&[2.0]))
            .unwrap();

        let best = storage.get_best_trial(sid).unwrap();
        assert_eq!(best.value().unwrap(), Some(1.0));
    }

    #[test]
    fn test_get_best_trial_maximize() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Maximize], None)
            .unwrap();

        let t0 = storage.create_new_trial(sid, None).unwrap();
        storage
            .set_trial_state_values(t0, TrialState::Complete, Some(&[1.0]))
            .unwrap();

        let t1 = storage.create_new_trial(sid, None).unwrap();
        storage
            .set_trial_state_values(t1, TrialState::Complete, Some(&[3.0]))
            .unwrap();

        let best = storage.get_best_trial(sid).unwrap();
        assert_eq!(best.value().unwrap(), Some(3.0));
    }

    #[test]
    fn test_get_best_trial_no_complete() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], None)
            .unwrap();
        assert!(storage.get_best_trial(sid).is_err());
    }

    #[test]
    fn test_study_user_attrs() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], None)
            .unwrap();
        storage
            .set_study_user_attr(sid, "key", serde_json::json!("value"))
            .unwrap();
        let attrs = storage.get_study_user_attrs(sid).unwrap();
        assert_eq!(attrs.get("key"), Some(&serde_json::json!("value")));
    }

    #[test]
    fn test_trial_user_attrs() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], None)
            .unwrap();
        let tid = storage.create_new_trial(sid, None).unwrap();
        storage
            .set_trial_user_attr(tid, "k", serde_json::json!(42))
            .unwrap();
        let trial = storage.get_trial(tid).unwrap();
        assert_eq!(trial.user_attrs.get("k"), Some(&serde_json::json!(42)));
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let storage = Arc::new(InMemoryStorage::new());
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], None)
            .unwrap();

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let s = Arc::clone(&storage);
                thread::spawn(move || {
                    let tid = s.create_new_trial(sid, None).unwrap();
                    s.set_trial_state_values(tid, TrialState::Complete, Some(&[1.0]))
                        .unwrap();
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let trials = storage.get_all_trials(sid, None).unwrap();
        assert_eq!(trials.len(), 10);
        assert!(trials.iter().all(|t| t.state == TrialState::Complete));
    }

    #[test]
    fn test_get_n_trials() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], None)
            .unwrap();
        let t0 = storage.create_new_trial(sid, None).unwrap();
        storage.create_new_trial(sid, None).unwrap();
        storage
            .set_trial_state_values(t0, TrialState::Complete, Some(&[1.0]))
            .unwrap();

        assert_eq!(storage.get_n_trials(sid, None).unwrap(), 2);
        assert_eq!(
            storage
                .get_n_trials(sid, Some(&[TrialState::Complete]))
                .unwrap(),
            1
        );
    }

    #[test]
    fn test_get_all_studies() {
        let storage = InMemoryStorage::new();
        storage
            .create_new_study(&[StudyDirection::Minimize], Some("a"))
            .unwrap();
        storage
            .create_new_study(&[StudyDirection::Maximize], Some("b"))
            .unwrap();

        let studies = storage.get_all_studies().unwrap();
        assert_eq!(studies.len(), 2);
    }

    #[test]
    fn test_template_trial() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], None)
            .unwrap();

        let mut template = FrozenTrial {
            number: 0,
            state: TrialState::Waiting,
            values: None,
            datetime_start: None,
            datetime_complete: None,
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
            trial_id: 0,
        };
        template
            .user_attrs
            .insert("preset".into(), serde_json::json!(true));

        let tid = storage.create_new_trial(sid, Some(&template)).unwrap();
        let trial = storage.get_trial(tid).unwrap();
        assert_eq!(trial.state, TrialState::Waiting);
        assert_eq!(
            trial.user_attrs.get("preset"),
            Some(&serde_json::json!(true))
        );
    }
}
