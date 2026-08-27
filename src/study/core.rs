use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use crate::callbacks::Callback;
use crate::distributions::Distribution;
use crate::error::{Error, Result};
use crate::pruners::Pruner;
use crate::samplers::Sampler;
use crate::storage::Storage;
use crate::study::StudyDirection;
use crate::terminators::Terminator;
use crate::trial::{FrozenTrial, Trial, TrialState};

/// An optimization study.
///
pub struct Study {
    study_name: String,
    study_id: i64,
    storage: Arc<dyn Storage>,
    directions: Vec<StudyDirection>,
    sampler: Arc<dyn Sampler>,
    pruner: Arc<dyn Pruner>,
    stop_flag: AtomicBool,
}

impl Study {
    /// Create a new Study. Prefer using `create_study()` instead.
    pub(crate) fn new(
        study_name: String,
        study_id: i64,
        storage: Arc<dyn Storage>,
        directions: Vec<StudyDirection>,
        sampler: Arc<dyn Sampler>,
        pruner: Arc<dyn Pruner>,
    ) -> Self {
        Self {
            study_name,
            study_id,
            storage,
            directions,
            sampler,
            pruner,
            stop_flag: AtomicBool::new(false),
        }
    }

    // ── Properties ──────────────────────────────────────────────────────

    pub fn study_name(&self) -> &str {
        &self.study_name
    }

    pub fn study_id(&self) -> i64 {
        self.study_id
    }

    pub fn directions(&self) -> &[StudyDirection] {
        &self.directions
    }

    /// Single-objective direction (errors if multi-objective).
    pub fn direction(&self) -> Result<StudyDirection> {
        if self.directions.len() != 1 {
            return Err(Error::ValueError(
                "direction is not supported for multi-objective studies".into(),
            ));
        }
        Ok(self.directions[0])
    }

    pub fn storage(&self) -> &Arc<dyn Storage> {
        &self.storage
    }

    /// Get the best trial for single-objective studies.
    pub fn best_trial(&self) -> Result<FrozenTrial> {
        self.storage.get_best_trial(self.study_id)
    }

    /// Get the Pareto-optimal trials for multi-objective studies.
    pub fn best_trials(&self) -> Result<Vec<FrozenTrial>> {
        let trials = self
            .storage
            .get_all_trials(self.study_id, Some(&[TrialState::Complete]))?;
        Ok(crate::multi_objective::get_pareto_front_trials(
            &trials,
            &self.directions,
        ))
    }

    /// Get the best value for single-objective studies.
    pub fn best_value(&self) -> Result<f64> {
        self.best_trial()?
            .value()?
            .ok_or_else(|| Error::ValueError("best trial has no value".into()))
    }

    /// Get the best params for single-objective studies.
    pub fn best_params(&self) -> Result<HashMap<String, crate::distributions::ParamValue>> {
        Ok(self.best_trial()?.params)
    }

    /// Get all trials.
    pub fn trials(&self) -> Result<Vec<FrozenTrial>> {
        self.storage.get_all_trials(self.study_id, None)
    }

    /// Get trials filtered by state.
    pub fn get_trials(&self, states: Option<&[TrialState]>) -> Result<Vec<FrozenTrial>> {
        self.storage.get_all_trials(self.study_id, states)
    }

    /// Get user attributes.
    pub fn user_attrs(&self) -> Result<HashMap<String, serde_json::Value>> {
        self.storage.get_study_user_attrs(self.study_id)
    }

    /// Set a user attribute.
    pub fn set_user_attr(&self, key: &str, value: serde_json::Value) -> Result<()> {
        self.storage.set_study_user_attr(self.study_id, key, value)
    }

    /// Signal the optimize loop to stop after the current trial.
    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::Release);
    }

    // ── Ask / Tell ──────────────────────────────────────────────────────

    /// Create a new trial and return a mutable handle.
    ///
    /// If `fixed_distributions` is provided, the trial's parameters will
    /// be pre-sampled for those distributions.
    pub fn ask(
        &self,
        fixed_distributions: Option<&HashMap<String, Distribution>>,
    ) -> Result<Trial> {
        // Try to pop a WAITING trial first
        let trial_id = match self.pop_waiting_trial()? {
            Some(tid) => tid,
            None => self.storage.create_new_trial(self.study_id, None)?,
        };

        let frozen = self.storage.get_trial(trial_id)?;
        let all_trials = self.storage.get_all_trials(self.study_id, None)?;

        // Run sampler hooks
        self.sampler.before_trial(&all_trials);

        // Enqueued trials carry their pre-specified parameters in the
        // `fixed_params` system attribute; they take precedence over any
        // sampler-proposed values. Read them up front: whether this trial's
        // point was chosen by the user or by the sampler changes what the
        // sampler should be asked to do below.
        let mut fixed_params: HashMap<String, crate::distributions::ParamValue> = HashMap::new();
        if let Some(json) = frozen.system_attrs.get("fixed_params") {
            fixed_params = serde_json::from_value(json.clone()).map_err(|e| {
                Error::StorageInternalError(format!("corrupt fixed_params attribute: {e}"))
            })?;
        }
        let is_enqueued = !fixed_params.is_empty();

        let mut trial = Trial::new(
            trial_id,
            self.study_id,
            frozen.number,
            Arc::clone(&self.storage),
            Arc::clone(&self.sampler),
            Arc::clone(&self.pruner),
            HashMap::new(),
            fixed_params,
        );

        // Let the sampler annotate the new trial before anything is sampled.
        // Skipped for enqueued trials: the sampler did not choose this point,
        // so reserving a resource for it (a GridSampler grid point, say)
        // would silently consume a candidate that never gets evaluated.
        if !is_enqueued {
            self.sampler.trial_created(&all_trials, &trial)?;
        }

        // Re-read the trial so the sampling below sees whatever the hook
        // recorded on it.
        let frozen = self.storage.get_trial(trial_id)?;

        // Determine search space: fixed_distributions override sampler's relative space
        let search_space = if let Some(fixed) = fixed_distributions {
            fixed.clone()
        } else {
            self.sampler.infer_relative_search_space(&all_trials)
        };

        // Sample relative params. An enqueued trial's values are already
        // decided, so don't ask the sampler to propose a point it would only
        // discard — a stateful sampler would otherwise book credit for a
        // candidate that is never evaluated.
        let mut relative_params = if is_enqueued || search_space.is_empty() {
            HashMap::new()
        } else {
            self.sampler.sample_relative(&all_trials, &search_space)?
        };

        // For any distributions not covered by relative sampling,
        // do independent sampling
        if let Some(fixed) = fixed_distributions
            && !is_enqueued
        {
            for (name, dist) in fixed {
                if !relative_params.contains_key(name) {
                    let v = self
                        .sampler
                        .sample_independent(&all_trials, &frozen, name, dist)?;
                    relative_params.insert(name.clone(), v);
                }
            }
        }

        trial.set_relative_params(relative_params);

        Ok(trial)
    }

    /// Finalize a trial with a value/values and state.
    ///
    /// This is the low-level tell; the optimize loop calls this internally.
    pub fn tell(
        &self,
        trial_id: i64,
        state: TrialState,
        values: Option<&[f64]>,
    ) -> Result<FrozenTrial> {
        // Write to storage
        self.storage
            .set_trial_state_values(trial_id, state, values)?;

        // Run after_trial hook
        let all_trials = self.storage.get_all_trials(self.study_id, None)?;
        let frozen = self.storage.get_trial(trial_id)?;
        self.sampler
            .after_trial(&all_trials, &frozen, state, values);

        Ok(frozen)
    }

    // ── Optimize loop ───────────────────────────────────────────────────

    /// Run the optimization loop.
    ///
    /// `func` is the objective function: given a `&mut Trial`, return the
    /// objective value(s). For single-objective, return a single `f64`.
    /// Raise `Error::TrialPruned` to prune.
    ///
    /// `n_trials`: max number of trials to run (None = unlimited).
    /// `timeout`: max duration (None = unlimited).
    /// `callbacks`: optional callbacks run after each trial.
    pub fn optimize<F>(
        &self,
        func: F,
        n_trials: Option<usize>,
        timeout: Option<Duration>,
        callbacks: Option<&[&dyn Callback]>,
    ) -> Result<()>
    where
        F: Fn(&mut Trial) -> Result<f64>,
    {
        self.stop_flag.store(false, Ordering::Release);
        let start = Instant::now();

        let mut i_trial: usize = 0;
        loop {
            // Check stop conditions
            if self.stop_flag.load(Ordering::Acquire) {
                break;
            }
            if n_trials.is_some_and(|n| i_trial >= n) {
                break;
            }
            if timeout.is_some_and(|t| start.elapsed() >= t) {
                break;
            }

            self.run_trial(&func, callbacks)?;
            i_trial += 1;
        }

        Ok(())
    }

    /// Run the optimization loop with a multi-objective function.
    pub fn optimize_multi<F>(
        &self,
        func: F,
        n_trials: Option<usize>,
        timeout: Option<Duration>,
        callbacks: Option<&[&dyn Callback]>,
    ) -> Result<()>
    where
        F: Fn(&mut Trial) -> Result<Vec<f64>>,
    {
        self.stop_flag.store(false, Ordering::Release);
        let start = Instant::now();

        let mut i_trial: usize = 0;
        loop {
            if self.stop_flag.load(Ordering::Acquire) {
                break;
            }
            if n_trials.is_some_and(|n| i_trial >= n) {
                break;
            }
            if timeout.is_some_and(|t| start.elapsed() >= t) {
                break;
            }

            self.run_trial_multi(&func, callbacks)?;
            i_trial += 1;
        }

        Ok(())
    }

    /// Run the optimization loop with terminator support.
    ///
    /// Like [`optimize`](Self::optimize), but also checks terminators after
    /// each trial. If any terminator returns `true`, optimization stops.
    pub fn optimize_with_terminators<F>(
        &self,
        func: F,
        n_trials: Option<usize>,
        timeout: Option<Duration>,
        callbacks: Option<&[&dyn Callback]>,
        terminators: Option<&[Arc<dyn Terminator>]>,
    ) -> Result<()>
    where
        F: Fn(&mut Trial) -> Result<f64>,
    {
        self.stop_flag.store(false, Ordering::Release);
        let start = Instant::now();

        let mut i_trial: usize = 0;
        loop {
            if self.stop_flag.load(Ordering::Acquire) {
                break;
            }
            if n_trials.is_some_and(|n| i_trial >= n) {
                break;
            }
            if timeout.is_some_and(|t| start.elapsed() >= t) {
                break;
            }
            if let Some(terms) = terminators
                && terms.iter().any(|t| t.should_terminate(self))
            {
                break;
            }

            self.run_trial(&func, callbacks)?;
            i_trial += 1;
        }

        Ok(())
    }

    /// Run the multi-objective optimization loop with terminator support.
    pub fn optimize_multi_with_terminators<F>(
        &self,
        func: F,
        n_trials: Option<usize>,
        timeout: Option<Duration>,
        callbacks: Option<&[&dyn Callback]>,
        terminators: Option<&[Arc<dyn Terminator>]>,
    ) -> Result<()>
    where
        F: Fn(&mut Trial) -> Result<Vec<f64>>,
    {
        self.stop_flag.store(false, Ordering::Release);
        let start = Instant::now();

        let mut i_trial: usize = 0;
        loop {
            if self.stop_flag.load(Ordering::Acquire) {
                break;
            }
            if n_trials.is_some_and(|n| i_trial >= n) {
                break;
            }
            if timeout.is_some_and(|t| start.elapsed() >= t) {
                break;
            }
            if let Some(terms) = terminators
                && terms.iter().any(|t| t.should_terminate(self))
            {
                break;
            }

            self.run_trial_multi(&func, callbacks)?;
            i_trial += 1;
        }

        Ok(())
    }

    /// Describe any enqueued parameters the objective never suggested.
    fn unconsumed_enqueued_reason(&self, trial: &Trial) -> Result<Option<String>> {
        let unconsumed = trial.unconsumed_fixed_params()?;
        if unconsumed.is_empty() {
            return Ok(None);
        }
        Ok(Some(format!(
            "enqueued parameter(s) never suggested by the objective: {}",
            unconsumed.join(", ")
        )))
    }

    /// Execute a single trial (single-objective).
    fn run_trial<F>(&self, func: &F, callbacks: Option<&[&dyn Callback]>) -> Result<()>
    where
        F: Fn(&mut Trial) -> Result<f64>,
    {
        let mut trial = self.ask(None)?;
        let trial_id = trial.trial_id();

        let (state, values) = match func(&mut trial) {
            Ok(value) => {
                // Validate the value
                if value.is_nan() {
                    let _ = self.storage.set_trial_system_attr(
                        trial_id,
                        "failure_reason",
                        serde_json::json!("objective returned NaN"),
                    );
                    (TrialState::Fail, None)
                } else if let Some(reason) = self.unconsumed_enqueued_reason(&trial)? {
                    let _ = self.storage.set_trial_system_attr(
                        trial_id,
                        "failure_reason",
                        serde_json::json!(reason),
                    );
                    (TrialState::Fail, None)
                } else {
                    (TrialState::Complete, Some(vec![value]))
                }
            }
            Err(Error::TrialPruned) => {
                // Use last intermediate value if available
                let frozen = self.storage.get_trial(trial_id)?;
                let last_value = frozen
                    .last_step()
                    .and_then(|step| frozen.intermediate_values.get(&step))
                    .copied()
                    .filter(|v| v.is_finite());
                (TrialState::Pruned, last_value.map(|v| vec![v]))
            }
            Err(e) => {
                // Record why the trial failed so it is not silently lost.
                let _ = self.storage.set_trial_system_attr(
                    trial_id,
                    "failure_reason",
                    serde_json::json!(e.to_string()),
                );
                (TrialState::Fail, None)
            }
        };

        let frozen = self.tell(trial_id, state, values.as_deref())?;

        // Run callbacks
        if let Some(cbs) = callbacks {
            let n_complete = self
                .storage
                .get_n_trials(self.study_id, Some(&[TrialState::Complete]))?;
            for cb in cbs {
                cb.on_trial_complete(n_complete, &frozen);
            }
        }

        Ok(())
    }

    /// Execute a single trial (multi-objective).
    fn run_trial_multi<F>(&self, func: &F, callbacks: Option<&[&dyn Callback]>) -> Result<()>
    where
        F: Fn(&mut Trial) -> Result<Vec<f64>>,
    {
        let mut trial = self.ask(None)?;
        let trial_id = trial.trial_id();

        let (state, values) = match func(&mut trial) {
            Ok(vals) => {
                if let Some(reason) = self.unconsumed_enqueued_reason(&trial)? {
                    let _ = self.storage.set_trial_system_attr(
                        trial_id,
                        "failure_reason",
                        serde_json::json!(reason),
                    );
                    (TrialState::Fail, None)
                } else if vals.iter().any(|v| v.is_nan()) || vals.len() != self.directions.len() {
                    let reason: std::borrow::Cow<'_, str> = if vals.iter().any(|v| v.is_nan()) {
                        "objective returned NaN".into()
                    } else {
                        format!(
                            "expected {} objective values, got {}",
                            self.directions.len(),
                            vals.len()
                        )
                        .into()
                    };
                    let _ = self.storage.set_trial_system_attr(
                        trial_id,
                        "failure_reason",
                        serde_json::json!(&*reason),
                    );
                    (TrialState::Fail, None)
                } else {
                    (TrialState::Complete, Some(vals))
                }
            }
            Err(Error::TrialPruned) => {
                // A single intermediate value cannot stand in for a vector of
                // objectives; storing one produced a trial whose value arity
                // disagreed with the study's, which then tripped up anything
                // comparing objective vectors. The intermediate values are
                // already recorded on the trial.
                (TrialState::Pruned, None)
            }
            Err(e) => {
                // Record why the trial failed so it is not silently lost.
                let _ = self.storage.set_trial_system_attr(
                    trial_id,
                    "failure_reason",
                    serde_json::json!(e.to_string()),
                );
                (TrialState::Fail, None)
            }
        };

        let frozen = self.tell(trial_id, state, values.as_deref())?;

        if let Some(cbs) = callbacks {
            let n_complete = self
                .storage
                .get_n_trials(self.study_id, Some(&[TrialState::Complete]))?;
            for cb in cbs {
                cb.on_trial_complete(n_complete, &frozen);
            }
        }

        Ok(())
    }

    /// Try to pop a WAITING trial and set it to RUNNING.
    fn pop_waiting_trial(&self) -> Result<Option<i64>> {
        let waiting = self
            .storage
            .get_all_trials(self.study_id, Some(&[TrialState::Waiting]))?;
        if let Some(t) = waiting.first() {
            let ok = self
                .storage
                .set_trial_state_values(t.trial_id, TrialState::Running, None)?;
            if ok {
                return Ok(Some(t.trial_id));
            }
        }
        Ok(None)
    }
}

/// Add a trial to a study directly.
impl Study {
    pub fn add_trial(&self, trial: &FrozenTrial) -> Result<()> {
        self.storage.create_new_trial(self.study_id, Some(trial))?;
        Ok(())
    }

    pub fn add_trials(&self, trials: &[FrozenTrial]) -> Result<()> {
        for trial in trials {
            self.add_trial(trial)?;
        }
        Ok(())
    }

    pub fn enqueue_trial(
        &self,
        params: HashMap<String, crate::distributions::ParamValue>,
        user_attrs: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<()> {
        let mut template = FrozenTrial {
            number: 0,
            state: TrialState::Waiting,
            values: None,
            datetime_start: None,
            datetime_complete: None,
            params,
            distributions: HashMap::new(),
            user_attrs: user_attrs.unwrap_or_default(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
            trial_id: 0,
        };
        // Distributions will be set when the trial is actually run
        // Clear params for WAITING trials (they'll be suggested later)
        let enqueued_params = template.params.clone();
        template.params = HashMap::new();
        template.system_attrs.insert(
            "fixed_params".to_string(),
            serde_json::to_value(enqueued_params)
                .map_err(|e| Error::StorageInternalError(e.to_string()))?,
        );
        self.storage
            .create_new_trial(self.study_id, Some(&template))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::InMemoryStorage;
    use crate::study::create_study;

    #[test]
    fn test_create_study_default() {
        let study = create_study(None, None, None, None, None, None, false).unwrap();
        assert_eq!(study.directions(), &[StudyDirection::Minimize]);
    }

    #[test]
    fn test_create_study_named() {
        let study = create_study(None, None, None, Some("my-study"), None, None, false).unwrap();
        assert_eq!(study.study_name(), "my-study");
    }

    #[test]
    fn test_create_study_maximize() {
        let study = create_study(
            None,
            None,
            None,
            None,
            Some(StudyDirection::Maximize),
            None,
            false,
        )
        .unwrap();
        assert_eq!(study.direction().unwrap(), StudyDirection::Maximize);
    }

    #[test]
    fn test_create_study_both_directions_errors() {
        assert!(
            create_study(
                None,
                None,
                None,
                None,
                Some(StudyDirection::Minimize),
                Some(vec![StudyDirection::Maximize]),
                false,
            )
            .is_err()
        );
    }

    #[test]
    fn test_create_study_load_if_exists() {
        let storage: Arc<dyn Storage> = Arc::new(InMemoryStorage::new());
        let _s1 = create_study(
            Some(Arc::clone(&storage)),
            None,
            None,
            Some("dup"),
            None,
            None,
            false,
        )
        .unwrap();
        // Without load_if_exists, duplicate errors
        assert!(
            create_study(
                Some(Arc::clone(&storage)),
                None,
                None,
                Some("dup"),
                None,
                None,
                false,
            )
            .is_err()
        );
        // With load_if_exists, succeeds
        let s2 = create_study(
            Some(Arc::clone(&storage)),
            None,
            None,
            Some("dup"),
            None,
            None,
            true,
        )
        .unwrap();
        assert_eq!(s2.study_name(), "dup");
    }

    #[test]
    fn test_optimize_quadratic() {
        let sampler: Arc<dyn Sampler> = Arc::new(crate::samplers::RandomSampler::new(Some(42)));
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
                    let y = trial.suggest_float("y", -10.0, 10.0, false, None)?;
                    Ok(x * x + y * y)
                },
                Some(50),
                None,
                None,
            )
            .unwrap();

        let trials = study.trials().unwrap();
        assert_eq!(trials.len(), 50);
        assert!(trials.iter().all(|t| t.state == TrialState::Complete));

        let best = study.best_value().unwrap();
        // With 50 random trials in [-10, 10]^2, best should be reasonably small
        assert!(best < 50.0, "best value {best} is too large");
    }

    #[test]
    fn test_optimize_maximize() {
        let sampler: Arc<dyn Sampler> = Arc::new(crate::samplers::RandomSampler::new(Some(42)));
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
                Some(20),
                None,
                None,
            )
            .unwrap();

        let best = study.best_value().unwrap();
        assert!(best > 0.5, "best value {best} should be > 0.5 for maximize");
    }

    #[test]
    fn test_optimize_with_pruning() {
        let sampler: Arc<dyn Sampler> = Arc::new(crate::samplers::RandomSampler::new(Some(42)));
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
                    trial.report(x * x, 0)?;
                    if x.abs() > 5.0 {
                        return Err(Error::TrialPruned);
                    }
                    Ok(x * x)
                },
                Some(30),
                None,
                None,
            )
            .unwrap();

        let trials = study.trials().unwrap();
        assert_eq!(trials.len(), 30);
        let n_pruned = trials
            .iter()
            .filter(|t| t.state == TrialState::Pruned)
            .count();
        let n_complete = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .count();
        assert!(n_pruned > 0, "expected some pruned trials");
        assert!(n_complete > 0, "expected some complete trials");
        assert_eq!(n_pruned + n_complete, 30);
    }

    #[test]
    fn test_ask_tell_lifecycle() {
        let sampler: Arc<dyn Sampler> = Arc::new(crate::samplers::RandomSampler::new(Some(42)));
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

        let mut trial = study.ask(None).unwrap();
        let x = trial.suggest_float("x", 0.0, 1.0, false, None).unwrap();
        let value = x * x;

        let frozen = study
            .tell(trial.trial_id(), TrialState::Complete, Some(&[value]))
            .unwrap();
        assert_eq!(frozen.state, TrialState::Complete);
        assert_eq!(frozen.values, Some(vec![value]));
    }

    #[test]
    fn test_enqueue_trial_uses_enqueued_params() {
        let sampler: Arc<dyn Sampler> = Arc::new(crate::samplers::RandomSampler::new(Some(42)));
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

        // Enqueue a specific parameter setting; it must run first, verbatim.
        let mut params = HashMap::new();
        params.insert(
            "x".to_string(),
            crate::distributions::ParamValue::Float(0.5),
        );
        params.insert(
            "y".to_string(),
            crate::distributions::ParamValue::Float(-2.0),
        );
        study.enqueue_trial(params, None).unwrap();

        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", -10.0, 10.0, false, None)?;
                    let y = trial.suggest_float("y", -10.0, 10.0, false, None)?;
                    Ok(x * x + y * y)
                },
                Some(3),
                None,
                None,
            )
            .unwrap();

        let trials = study.trials().unwrap();
        assert_eq!(trials.len(), 3);

        // The enqueued trial runs first and uses the exact values given.
        let first = &trials[0];
        assert_eq!(
            first.params.get("x"),
            Some(&crate::distributions::ParamValue::Float(0.5))
        );
        assert_eq!(
            first.params.get("y"),
            Some(&crate::distributions::ParamValue::Float(-2.0))
        );
        assert_eq!(first.values.as_ref().unwrap()[0], 0.5 * 0.5 + 4.0);
    }

    #[test]
    fn test_failed_trial_records_failure_reason() {
        let sampler: Arc<dyn Sampler> = Arc::new(crate::samplers::RandomSampler::new(Some(42)));
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
                    if x < 0.5 {
                        Err(Error::ValueError("boom".into()))
                    } else {
                        Ok(x)
                    }
                },
                Some(10),
                None,
                None,
            )
            .unwrap();

        let trials = study.trials().unwrap();
        let failed: Vec<&FrozenTrial> = trials
            .iter()
            .filter(|t| t.state == TrialState::Fail)
            .collect();
        assert!(!failed.is_empty(), "some trials should fail");
        for t in &failed {
            assert_eq!(
                t.system_attrs
                    .get("failure_reason")
                    .and_then(|v| v.as_str()),
                Some("boom"),
                "failed trials must record why they failed"
            );
        }
    }

    #[test]
    fn test_study_stop() {
        let sampler: Arc<dyn Sampler> = Arc::new(crate::samplers::RandomSampler::new(Some(42)));
        let study = Arc::new(
            create_study(
                None,
                Some(sampler),
                None,
                None,
                Some(StudyDirection::Minimize),
                None,
                false,
            )
            .unwrap(),
        );

        // Stop after first trial via callback-like logic
        // We'll test the stop flag directly
        let study_ref = Arc::clone(&study);
        std::thread::spawn(move || {
            // Wait a tiny bit then stop
            std::thread::sleep(std::time::Duration::from_millis(10));
            study_ref.stop();
        });

        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    Ok(x * x)
                },
                Some(10000), // Would take a long time without stop
                None,
                None,
            )
            .unwrap();

        // Should have been stopped early
        let n_trials = study.trials().unwrap().len();
        assert!(
            n_trials < 10000,
            "study should have stopped early, got {n_trials} trials"
        );
    }

    #[test]
    fn test_optimize_with_int_param() {
        let sampler: Arc<dyn Sampler> = Arc::new(crate::samplers::RandomSampler::new(Some(42)));
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
                    let n = trial.suggest_int("n_layers", 1, 5, false, 1)?;
                    Ok((n as f64 - 3.0).powi(2))
                },
                Some(20),
                None,
                None,
            )
            .unwrap();

        let best = study.best_trial().unwrap();
        assert_eq!(best.state, TrialState::Complete);
    }

    #[test]
    fn test_optimize_timeout() {
        let sampler: Arc<dyn Sampler> = Arc::new(crate::samplers::RandomSampler::new(Some(42)));
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

        let start = std::time::Instant::now();
        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    Ok(x * x)
                },
                None, // unlimited trials
                Some(std::time::Duration::from_millis(100)),
                None,
            )
            .unwrap();

        let elapsed = start.elapsed();
        assert!(elapsed < std::time::Duration::from_secs(2));
        assert!(!study.trials().unwrap().is_empty());
    }

    #[test]
    fn test_best_params() {
        let sampler: Arc<dyn Sampler> = Arc::new(crate::samplers::RandomSampler::new(Some(42)));
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
                    let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                    Ok(x * x)
                },
                Some(30),
                None,
                None,
            )
            .unwrap();

        let params = study.best_params().unwrap();
        assert!(params.contains_key("x"));
    }

    #[test]
    fn test_study_user_attrs() {
        let study = create_study(None, None, None, None, None, None, false).unwrap();
        study
            .set_user_attr("key", serde_json::json!("value"))
            .unwrap();
        let attrs = study.user_attrs().unwrap();
        assert_eq!(attrs.get("key"), Some(&serde_json::json!("value")));
    }

    #[test]
    fn test_optimize_with_median_pruner() {
        let sampler: Arc<dyn Sampler> = Arc::new(crate::samplers::RandomSampler::new(Some(42)));
        let pruner: Arc<dyn crate::pruners::Pruner> = Arc::new(crate::pruners::MedianPruner::new(
            3,
            0,
            1,
            1,
            StudyDirection::Minimize,
        ));
        let study = create_study(
            None,
            Some(sampler),
            Some(pruner),
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
                    for step in 0..5 {
                        let v = x * x + step as f64;
                        trial.report(v, step)?;
                        if trial.should_prune()? {
                            return Err(Error::TrialPruned);
                        }
                    }
                    Ok(x * x)
                },
                Some(50),
                None,
                None,
            )
            .unwrap();

        let trials = study.trials().unwrap();
        assert_eq!(trials.len(), 50);
        let n_pruned = trials
            .iter()
            .filter(|t| t.state == TrialState::Pruned)
            .count();
        let n_complete = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .count();
        assert!(n_complete > 0, "expected some complete trials");
        // With median pruner active after 3 startup trials, some should be pruned
        assert!(
            n_pruned > 0,
            "expected some pruned trials with median pruner, got {n_pruned} pruned / {n_complete} complete"
        );
    }

    #[test]
    fn test_end_to_end_with_search_space_transform() {
        // Verify the full pipeline: RandomSampler -> SearchSpaceTransform -> optimize
        // with mixed param types
        let sampler: Arc<dyn Sampler> = Arc::new(crate::samplers::RandomSampler::new(Some(99)));
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
                    let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                    let n = trial.suggest_int("n", 1, 10, false, 1)?;
                    Ok(x * x + (n as f64 - 5.0).powi(2))
                },
                Some(40),
                None,
                None,
            )
            .unwrap();

        let best = study.best_trial().unwrap();
        assert_eq!(best.state, TrialState::Complete);
        assert!(best.params.contains_key("x"));
        assert!(best.params.contains_key("n"));
        // With 40 random trials, should find something reasonable
        let best_val = study.best_value().unwrap();
        assert!(best_val < 30.0, "best value {best_val} should be < 30");
    }

    #[test]
    fn test_best_trial_keeps_infinite_objective() {
        // Regression: filtering to finite values silently dropped the true
        // optimum when an infinite penalty was the best (or only) score.
        let sampler: Arc<dyn Sampler> = Arc::new(crate::samplers::RandomSampler::new(Some(42)));
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

        let values = [1.0, 1.0, f64::INFINITY, 1.0];
        let i = std::sync::atomic::AtomicUsize::new(0);
        study
            .optimize(
                |trial| {
                    let _x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    Ok(values[i.fetch_add(1, std::sync::atomic::Ordering::SeqCst)])
                },
                Some(4),
                None,
                None,
            )
            .unwrap();

        assert_eq!(study.best_trial().unwrap().number, 2);
        assert_eq!(study.best_value().unwrap(), f64::INFINITY);
    }

    #[test]
    fn test_enqueued_param_never_suggested_fails_trial() {
        // An enqueued trial is a request to evaluate one specific
        // configuration. If a name is never suggested — a typo, say — that
        // configuration was not what ran, and the trial must say so instead
        // of silently completing as an ordinary sampled trial.
        let sampler: Arc<dyn Sampler> = Arc::new(crate::samplers::RandomSampler::new(Some(42)));
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

        let mut params = HashMap::new();
        params.insert(
            "typo".to_string(),
            crate::distributions::ParamValue::Float(0.5),
        );
        study.enqueue_trial(params, None).unwrap();

        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    Ok(x)
                },
                Some(1),
                None,
                None,
            )
            .unwrap();

        let first = &study.trials().unwrap()[0];
        assert_eq!(first.state, TrialState::Fail);
        assert_eq!(
            first
                .system_attrs
                .get("failure_reason")
                .and_then(|v| v.as_str()),
            Some("enqueued parameter(s) never suggested by the objective: typo")
        );
    }

    #[test]
    fn test_enqueued_non_integral_float_for_int_param_fails() {
        // Regression: Float(2.7) for an integer parameter was truncated to 2
        // and reported as a success, evaluating a different configuration
        // than the one enqueued.
        let sampler: Arc<dyn Sampler> = Arc::new(crate::samplers::RandomSampler::new(Some(42)));
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

        let mut params = HashMap::new();
        params.insert(
            "x".to_string(),
            crate::distributions::ParamValue::Float(2.7),
        );
        study.enqueue_trial(params, None).unwrap();

        study
            .optimize(
                |trial| {
                    let x = trial.suggest_int("x", 0, 10, false, 1)?;
                    Ok(x as f64)
                },
                Some(1),
                None,
                None,
            )
            .unwrap();

        let first = &study.trials().unwrap()[0];
        assert_eq!(first.state, TrialState::Fail);
        assert!(
            first
                .system_attrs
                .get("failure_reason")
                .and_then(|v| v.as_str())
                .unwrap_or_default()
                .contains("enqueued value for parameter 'x'"),
            "reason should point at the enqueued value: {:?}",
            first.system_attrs.get("failure_reason")
        );
    }

    #[test]
    fn test_stepped_distributions_with_off_grid_high_complete() {
        // Regression: samplers clamp to `high`, but for a step whose grid
        // does not land on `high` that clamp target was not a value the
        // distribution contained, so storage rejected it and trials failed
        // silently — `optimize` still returned Ok.
        let samplers: Vec<(&str, Arc<dyn Sampler>)> = vec![
            (
                "random",
                Arc::new(crate::samplers::RandomSampler::new(Some(42))),
            ),
            (
                "tpe",
                Arc::new(crate::samplers::TpeSampler::with_defaults(
                    StudyDirection::Minimize,
                    Some(42),
                )),
            ),
            (
                "qmc",
                Arc::new(crate::samplers::QmcSampler::new(None, Some(42))),
            ),
        ];

        for (name, sampler) in samplers {
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
                        // Neither grid lands on its declared `high`.
                        let x = trial.suggest_float("x", 0.0, 1.0, false, Some(0.3))?;
                        let n = trial.suggest_int("n", 0, 10, false, 3)?;
                        Ok(-(x + n as f64))
                    },
                    Some(60),
                    None,
                    None,
                )
                .unwrap();

            let failed: Vec<_> = study
                .trials()
                .unwrap()
                .into_iter()
                .filter(|t| t.state == TrialState::Fail)
                .collect();
            assert!(
                failed.is_empty(),
                "{name}: {} trials failed, e.g. {:?}",
                failed.len(),
                failed[0].system_attrs.get("failure_reason")
            );
        }
    }

    #[test]
    fn test_multi_objective_pruned_trial_stores_no_values() {
        // Regression: a pruned multi-objective trial stored a single
        // intermediate value, producing a trial whose value arity disagreed
        // with the study's — which then tripped up objective-vector
        // comparisons on public API.
        let sampler: Arc<dyn Sampler> = Arc::new(crate::samplers::RandomSampler::new(Some(42)));
        let study = create_study(
            None,
            Some(sampler),
            None,
            None,
            None,
            Some(vec![StudyDirection::Minimize, StudyDirection::Minimize]),
            false,
        )
        .unwrap();

        study
            .optimize_multi(
                |trial| {
                    let _x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    trial.report(0.5, 0)?;
                    Err(Error::TrialPruned)
                },
                Some(2),
                None,
                None,
            )
            .unwrap();

        for t in study.trials().unwrap() {
            assert_eq!(t.state, TrialState::Pruned);
            assert_eq!(t.values, None, "a pruned MO trial must carry no values");
            // The intermediate report is still there.
            assert_eq!(t.intermediate_values.get(&0), Some(&0.5));
        }
    }
}
