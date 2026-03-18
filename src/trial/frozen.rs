use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::distributions::{Distribution, ParamValue};
use crate::error::{Error, Result};
use crate::trial::TrialState;

/// An immutable snapshot of a trial's state.
///
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenTrial {
    pub number: i64,
    pub state: TrialState,
    pub values: Option<Vec<f64>>,
    pub datetime_start: Option<DateTime<Utc>>,
    pub datetime_complete: Option<DateTime<Utc>>,
    pub params: HashMap<String, ParamValue>,
    pub distributions: HashMap<String, Distribution>,
    pub user_attrs: HashMap<String, serde_json::Value>,
    pub system_attrs: HashMap<String, serde_json::Value>,
    pub intermediate_values: HashMap<i64, f64>,
    pub trial_id: i64,
}

impl FrozenTrial {
    /// Create a new `FrozenTrial`.
    ///
    /// Either `value` or `values` may be provided, but not both.
    /// A single `value` is stored as `Some(vec![value])`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        number: i64,
        state: TrialState,
        value: Option<f64>,
        values: Option<Vec<f64>>,
        datetime_start: Option<DateTime<Utc>>,
        datetime_complete: Option<DateTime<Utc>>,
        params: HashMap<String, ParamValue>,
        distributions: HashMap<String, Distribution>,
        user_attrs: HashMap<String, serde_json::Value>,
        system_attrs: HashMap<String, serde_json::Value>,
        intermediate_values: HashMap<i64, f64>,
        trial_id: i64,
    ) -> Result<Self> {
        if value.is_some() && values.is_some() {
            return Err(Error::ValueError(
                "specify either `value` or `values`, not both".into(),
            ));
        }

        let merged_values = if let Some(v) = value {
            Some(vec![v])
        } else {
            values
        };

        let trial = Self {
            number,
            state,
            values: merged_values,
            datetime_start,
            datetime_complete,
            params,
            distributions,
            user_attrs,
            system_attrs,
            intermediate_values,
            trial_id,
        };

        trial.validate()?;
        Ok(trial)
    }

    /// Single-objective accessor. Returns the value if there is exactly one.
    pub fn value(&self) -> Result<Option<f64>> {
        match &self.values {
            None => Ok(None),
            Some(vs) if vs.len() == 1 => Ok(Some(vs[0])),
            Some(vs) => Err(Error::ValueError(format!(
                "trial has {} values; use `values` for multi-objective",
                vs.len()
            ))),
        }
    }

    /// The last intermediate value step, if any.
    pub fn last_step(&self) -> Option<i64> {
        self.intermediate_values.keys().max().copied()
    }

    /// Duration of the trial (complete_time - start_time).
    pub fn duration(&self) -> Option<chrono::Duration> {
        match (self.datetime_start, self.datetime_complete) {
            (Some(start), Some(end)) => Some(end - start),
            _ => None,
        }
    }

    /// Validate the invariants of this trial.
    pub fn validate(&self) -> Result<()> {
        // 1. Non-waiting trials must have a start time.
        if self.state != TrialState::Waiting && self.datetime_start.is_none() {
            return Err(Error::ValueError(format!(
                "trial {} has state {} but no datetime_start",
                self.number, self.state
            )));
        }

        // 2/3. Finished trials must have complete time; non-finished must not.
        if self.state.is_finished() && self.datetime_complete.is_none() {
            return Err(Error::ValueError(format!(
                "trial {} is finished ({}) but has no datetime_complete",
                self.number, self.state
            )));
        }
        if !self.state.is_finished() && self.datetime_complete.is_some() {
            return Err(Error::ValueError(format!(
                "trial {} is not finished ({}) but has datetime_complete",
                self.number, self.state
            )));
        }

        // 4. Failed trials must not have values.
        if self.state == TrialState::Fail && self.values.is_some() {
            return Err(Error::ValueError(format!(
                "trial {} is FAIL but has values",
                self.number
            )));
        }

        // 5. Completed trials must have values with no NaN.
        if self.state == TrialState::Complete {
            match &self.values {
                None => {
                    return Err(Error::ValueError(format!(
                        "trial {} is COMPLETE but has no values",
                        self.number
                    )));
                }
                Some(vs) => {
                    if vs.iter().any(|v| v.is_nan()) {
                        return Err(Error::ValueError(format!(
                            "trial {} is COMPLETE but contains NaN values",
                            self.number
                        )));
                    }
                }
            }
        }

        // 6. params and distributions must have the same keys.
        if self.params.len() != self.distributions.len() {
            return Err(Error::ValueError(format!(
                "trial {} has {} params but {} distributions",
                self.number,
                self.params.len(),
                self.distributions.len()
            )));
        }
        for key in self.params.keys() {
            if !self.distributions.contains_key(key) {
                return Err(Error::ValueError(format!(
                    "trial {}: param '{}' has no matching distribution",
                    self.number, key
                )));
            }
        }

        // 7. Each param value must be contained in its distribution.
        for (name, value) in &self.params {
            let dist = &self.distributions[name];
            let internal = dist.to_internal_repr(value)?;
            if !dist.contains(internal) {
                return Err(Error::ValueError(format!(
                    "trial {}: param '{}' value is not contained in its distribution",
                    self.number, name
                )));
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::{FloatDistribution, IntDistribution};

    fn empty_maps() -> (
        HashMap<String, ParamValue>,
        HashMap<String, Distribution>,
        HashMap<String, serde_json::Value>,
        HashMap<String, serde_json::Value>,
        HashMap<i64, f64>,
    ) {
        (
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        )
    }

    #[test]
    fn test_complete_trial() {
        let now = Utc::now();
        let (params, dists, ua, sa, iv) = empty_maps();
        let trial = FrozenTrial::new(
            0,
            TrialState::Complete,
            Some(1.0),
            None,
            Some(now),
            Some(now),
            params,
            dists,
            ua,
            sa,
            iv,
            0,
        )
        .unwrap();
        assert_eq!(trial.value().unwrap(), Some(1.0));
        assert_eq!(trial.values, Some(vec![1.0]));
    }

    #[test]
    fn test_value_and_values_both_provided() {
        let now = Utc::now();
        let (params, dists, ua, sa, iv) = empty_maps();
        assert!(FrozenTrial::new(
            0,
            TrialState::Complete,
            Some(1.0),
            Some(vec![2.0]),
            Some(now),
            Some(now),
            params,
            dists,
            ua,
            sa,
            iv,
            0,
        )
        .is_err());
    }

    #[test]
    fn test_complete_without_values_rejected() {
        let now = Utc::now();
        let (params, dists, ua, sa, iv) = empty_maps();
        assert!(FrozenTrial::new(
            0, TrialState::Complete, None, None, Some(now), Some(now), params, dists, ua, sa,
            iv, 0,
        )
        .is_err());
    }

    #[test]
    fn test_fail_with_values_rejected() {
        let now = Utc::now();
        let (params, dists, ua, sa, iv) = empty_maps();
        assert!(FrozenTrial::new(
            0,
            TrialState::Fail,
            Some(1.0),
            None,
            Some(now),
            Some(now),
            params,
            dists,
            ua,
            sa,
            iv,
            0,
        )
        .is_err());
    }

    #[test]
    fn test_running_without_start_rejected() {
        let (params, dists, ua, sa, iv) = empty_maps();
        assert!(FrozenTrial::new(
            0,
            TrialState::Running,
            None,
            None,
            None,
            None,
            params,
            dists,
            ua,
            sa,
            iv,
            0,
        )
        .is_err());
    }

    #[test]
    fn test_waiting_without_start_ok() {
        let (params, dists, ua, sa, iv) = empty_maps();
        let trial = FrozenTrial::new(
            0,
            TrialState::Waiting,
            None,
            None,
            None,
            None,
            params,
            dists,
            ua,
            sa,
            iv,
            0,
        )
        .unwrap();
        assert_eq!(trial.state, TrialState::Waiting);
    }

    #[test]
    fn test_param_distribution_mismatch() {
        let now = Utc::now();
        let mut params = HashMap::new();
        params.insert("x".into(), ParamValue::Float(0.5));
        let (_, dists, ua, sa, iv) = empty_maps();
        // params has "x" but dists is empty
        assert!(FrozenTrial::new(
            0,
            TrialState::Complete,
            Some(1.0),
            None,
            Some(now),
            Some(now),
            params,
            dists,
            ua,
            sa,
            iv,
            0,
        )
        .is_err());
    }

    #[test]
    fn test_param_value_out_of_range() {
        let now = Utc::now();
        let mut params = HashMap::new();
        params.insert("x".into(), ParamValue::Float(2.0)); // out of [0, 1]
        let mut dists = HashMap::new();
        dists.insert(
            "x".into(),
            Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap()),
        );
        let (_, _, ua, sa, iv) = empty_maps();
        assert!(FrozenTrial::new(
            0,
            TrialState::Complete,
            Some(1.0),
            None,
            Some(now),
            Some(now),
            params,
            dists,
            ua,
            sa,
            iv,
            0,
        )
        .is_err());
    }

    #[test]
    fn test_with_params() {
        let now = Utc::now();
        let mut params = HashMap::new();
        params.insert("x".into(), ParamValue::Float(0.5));
        params.insert("n".into(), ParamValue::Int(3));
        let mut dists = HashMap::new();
        dists.insert(
            "x".into(),
            Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap()),
        );
        dists.insert(
            "n".into(),
            Distribution::IntDistribution(IntDistribution::new(1, 10, false, 1).unwrap()),
        );
        let (_, _, ua, sa, iv) = empty_maps();
        let trial = FrozenTrial::new(
            0,
            TrialState::Complete,
            Some(42.0),
            None,
            Some(now),
            Some(now),
            params,
            dists,
            ua,
            sa,
            iv,
            0,
        )
        .unwrap();
        assert_eq!(trial.params.len(), 2);
    }

    #[test]
    fn test_last_step() {
        let now = Utc::now();
        let (params, dists, ua, sa, _) = empty_maps();
        let mut iv = HashMap::new();
        iv.insert(0, 0.5);
        iv.insert(5, 0.3);
        iv.insert(2, 0.4);
        let trial = FrozenTrial::new(
            0,
            TrialState::Running,
            None,
            None,
            Some(now),
            None,
            params,
            dists,
            ua,
            sa,
            iv,
            0,
        )
        .unwrap();
        assert_eq!(trial.last_step(), Some(5));
    }

    #[test]
    fn test_nan_values_rejected() {
        let now = Utc::now();
        let (params, dists, ua, sa, iv) = empty_maps();
        assert!(FrozenTrial::new(
            0,
            TrialState::Complete,
            Some(f64::NAN),
            None,
            Some(now),
            Some(now),
            params,
            dists,
            ua,
            sa,
            iv,
            0,
        )
        .is_err());
    }
}
