use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// A distribution over floating-point values.
///
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FloatDistribution {
    pub low: f64,
    pub high: f64,
    #[serde(default)]
    pub log: bool,
    #[serde(default)]
    pub step: Option<f64>,
}

impl FloatDistribution {
    /// Create a new `FloatDistribution` with validation.
    pub fn new(low: f64, high: f64, log: bool, step: Option<f64>) -> Result<Self> {
        if log && step.is_some() {
            return Err(Error::InvalidDistribution(
                "cannot combine log-scale with discretization step".into(),
            ));
        }
        if low > high {
            return Err(Error::InvalidDistribution(format!(
                "low must be <= high, got low={low}, high={high}"
            )));
        }
        if log && low <= 0.0 {
            return Err(Error::InvalidDistribution(format!(
                "low must be > 0 for log-scale, got low={low}"
            )));
        }
        if let Some(s) = step
            && s <= 0.0
        {
            return Err(Error::InvalidDistribution(format!(
                "step must be > 0, got step={s}"
            )));
        }
        Ok(Self {
            low,
            high,
            log,
            step,
        })
    }

    /// Check if `value` (in internal representation) is contained in this distribution.
    pub fn contains(&self, value: f64) -> bool {
        if value < self.low || value > self.high {
            return false;
        }
        if let Some(step) = self.step {
            let k = (value - self.low) / step;
            (k - k.round()).abs() < 1.0e-8
        } else {
            true
        }
    }

    /// Convert an external value to internal representation (f64).
    pub fn to_internal_repr(&self, value: f64) -> Result<f64> {
        if value.is_nan() {
            return Err(Error::ValueError("NaN is not allowed".into()));
        }
        if self.log && value <= 0.0 {
            return Err(Error::ValueError(format!(
                "value must be > 0 for log-scale, got {value}"
            )));
        }
        Ok(value)
    }

    /// Convert an internal representation back to an external value.
    pub fn to_external_repr(&self, value: f64) -> f64 {
        value
    }

    /// True if this distribution contains exactly one value.
    pub fn single(&self) -> bool {
        self.low == self.high
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_distribution() {
        let d = FloatDistribution::new(0.0, 1.0, false, None).unwrap();
        assert_eq!(d.low, 0.0);
        assert_eq!(d.high, 1.0);
        assert!(!d.log);
        assert!(d.step.is_none());
    }

    #[test]
    fn test_log_with_step_rejected() {
        assert!(FloatDistribution::new(0.1, 1.0, true, Some(0.1)).is_err());
    }

    #[test]
    fn test_low_greater_than_high() {
        assert!(FloatDistribution::new(2.0, 1.0, false, None).is_err());
    }

    #[test]
    fn test_log_non_positive_low() {
        assert!(FloatDistribution::new(0.0, 1.0, true, None).is_err());
        assert!(FloatDistribution::new(-1.0, 1.0, true, None).is_err());
    }

    #[test]
    fn test_negative_step() {
        assert!(FloatDistribution::new(0.0, 1.0, false, Some(-0.1)).is_err());
        assert!(FloatDistribution::new(0.0, 1.0, false, Some(0.0)).is_err());
    }

    #[test]
    fn test_contains() {
        let d = FloatDistribution::new(0.0, 1.0, false, None).unwrap();
        assert!(d.contains(0.0));
        assert!(d.contains(0.5));
        assert!(d.contains(1.0));
        assert!(!d.contains(-0.1));
        assert!(!d.contains(1.1));
    }

    #[test]
    fn test_contains_with_step() {
        let d = FloatDistribution::new(0.0, 1.0, false, Some(0.25)).unwrap();
        assert!(d.contains(0.0));
        assert!(d.contains(0.25));
        assert!(d.contains(0.5));
        assert!(d.contains(1.0));
        assert!(!d.contains(0.1));
    }

    #[test]
    fn test_single() {
        assert!(
            FloatDistribution::new(1.0, 1.0, false, None)
                .unwrap()
                .single()
        );
        assert!(
            !FloatDistribution::new(0.0, 1.0, false, None)
                .unwrap()
                .single()
        );
    }

    #[test]
    fn test_to_internal_repr_nan() {
        let d = FloatDistribution::new(0.0, 1.0, false, None).unwrap();
        assert!(d.to_internal_repr(f64::NAN).is_err());
    }
}
