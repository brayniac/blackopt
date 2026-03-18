use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// A distribution over integer values.
///
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IntDistribution {
    pub low: i64,
    pub high: i64,
    #[serde(default)]
    pub log: bool,
    #[serde(default = "default_step")]
    pub step: i64,
}

fn default_step() -> i64 {
    1
}

impl IntDistribution {
    /// Create a new `IntDistribution` with validation.
    pub fn new(low: i64, high: i64, log: bool, step: i64) -> Result<Self> {
        if log && step != 1 {
            return Err(Error::InvalidDistribution(
                "cannot combine log-scale with step != 1".into(),
            ));
        }
        if low > high {
            return Err(Error::InvalidDistribution(format!(
                "low must be <= high, got low={low}, high={high}"
            )));
        }
        if log && low < 1 {
            return Err(Error::InvalidDistribution(format!(
                "low must be >= 1 for log-scale, got low={low}"
            )));
        }
        if step <= 0 {
            return Err(Error::InvalidDistribution(format!(
                "step must be > 0, got step={step}"
            )));
        }
        Ok(Self {
            low,
            high,
            log,
            step,
        })
    }

    /// Convenience constructor with default step=1 and log=false.
    pub fn with_range(low: i64, high: i64) -> Result<Self> {
        Self::new(low, high, false, 1)
    }

    /// Check if `value` (in internal representation as f64) is contained in this distribution.
    pub fn contains(&self, value: f64) -> bool {
        let v = value as i64;
        if (v as f64 - value).abs() > 1e-8 {
            return false;
        }
        v >= self.low && v <= self.high && (v - self.low) % self.step == 0
    }

    /// Convert an external value to internal representation (f64).
    pub fn to_internal_repr(&self, value: i64) -> Result<f64> {
        let f = value as f64;
        if f.is_nan() {
            return Err(Error::ValueError("NaN is not allowed".into()));
        }
        if self.log && value <= 0 {
            return Err(Error::ValueError(format!(
                "value must be > 0 for log-scale, got {value}"
            )));
        }
        Ok(f)
    }

    /// Convert an internal representation back to an external value.
    pub fn to_external_repr(&self, value: f64) -> i64 {
        value as i64
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
        let d = IntDistribution::new(1, 10, false, 1).unwrap();
        assert_eq!(d.low, 1);
        assert_eq!(d.high, 10);
    }

    #[test]
    fn test_log_with_step_rejected() {
        assert!(IntDistribution::new(1, 10, true, 2).is_err());
    }

    #[test]
    fn test_low_greater_than_high() {
        assert!(IntDistribution::new(10, 1, false, 1).is_err());
    }

    #[test]
    fn test_log_low_less_than_one() {
        assert!(IntDistribution::new(0, 10, true, 1).is_err());
        assert!(IntDistribution::new(-1, 10, true, 1).is_err());
    }

    #[test]
    fn test_non_positive_step() {
        assert!(IntDistribution::new(0, 10, false, 0).is_err());
        assert!(IntDistribution::new(0, 10, false, -1).is_err());
    }

    #[test]
    fn test_contains() {
        let d = IntDistribution::new(0, 10, false, 2).unwrap();
        assert!(d.contains(0.0));
        assert!(d.contains(2.0));
        assert!(d.contains(10.0));
        assert!(!d.contains(1.0));
        assert!(!d.contains(0.5));
        assert!(!d.contains(-1.0));
        assert!(!d.contains(11.0));
    }

    #[test]
    fn test_single() {
        assert!(IntDistribution::new(5, 5, false, 1).unwrap().single());
        assert!(!IntDistribution::new(1, 10, false, 1).unwrap().single());
    }

    #[test]
    fn test_to_external_repr() {
        let d = IntDistribution::new(0, 10, false, 1).unwrap();
        assert_eq!(d.to_external_repr(5.0), 5);
    }
}
