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
        if low.is_nan() || high.is_nan() {
            return Err(Error::InvalidDistribution(format!(
                "low and high must not be NaN, got low={low}, high={high}"
            )));
        }
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
            && (s.is_nan() || s <= 0.0)
        {
            return Err(Error::InvalidDistribution(format!(
                "step must be finite and > 0, got step={s}"
            )));
        }
        // Snap `high` down onto the step grid so that every value the
        // samplers can produce — including the clamp target `high` itself —
        // satisfies `contains`. Without this, a distribution such as
        // (low=0, high=1, step=0.3) has an unreachable upper bound that
        // storage validation would reject.
        let high = match step {
            Some(s) => {
                let k = (high - low) / s;
                let k = if (k - k.round()).abs() < 1e-8 {
                    k.round()
                } else {
                    k.floor()
                };
                low + k * s
            }
            None => high,
        };
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
            // The tolerance has to scale with the values involved: when
            // `low / step` is large the grid points are not exactly
            // representable, and a fixed absolute tolerance rejects almost
            // every legitimate value.
            let tol = 1.0e-8_f64.max(4.0 * f64::EPSILON * value.abs() / step);
            (k - k.round()).abs() < tol
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
    fn test_nan_bounds_rejected() {
        assert!(FloatDistribution::new(f64::NAN, 1.0, false, None).is_err());
        assert!(FloatDistribution::new(0.0, f64::NAN, false, None).is_err());
        assert!(FloatDistribution::new(0.0, 1.0, false, Some(f64::NAN)).is_err());
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

    #[test]
    fn test_step_snaps_high_onto_grid() {
        // Regression: `high` must be a value the distribution contains.
        // Samplers clamp to `high`, and storage rejects anything `contains`
        // refuses, so an off-grid `high` silently failed trials.
        let d = FloatDistribution::new(0.0, 1.0, false, Some(0.3)).unwrap();
        assert!((d.high - 0.9).abs() < 1e-12, "high={}", d.high);
        assert!(d.contains(d.high));

        // An already-on-grid high is left alone.
        let d = FloatDistribution::new(0.0, 1.0, false, Some(0.1)).unwrap();
        assert!((d.high - 1.0).abs() < 1e-12, "high={}", d.high);
        assert!(d.contains(d.high));

        // A high that lands on the grid only up to float error still snaps
        // to the grid point rather than a step below it.
        let d = FloatDistribution::new(0.0, 0.3, false, Some(0.1)).unwrap();
        assert!((d.high - 0.3).abs() < 1e-12, "high={}", d.high);
        assert!(d.contains(d.high));
    }

    #[test]
    fn test_step_grid_tolerance_scales_with_magnitude() {
        // Regression: a fixed 1e-8 tolerance on (value - low) / step rejected
        // almost every on-grid value once low/step grew large, because the
        // grid points are not exactly representable there.
        let d = FloatDistribution::new(1e6, 1e6 + 1.0, false, Some(1e-6)).unwrap();
        let rejected = (0..1000)
            .filter(|k| !d.contains(d.low + (*k as f64) * 1e-6))
            .count();
        assert_eq!(rejected, 0, "{rejected}/1000 on-grid values were rejected");

        // Values genuinely off the grid are still rejected.
        assert!(!d.contains(d.low + 0.5e-6));
    }
}
