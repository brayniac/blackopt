use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// A value that can appear as a categorical choice.
///
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum CategoricalChoice {
    None,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
}

impl PartialEq for CategoricalChoice {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::None, Self::None) => true,
            (Self::Bool(a), Self::Bool(b)) => a == b,
            (Self::Int(a), Self::Int(b)) => a == b,
            (Self::Float(a), Self::Float(b)) => a == b,
            (Self::Str(a), Self::Str(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for CategoricalChoice {}

impl std::fmt::Display for CategoricalChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::None => write!(f, "None"),
            Self::Bool(b) => write!(f, "{b}"),
            Self::Int(i) => write!(f, "{i}"),
            Self::Float(v) => write!(f, "{v}"),
            Self::Str(s) => write!(f, "{s}"),
        }
    }
}

/// A distribution over categorical values.
///
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CategoricalDistribution {
    pub choices: Vec<CategoricalChoice>,
}

impl CategoricalDistribution {
    /// Create a new `CategoricalDistribution` with validation.
    pub fn new(choices: Vec<CategoricalChoice>) -> Result<Self> {
        if choices.is_empty() {
            return Err(Error::InvalidDistribution(
                "choices must not be empty".into(),
            ));
        }
        Ok(Self { choices })
    }

    /// Check if `value` (an index as f64) is contained in this distribution.
    pub fn contains(&self, value: f64) -> bool {
        let index = value as usize;
        index < self.choices.len() && (index as f64 - value).abs() < 1e-8
    }

    /// Convert an external value to internal representation (the index as f64).
    pub fn to_internal_repr(&self, value: &CategoricalChoice) -> Result<f64> {
        self.choices
            .iter()
            .position(|c| c == value)
            .map(|i| i as f64)
            .ok_or_else(|| Error::ValueError(format!("value {value} not found in choices")))
    }

    /// Convert an internal representation (index as f64) back to an external value.
    pub fn to_external_repr(&self, value: f64) -> Result<CategoricalChoice> {
        let index = value as usize;
        self.choices
            .get(index)
            .cloned()
            .ok_or_else(|| Error::ValueError(format!("index {index} out of range for choices")))
    }

    /// True if this distribution contains exactly one choice.
    pub fn single(&self) -> bool {
        self.choices.len() == 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_choices_rejected() {
        assert!(CategoricalDistribution::new(vec![]).is_err());
    }

    #[test]
    fn test_valid_distribution() {
        let d = CategoricalDistribution::new(vec![
            CategoricalChoice::Str("a".into()),
            CategoricalChoice::Str("b".into()),
        ])
        .unwrap();
        assert_eq!(d.choices.len(), 2);
    }

    #[test]
    fn test_contains() {
        let d = CategoricalDistribution::new(vec![
            CategoricalChoice::Str("a".into()),
            CategoricalChoice::Str("b".into()),
            CategoricalChoice::Str("c".into()),
        ])
        .unwrap();
        assert!(d.contains(0.0));
        assert!(d.contains(1.0));
        assert!(d.contains(2.0));
        assert!(!d.contains(3.0));
        assert!(!d.contains(-1.0));
        assert!(!d.contains(0.5));
    }

    #[test]
    fn test_internal_external_roundtrip() {
        let d = CategoricalDistribution::new(vec![
            CategoricalChoice::Str("x".into()),
            CategoricalChoice::Int(42),
            CategoricalChoice::None,
        ])
        .unwrap();

        let internal = d.to_internal_repr(&CategoricalChoice::Int(42)).unwrap();
        assert_eq!(internal, 1.0);

        let external = d.to_external_repr(1.0).unwrap();
        assert_eq!(external, CategoricalChoice::Int(42));
    }

    #[test]
    fn test_single() {
        assert!(
            CategoricalDistribution::new(vec![CategoricalChoice::Bool(true)])
                .unwrap()
                .single()
        );
    }
}
