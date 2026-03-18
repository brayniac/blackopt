use serde::{Deserialize, Serialize};

/// The optimization direction of a study.
///
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum StudyDirection {
    /// Direction has not been set yet.
    NotSet = 0,
    /// Minimize the objective value.
    Minimize = 1,
    /// Maximize the objective value.
    Maximize = 2,
}

impl std::fmt::Display for StudyDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NotSet => write!(f, "NOT_SET"),
            Self::Minimize => write!(f, "MINIMIZE"),
            Self::Maximize => write!(f, "MAXIMIZE"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", StudyDirection::NotSet), "NOT_SET");
        assert_eq!(format!("{}", StudyDirection::Minimize), "MINIMIZE");
        assert_eq!(format!("{}", StudyDirection::Maximize), "MAXIMIZE");
    }

    #[test]
    fn test_equality() {
        assert_eq!(StudyDirection::Minimize, StudyDirection::Minimize);
        assert_ne!(StudyDirection::Minimize, StudyDirection::Maximize);
    }
}
