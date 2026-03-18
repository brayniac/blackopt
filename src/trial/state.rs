use serde::{Deserialize, Serialize};

/// The state of a trial.
///
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum TrialState {
    /// The trial is currently executing.
    Running = 0,
    /// The trial finished without error and has values.
    Complete = 1,
    /// The trial was pruned via `TrialPruned`.
    Pruned = 2,
    /// The trial failed due to an uncaught error.
    Fail = 3,
    /// The trial is queued but not yet started.
    Waiting = 4,
}

impl TrialState {
    /// Returns `true` for terminal states: `Complete`, `Pruned`, `Fail`.
    pub fn is_finished(self) -> bool {
        matches!(self, Self::Complete | Self::Pruned | Self::Fail)
    }
}

impl std::fmt::Display for TrialState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Running => write!(f, "RUNNING"),
            Self::Complete => write!(f, "COMPLETE"),
            Self::Pruned => write!(f, "PRUNED"),
            Self::Fail => write!(f, "FAIL"),
            Self::Waiting => write!(f, "WAITING"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_finished() {
        assert!(!TrialState::Running.is_finished());
        assert!(TrialState::Complete.is_finished());
        assert!(TrialState::Pruned.is_finished());
        assert!(TrialState::Fail.is_finished());
        assert!(!TrialState::Waiting.is_finished());
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", TrialState::Running), "RUNNING");
        assert_eq!(format!("{}", TrialState::Complete), "COMPLETE");
    }
}
