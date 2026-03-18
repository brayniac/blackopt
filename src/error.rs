use thiserror::Error as ThiserrorDerive;

/// All errors that can occur in blackopt.
#[derive(Debug, ThiserrorDerive)]
pub enum Error {
    /// Raised when a trial is pruned.
    #[error("trial was pruned")]
    TrialPruned,

    /// Raised when a study with the same name already exists.
    #[error("duplicated study: {0}")]
    DuplicatedStudyError(String),

    /// Raised when a storage operation fails.
    #[error("storage internal error: {0}")]
    StorageInternalError(String),

    /// Raised when attempting to update a finished trial.
    #[error("cannot update a finished trial: {0}")]
    UpdateFinishedTrialError(String),

    /// Raised for general value/argument errors.
    #[error("{0}")]
    ValueError(String),

    /// Raised when a distribution is invalid.
    #[error("invalid distribution: {0}")]
    InvalidDistribution(String),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, Error>;
