use crate::trial::FrozenTrial;

/// A callback invoked after each trial completes.
///
pub trait Callback: Send + Sync {
    /// Called after each trial finishes.
    ///
    /// `n_complete` is the total number of complete trials so far.
    /// `trial` is the just-finished trial.
    fn on_trial_complete(&self, n_complete: usize, trial: &FrozenTrial);
}
