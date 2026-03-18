mod median;
mod nop;
mod percentile;

pub use median::MedianPruner;
pub use nop::NopPruner;
pub use percentile::PercentilePruner;

use crate::error::Result;
use crate::trial::FrozenTrial;

/// The pruner trait: decides whether a running trial should be stopped early.
///
pub trait Pruner: Send + Sync {
    /// Returns `true` if the trial should be pruned.
    ///
    /// `study_trials` are the completed/pruned trials from the study.
    /// `trial` is the current running trial with intermediate values reported so far.
    fn prune(&self, study_trials: &[FrozenTrial], trial: &FrozenTrial) -> Result<bool>;
}
