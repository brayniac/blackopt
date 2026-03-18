use crate::error::Result;
use crate::pruners::Pruner;
use crate::trial::FrozenTrial;

/// A pruner that never prunes.
///
#[derive(Debug, Default)]
pub struct NopPruner;

impl NopPruner {
    pub fn new() -> Self {
        Self
    }
}

impl Pruner for NopPruner {
    fn prune(&self, _study_trials: &[FrozenTrial], _trial: &FrozenTrial) -> Result<bool> {
        Ok(false)
    }
}
