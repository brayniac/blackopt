mod brute_force;
mod cmaes;
mod grid;
pub mod morbo;
pub mod nsgaii;
pub mod nsgaiii;
mod partial_fixed;
mod qmc;
pub(crate) mod random;
mod tpe;

pub use brute_force::BruteForceSampler;
pub use cmaes::{CmaEsSampler, CmaEsSamplerBuilder};
pub use grid::GridSampler;
pub use morbo::{MorboSampler, MorboSamplerBuilder};
pub use nsgaii::{CmaEsCrossover, NSGAIISampler, NSGAIISamplerBuilder};
pub use nsgaiii::{NSGAIIISampler, NSGAIIISamplerBuilder};
pub use partial_fixed::PartialFixedSampler;
pub use qmc::QmcSampler;
pub use random::RandomSampler;
pub use tpe::{TpeSampler, TpeSamplerBuilder};

use std::collections::HashMap;

use crate::distributions::Distribution;
use crate::error::Result;
use crate::trial::{FrozenTrial, TrialState};

/// The sampler trait: decides which parameter values to try next.
///
pub trait Sampler: Send + Sync {
    /// Infer the search space for relative sampling from completed trials.
    ///
    /// Returns a map from param name to distribution for parameters that
    /// should be sampled together (relative sampling).
    fn infer_relative_search_space(&self, trials: &[FrozenTrial]) -> HashMap<String, Distribution> {
        let _ = trials;
        HashMap::new()
    }

    /// Sample parameters jointly in the relative search space.
    ///
    /// Returns a map from param name to internal f64 value.
    fn sample_relative(
        &self,
        trials: &[FrozenTrial],
        search_space: &HashMap<String, Distribution>,
    ) -> Result<HashMap<String, f64>> {
        let _ = (trials, search_space);
        Ok(HashMap::new())
    }

    /// Sample a single parameter independently.
    ///
    /// `trials` is the study's full trial history (including the current
    /// running trial), `trial` is the current trial. Samplers that need
    /// history (e.g. univariate TPE) use `trials`.
    fn sample_independent(
        &self,
        trials: &[FrozenTrial],
        trial: &FrozenTrial,
        param_name: &str,
        distribution: &Distribution,
    ) -> Result<f64>;

    /// Called before a trial starts (optional hook).
    fn before_trial(&self, _trials: &[FrozenTrial]) {}

    /// Called right after a new trial has been created and started, before
    /// any parameters are sampled or suggested.
    ///
    /// `trials` is the study's trial history, including the newly created
    /// trial. Implementations may annotate the new trial through storage
    /// (e.g. `GridSampler` records which grid point this trial will
    /// evaluate); returning an error aborts the trial.
    ///
    /// Not called for trials whose parameters were pre-specified through
    /// [`Study::enqueue_trial`](crate::study::Study::enqueue_trial): the
    /// sampler does not choose those points, so it must not reserve
    /// resources for them.
    fn trial_created(&self, _trials: &[FrozenTrial], _trial: &crate::trial::Trial) -> Result<()> {
        Ok(())
    }

    /// Called after a trial finishes (optional hook).
    fn after_trial(
        &self,
        _trials: &[FrozenTrial],
        _trial: &FrozenTrial,
        _state: TrialState,
        _values: Option<&[f64]>,
    ) {
    }
}
