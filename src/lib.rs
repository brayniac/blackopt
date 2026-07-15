//! # blackopt
//!
//! Black-box optimization in Rust. Provides a define-by-run API for
//! hyperparameter and function optimization, supporting both single-objective
//! and multi-objective optimization with state-of-the-art algorithms.
//!
//! The API and several algorithm implementations are modeled on and adapted from
//! [Optuna](https://github.com/optuna/optuna) (MIT, © Preferred Networks, Inc.);
//! see `THIRD_PARTY_NOTICES.md`.
//!
//! ## Quick Start
//!
//! ```rust
//! use blackopt::{create_study, StudyDirection};
//!
//! let study = create_study(None, None, None, None,
//!     Some(StudyDirection::Minimize), None, false).unwrap();
//!
//! study.optimize(|trial| {
//!     let x = trial.suggest_float("x", -10.0, 10.0, false, None)?;
//!     Ok(x * x)
//! }, Some(100), None, None).unwrap();
//!
//! println!("Best value: {}", study.best_value().unwrap());
//! ```
//!
//! ## Samplers
//!
//! - [`RandomSampler`] — uniform random sampling
//! - [`TpeSampler`] — Tree-structured Parzen Estimator (Bergstra et al. 2011)
//! - [`CmaEsSampler`] — CMA-ES (Hansen & Ostermeier 2001)
//! - [`MorboSampler`] — MORBO trust-region Bayesian optimization (Daulton et al. 2022)
//! - [`NSGAIISampler`] — NSGA-II for multi-objective optimization (Deb et al. 2002)
//! - [`NSGAIIISampler`] — NSGA-III for many-objective optimization (Deb & Jain 2014)
//! - [`GridSampler`] — exhaustive grid search
//! - [`QmcSampler`] — quasi-Monte Carlo sampling (Halton sequences)
//! - [`BruteForceSampler`] — brute-force enumeration
//! - [`PartialFixedSampler`] — fix some parameters while optimizing the rest
//!
//! ## Pruners
//!
//! - [`MedianPruner`] — prune trials below the median of previous trials
//! - [`PercentilePruner`] — prune trials below a given percentile
//! - [`NopPruner`] — no pruning (default)

pub mod callbacks;
pub mod distributions;
pub mod error;
pub mod importance;
pub mod multi_objective;
pub mod pruners;
pub mod samplers;
pub mod search_space;
pub mod storage;
pub mod study;
pub mod terminators;
pub mod trial;

// Re-export key types at the crate root for convenience.
pub use distributions::{
    CategoricalChoice, CategoricalDistribution, Distribution, FloatDistribution, IntDistribution,
    ParamValue,
};
pub use error::{Error, Result};
pub use importance::{FanovaEvaluator, ImportanceEvaluator, get_param_importances};
pub use multi_objective::{
    crowding_distance, dominates, fast_non_dominated_sort, get_pareto_front_trials, hypervolume_2d,
    is_pareto_front,
};
pub use pruners::{MedianPruner, NopPruner, PercentilePruner, Pruner};
pub use samplers::{
    BruteForceSampler, CmaEsCrossover, CmaEsSampler, CmaEsSamplerBuilder, GridSampler,
    MorboSampler, MorboSamplerBuilder, NSGAIIISampler, NSGAIIISamplerBuilder, NSGAIISampler,
    NSGAIISamplerBuilder, PartialFixedSampler, QmcSampler, RandomSampler, Sampler, TpeSampler,
    TpeSamplerBuilder,
};
pub use search_space::{IntersectionSearchSpace, SearchSpaceTransform};
pub use storage::{InMemoryStorage, Storage};
pub use study::{FrozenStudy, Study, StudyDirection, create_study};
pub use terminators::{
    MaxTrialsTerminator, NoImprovementTerminator, TargetValueTerminator, Terminator,
};
pub use trial::{FixedTrial, FrozenTrial, Trial, TrialState};
