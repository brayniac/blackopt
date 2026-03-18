//! MORBO: Multi-Objective Bayesian Optimization with multiple trust regions.
//!
//! Based on Daulton et al. 2022. Achieves order-of-magnitude sample efficiency
//! improvements over evolutionary methods (NSGA-II) by using local Gaussian
//! Processes within adaptively-sized trust regions.

mod gp;
mod sampler;
mod trust_region;

pub use sampler::{MorboSampler, MorboSamplerBuilder};
pub use trust_region::TrustRegionConfig;
