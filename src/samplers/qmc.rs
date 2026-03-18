use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};

use indexmap::IndexMap;
use parking_lot::Mutex;

use crate::distributions::Distribution;
use crate::error::Result;
use crate::samplers::random::RandomSampler;
use crate::samplers::Sampler;
use crate::search_space::{IntersectionSearchSpace, SearchSpaceTransform};
use crate::trial::FrozenTrial;

/// First 20 primes for Halton sequence bases.
const PRIMES: [u64; 20] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71];

/// Quasi-Monte Carlo sampler using Halton sequences.
///
/// Produces low-discrepancy samples that fill the search space more uniformly
/// than pseudo-random sampling.
pub struct QmcSampler {
    independent_sampler: RandomSampler,
    next_index: AtomicU64,
    scramble: bool,
    seed: Option<u64>,
    search_space: Mutex<IntersectionSearchSpace>,
}

impl std::fmt::Debug for QmcSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QmcSampler")
            .field("scramble", &self.scramble)
            .finish()
    }
}

impl QmcSampler {
    pub fn new(scramble: Option<bool>, seed: Option<u64>) -> Self {
        Self {
            independent_sampler: RandomSampler::new(seed),
            next_index: AtomicU64::new(1), // Start at 1 to avoid 0
            scramble: scramble.unwrap_or(true),
            seed,
            search_space: Mutex::new(IntersectionSearchSpace::new(false)),
        }
    }
}

/// Compute the van der Corput sequence value for index `n` in base `base`.
fn van_der_corput(mut n: u64, base: u64) -> f64 {
    let mut result = 0.0;
    let mut denom = 1.0;
    while n > 0 {
        denom *= base as f64;
        let digit = n % base;
        n /= base;
        result += digit as f64 / denom;
    }
    result
}

/// Compute a scrambled van der Corput value using a simple seed-based permutation.
fn van_der_corput_scrambled(mut n: u64, base: u64, seed: u64) -> f64 {
    let mut result = 0.0;
    let mut denom = 1.0;
    while n > 0 {
        denom *= base as f64;
        let digit = n % base;
        n /= base;
        // Simple scrambling: permute digit using seed
        let scrambled = (digit.wrapping_add(seed).wrapping_mul(2654435761)) % base;
        result += scrambled as f64 / denom;
    }
    result
}

/// Generate a Halton point in [0,1]^d for the given index.
fn halton_point(index: u64, dim: usize, scramble: bool, seed: u64) -> Vec<f64> {
    (0..dim)
        .map(|d| {
            let base = if d < PRIMES.len() {
                PRIMES[d]
            } else {
                // For high dimensions, use odd numbers as bases
                2 * d as u64 + 3
            };
            if scramble {
                van_der_corput_scrambled(index, base, seed.wrapping_add(d as u64))
            } else {
                van_der_corput(index, base)
            }
        })
        .collect()
}

impl Sampler for QmcSampler {
    fn infer_relative_search_space(
        &self,
        trials: &[FrozenTrial],
    ) -> HashMap<String, Distribution> {
        self.search_space.lock().calculate(trials)
    }

    fn sample_relative(
        &self,
        _trials: &[FrozenTrial],
        search_space: &HashMap<String, Distribution>,
    ) -> Result<HashMap<String, f64>> {
        if search_space.is_empty() {
            return Ok(HashMap::new());
        }

        // Build ordered search space
        let mut ordered_space = IndexMap::new();
        let mut param_names: Vec<String> = search_space.keys().cloned().collect();
        param_names.sort();
        for name in &param_names {
            ordered_space.insert(name.clone(), search_space[name].clone());
        }

        let transform = SearchSpaceTransform::new(ordered_space.clone(), true, true, true);
        let n_dims = transform.n_encoded();

        // Get next Halton index
        let index = self.next_index.fetch_add(1, Ordering::Relaxed);
        let seed = self.seed.unwrap_or(0);

        // Generate Halton point in [0,1]^d
        let point = halton_point(index, n_dims, self.scramble, seed);

        // Untransform back to parameter space
        let decoded = transform.untransform(&point)?;
        let mut result = HashMap::new();
        for (name, dist) in &ordered_space {
            if let Some(pv) = decoded.get(name) {
                let internal = dist.to_internal_repr(pv)?;
                result.insert(name.clone(), internal);
            }
        }

        Ok(result)
    }

    fn sample_independent(
        &self,
        trial: &FrozenTrial,
        param_name: &str,
        distribution: &Distribution,
    ) -> Result<f64> {
        self.independent_sampler
            .sample_independent(trial, param_name, distribution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::study::{create_study, StudyDirection};
    use std::sync::Arc;

    #[test]
    fn test_van_der_corput() {
        // Base 2: 1/2, 1/4, 3/4, 1/8, 5/8, ...
        assert!((van_der_corput(1, 2) - 0.5).abs() < 1e-10);
        assert!((van_der_corput(2, 2) - 0.25).abs() < 1e-10);
        assert!((van_der_corput(3, 2) - 0.75).abs() < 1e-10);
        // Base 3: 1/3, 2/3, 1/9, ...
        assert!((van_der_corput(1, 3) - 1.0 / 3.0).abs() < 1e-10);
        assert!((van_der_corput(2, 3) - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_halton_point_bounds() {
        for i in 1..100 {
            let pt = halton_point(i, 5, false, 0);
            assert_eq!(pt.len(), 5);
            for &v in &pt {
                assert!((0.0..1.0).contains(&v), "value {v} out of [0, 1)");
            }
        }
    }

    #[test]
    fn test_halton_uniqueness() {
        let points: Vec<Vec<f64>> = (1..50).map(|i| halton_point(i, 3, false, 0)).collect();
        // All points should be unique
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                assert_ne!(points[i], points[j]);
            }
        }
    }

    #[test]
    fn test_qmc_sampler_basic() {
        let sampler: Arc<dyn Sampler> = Arc::new(QmcSampler::new(Some(false), Some(42)));

        let study = create_study(
            None,
            Some(sampler),
            None,
            None,
            Some(StudyDirection::Minimize),
            None,
            false,
        )
        .unwrap();

        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    let y = trial.suggest_float("y", 0.0, 1.0, false, None)?;
                    Ok(x * x + y * y)
                },
                Some(30),
                None,
                None,
            )
            .unwrap();

        let trials = study.trials().unwrap();
        assert_eq!(trials.len(), 30);
    }

    #[test]
    fn test_qmc_sampler_scrambled() {
        let sampler: Arc<dyn Sampler> = Arc::new(QmcSampler::new(Some(true), Some(42)));

        let study = create_study(
            None,
            Some(sampler),
            None,
            None,
            Some(StudyDirection::Minimize),
            None,
            false,
        )
        .unwrap();

        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    Ok(x * x)
                },
                Some(20),
                None,
                None,
            )
            .unwrap();

        assert_eq!(study.trials().unwrap().len(), 20);
    }
}
