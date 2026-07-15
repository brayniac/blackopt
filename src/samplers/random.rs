use indexmap::IndexMap;
use parking_lot::Mutex;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::distributions::Distribution;
use crate::error::Result;
use crate::samplers::Sampler;
use crate::search_space::SearchSpaceTransform;
use crate::trial::FrozenTrial;

/// A sampler that draws parameters uniformly at random.
///
///
/// Uses `SearchSpaceTransform` to convert distributions into a continuous
/// uniform space, samples uniformly, then untransforms back.
pub struct RandomSampler {
    rng: Mutex<ChaCha8Rng>,
}

impl std::fmt::Debug for RandomSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RandomSampler").finish()
    }
}

impl RandomSampler {
    /// Create a new `RandomSampler` with an optional seed.
    pub fn new(seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };
        Self {
            rng: Mutex::new(rng),
        }
    }

    /// Sample a single parameter using SearchSpaceTransform.
    fn sample_with_transform(&self, distribution: &Distribution) -> Result<f64> {
        let mut space = IndexMap::new();
        space.insert("_param".to_string(), distribution.clone());
        let transform = SearchSpaceTransform::with_defaults(space);
        let bounds = transform.bounds();

        let mut rng = self.rng.lock();
        let mut encoded = Vec::with_capacity(bounds.len());
        for [lo, hi] in &bounds {
            let v: f64 = rng.gen_range(*lo..=*hi);
            encoded.push(v);
        }
        drop(rng);

        let params = transform.untransform(&encoded)?;
        let value = &params["_param"];
        distribution.to_internal_repr(value)
    }
}

impl Sampler for RandomSampler {
    fn sample_independent(
        &self,
        _trial: &FrozenTrial,
        _param_name: &str,
        distribution: &Distribution,
    ) -> Result<f64> {
        self.sample_with_transform(distribution)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::*;

    fn dummy_trial() -> FrozenTrial {
        use chrono::Utc;
        use std::collections::HashMap;
        FrozenTrial {
            number: 0,
            state: crate::trial::TrialState::Running,
            values: None,
            datetime_start: Some(Utc::now()),
            datetime_complete: None,
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
            trial_id: 0,
        }
    }

    #[test]
    fn test_sample_float_bounds() {
        let sampler = RandomSampler::new(Some(42));
        let dist =
            Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
        let trial = dummy_trial();
        for _ in 0..100 {
            let v = sampler.sample_independent(&trial, "x", &dist).unwrap();
            assert!((0.0..=1.0).contains(&v), "value {v} out of [0, 1]");
        }
    }

    #[test]
    fn test_sample_float_log() {
        let sampler = RandomSampler::new(Some(42));
        let dist = Distribution::FloatDistribution(
            FloatDistribution::new(0.001, 1.0, true, None).unwrap(),
        );
        let trial = dummy_trial();
        for _ in 0..100 {
            let v = sampler.sample_independent(&trial, "lr", &dist).unwrap();
            assert!((0.001..=1.0).contains(&v), "value {v} out of [0.001, 1.0]");
        }
    }

    #[test]
    fn test_sample_float_step() {
        let sampler = RandomSampler::new(Some(42));
        let fd = FloatDistribution::new(0.0, 1.0, false, Some(0.25)).unwrap();
        let dist = Distribution::FloatDistribution(fd.clone());
        let trial = dummy_trial();
        for _ in 0..100 {
            let v = sampler.sample_independent(&trial, "x", &dist).unwrap();
            assert!(fd.contains(v), "value {v} not on step grid");
        }
    }

    #[test]
    fn test_sample_int_bounds() {
        let sampler = RandomSampler::new(Some(42));
        let dist = Distribution::IntDistribution(IntDistribution::new(1, 10, false, 1).unwrap());
        let trial = dummy_trial();
        for _ in 0..100 {
            let v = sampler.sample_independent(&trial, "n", &dist).unwrap() as i64;
            assert!((1..=10).contains(&v), "value {v} out of [1, 10]");
        }
    }

    #[test]
    fn test_sample_int_step() {
        let sampler = RandomSampler::new(Some(42));
        let id = IntDistribution::new(0, 10, false, 2).unwrap();
        let dist = Distribution::IntDistribution(id.clone());
        let trial = dummy_trial();
        for _ in 0..100 {
            let v = sampler.sample_independent(&trial, "n", &dist).unwrap();
            assert!(id.contains(v), "value {v} not on step grid");
        }
    }

    #[test]
    fn test_sample_int_log() {
        let sampler = RandomSampler::new(Some(42));
        let dist = Distribution::IntDistribution(IntDistribution::new(1, 100, true, 1).unwrap());
        let trial = dummy_trial();
        for _ in 0..100 {
            let v = sampler.sample_independent(&trial, "n", &dist).unwrap() as i64;
            assert!((1..=100).contains(&v), "value {v} out of [1, 100]");
        }
    }

    #[test]
    fn test_sample_categorical() {
        let sampler = RandomSampler::new(Some(42));
        let dist = Distribution::CategoricalDistribution(
            CategoricalDistribution::new(vec![
                CategoricalChoice::Str("a".into()),
                CategoricalChoice::Str("b".into()),
                CategoricalChoice::Str("c".into()),
            ])
            .unwrap(),
        );
        let trial = dummy_trial();
        for _ in 0..100 {
            let v = sampler.sample_independent(&trial, "opt", &dist).unwrap() as usize;
            assert!(v < 3, "index {v} out of range");
        }
    }

    #[test]
    fn test_single_distributions() {
        let sampler = RandomSampler::new(Some(42));
        let trial = dummy_trial();

        let fd =
            Distribution::FloatDistribution(FloatDistribution::new(5.0, 5.0, false, None).unwrap());
        let v = sampler.sample_independent(&trial, "x", &fd).unwrap();
        assert!((v - 5.0).abs() < 1e-10);

        let id = Distribution::IntDistribution(IntDistribution::new(3, 3, false, 1).unwrap());
        let v = sampler.sample_independent(&trial, "n", &id).unwrap();
        assert!((v - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_deterministic_with_seed() {
        let s1 = RandomSampler::new(Some(123));
        let s2 = RandomSampler::new(Some(123));
        let dist =
            Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
        let trial = dummy_trial();
        for _ in 0..20 {
            let v1 = s1.sample_independent(&trial, "x", &dist).unwrap();
            let v2 = s2.sample_independent(&trial, "x", &dist).unwrap();
            assert_eq!(v1, v2);
        }
    }
}
