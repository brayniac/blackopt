use rand::Rng;

/// Trait for crossover operators used in evolutionary algorithms.
pub trait Crossover: Send + Sync {
    /// Number of parent solutions required.
    fn n_parents(&self) -> usize;

    /// Perform crossover on parent vectors (in \[0,1\] space) to produce a child.
    fn crossover(&self, parents: &[Vec<f64>], rng: &mut dyn RngCore) -> Vec<f64>;
}

/// Trait alias to allow `&mut dyn RngCore` usage.
pub use rand::RngCore;

/// Uniform crossover: randomly selects each gene from one of the two parents.
pub struct UniformCrossover {
    /// Probability of swapping each gene.
    pub swapping_prob: f64,
}

impl UniformCrossover {
    pub fn new(swapping_prob: Option<f64>) -> Self {
        Self {
            swapping_prob: swapping_prob.unwrap_or(0.5),
        }
    }
}

impl Default for UniformCrossover {
    fn default() -> Self {
        Self::new(None)
    }
}

impl Crossover for UniformCrossover {
    fn n_parents(&self) -> usize {
        2
    }

    fn crossover(&self, parents: &[Vec<f64>], rng: &mut dyn RngCore) -> Vec<f64> {
        parents[0]
            .iter()
            .zip(parents[1].iter())
            .map(|(&p0, &p1)| {
                if rng.r#gen::<f64>() < self.swapping_prob {
                    p1
                } else {
                    p0
                }
            })
            .collect()
    }
}

/// BLX-alpha crossover: samples from an expanded range between parents.
pub struct BLXAlphaCrossover {
    pub alpha: f64,
}

impl BLXAlphaCrossover {
    pub fn new(alpha: Option<f64>) -> Self {
        Self {
            alpha: alpha.unwrap_or(0.5),
        }
    }
}

impl Default for BLXAlphaCrossover {
    fn default() -> Self {
        Self::new(None)
    }
}

impl Crossover for BLXAlphaCrossover {
    fn n_parents(&self) -> usize {
        2
    }

    fn crossover(&self, parents: &[Vec<f64>], rng: &mut dyn RngCore) -> Vec<f64> {
        parents[0]
            .iter()
            .zip(parents[1].iter())
            .map(|(&p0, &p1)| {
                let lo = p0.min(p1);
                let hi = p0.max(p1);
                let d = hi - lo;
                let lower = lo - self.alpha * d;
                let upper = hi + self.alpha * d;
                let v: f64 = rng.r#gen::<f64>() * (upper - lower) + lower;
                v.clamp(0.0, 1.0)
            })
            .collect()
    }
}

/// Simulated Binary Crossover (SBX).
pub struct SBXCrossover {
    /// Distribution index (eta). Higher values produce children closer to parents.
    pub eta: f64,
}

impl SBXCrossover {
    pub fn new(eta: Option<f64>) -> Self {
        Self {
            eta: eta.unwrap_or(2.0),
        }
    }
}

impl Default for SBXCrossover {
    fn default() -> Self {
        Self::new(None)
    }
}

impl Crossover for SBXCrossover {
    fn n_parents(&self) -> usize {
        2
    }

    fn crossover(&self, parents: &[Vec<f64>], rng: &mut dyn RngCore) -> Vec<f64> {
        parents[0]
            .iter()
            .zip(parents[1].iter())
            .map(|(&p0, &p1)| {
                if (p0 - p1).abs() < 1e-14 {
                    return p0;
                }

                let u: f64 = rng.r#gen();
                let beta = if u <= 0.5 {
                    (2.0 * u).powf(1.0 / (self.eta + 1.0))
                } else {
                    (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (self.eta + 1.0))
                };

                let c = if rng.r#gen::<bool>() {
                    0.5 * ((1.0 + beta) * p0 + (1.0 - beta) * p1)
                } else {
                    0.5 * ((1.0 - beta) * p0 + (1.0 + beta) * p1)
                };

                c.clamp(0.0, 1.0)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    #[test]
    fn test_uniform_crossover() {
        let cx = UniformCrossover::new(Some(0.5));
        assert_eq!(cx.n_parents(), 2);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let p0 = vec![0.0, 0.0, 0.0, 0.0];
        let p1 = vec![1.0, 1.0, 1.0, 1.0];
        let child = cx.crossover(&[p0, p1], &mut rng);
        assert_eq!(child.len(), 4);
        // Each gene should be 0.0 or 1.0
        for &v in &child {
            assert!(v == 0.0 || v == 1.0);
        }
    }

    #[test]
    fn test_blx_alpha_crossover() {
        let cx = BLXAlphaCrossover::new(Some(0.5));
        assert_eq!(cx.n_parents(), 2);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let p0 = vec![0.3, 0.3];
        let p1 = vec![0.7, 0.7];
        let child = cx.crossover(&[p0, p1], &mut rng);
        assert_eq!(child.len(), 2);
        for &v in &child {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_sbx_crossover() {
        let cx = SBXCrossover::new(Some(2.0));
        assert_eq!(cx.n_parents(), 2);
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let p0 = vec![0.3, 0.3];
        let p1 = vec![0.7, 0.7];
        let child = cx.crossover(&[p0, p1], &mut rng);
        assert_eq!(child.len(), 2);
        for &v in &child {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_sbx_identical_parents() {
        let cx = SBXCrossover::new(Some(2.0));
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let p0 = vec![0.5, 0.5];
        let p1 = vec![0.5, 0.5];
        let child = cx.crossover(&[p0, p1], &mut rng);
        for &v in &child {
            assert!((v - 0.5).abs() < 1e-10);
        }
    }
}
