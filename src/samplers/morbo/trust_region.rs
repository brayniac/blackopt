//! Trust region management for MORBO.

/// Configuration for trust region behavior.
#[derive(Debug, Clone)]
pub struct TrustRegionConfig {
    /// Initial edge length in \[0,1\] space.
    pub l_init: f64,
    /// Minimum edge length — below this, the TR is terminated.
    pub l_min: f64,
    /// Maximum edge length.
    pub l_max: f64,
    /// Number of consecutive successes before expansion.
    pub tau_succ: usize,
    /// Number of consecutive failures before shrinking.
    pub tau_fail: usize,
}

impl TrustRegionConfig {
    /// Create config with MORBO paper defaults for the given dimensionality.
    pub fn new(n_dims: usize) -> Self {
        Self {
            l_init: 0.8,
            l_min: 0.5_f64.powi(7), // ~0.0078
            l_max: 1.6,
            tau_succ: usize::MAX, // paper default: never expand
            tau_fail: n_dims.max(1),
        }
    }
}

/// A single trust region in [0,1]^d space.
#[derive(Debug, Clone)]
pub struct TrustRegion {
    /// Center point in [0,1]^d.
    pub center: Vec<f64>,
    /// Current edge length (half-width of the hypercube is L).
    pub edge_length: f64,
    /// Consecutive successes (HV improvement).
    pub n_successes: usize,
    /// Consecutive failures (no HV improvement).
    pub n_failures: usize,
    /// Whether this TR is still active.
    pub active: bool,
    /// Config reference.
    config: TrustRegionConfig,
}

impl TrustRegion {
    /// Create a new trust region centered at `center`.
    pub fn new(center: Vec<f64>, config: TrustRegionConfig) -> Self {
        let edge_length = config.l_init;
        Self {
            center,
            edge_length,
            n_successes: 0,
            n_failures: 0,
            active: true,
            config,
        }
    }

    /// Lower and upper bounds for each dimension, clamped to [0, 1].
    pub fn bounds(&self) -> Vec<[f64; 2]> {
        self.center
            .iter()
            .map(|&c| {
                let lo = (c - self.edge_length).max(0.0);
                let hi = (c + self.edge_length).min(1.0);
                [lo, hi]
            })
            .collect()
    }

    /// Check if a point is inside this trust region (within 2L hypercube for local data).
    pub fn contains_local(&self, point: &[f64]) -> bool {
        let radius = 2.0 * self.edge_length;
        point
            .iter()
            .zip(self.center.iter())
            .all(|(&p, &c)| (p - c).abs() <= radius)
    }

    /// Get indices of data points within the local 2L neighborhood.
    pub fn local_data_indices(&self, all_x: &[Vec<f64>]) -> Vec<usize> {
        all_x
            .iter()
            .enumerate()
            .filter(|(_, x)| self.contains_local(x))
            .map(|(i, _)| i)
            .collect()
    }

    /// Record a success (HV improved). Expand if threshold met.
    pub fn record_success(&mut self) {
        self.n_successes += 1;
        self.n_failures = 0;
        if self.n_successes >= self.config.tau_succ {
            self.edge_length = (self.edge_length * 2.0).min(self.config.l_max);
            self.n_successes = 0;
        }
    }

    /// Record a failure (no HV improvement). Shrink if threshold met.
    pub fn record_failure(&mut self) {
        self.n_failures += 1;
        self.n_successes = 0;
        if self.n_failures >= self.config.tau_fail {
            self.edge_length /= 2.0;
            self.n_failures = 0;
            if self.edge_length < self.config.l_min {
                self.active = false;
            }
        }
    }

}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trust_region_bounds() {
        let config = TrustRegionConfig {
            l_init: 0.3,
            l_min: 0.01,
            l_max: 1.0,
            tau_succ: 3,
            tau_fail: 2,
        };
        let tr = TrustRegion::new(vec![0.5, 0.1], config);
        let bounds = tr.bounds();
        assert!((bounds[0][0] - 0.2).abs() < 1e-10);
        assert!((bounds[0][1] - 0.8).abs() < 1e-10);
        // Clamped at 0.
        assert!((bounds[1][0] - 0.0).abs() < 1e-10);
        assert!((bounds[1][1] - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_shrinking() {
        let config = TrustRegionConfig {
            l_init: 0.4,
            l_min: 0.01,
            l_max: 1.0,
            tau_succ: 3,
            tau_fail: 2,
        };
        let mut tr = TrustRegion::new(vec![0.5], config);
        assert!((tr.edge_length - 0.4).abs() < 1e-10);
        tr.record_failure();
        assert!((tr.edge_length - 0.4).abs() < 1e-10); // not yet
        tr.record_failure();
        assert!((tr.edge_length - 0.2).abs() < 1e-10); // shrunk
        assert!(tr.active);
    }

    #[test]
    fn test_expansion() {
        let config = TrustRegionConfig {
            l_init: 0.4,
            l_min: 0.01,
            l_max: 1.0,
            tau_succ: 2,
            tau_fail: 5,
        };
        let mut tr = TrustRegion::new(vec![0.5], config);
        tr.record_success();
        assert!((tr.edge_length - 0.4).abs() < 1e-10);
        tr.record_success();
        assert!((tr.edge_length - 0.8).abs() < 1e-10); // expanded
    }

    #[test]
    fn test_termination() {
        let config = TrustRegionConfig {
            l_init: 0.02,
            l_min: 0.01,
            l_max: 1.0,
            tau_succ: usize::MAX,
            tau_fail: 1,
        };
        let mut tr = TrustRegion::new(vec![0.5], config);
        tr.record_failure(); // 0.02 -> 0.01
        assert!(tr.active); // exactly at l_min, edge_length >= l_min
        tr.record_failure(); // 0.01 -> 0.005
        assert!(!tr.active);
    }

    #[test]
    fn test_success_resets_failures() {
        let config = TrustRegionConfig {
            l_init: 0.4,
            l_min: 0.01,
            l_max: 1.0,
            tau_succ: 3,
            tau_fail: 2,
        };
        let mut tr = TrustRegion::new(vec![0.5], config);
        tr.record_failure();
        assert_eq!(tr.n_failures, 1);
        tr.record_success();
        assert_eq!(tr.n_failures, 0);
    }

    #[test]
    fn test_local_data_indices() {
        let config = TrustRegionConfig::new(1);
        let mut tr = TrustRegion::new(vec![0.5], config);
        tr.edge_length = 0.1;
        let data = vec![vec![0.5], vec![0.6], vec![0.7], vec![0.8], vec![1.0]];
        let local = tr.local_data_indices(&data);
        // 2L = 0.2, so within [0.3, 0.7]
        assert_eq!(local, vec![0, 1, 2]);
    }

}
