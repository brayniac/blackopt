use indexmap::IndexMap;

use crate::distributions::{Distribution, FloatDistribution, IntDistribution, ParamValue};
use crate::error::Result;

/// Transforms a search space of distributions into a flat continuous representation.
///
///
/// Handles:
/// - Log distributions → apply `ln()` to bounds and values
/// - Step distributions → add ±0.5*step padding to bounds
/// - Categorical → one-hot encode (N choices → N columns, each [0, 1])
///
/// The result is a flat `Vec<f64>` that can be sampled uniformly from `bounds`.
#[derive(Debug, Clone)]
pub struct SearchSpaceTransform {
    search_space: IndexMap<String, Distribution>,
    raw_bounds: Vec<[f64; 2]>,
    /// Maps original param index → range of encoded column indices.
    column_to_encoded_columns: Vec<std::ops::Range<usize>>,
    /// Maps encoded column index → original param index.
    #[allow(dead_code)]
    encoded_column_to_column: Vec<usize>,
    transform_log: bool,
    transform_0_1: bool,
}

impl SearchSpaceTransform {
    /// Create a new transform from a search space.
    ///
    /// # Arguments
    /// * `search_space` - Ordered map of param name → distribution
    /// * `transform_log` - Apply log to log-scale distributions (default: true)
    /// * `transform_step` - Add ±half_step padding to stepped distributions (default: true)
    /// * `transform_0_1` - Rescale all bounds to [0, 1] (default: false)
    pub fn new(
        search_space: IndexMap<String, Distribution>,
        transform_log: bool,
        transform_step: bool,
        transform_0_1: bool,
    ) -> Self {
        let mut raw_bounds = Vec::new();
        let mut column_to_encoded_columns = Vec::new();
        let mut encoded_column_to_column = Vec::new();

        for (col_idx, (_, dist)) in search_space.iter().enumerate() {
            let start = raw_bounds.len();
            match dist {
                Distribution::FloatDistribution(d) => {
                    let (lo, hi) = float_bounds(d, transform_log, transform_step);
                    raw_bounds.push([lo, hi]);
                    encoded_column_to_column.push(col_idx);
                }
                Distribution::IntDistribution(d) => {
                    let (lo, hi) = int_bounds(d, transform_log, transform_step);
                    raw_bounds.push([lo, hi]);
                    encoded_column_to_column.push(col_idx);
                }
                Distribution::CategoricalDistribution(d) => {
                    for _ in 0..d.choices.len() {
                        raw_bounds.push([0.0, 1.0]);
                        encoded_column_to_column.push(col_idx);
                    }
                }
            }
            let end = raw_bounds.len();
            column_to_encoded_columns.push(start..end);
        }

        Self {
            search_space,
            raw_bounds,
            column_to_encoded_columns,
            encoded_column_to_column,
            transform_log,
            transform_0_1,
        }
    }

    /// Create with default settings (transform_log=true, transform_step=true, transform_0_1=false).
    pub fn with_defaults(search_space: IndexMap<String, Distribution>) -> Self {
        Self::new(search_space, true, true, false)
    }

    /// The bounds for each encoded column.
    ///
    /// If `transform_0_1`, all bounds are [0, 1]. Otherwise raw bounds.
    pub fn bounds(&self) -> Vec<[f64; 2]> {
        if self.transform_0_1 {
            vec![[0.0, 1.0]; self.raw_bounds.len()]
        } else {
            self.raw_bounds.clone()
        }
    }

    /// Number of encoded columns.
    pub fn n_encoded(&self) -> usize {
        self.raw_bounds.len()
    }

    /// Transform parameter values into encoded space.
    pub fn transform(&self, params: &IndexMap<String, ParamValue>) -> Vec<f64> {
        let mut encoded = vec![0.0; self.raw_bounds.len()];

        for (col_idx, (name, dist)) in self.search_space.iter().enumerate() {
            let range = &self.column_to_encoded_columns[col_idx];
            let value = &params[name];

            match (dist, value) {
                (Distribution::CategoricalDistribution(d), ParamValue::Categorical(choice)) => {
                    let idx = d
                        .choices
                        .iter()
                        .position(|c| c == choice)
                        .unwrap_or(0);
                    encoded[range.start + idx] = 1.0;
                }
                (Distribution::FloatDistribution(d), ParamValue::Float(v)) => {
                    encoded[range.start] = transform_numerical_float(*v, d, self.transform_log);
                }
                (Distribution::FloatDistribution(d), ParamValue::Int(v)) => {
                    encoded[range.start] =
                        transform_numerical_float(*v as f64, d, self.transform_log);
                }
                (Distribution::IntDistribution(d), ParamValue::Int(v)) => {
                    encoded[range.start] = transform_numerical_int(*v, d, self.transform_log);
                }
                (Distribution::IntDistribution(d), ParamValue::Float(v)) => {
                    encoded[range.start] =
                        transform_numerical_int(*v as i64, d, self.transform_log);
                }
                _ => {}
            }
        }

        if self.transform_0_1 {
            for (i, val) in encoded.iter_mut().enumerate() {
                let [lo, hi] = self.raw_bounds[i];
                if (hi - lo).abs() < f64::EPSILON {
                    *val = 0.5;
                } else {
                    *val = (*val - lo) / (hi - lo);
                }
            }
        }

        encoded
    }

    /// Untransform encoded values back to parameter values.
    pub fn untransform(&self, encoded: &[f64]) -> Result<IndexMap<String, ParamValue>> {
        let mut working = encoded.to_vec();

        // Reverse 0-1 scaling
        if self.transform_0_1 {
            for (i, val) in working.iter_mut().enumerate() {
                let [lo, hi] = self.raw_bounds[i];
                *val = lo + *val * (hi - lo);
            }
        }

        let mut result = IndexMap::new();

        for (col_idx, (name, dist)) in self.search_space.iter().enumerate() {
            let range = &self.column_to_encoded_columns[col_idx];

            let value = match dist {
                Distribution::CategoricalDistribution(d) => {
                    let slice = &working[range.clone()];
                    let idx = argmax(slice);
                    let choice = d.to_external_repr(idx as f64)?;
                    ParamValue::Categorical(choice)
                }
                Distribution::FloatDistribution(d) => {
                    let v = working[range.start];
                    ParamValue::Float(untransform_numerical_float(v, d, self.transform_log))
                }
                Distribution::IntDistribution(d) => {
                    let v = working[range.start];
                    ParamValue::Int(untransform_numerical_int(v, d, self.transform_log))
                }
            };

            result.insert(name.clone(), value);
        }

        Ok(result)
    }

    /// Access the search space.
    pub fn search_space(&self) -> &IndexMap<String, Distribution> {
        &self.search_space
    }
}

// ── Numerical transform helpers ─────────────────────────────────────────────

fn transform_numerical_float(value: f64, d: &FloatDistribution, transform_log: bool) -> f64 {
    if d.log && transform_log {
        value.ln()
    } else {
        value
    }
}

fn transform_numerical_int(value: i64, d: &IntDistribution, transform_log: bool) -> f64 {
    if d.log && transform_log {
        (value as f64).ln()
    } else {
        value as f64
    }
}

fn untransform_numerical_float(trans: f64, d: &FloatDistribution, transform_log: bool) -> f64 {
    if d.log {
        let v = if transform_log { trans.exp() } else { trans };
        if d.single() {
            v
        } else {
            // Half-open [low, high): clamp to just below high
            v.clamp(d.low, next_down(d.high))
        }
    } else if let Some(step) = d.step {
        let v = ((trans - d.low) / step).round() * step + d.low;
        v.clamp(d.low, d.high)
    } else if d.single() {
        trans
    } else {
        // Half-open [low, high)
        trans.clamp(d.low, next_down(d.high))
    }
}

fn untransform_numerical_int(trans: f64, d: &IntDistribution, transform_log: bool) -> i64 {
    if d.log {
        let v = if transform_log { trans.exp() } else { trans };
        (v.round() as i64).clamp(d.low, d.high)
    } else {
        let v = ((trans - d.low as f64) / d.step as f64).round() * d.step as f64 + d.low as f64;
        (v.round() as i64).clamp(d.low, d.high)
    }
}

fn float_bounds(d: &FloatDistribution, transform_log: bool, transform_step: bool) -> (f64, f64) {
    let half_step = if transform_step {
        d.step.map(|s| 0.5 * s).unwrap_or(0.0)
    } else {
        0.0
    };

    if d.log && transform_log {
        let lo = (d.low).ln();
        let hi = (d.high).ln();
        (lo - half_step, hi + half_step)
    } else {
        let lo = d.low;
        let hi = d.high;
        (lo - half_step, hi + half_step)
    }
}

fn int_bounds(d: &IntDistribution, transform_log: bool, transform_step: bool) -> (f64, f64) {
    let half_step = if transform_step {
        0.5 * d.step as f64
    } else {
        0.0
    };

    if d.log && transform_log {
        // Half-step applied BEFORE log transform
        let lo = (d.low as f64 - half_step).ln();
        let hi = (d.high as f64 + half_step).ln();
        (lo, hi)
    } else {
        let lo = d.low as f64;
        let hi = d.high as f64;
        (lo - half_step, hi + half_step)
    }
}

/// Return the largest f64 strictly less than `x`.
fn next_down(x: f64) -> f64 {
    // f64::next_down is unstable; use bit manipulation
    if x.is_nan() || (x.is_infinite() && x < 0.0) {
        return x;
    }
    if x == 0.0 {
        return -f64::MIN_POSITIVE;
    }
    let bits = x.to_bits();
    let prev_bits = if x > 0.0 { bits - 1 } else { bits + 1 };
    f64::from_bits(prev_bits)
}

fn argmax(slice: &[f64]) -> usize {
    slice
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::{CategoricalChoice, CategoricalDistribution};

    fn make_space() -> IndexMap<String, Distribution> {
        let mut space = IndexMap::new();
        space.insert(
            "x".into(),
            Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap()),
        );
        space.insert(
            "y".into(),
            Distribution::IntDistribution(IntDistribution::new(1, 10, false, 1).unwrap()),
        );
        space.insert(
            "z".into(),
            Distribution::CategoricalDistribution(
                CategoricalDistribution::new(vec![
                    CategoricalChoice::Str("a".into()),
                    CategoricalChoice::Str("b".into()),
                    CategoricalChoice::Str("c".into()),
                ])
                .unwrap(),
            ),
        );
        space
    }

    #[test]
    fn test_n_encoded() {
        let t = SearchSpaceTransform::with_defaults(make_space());
        // x=1, y=1, z=3 (one-hot)
        assert_eq!(t.n_encoded(), 5);
    }

    #[test]
    fn test_bounds_shape() {
        let t = SearchSpaceTransform::with_defaults(make_space());
        let bounds = t.bounds();
        assert_eq!(bounds.len(), 5);
    }

    #[test]
    fn test_transform_untransform_float() {
        let mut space = IndexMap::new();
        space.insert(
            "x".into(),
            Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap()),
        );
        let t = SearchSpaceTransform::with_defaults(space);

        let mut params = IndexMap::new();
        params.insert("x".into(), ParamValue::Float(0.5));
        let encoded = t.transform(&params);
        assert_eq!(encoded.len(), 1);
        assert!((encoded[0] - 0.5).abs() < 1e-10);

        let decoded = t.untransform(&encoded).unwrap();
        match decoded.get("x").unwrap() {
            ParamValue::Float(v) => assert!((*v - 0.5).abs() < 1e-10),
            _ => panic!("expected float"),
        }
    }

    #[test]
    fn test_transform_untransform_log_float() {
        let mut space = IndexMap::new();
        space.insert(
            "lr".into(),
            Distribution::FloatDistribution(
                FloatDistribution::new(0.001, 1.0, true, None).unwrap(),
            ),
        );
        let t = SearchSpaceTransform::with_defaults(space);

        let mut params = IndexMap::new();
        params.insert("lr".into(), ParamValue::Float(0.01));
        let encoded = t.transform(&params);
        // ln(0.01) ≈ -4.605
        assert!((encoded[0] - 0.01_f64.ln()).abs() < 1e-10);

        let decoded = t.untransform(&encoded).unwrap();
        match decoded.get("lr").unwrap() {
            ParamValue::Float(v) => assert!((*v - 0.01).abs() < 1e-8),
            _ => panic!("expected float"),
        }
    }

    #[test]
    fn test_transform_untransform_step_float() {
        let mut space = IndexMap::new();
        space.insert(
            "x".into(),
            Distribution::FloatDistribution(
                FloatDistribution::new(0.0, 1.0, false, Some(0.25)).unwrap(),
            ),
        );
        let t = SearchSpaceTransform::with_defaults(space);

        // Bounds should have ±0.125 padding
        let bounds = t.bounds();
        assert!((bounds[0][0] - (-0.125)).abs() < 1e-10);
        assert!((bounds[0][1] - 1.125).abs() < 1e-10);

        // Untransform should snap to step grid
        let decoded = t.untransform(&[0.3]).unwrap();
        match decoded.get("x").unwrap() {
            ParamValue::Float(v) => assert!(
                (*v - 0.25).abs() < 1e-10,
                "expected 0.25 (nearest step), got {v}"
            ),
            _ => panic!("expected float"),
        }
    }

    #[test]
    fn test_transform_untransform_int() {
        let mut space = IndexMap::new();
        space.insert(
            "n".into(),
            Distribution::IntDistribution(IntDistribution::new(1, 10, false, 1).unwrap()),
        );
        let t = SearchSpaceTransform::with_defaults(space);

        let mut params = IndexMap::new();
        params.insert("n".into(), ParamValue::Int(5));
        let encoded = t.transform(&params);
        assert!((encoded[0] - 5.0).abs() < 1e-10);

        let decoded = t.untransform(&encoded).unwrap();
        assert_eq!(decoded.get("n").unwrap(), &ParamValue::Int(5));
    }

    #[test]
    fn test_transform_untransform_categorical() {
        let mut space = IndexMap::new();
        space.insert(
            "opt".into(),
            Distribution::CategoricalDistribution(
                CategoricalDistribution::new(vec![
                    CategoricalChoice::Str("sgd".into()),
                    CategoricalChoice::Str("adam".into()),
                ])
                .unwrap(),
            ),
        );
        let t = SearchSpaceTransform::with_defaults(space);

        let mut params = IndexMap::new();
        params.insert(
            "opt".into(),
            ParamValue::Categorical(CategoricalChoice::Str("adam".into())),
        );
        let encoded = t.transform(&params);
        assert_eq!(encoded.len(), 2);
        assert!((encoded[0] - 0.0).abs() < 1e-10); // sgd = 0
        assert!((encoded[1] - 1.0).abs() < 1e-10); // adam = 1

        let decoded = t.untransform(&encoded).unwrap();
        assert_eq!(
            decoded.get("opt").unwrap(),
            &ParamValue::Categorical(CategoricalChoice::Str("adam".into()))
        );
    }

    #[test]
    fn test_transform_0_1() {
        let mut space = IndexMap::new();
        space.insert(
            "x".into(),
            Distribution::FloatDistribution(FloatDistribution::new(2.0, 8.0, false, None).unwrap()),
        );
        let t = SearchSpaceTransform::new(space, true, true, true);

        let bounds = t.bounds();
        assert!((bounds[0][0] - 0.0).abs() < 1e-10);
        assert!((bounds[0][1] - 1.0).abs() < 1e-10);

        let mut params = IndexMap::new();
        params.insert("x".into(), ParamValue::Float(5.0));
        let encoded = t.transform(&params);
        assert!((encoded[0] - 0.5).abs() < 1e-10);

        let decoded = t.untransform(&encoded).unwrap();
        match decoded.get("x").unwrap() {
            ParamValue::Float(v) => assert!((*v - 5.0).abs() < 1e-8),
            _ => panic!("expected float"),
        }
    }

    #[test]
    fn test_mixed_roundtrip() {
        let t = SearchSpaceTransform::with_defaults(make_space());

        let mut params = IndexMap::new();
        params.insert("x".into(), ParamValue::Float(0.5));
        params.insert("y".into(), ParamValue::Int(5));
        params.insert(
            "z".into(),
            ParamValue::Categorical(CategoricalChoice::Str("b".into())),
        );

        let encoded = t.transform(&params);
        assert_eq!(encoded.len(), 5);

        let decoded = t.untransform(&encoded).unwrap();
        match decoded.get("x").unwrap() {
            ParamValue::Float(v) => assert!((*v - 0.5).abs() < 1e-8),
            _ => panic!("expected float"),
        }
        assert_eq!(decoded.get("y").unwrap(), &ParamValue::Int(5));
        assert_eq!(
            decoded.get("z").unwrap(),
            &ParamValue::Categorical(CategoricalChoice::Str("b".into()))
        );
    }

    #[test]
    fn test_single_value_distribution() {
        let mut space = IndexMap::new();
        space.insert(
            "x".into(),
            Distribution::FloatDistribution(FloatDistribution::new(5.0, 5.0, false, None).unwrap()),
        );
        let t = SearchSpaceTransform::with_defaults(space);
        let decoded = t.untransform(&[5.0]).unwrap();
        match decoded.get("x").unwrap() {
            ParamValue::Float(v) => assert!((*v - 5.0).abs() < 1e-10),
            _ => panic!("expected float"),
        }
    }

    #[test]
    fn test_log_int_bounds() {
        let mut space = IndexMap::new();
        space.insert(
            "n".into(),
            Distribution::IntDistribution(IntDistribution::new(1, 100, true, 1).unwrap()),
        );
        let t = SearchSpaceTransform::with_defaults(space);
        let bounds = t.bounds();
        // With half_step=0.5 applied before log: ln(0.5) and ln(100.5)
        assert!((bounds[0][0] - 0.5_f64.ln()).abs() < 1e-10);
        assert!((bounds[0][1] - 100.5_f64.ln()).abs() < 1e-10);
    }
}
