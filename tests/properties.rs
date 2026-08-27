//! Property tests for invariants that hold across *all* distributions.
//!
//! Every bug these encode was found by hand during review, one concrete case
//! at a time. Stated as properties, a generator explores the same space
//! without anyone having to guess which constants are interesting — the
//! failures were things like `(1e6, 1e6 + 1, step 1e-6)` and
//! `suggest_int("n", 100, 300, log = true)`, which nobody picks by intuition.

use blackopt::distributions::{Distribution, FloatDistribution, IntDistribution};
use blackopt::samplers::{RandomSampler, Sampler, TpeSampler};
use blackopt::study::{StudyDirection, create_study};
use blackopt::trial::{FrozenTrial, TrialState};
use proptest::prelude::*;
use std::collections::HashMap;
use std::sync::Arc;

fn dummy_trial() -> FrozenTrial {
    FrozenTrial {
        number: 0,
        state: TrialState::Running,
        values: None,
        datetime_start: None,
        datetime_complete: None,
        params: HashMap::new(),
        distributions: HashMap::new(),
        user_attrs: HashMap::new(),
        system_attrs: HashMap::new(),
        intermediate_values: HashMap::new(),
        trial_id: 0,
    }
}

/// Float distributions spanning the awkward corners: tiny and huge bounds,
/// steps that do not divide the range, single-valued ranges, log scale.
fn any_float_dist() -> impl Strategy<Value = FloatDistribution> {
    (
        prop_oneof![
            -1.0e6..1.0e6f64,
            1.0e-6..1.0e-3f64,
            0.5..2.0f64,
            1.0e5..1.0e6f64,
        ],
        prop_oneof![Just(0.0), 1.0e-6..1.0e3f64],
        prop_oneof![Just(None), (1.0e-6..10.0f64).prop_map(Some)],
        any::<bool>(),
    )
        .prop_filter_map("valid float distribution", |(low, width, step, log)| {
            let high = low + width;
            // log scale requires low > 0 and forbids a step
            let (log, step) = if log && low > 0.0 {
                (true, None)
            } else {
                (false, step)
            };
            FloatDistribution::new(low, high, log, step).ok()
        })
}

fn any_int_dist() -> impl Strategy<Value = IntDistribution> {
    (
        prop_oneof![-1000..1000i64, 1..1000i64, 100..100_000i64],
        0..1000i64,
        1..17i64,
        any::<bool>(),
    )
        .prop_filter_map("valid int distribution", |(low, width, step, log)| {
            let high = low + width;
            // log scale requires low >= 1 and step == 1
            let (log, step) = if log && low >= 1 {
                (true, 1)
            } else {
                (false, step)
            };
            IntDistribution::new(low, high, log, step).ok()
        })
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 400, ..ProptestConfig::default() })]

    /// The bounds a distribution reports must be values it accepts. Samplers
    /// clamp to them, and storage rejects anything `contains` refuses, so a
    /// bound outside its own distribution fails trials.
    #[test]
    fn float_bounds_are_contained(d in any_float_dist()) {
        prop_assert!(d.contains(d.low), "low {} not contained in {:?}", d.low, d);
        prop_assert!(d.contains(d.high), "high {} not contained in {:?}", d.high, d);
    }

    #[test]
    fn int_bounds_are_contained(d in any_int_dist()) {
        prop_assert!(d.contains(d.low as f64), "low {} not contained in {:?}", d.low, d);
        prop_assert!(d.contains(d.high as f64), "high {} not contained in {:?}", d.high, d);
    }

    /// Anything a sampler proposes must be storable. This is the invariant the
    /// storage-validation gate assumes, and the one that was violated for
    /// off-grid steps, pinned log floats, and log-scale integers.
    #[test]
    fn random_samples_are_contained(d in any_float_dist(), seed in any::<u64>()) {
        let sampler = RandomSampler::new(Some(seed));
        let dist = Distribution::FloatDistribution(d.clone());
        let trial = dummy_trial();
        for _ in 0..20 {
            let v = sampler.sample_independent(&[], &trial, "x", &dist).unwrap();
            prop_assert!(d.contains(v), "sampled {v} not contained in {d:?}");
        }
    }

    #[test]
    fn random_int_samples_are_contained(d in any_int_dist(), seed in any::<u64>()) {
        let sampler = RandomSampler::new(Some(seed));
        let dist = Distribution::IntDistribution(d.clone());
        let trial = dummy_trial();
        for _ in 0..20 {
            let v = sampler.sample_independent(&[], &trial, "n", &dist).unwrap();
            prop_assert!(d.contains(v), "sampled {v} not contained in {d:?}");
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 60, ..ProptestConfig::default() })]

    /// A study over any valid distribution must not silently fail trials.
    /// `optimize` returns Ok even when every trial failed, so a caller cannot
    /// tell from the return value alone; that is exactly how the storage-gate
    /// bugs stayed invisible.
    #[test]
    fn tpe_study_completes_every_trial(fd in any_float_dist(), id in any_int_dist()) {
        let sampler: Arc<dyn Sampler> =
            Arc::new(TpeSampler::with_defaults(StudyDirection::Minimize, Some(7)));
        let study = create_study(
            None, Some(sampler), None, None,
            Some(StudyDirection::Minimize), None, false,
        ).unwrap();

        let (flo, fhi, flog, fstep) = (fd.low, fd.high, fd.log, fd.step);
        let (ilo, ihi, ilog, istep) = (id.low, id.high, id.log, id.step);

        study.optimize(|t| {
            let x = t.suggest_float("x", flo, fhi, flog, fstep)?;
            let n = t.suggest_int("n", ilo, ihi, ilog, istep)?;
            Ok(x + n as f64)
        }, Some(25), None, None).unwrap();

        let trials = study.trials().unwrap();
        let failed: Vec<_> = trials.iter()
            .filter(|t| t.state == TrialState::Fail)
            .filter_map(|t| t.system_attrs.get("failure_reason").and_then(|v| v.as_str()))
            .collect();
        prop_assert!(
            failed.is_empty(),
            "{} of {} trials failed for float={:?} int={:?}; first: {:?}",
            failed.len(), trials.len(), fd, id, failed.first()
        );
    }
}
