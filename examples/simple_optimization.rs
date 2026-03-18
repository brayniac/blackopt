//! Simple single-objective optimization with RandomSampler.
//!
//! Minimizes f(x, y) = (x - 3)^2 + (y + 2)^2 over [-10, 10]^2.

use blackopt::{create_study, RandomSampler, Sampler, StudyDirection};
use std::sync::Arc;

fn main() {
    let sampler: Arc<dyn Sampler> = Arc::new(RandomSampler::new(Some(42)));
    let study = create_study(
        None,
        Some(sampler),
        None,
        Some("simple-example"),
        Some(StudyDirection::Minimize),
        None,
        false,
    )
    .unwrap();

    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -10.0, 10.0, false, None)?;
                let y = trial.suggest_float("y", -10.0, 10.0, false, None)?;
                Ok((x - 3.0).powi(2) + (y + 2.0).powi(2))
            },
            Some(100),
            None,
            None,
        )
        .unwrap();

    let best = study.best_trial().unwrap();
    println!("Best trial: #{}", best.number);
    println!("  Value: {:.4}", best.value().unwrap().unwrap());
    println!("  Params: {:?}", best.params);
}
