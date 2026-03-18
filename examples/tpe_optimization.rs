//! TPE sampler optimization with progress logging.
//!
//! Uses the Tree-structured Parzen Estimator to minimize a quadratic function.

use blackopt::{create_study, Sampler, StudyDirection, TpeSamplerBuilder};
use std::sync::Arc;

fn main() {
    let sampler: Arc<dyn Sampler> = Arc::new(
        TpeSamplerBuilder::new(StudyDirection::Minimize)
            .seed(42)
            .n_startup_trials(10)
            .n_ei_candidates(24)
            .build(),
    );

    let study = create_study(
        None,
        Some(sampler),
        None,
        Some("tpe-example"),
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
                let value = x * x + y * y;
                Ok(value)
            },
            Some(100),
            None,
            None,
        )
        .unwrap();

    println!("Completed {} trials", study.trials().unwrap().len());

    let best = study.best_trial().unwrap();
    println!("Best trial: #{}", best.number);
    println!("  Value: {:.6}", best.value().unwrap().unwrap());
    for (name, val) in &best.params {
        println!("  {name}: {val:?}");
    }
}
