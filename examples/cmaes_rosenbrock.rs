//! CMA-ES optimization on the Rosenbrock function.
//!
//! Minimizes f(x, y) = (1-x)^2 + 100*(y-x^2)^2, which has a global
//! minimum of 0 at (1, 1).

use blackopt::{create_study, CmaEsSamplerBuilder, Sampler, StudyDirection};
use std::sync::Arc;

fn main() {
    let sampler: Arc<dyn Sampler> = Arc::new(
        CmaEsSamplerBuilder::new(StudyDirection::Minimize)
            .sigma0(0.5)
            .n_startup_trials(20)
            .seed(42)
            .build(),
    );

    let study = create_study(
        None,
        Some(sampler),
        None,
        Some("cmaes-rosenbrock"),
        Some(StudyDirection::Minimize),
        None,
        false,
    )
    .unwrap();

    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                let y = trial.suggest_float("y", -5.0, 5.0, false, None)?;
                Ok((1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2))
            },
            Some(200),
            None,
            None,
        )
        .unwrap();

    let best = study.best_trial().unwrap();
    println!("Best trial: #{}", best.number);
    println!("  Value: {:.6}", best.value().unwrap().unwrap());
    for (name, val) in &best.params {
        println!("  {name}: {val:?}");
    }
}
