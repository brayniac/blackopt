//! Multi-objective optimization with NSGA-II.
//!
//! Optimizes a bi-objective problem: minimize (x^2, (x-2)^2).

use blackopt::{create_study, NSGAIISamplerBuilder, Sampler, StudyDirection};
use std::sync::Arc;

fn main() {
    let directions = vec![StudyDirection::Minimize, StudyDirection::Minimize];

    let sampler: Arc<dyn Sampler> = Arc::new(
        NSGAIISamplerBuilder::new(directions.clone())
            .population_size(20)
            .seed(42)
            .build(),
    );

    let study = create_study(
        None,
        Some(sampler),
        None,
        Some("nsga2-example"),
        None,
        Some(directions),
        false,
    )
    .unwrap();

    study
        .optimize_multi(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                let y = trial.suggest_float("y", -5.0, 5.0, false, None)?;
                Ok(vec![x * x + y * y, (x - 2.0).powi(2) + (y - 1.0).powi(2)])
            },
            Some(100),
            None,
            None,
        )
        .unwrap();

    let pareto = study.best_trials().unwrap();
    println!("Found {} Pareto-optimal trials:", pareto.len());
    for trial in &pareto {
        let values = trial.values.as_ref().unwrap();
        println!("  Trial #{}: f1={:.4}, f2={:.4}", trial.number, values[0], values[1]);
    }
}
