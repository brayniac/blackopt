//! Pruning example with MedianPruner.
//!
//! Simulates a training loop where unpromising trials are pruned early.

use blackopt::{
    create_study, MedianPruner, Error, Pruner, RandomSampler, Sampler, StudyDirection,
    TrialState,
};
use std::sync::Arc;

fn main() {
    let sampler: Arc<dyn Sampler> = Arc::new(RandomSampler::new(Some(42)));
    let pruner: Arc<dyn Pruner> = Arc::new(MedianPruner::new(
        5, // n_startup_trials
        0, // n_warmup_steps
        1, // interval_steps
        1, // n_min_trials
        StudyDirection::Minimize,
    ));

    let study = create_study(
        None,
        Some(sampler),
        Some(pruner),
        Some("pruning-example"),
        Some(StudyDirection::Minimize),
        None,
        false,
    )
    .unwrap();

    study
        .optimize(
            |trial| {
                let lr = trial.suggest_float("learning_rate", 1e-4, 1.0, true, None)?;
                let n_layers = trial.suggest_int("n_layers", 1, 5, false, 1)?;

                // Simulate a training loop
                let mut loss = 10.0;
                for epoch in 0..20 {
                    // Simulated loss decay — good lr decays fast, bad lr diverges
                    loss *= 1.0 - lr * 0.5 / n_layers as f64;
                    loss += 0.01 * n_layers as f64;

                    trial.report(loss, epoch as i64)?;

                    if trial.should_prune()? {
                        return Err(Error::TrialPruned);
                    }
                }

                Ok(loss)
            },
            Some(50),
            None,
            None,
        )
        .unwrap();

    let trials = study.trials().unwrap();
    let n_complete = trials.iter().filter(|t| t.state == TrialState::Complete).count();
    let n_pruned = trials.iter().filter(|t| t.state == TrialState::Pruned).count();

    println!("Completed: {n_complete}, Pruned: {n_pruned}, Total: {}", trials.len());

    let best = study.best_trial().unwrap();
    println!("Best trial: #{}", best.number);
    println!("  Value: {:.4}", best.value().unwrap().unwrap());
    for (name, val) in &best.params {
        println!("  {name}: {val:?}");
    }
}
