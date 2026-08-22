# blackopt

Black-box optimization in Rust. Provides automatic hyperparameter search with single and multi-objective optimization, pruning, and a variety of sampling algorithms.

## Features

- **Single and multi-objective optimization** with Pareto front analysis
- **10 built-in samplers**: Random, TPE, Grid, QMC (Halton), CMA-ES, MORBO, NSGA-II, NSGA-III, BruteForce, PartialFixed
- **3 pruners**: Median, Percentile, Nop
- **Pluggable storage** via trait (in-memory included)
- **Ask-and-tell interface** for external optimization loops
- **Enqueue** specific parameter settings to evaluate (`enqueue_trial`)
- **Parameter importance** analysis (between-group variance, a fast fANOVA-style approximation)
- **Terminators** for early stopping (max trials, no improvement, target value)
- Define-by-run API

## Quick Start

```rust
use blackopt::{create_study, RandomSampler, Sampler, StudyDirection};
use std::sync::Arc;

fn main() {
    let sampler: Arc<dyn Sampler> = Arc::new(RandomSampler::new(Some(42)));
    let study = create_study(
        None,              // storage (default: in-memory)
        Some(sampler),     // sampler
        None,              // pruner
        Some("my-study"),  // study name
        Some(StudyDirection::Minimize),
        None,              // directions (for multi-objective)
        false,             // load_if_exists
    ).unwrap();

    study.optimize(
        |trial| {
            let x = trial.suggest_float("x", -10.0, 10.0, false, None)?;
            let y = trial.suggest_float("y", -10.0, 10.0, false, None)?;
            Ok(x * x + y * y)
        },
        Some(100),  // n_trials
        None,       // timeout
        None,       // callbacks
    ).unwrap();

    println!("Best value: {}", study.best_value().unwrap());
    println!("Best params: {:?}", study.best_params().unwrap());
}
```

## Enqueueing Trials

You can queue a specific parameter setting so it is evaluated (verbatim, ahead of
sampler-drawn points). This is useful for evaluating a known-good configuration
or seeding a search.

```rust
use std::collections::HashMap;
use blackopt::ParamValue;

let mut params = HashMap::new();
params.insert("x".to_string(), ParamValue::Float(0.5));
params.insert("y".to_string(), ParamValue::Float(-2.0));
study.enqueue_trial(params, None)?;
// The next `ask`/`optimize` step runs this exact setting first.
```

Enqueued values are checked against the distribution the objective declares
when it suggests the parameter, so a value outside those bounds — or a
non-integer for an integer parameter — fails the trial rather than being
quietly adjusted. A name the objective never suggests fails the trial too:
the configuration you asked for was not the one that ran.

Because the point is already chosen, the sampler is not consulted for an
enqueued trial. With `GridSampler` that means an enqueued trial does not
consume a grid point, and a grid trial's parameters must all come from the
grid — enqueue every parameter, or none.

## Multi-Objective Optimization

```rust
use blackopt::{create_study, NSGAIISampler, Sampler, StudyDirection};
use std::sync::Arc;

let directions = vec![StudyDirection::Minimize, StudyDirection::Minimize];
let sampler: Arc<dyn Sampler> = Arc::new(NSGAIISampler::new(
    directions.clone(), None, None, None, None, Some(42),
));

let study = create_study(
    None, Some(sampler), None, None, None, Some(directions), false,
).unwrap();

study.optimize_multi(
    |trial| {
        let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
        Ok(vec![x, 1.0 - x])  // two conflicting objectives
    },
    Some(100), None, None,
).unwrap();

let pareto_front = study.best_trials().unwrap();
println!("Pareto front size: {}", pareto_front.len());
```

## Samplers

| Sampler | Use Case | Reference |
|---------|----------|-----------|
| `RandomSampler` | Baseline, no assumptions about the objective | |
| `TpeSampler` | General-purpose Bayesian optimization (univariate by default; multivariate optional) | Bergstra et al. (2011) |
| `GridSampler` | Exhaustive search over discrete parameters; each point is attempted once, and new trials fail once the grid is exhausted | |
| `QmcSampler` | Low-discrepancy sampling (Halton sequences) | Halton (1964) |
| `CmaEsSampler` | Continuous optimization with covariance adaptation | Hansen & Ostermeier (2001) |
| `MorboSampler` | Multi-objective Bayesian optimization with trust regions (requires exactly 2 objectives) | Daulton et al. (2022) |
| `NSGAIISampler` | Multi-objective optimization (2-3 objectives) | Deb et al. (2002) |
| `NSGAIIISampler` | Many-objective optimization (3+ objectives) | Deb & Jain (2014) |
| `BruteForceSampler` | Enumerate all discrete parameter combinations | |
| `PartialFixedSampler` | Fix some parameters, optimize the rest | |

## Parameter Types

- `suggest_float(name, low, high, log, step)` -- continuous or stepped floats
- `suggest_int(name, low, high, log, step)` -- integers with optional step
- `suggest_categorical(name, choices)` -- categorical choices (strings, numbers, bools)

## References

- Bergstra, J., Bardenet, R., Bengio, Y., & Kegl, B. (2011). "Algorithms for Hyper-Parameter Optimization." *NIPS*.
- Hansen, N. & Ostermeier, A. (2001). "Completely Derandomized Self-Adaptation in Evolution Strategies." *Evolutionary Computation*.
- Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). "A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II." *IEEE Transactions on Evolutionary Computation*.
- Deb, K. & Jain, H. (2014). "An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting Approach." *IEEE Transactions on Evolutionary Computation*.
- Daulton, S., Eriksson, D., Balandat, M., & Bakshy, E. (2022). "Multi-Objective Bayesian Optimization over High-Dimensional Search Spaces." *UAI*.

## Acknowledgments

blackopt's API and several of its algorithm implementations are modeled on and
adapted from [Optuna](https://github.com/optuna/optuna) (the study/trial/sampler
model, the define-by-run and ask-and-tell interfaces, the sampler lineup, pruners,
and fANOVA importance). Optuna is © 2018 Preferred Networks, Inc., MIT-licensed;
its notice is reproduced in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.
