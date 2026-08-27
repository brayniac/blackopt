# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the crate is below `0.1.0`, breaking changes may land in any release; they
are always listed under **Breaking** so downstream users can see what moved.

## [Unreleased]

Everything since the initial commit. No version has been tagged yet.

### Breaking

#### Serialized format

- `ParamValue` and `CategoricalChoice` are now **adjacently tagged** in serde
  (`{"type": "Int", "value": 5}`) rather than untagged. Previously `Int(5)`
  serialized to the bare `5` and deserialized back as `Float(5.0)`, because the
  first matching variant wins — so an integer parameter silently changed type
  across a round trip.

  There is no automatic migration and no format version marker. Data written by
  an external `Storage` implementation before this change **will not
  deserialize**. The in-tree `InMemoryStorage` does not persist, so it is
  unaffected.

#### API

- `Sampler::sample_independent` takes the study's trial history as a new first
  argument (`trials: &[FrozenTrial]`). Every implementation must be updated.
  Univariate TPE needs the history; without it, it silently fell back to random
  sampling.
- `Sampler::trial_created` is a new hook, called after a trial is created and
  before any parameter is sampled. It has a default no-op implementation, so
  existing implementations still compile.
- `MorboSamplerBuilder::build` returns `Result<MorboSampler>`. `MorboSampler`
  now requires **exactly two objectives**; its hypervolume-based trust-region
  management does not support any other count.
- `SearchSpaceTransform::transform` returns `Result<Vec<f64>>` instead of
  panicking on a missing parameter.
- `FixedTrial` method signatures now match `Trial`: `set_user_attr(&mut self,
  &str, Value) -> Result<()>`, `report(...) -> Result<()>`, `should_prune() ->
  Result<bool>`.
- Removed `Storage::remove_session` (an unused no-op with no implementors) and
  `GridSampler::suggest_grid_id` (superseded by the `trial_created` hook).
- Minimum supported Rust version corrected to **1.88**. It was declared as
  1.85, then 1.86; both were wrong. The crate has used `let` chains
  (`if let ... && let ...`) since its first commit, and those stabilized in
  1.88 — so it never built on the version it advertised. CI now checks the
  declared MSRV on every run, reading it from `Cargo.toml` so the two cannot
  drift apart again.

#### Behavior

- `FloatDistribution::new` and `IntDistribution::new` snap `high` down onto the
  step grid, so `high` is always a value the distribution contains. A
  distribution declared as `(0.0, 1.0, step 0.3)` now reports `high == 0.9`.
  Previously `high` was unreachable but was exactly what samplers clamped to.
- `GridSampler` attempts each grid point **exactly once**, whatever the
  outcome. A failed or pruned point is not retried; retrying a deterministic
  point only repeats the same evaluation.
- Trials enqueued with `enqueue_trial` bypass the sampler entirely — no
  `trial_created`, no `sample_relative`. With `GridSampler` this means an
  enqueued trial does not consume a grid point, and a grid trial's parameters
  must all come from the grid: enqueue every parameter, or none.
- An enqueued value that the objective never suggests, or that does not fit the
  distribution the objective declares, now fails the trial instead of quietly
  running an ordinary sampled trial. Non-integer floats are no longer truncated
  for integer parameters.
- Pruned trials in a multi-objective study no longer record their last
  intermediate value as the trial's value; one scalar cannot stand in for a
  vector of objectives. Intermediate values remain on the trial.
- `NoImprovementTerminator` counts all *finished* trials (complete, failed, or
  pruned) against `patience`, so a perpetually-failing sampler stops the loop.
  Improvement itself is measured only over *completed* trials.
- `Storage::get_best_trial` skips trials whose value cannot be compared (NaN,
  or the wrong number of objectives) rather than panicking. Infinite objective
  values are kept, since an infinite penalty for an infeasible configuration is
  a legitimate result.
- `InMemoryStorage` validates on write: `set_trial_param` rejects NaN and
  out-of-distribution values, `set_trial_state_values` enforces the
  complete-requires-values / no-NaN / objective-count / fail-rejects-values
  invariants, and `create_new_trial` validates templates.
- Failed trials record a `failure_reason` system attribute instead of
  discarding the error.

### Added

- `enqueue_trial` now works end to end: enqueued parameter settings are
  evaluated verbatim, ahead of sampler-drawn points.
- Grid exhaustion distinguishes "every grid point has been visited" from "every
  remaining point is held by an unfinished trial", which need different action
  from the caller.

### Fixed

- **TPE performed no TPE.** The default (univariate) sampler silently returned
  random samples for every parameter. It now implements univariate TPE (Parzen
  KDE with expected improvement), and the previously unreachable `Maximize`
  trial-split path works.
- **`GridSampler` was entirely unwired.** Every trial failed while `optimize`
  returned `Ok`. Grid points are now assigned through `trial_created`, and the
  sampler owns its allocation rather than re-deriving it from a storage
  snapshot — which raced with concurrent `ask` calls and made "consumed" depend
  on how far the objective got before it threw.
- **Enqueued parameters were stored but never read.**
- Stepped distributions whose `high` sat off the grid failed a share of every
  sampler's trials, because storage rejected the value the sampler clamped to.
- A pinned log-scale float (`suggest_float("lr", 1e-3, 1e-3, true, None)`)
  failed every trial: `exp(ln(c))` lands an ULP or two off `c`.
- TPE returned `NaN` for a single-valued distribution, giving every kernel zero
  bandwidth.
- TPE's Parzen estimator reconstructed the original bounds from the transformed
  ones, which is exact in the reals but not in `f64`; the residue pushed
  log-scale integer samples off the integer grid entirely.
- TPE built a model out of unrankable trials in multi-objective studies, where
  every `value()` is an error, degenerating the good/bad split into "the oldest
  trials are good". It now falls back instead.
- `FloatDistribution::contains` used a fixed grid tolerance, rejecting nearly
  every legitimate value once `low / step` grew large.
- `CategoricalDistribution::new` accepted non-finite float choices, which can
  never be matched back to their own index.
- `FanovaEvaluator::new(0)` panicked with an arithmetic underflow, and
  `ImportanceEvaluator::evaluate` indexed `target_values` by trial position,
  reading out of bounds on a length mismatch.
- Parameter importance grouped categorical values by a hash that was not stable
  across runs; it now uses the choice index from the trial's own distribution,
  and an unrecognized choice is skipped rather than merged with the first one.
- `add_trial` rejected hand-built templates that omit timestamps, and accepted
  templates whose objective count disagreed with the study's.
- `FloatDistribution::new` accepted NaN bounds and step.

### Infrastructure

- GitHub Actions CI: rustfmt, clippy (warnings denied), tests on Linux and
  macOS, a build against the declared MSRV, rustdoc with warnings denied, and
  `cargo package`.
- `Cargo.lock` is now committed. Consumers of a library ignore it, but CI needs
  it: without a lock file every run resolves dependencies afresh, so a
  dependency that raises its own minimum Rust version would fail the MSRV job
  for reasons unrelated to this crate.

### Documentation

- `FanovaEvaluator` is described as what it is — a between-group-variance
  approximation, not a true functional ANOVA decomposition.
- README documents the `MorboSampler` two-objective requirement, grid
  exhaustion behavior, and enqueueing.
