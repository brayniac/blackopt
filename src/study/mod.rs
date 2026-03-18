mod direction;
mod frozen;
mod core;

pub use core::Study;
pub use direction::StudyDirection;
pub use frozen::FrozenStudy;

use std::sync::Arc;

use crate::error::{Error, Result};
use crate::pruners::{NopPruner, Pruner};
use crate::samplers::{RandomSampler, Sampler};
use crate::storage::{InMemoryStorage, Storage};

/// Create a new study.
///
///
/// # Arguments
/// * `storage` - Storage backend. If `None`, uses `InMemoryStorage`.
/// * `sampler` - Sampler. If `None`, uses `RandomSampler` (TPE in later phases).
/// * `pruner` - Pruner. If `None`, uses `NopPruner` (MedianPruner in later phases).
/// * `study_name` - Study name. If `None`, auto-generated.
/// * `direction` - Single-objective direction. Mutually exclusive with `directions`.
/// * `directions` - Multi-objective directions. Mutually exclusive with `direction`.
/// * `load_if_exists` - If `true`, load existing study instead of erroring on duplicate.
#[allow(clippy::too_many_arguments)]
pub fn create_study(
    storage: Option<Arc<dyn Storage>>,
    sampler: Option<Arc<dyn Sampler>>,
    pruner: Option<Arc<dyn Pruner>>,
    study_name: Option<&str>,
    direction: Option<StudyDirection>,
    directions: Option<Vec<StudyDirection>>,
    load_if_exists: bool,
) -> Result<Study> {
    // Validate direction / directions
    if direction.is_some() && directions.is_some() {
        return Err(Error::ValueError(
            "specify either `direction` or `directions`, not both".into(),
        ));
    }

    let dirs = if let Some(d) = direction {
        vec![d]
    } else if let Some(ds) = directions {
        ds
    } else {
        vec![StudyDirection::Minimize]
    };

    let storage = storage.unwrap_or_else(|| Arc::new(InMemoryStorage::new()));

    // Create or load
    let study_id = match storage.create_new_study(&dirs, study_name) {
        Ok(id) => id,
        Err(Error::DuplicatedStudyError(_)) if load_if_exists => {
            let name = study_name.ok_or_else(|| {
                Error::ValueError(
                    "load_if_exists requires a study_name".into(),
                )
            })?;
            storage.get_study_id_from_name(name)?
        }
        Err(e) => return Err(e),
    };

    let name = storage.get_study_name_from_id(study_id)?;
    let stored_dirs = storage.get_study_directions(study_id)?;

    let sampler =
        sampler.unwrap_or_else(|| Arc::new(RandomSampler::new(None)));
    let pruner = pruner.unwrap_or_else(|| Arc::new(NopPruner::new()));

    Ok(Study::new(
        name,
        study_id,
        storage,
        stored_dirs,
        sampler,
        pruner,
    ))
}
