use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::study::StudyDirection;

/// An immutable snapshot of a study's metadata.
///
/// Returned by `Storage::get_all_studies()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenStudy {
    pub study_id: i64,
    pub study_name: String,
    pub directions: Vec<StudyDirection>,
    pub user_attrs: HashMap<String, serde_json::Value>,
    pub system_attrs: HashMap<String, serde_json::Value>,
}
