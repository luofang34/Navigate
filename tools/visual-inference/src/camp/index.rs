use crate::InferenceError;
use navigate_visual::place_retrieval::{ReferenceCatalog, ReferenceId, RetrievalError};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

/// Descriptor data for the fixed CAMP global-feature export.
///
/// This format belongs to the adapter. Other retrieval methods need not use it.
/// The host verifies the catalog and descriptor bytes before it loads the index.
pub struct CampIndex {
    pub(super) catalog: ReferenceCatalog,
    pub(super) model_sha256: String,
    pub(super) digest: String,
    rows: BTreeMap<ReferenceId, usize>,
    descriptors: Vec<f32>,
}
impl CampIndex {
    /// Bind normalized 1024-value descriptor rows to their model and catalog.
    ///
    /// # Errors
    /// Rejects duplicate IDs, malformed identities, nonfinite or unnormalized rows.
    pub fn new(
        catalog: ReferenceCatalog,
        model_sha256: String,
        ids: Vec<ReferenceId>,
        descriptors: Vec<f32>,
    ) -> Result<Self, InferenceError> {
        if catalog.map.release_id.is_empty()
            || [
                &catalog.manifest_sha256,
                &catalog.map.manifest_sha256,
                &model_sha256,
            ]
            .iter()
            .any(|s| s.len() != 64 || !s.bytes().all(|b| b.is_ascii_hexdigit()))
            || ids.is_empty()
            || ids.len().checked_mul(1024) != Some(descriptors.len())
        {
            return Err(InferenceError::Invalid(
                "invalid CAMP catalog identity or row count".into(),
            ));
        }
        let mut rows = BTreeMap::new();
        let mut hash = Sha256::new();
        hash.update(b"navigate-camp-index-v1");
        for value in [
            &catalog.map.release_id,
            &catalog.map.manifest_sha256,
            &catalog.manifest_sha256,
            &model_sha256,
        ] {
            hash.update((value.len() as u64).to_le_bytes());
            hash.update(value.as_bytes());
        }
        for (row, (id, values)) in ids
            .into_iter()
            .zip(descriptors.as_chunks::<1024>().0)
            .enumerate()
        {
            if rows.insert(id, row).is_some() {
                return Err(InferenceError::Invalid(format!(
                    "duplicate CAMP reference ID {}",
                    id.0
                )));
            }
            validate_descriptor(values)?;
            hash.update(id.0.to_le_bytes());
            for v in values {
                hash.update(v.to_le_bytes());
            }
        }
        Ok(Self {
            catalog,
            model_sha256,
            rows,
            descriptors,
            digest: format!("{:x}", hash.finalize()),
        })
    }

    pub(super) fn eligible_rows(
        &self,
        eligible: &[ReferenceId],
        limit: usize,
    ) -> Result<Vec<(ReferenceId, usize)>, RetrievalError> {
        if limit > 4096 {
            return Err(RetrievalError::Invalid(
                "reference limit exceeds 4096".into(),
            ));
        }
        let mut seen = BTreeSet::new();
        eligible
            .iter()
            .map(|&id| {
                if !seen.insert(id) {
                    return Err(RetrievalError::Invalid(format!(
                        "duplicate eligible ID {}",
                        id.0
                    )));
                }
                self.rows.get(&id).map(|&row| (id, row)).ok_or_else(|| {
                    RetrievalError::Invalid(format!(
                        "reference ID {} is absent from the pinned catalog",
                        id.0
                    ))
                })
            })
            .collect()
    }

    pub(super) fn rank(
        &self,
        query: &[f32],
        rows: &[(ReferenceId, usize)],
        limit: usize,
    ) -> Vec<ReferenceId> {
        let mut ranked: Vec<_> = rows
            .iter()
            .map(|&(id, row)| {
                let values = &self.descriptors[row * 1024..(row + 1) * 1024];
                let score: f32 = values.iter().zip(query).map(|(a, b)| a * b).sum();
                (id, score)
            })
            .collect();
        ranked.sort_by(|(aid, a), (bid, b)| b.total_cmp(a).then(aid.cmp(bid)));
        ranked.into_iter().take(limit).map(|(id, _)| id).collect()
    }
}

pub(super) fn validate_descriptor(values: &[f32]) -> Result<(), InferenceError> {
    let norm = values.iter().map(|v| v * v).sum::<f32>();
    if values.len() != 1024 || values.iter().any(|v| !v.is_finite()) || (norm - 1.0).abs() > 0.001 {
        return Err(InferenceError::Invalid(
            "invalid CAMP descriptor length or unit norm".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
