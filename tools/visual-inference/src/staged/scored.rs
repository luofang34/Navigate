//! Preserve assignment scores for diagnostics without changing geometric admission.
use super::{XFeatFeatures, XFeatStages};
use crate::{InferenceError, lighterglue};
use navigate_visual::{PixelMatch, VisualError};

/// A retained correspondence with its model assignment score.
///
/// The score is not a calibrated probability, an independence label or a position
/// uncertainty. It must not replace geometric verification.
#[derive(Clone, Copy, Debug)]
pub struct ScoredPixelMatch {
    /// Coordinates in the unchanged source images.
    pub pair: PixelMatch,
    /// Raw assignment value for the retained mutual match, greater than 0.1.
    pub assignment_score: f32,
}
impl XFeatStages {
    /// Match features and retain the compact masked matcher's assignment scores.
    ///
    /// Coordinates, mutuality, threshold and output order match the ordinary path.
    /// The returned scores can support a sampling experiment; they do not admit a pose.
    ///
    /// # Errors
    /// Rejects incompatible features, a non-compact masked matcher, or inference errors.
    pub fn match_features_scored_blocking(
        &mut self,
        reference: &XFeatFeatures,
        query: &XFeatFeatures,
    ) -> Result<Vec<ScoredPixelMatch>, VisualError> {
        self.check_contract(reference).map_err(|e| self.error(e))?;
        self.check_contract(query).map_err(|e| self.error(e))?;
        let result = match &mut self.matcher {
            Some(session) => lighterglue::match_prepared_scored(
                session,
                &reference.features,
                &query.features,
                &reference.normalized_pixels,
                &query.normalized_pixels,
            ),
            None => Err(InferenceError::Invalid(
                "assignment scores require a compact masked matcher".into(),
            )),
        };
        result.map_err(|e| self.error(e))
    }
}
