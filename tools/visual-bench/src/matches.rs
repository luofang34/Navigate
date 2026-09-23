//! Image-bound correspondences from an external feature matcher.

use crate::{BenchError, package::digest, read_blocking};
use image::GrayImage;
use nalgebra::Vector2;
use navigate_visual::{ImageMatcher, PixelMatch, ReferenceView, VisualError};
use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Pair {
    reference: [f64; 2],
    query: [f64; 2],
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct VerifiedMatches {
    backend_identity: String,
    reference_image_sha256: String,
    reference_depth_sha256: String,
    query_image_sha256: String,
    matches: Vec<Pair>,
}

impl VerifiedMatches {
    pub fn open_blocking(path: &Path) -> Result<Self, BenchError> {
        let input: Self =
            serde_json::from_slice(&read_blocking(path)?).map_err(|source| BenchError::Json {
                path: path.to_owned(),
                source,
            })?;
        if input.backend_identity.is_empty() || input.matches.len() > 16384 {
            return Err(BenchError::Record {
                reason: "matcher identity is empty or correspondence count exceeds 16384".into(),
            });
        }
        Ok(input)
    }

    pub fn validate_depth(&self, reference: &ReferenceView) -> Result<(), VisualError> {
        if depth_digest(&reference.depth_m) != self.reference_depth_sha256 {
            return Err(VisualError::Invalid {
                field: "correspondence reference depth digest",
            });
        }
        Ok(())
    }
}

pub(crate) fn depth_digest(depth: &[f32]) -> String {
    digest(
        &depth
            .iter()
            .flat_map(|d| d.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

impl ImageMatcher for VerifiedMatches {
    fn identity(&self) -> &str {
        &self.backend_identity
    }

    fn match_images_blocking(
        &mut self,
        reference: &GrayImage,
        query: &GrayImage,
    ) -> Result<Vec<PixelMatch>, VisualError> {
        if digest(reference.as_raw()) != self.reference_image_sha256
            || digest(query.as_raw()) != self.query_image_sha256
        {
            return Err(VisualError::Invalid {
                field: "correspondence image digest",
            });
        }
        Ok(self
            .matches
            .iter()
            .map(|pair| PixelMatch {
                reference: Vector2::from_row_slice(&pair.reference),
                query: Vector2::from_row_slice(&pair.query),
            })
            .collect())
    }
}

#[cfg(test)]
mod tests;
