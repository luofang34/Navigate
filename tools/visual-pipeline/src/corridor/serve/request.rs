//! Request schema and `CorridorTrackOptions` builder for `corridor-serve`.

use super::CorridorServeOptions;
use crate::{
    CorridorTrackOptions, PipelineError, digest::file_sha256, metadata::extract_photo_claims,
    testset::PhotoCase,
};
use serde::Deserialize;
use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

#[derive(Debug, Default, Deserialize)]
pub(super) struct ServeRequest {
    #[serde(default)]
    pub op: Option<String>,
    #[serde(default)]
    pub request_id: Option<String>,
    #[serde(default)]
    pub image: Option<PathBuf>,
    #[serde(default)]
    pub id: Option<String>,
    #[serde(default)]
    pub valid_mask: Option<PathBuf>,
    #[serde(default)]
    pub dataset: Option<PathBuf>,
    #[serde(default)]
    pub profile: Option<PathBuf>,
    #[serde(default)]
    pub output: Option<PathBuf>,
    #[serde(default)]
    pub only: Option<Vec<String>>,
    #[serde(default)]
    pub prior_lat: Option<f64>,
    #[serde(default)]
    pub prior_lon: Option<f64>,
    #[serde(default)]
    pub prior_alt: Option<f64>,
    #[serde(default)]
    pub prior_heading: Option<f64>,
    #[serde(default)]
    pub focal_35mm: Option<f64>,
    #[serde(default)]
    pub heading_step: Option<f64>,
    #[serde(default)]
    pub height_factors: Option<Vec<f64>>,
    #[serde(default)]
    pub attempts: Option<usize>,
    #[serde(default)]
    pub query_keypoints: Option<usize>,
    #[serde(default)]
    pub polish: Option<bool>,
    #[serde(default)]
    pub pace_realtime: Option<bool>,
    #[serde(default)]
    pub sync_refresh: Option<bool>,
}

pub(super) struct PreparedServeRequest {
    pub track_opts: CorridorTrackOptions,
    pub single_case: Option<PhotoCase>,
    pub persist_output: Option<PathBuf>,
}

impl ServeRequest {
    pub(super) fn into_prepared(
        self,
        base: &CorridorServeOptions,
    ) -> Result<PreparedServeRequest, PipelineError> {
        let persist_output = self.output.clone();
        let output_dir = self.output.unwrap_or_default();

        let (dataset_dir, profile_path, single_case) = if let Some(ref img_path) = self.image {
            let parent = self
                .dataset
                .or_else(|| img_path.parent().map(Path::to_path_buf))
                .unwrap_or_else(|| PathBuf::from("."));
            let rel_img = img_path
                .strip_prefix(&parent)
                .unwrap_or(img_path.as_path())
                .to_string_lossy()
                .into_owned();
            let (w, h) =
                image::image_dimensions(img_path).map_err(|source| PipelineError::Image {
                    path: img_path.clone(),
                    source,
                })?;
            let (_, claims, _) = extract_photo_claims(img_path);
            let case_id = self.id.unwrap_or_else(|| {
                img_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("frame-0")
                    .to_owned()
            });
            let (valid_mask, mask_sha256) = if let Some(ref mp) = self.valid_mask {
                let m_sha = file_sha256(mp)?;
                let rel_m = mp
                    .strip_prefix(&parent)
                    .unwrap_or(mp.as_path())
                    .to_string_lossy()
                    .into_owned();
                (rel_m, m_sha)
            } else {
                (String::new(), String::new())
            };
            let case = PhotoCase {
                id: case_id,
                image: rel_img,
                image_sha256: String::new(),
                size: [w, h],
                valid_mask,
                mask_sha256,
                mask_status: String::new(),
                semantic_labels: None,
                semantic_sha256: None,
                prior: Some(claims),
                sequence: Some(0),
                time_s: Some(0.0),
            };
            (parent, PathBuf::new(), Some(case))
        } else {
            let ds = self.dataset.ok_or_else(|| {
                PipelineError::Invalid("request requires `image` or `dataset`".into())
            })?;
            let pf = self.profile.ok_or_else(|| {
                PipelineError::Invalid("dataset request requires `profile`".into())
            })?;
            (ds, pf, None)
        };

        let prior_override = match (self.prior_lat, self.prior_lon) {
            (Some(lat), Some(lon)) => Some((
                [lat, lon, self.prior_alt.unwrap_or(950.0)],
                self.prior_heading,
            )),
            _ => None,
        };
        let height_factors = self
            .height_factors
            .filter(|v| !v.is_empty())
            .unwrap_or_else(|| vec![1.0]);
        let only_cases = self
            .only
            .map(|items| items.into_iter().collect::<BTreeSet<_>>());
        let polish_anchors = self.polish.unwrap_or(base.polish_anchors);
        let track_opts = CorridorTrackOptions {
            dataset_dir,
            profile_path,
            catalog_path: base.catalog_path.clone(),
            corridor_bank_path: base.corridor_bank_path.clone(),
            xfeat_weights_path: base.xfeat_weights_path.clone(),
            lighterglue_weights_path: base.lighterglue_weights_path.clone(),
            seg_model_dir: base.seg_model_dir.clone(),
            dinov2_weights_path: base.dinov2_weights_path.clone(),
            place_index_path: base.place_index_path.clone(),
            dynamics_profile: base.dynamics_profile,
            mask_policy: base.mask_policy.clone(),
            mask_interval: 10,
            mask_interval_s: None,
            output_dir,
            anchor_interval: 30,
            anchor_interval_s: None,
            anchor_uncertainty_m: None,
            anchor_on_cell_boundary: false,
            min_track_inliers: 32,
            map_cell_m: 500.0,
            anchor_attempts: self.attempts.unwrap_or(12),
            heading_step_deg: self.heading_step.unwrap_or(0.0),
            height_factors,
            query_keypoints: self.query_keypoints.unwrap_or(base.query_keypoints),
            polish_anchors,
            prior_override,
            focal_35mm_override: self.focal_35mm,
            only_cases,
            background_refresh: !self.sync_refresh.unwrap_or(false),
            pace_realtime: self.pace_realtime.unwrap_or(false),
            xfeat_runtime: base.xfeat_runtime.clone(),
        };

        Ok(PreparedServeRequest {
            track_opts,
            single_case,
            persist_output,
        })
    }
}
