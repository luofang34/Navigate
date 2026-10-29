//! Semantic mask policies for aerial photo and video terrain matching (`photo_masks`).

use crate::{
    SegmentError,
    airframe::{border_support_mask, support_mask},
    morphology::{connected_components_8, erode_rect, morph_close},
};
use image::RgbImage;
use serde::{Deserialize, Serialize};

/// ADE20K class indices accepted as matchable ground/water surfaces.
pub const SURFACE_LABELS: [u8; 25] = [
    1, 4, 6, 9, 11, 13, 16, 17, 21, 25, 26, 29, 34, 46, 48, 52, 54, 60, 61, 68, 72, 79, 84, 91, 94,
];
/// Additional ADE20K lake class index (`128`) accepted in surface mode.
pub const LAKE_LABEL: u8 = 128;

/// ADE20K class indices excluded as sky, cabin, window, person, or aircraft obstructions.
pub const OBSTRUCTION_LABELS: [u8; 7] = [0, 2, 5, 8, 12, 18, 90];

/// ADE20K classes that count as cabin/airframe only when connected to an image border.
const BORDER_OBSTRUCTION_LABELS: [u8; 5] = [0, 5, 8, 18, 90];

/// Semantic occlusion policy controlling which pixels remain valid for terrain matching.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MaskPolicy {
    /// Exclude all [`OBSTRUCTION_LABELS`] above `confidence >= 0.35`.
    Occlusions,
    /// Retain only [`SURFACE_LABELS`] (plus lake `128`) and exclude [`OBSTRUCTION_LABELS`].
    Surfaces,
    /// Exclude sky/person plus border-connected cabin/airframe components.
    AttachedOcclusions,
    /// `AttachedOcclusions` plus elongated strut components (`support_mask`).
    AirframeOcclusions,
    /// `AirframeOcclusions` plus straight border line pairs bracketing airplane pixels.
    AirframeLines,
}

impl MaskPolicy {
    /// Parse a CLI policy identifier.
    ///
    /// # Errors
    ///
    /// Returns [`SegmentError::Invalid`] when `value` is unknown.
    pub fn parse(value: &str) -> Result<Self, SegmentError> {
        match value {
            "occlusions" => Ok(Self::Occlusions),
            "surfaces" => Ok(Self::Surfaces),
            "attached-occlusions" => Ok(Self::AttachedOcclusions),
            "airframe-occlusions" => Ok(Self::AirframeOcclusions),
            "airframe-lines" => Ok(Self::AirframeLines),
            other => Err(SegmentError::Invalid(format!(
                "unknown mask policy '{other}'"
            ))),
        }
    }

    /// Return the canonical kebab-case policy identifier.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Occlusions => "occlusions",
            Self::Surfaces => "surfaces",
            Self::AttachedOcclusions => "attached-occlusions",
            Self::AirframeOcclusions => "airframe-occlusions",
            Self::AirframeLines => "airframe-lines",
        }
    }
}

/// Coverage fractions produced alongside a binary valid-terrain mask.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaskSummary {
    /// Policy used to derive the mask.
    pub policy: MaskPolicy,
    /// Fraction of pixels retained for terrain matching (`[0, 1]`).
    pub valid_fraction: f64,
    /// Fraction of pixels classified as sky (`label == 2`).
    pub sky_fraction: f64,
    /// Fraction of pixels excluded as airframe/cabin obstructions.
    pub obstruction_fraction: f64,
}

/// Keep 8-connected components of `mask` (`width × height`) that touch the outer 2-pixel border.
pub fn attached_mask(mask: &[bool], width: usize, height: usize) -> Vec<bool> {
    let (comp_labels, stats) = connected_components_8(mask, width, height);
    let mut keep = vec![false; stats.len() + 1];
    for stat in &stats {
        let touches = stat.left <= 2
            || stat.top <= 2
            || stat.left + stat.width >= width.saturating_sub(2)
            || stat.top + stat.height >= height.saturating_sub(2);
        if touches {
            keep[stat.label as usize] = true;
        }
    }
    comp_labels
        .into_iter()
        .map(|l| l != 0 && keep[l as usize])
        .collect()
}

/// Compute the binary valid-terrain mask (`true` = usable ground/water pixel) and summary metrics.
///
/// # Errors
///
/// Returns [`SegmentError::Shape`] when `labels` or `confidence` lengths do not match `rgb`.
pub fn surface_mask(
    rgb: &RgbImage,
    labels: &[u8],
    confidence: &[f32],
    airplane_probs: Option<&[f32]>,
    policy: MaskPolicy,
    margin_pixels: usize,
) -> Result<(Vec<bool>, MaskSummary), SegmentError> {
    let w = rgb.width() as usize;
    let h = rgb.height() as usize;
    let n = w * h;
    if labels.len() != n {
        return Err(SegmentError::Shape {
            name: "labels".into(),
            expected: vec![h, w],
            actual: vec![labels.len()],
        });
    }
    if confidence.len() != n {
        return Err(SegmentError::Shape {
            name: "confidence".into(),
            expected: vec![h, w],
            actual: vec![confidence.len()],
        });
    }

    let sky_count = labels
        .iter()
        .zip(confidence)
        .filter(|&(&l, &c)| l == 2 && c >= 0.35)
        .count();

    let mut excluded = vec![false; n];
    let mut airframe = vec![false; n];

    match policy {
        MaskPolicy::Occlusions => {
            for i in 0..n {
                if OBSTRUCTION_LABELS.contains(&labels[i]) && confidence[i] >= 0.35 {
                    excluded[i] = true;
                    if labels[i] != 2 {
                        airframe[i] = true;
                    }
                }
            }
        }
        MaskPolicy::Surfaces => {
            for i in 0..n {
                let is_surface = (SURFACE_LABELS.contains(&labels[i]) || labels[i] == LAKE_LABEL)
                    && confidence[i] >= 0.25;
                let is_obstruction =
                    OBSTRUCTION_LABELS.contains(&labels[i]) && confidence[i] >= 0.35;
                if !is_surface || is_obstruction {
                    excluded[i] = true;
                    if is_obstruction && labels[i] != 2 {
                        airframe[i] = true;
                    }
                }
            }
        }
        MaskPolicy::AttachedOcclusions
        | MaskPolicy::AirframeOcclusions
        | MaskPolicy::AirframeLines => {
            let border_candidates: Vec<bool> = (0..n)
                .map(|i| BORDER_OBSTRUCTION_LABELS.contains(&labels[i]) && confidence[i] >= 0.35)
                .collect();
            airframe = attached_mask(&border_candidates, w, h);
            if matches!(
                policy,
                MaskPolicy::AirframeOcclusions | MaskPolicy::AirframeLines
            ) {
                let struts = support_mask(rgb, labels, confidence, &airframe);
                for i in 0..n {
                    airframe[i] |= struts[i];
                }
            }
            if policy == MaskPolicy::AirframeLines {
                let wings = border_support_mask(rgb, labels, airplane_probs);
                for i in 0..n {
                    airframe[i] |= wings[i];
                }
            }
            for i in 0..n {
                let sky_or_person = matches!(labels[i], 2 | 12) && confidence[i] >= 0.35;
                excluded[i] = airframe[i] || sky_or_person;
            }
        }
    }

    let closed_excluded = morph_close(&excluded, w, h, 4);
    let raw_keep: Vec<bool> = closed_excluded.into_iter().map(|e| !e).collect();
    let keep = erode_rect(&raw_keep, w, h, margin_pixels, true);

    let valid_count = keep.iter().filter(|&&b| b).count();
    let obstruction_count = airframe.iter().filter(|&&b| b).count();
    let denom = (n.max(1)) as f64;
    Ok((
        keep,
        MaskSummary {
            policy,
            valid_fraction: valid_count as f64 / denom,
            sky_fraction: sky_count as f64 / denom,
            obstruction_fraction: obstruction_count as f64 / denom,
        },
    ))
}
