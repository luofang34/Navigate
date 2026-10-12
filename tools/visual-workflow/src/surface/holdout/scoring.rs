//! Keep test outliers and behind-camera projections in the residual distribution.
use crate::evaluate::metrics::partition;
use navigate_visual::{CameraPose, Frame, OrthographicReference, PixelMatch};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

pub(super) struct Split {
    pub training: Vec<PixelMatch>,
    pub heldout: Vec<PixelMatch>,
    pub boundary: usize,
    pub invalid: usize,
}
pub(super) fn split(pairs: &[PixelMatch], window: Option<[u32; 4]>, fold: u32) -> Split {
    let offset = window.map_or([0.0, 0.0], |w| [f64::from(w[0]), f64::from(w[1])]);
    let mut result = Split {
        training: Vec::new(),
        heldout: Vec::new(),
        boundary: 0,
        invalid: 0,
    };
    for pair in pairs {
        let global = PixelMatch {
            reference: pair.reference + nalgebra::Vector2::from(offset),
            ..*pair
        };
        if !global
            .reference
            .iter()
            .all(|v| v.is_finite() && (0.0..8192.0).contains(v))
        {
            result.invalid = result.invalid.wrapping_add(1);
            continue;
        }
        match partition(&global, fold) {
            Some(true) => result.heldout.push(*pair),
            Some(false) => result.training.push(*pair),
            None => result.boundary = result.boundary.wrapping_add(1),
        }
    }
    result
}
pub(super) fn digest(pairs: &[PixelMatch]) -> String {
    let mut hash = Sha256::new();
    hash.update(b"orthographic-heldout-ordered-local-pairs-v1");
    hash.update((pairs.len() as u64).to_le_bytes());
    for pair in pairs {
        for v in [
            pair.reference.x,
            pair.reference.y,
            pair.query.x,
            pair.query.y,
        ] {
            hash.update(v.to_le_bytes());
        }
    }
    hex::encode(hash.finalize())
}
pub(super) fn score(
    pairs: &[PixelMatch],
    frame: &Frame,
    reference: &OrthographicReference,
    pose: &CameraPose,
) -> Value {
    let mut errors = Vec::new();
    let mut invalid_query = 0usize;
    let mut missing_surface = 0usize;
    let mut behind_camera = 0usize;
    let mut non_finite_projection = 0usize;
    for pair in pairs {
        let q = pair.query;
        if !q.iter().all(|v| v.is_finite())
            || q.x < 0.0
            || q.y < 0.0
            || q.x >= f64::from(frame.camera.width - 1)
            || q.y >= f64::from(frame.camera.height - 1)
        {
            invalid_query = invalid_query.wrapping_add(1);
            continue;
        }
        let Some(world) = reference.world_at(pair.reference) else {
            missing_surface = missing_surface.wrapping_add(1);
            continue;
        };
        let error = if let Some(pixel) = frame.camera.project(pose, world) {
            let error = (pixel - q).norm();
            if error.is_finite() {
                error
            } else {
                non_finite_projection = non_finite_projection.wrapping_add(1);
                f64::INFINITY
            }
        } else {
            behind_camera = behind_camera.wrapping_add(1);
            f64::INFINITY
        };
        errors.push(error);
    }
    errors.sort_by(f64::total_cmp);
    let quantile = |q: f64| {
        errors
            .get(((errors.len().saturating_sub(1) as f64) * q).ceil() as usize)
            .copied()
    };
    json!({"raw_matches":pairs.len(),"invalid_query":invalid_query,"missing_surface":missing_surface,"valid_3d":errors.len(),
        "behind_camera":behind_camera,"non_finite_projection":non_finite_projection,"within_2_5_px":errors.iter().filter(|e|**e<=2.5).count(),
        "median_px":quantile(0.5).filter(|v|v.is_finite()),"p90_px":quantile(0.9).filter(|v|v.is_finite()),
        "median_is_infinite":quantile(0.5).is_some_and(|v|!v.is_finite()),"p90_is_infinite":quantile(0.9).is_some_and(|v|!v.is_finite())})
}
