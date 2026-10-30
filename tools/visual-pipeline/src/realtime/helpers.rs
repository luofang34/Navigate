//! Helper routines for [`super::build_reference_bank`] and [`super::run_realtime_bank_match`].

use crate::{
    PipelineError,
    matcher_client::default_burn_device,
    pose::{PoseCandidate, fit_pose_hybrid},
    surface::{Surface, SurfaceRayHit},
};
use image::DynamicImage;
use nalgebra::Matrix3;
use navigate_match::{
    FastMatchConfig, IndexedMatch, LighterGlue, PrecomputedMatch, PrecomputedViewEntry,
    SparseFeatures, Weights, XFeat,
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{fs, path::Path};

pub(crate) fn polish_pose_sparse(
    surface: &Surface,
    xfeat: &XFeat,
    lighterglue: Option<&LighterGlue>,
    qry_feat: &SparseFeatures,
    pose: ([f64; 3], Matrix3<f64>),
    k_render: &Matrix3<f64>,
    render_size: [u32; 2],
) -> Result<Option<(PoseCandidate, usize, usize)>, PipelineError> {
    let kp_cap = qry_feat.len().clamp(1536, 4096);
    let ((ref_rgb, _, _), render_stats) =
        surface.render_adaptive(pose.0, &pose.1, k_render, render_size, 4, 8, 0.025);
    let ref_gray = DynamicImage::ImageRgb8(ref_rgb).into_luma8();
    let ref_feat = pollster::block_on(xfeat.extract(&ref_gray, kp_cap))?;
    let (matches, layers) = if let Some(lg) = lighterglue {
        let cfg = FastMatchConfig {
            max_keypoints: kp_cap,
            ..FastMatchConfig::balanced_search()
        };
        let (m, stats) = pollster::block_on(lg.match_features_fast(&ref_feat, qry_feat, &cfg))?;
        (m, stats.layers_executed)
    } else {
        (ref_feat.mutual_matches(qry_feat), 0)
    };
    let ref_pixels: Vec<[f64; 2]> = matches
        .iter()
        .map(|m| [m.reference.x, m.reference.y])
        .collect();
    let hits = surface.raycast_pixels(pose.0, &pose.1, k_render, &ref_pixels);
    let mut world_pts = Vec::with_capacity(matches.len());
    let mut img_px = Vec::with_capacity(matches.len());
    for (m, hit) in matches.iter().zip(hits) {
        if let Some(h) = hit {
            world_pts.push(h.world_enu);
            img_px.push([m.query.x, m.query.y]);
        }
    }
    let cand = fit_pose_hybrid(&world_pts, &img_px, k_render, render_size, Some(pose));
    Ok(cand.map(|c| (c, render_stats.rays_cast + ref_pixels.len(), layers)))
}

pub(crate) fn filter_terrain_features(
    raw: &SparseFeatures,
    hits: &[Option<SurfaceRayHit>],
) -> (SparseFeatures, Vec<[f64; 3]>) {
    let mut out = SparseFeatures::empty();
    let mut world = Vec::new();
    for (i, hit) in hits.iter().enumerate() {
        if let Some(h) = hit {
            out.pixels.push(raw.pixels[i]);
            out.model_pixels.push(raw.model_pixels[i]);
            out.scores.push(raw.scores[i]);
            out.descriptors
                .extend_from_slice(&raw.descriptors[i * 64..(i + 1) * 64]);
            world.push(h.world_enu);
        }
    }
    (out, world)
}

pub(super) fn collect_bank_correspondences(
    matches: &[PrecomputedMatch],
) -> (Vec<[f64; 3]>, Vec<[f64; 2]>) {
    let mut world = Vec::with_capacity(matches.len());
    let mut px = Vec::with_capacity(matches.len());
    for m in matches {
        if let Some(w) = m.world_point_enu {
            world.push(w);
            px.push(m.query_pixel);
        }
    }
    (world, px)
}

pub(super) fn lookup_indexed_3d_matches(
    entry: &PrecomputedViewEntry,
    matches: &[IndexedMatch],
) -> (Vec<[f64; 3]>, Vec<[f64; 2]>) {
    let mut world = Vec::with_capacity(matches.len());
    let mut px = Vec::with_capacity(matches.len());
    for m in matches {
        if let Some(&w) = entry.world_points_enu.get(m.ref_index) {
            world.push(w);
            px.push([m.pixel_match.query.x, m.pixel_match.query.y]);
        }
    }
    (world, px)
}

pub(crate) fn log_attempt(
    log: &mut Vec<Value>,
    backend: &str,
    seed: usize,
    step: usize,
    sketch_score: f32,
    pairs: usize,
    cand: Option<&PoseCandidate>,
) {
    if let Some(c) = cand {
        log.push(json!({
            "backend": backend,
            "seed": seed,
            "step": step,
            "sketch_score": sketch_score,
            "pairs": c.score.pairs,
            "inliers": c.score.inliers,
            "train_inliers": c.score.train_inliers,
            "train_rms_px": c.score.train_rms_px,
            "holdout_count": c.score.holdout_count,
            "holdout_inlier_fraction": c.score.holdout_inlier_fraction,
            "holdout_median_px": c.score.holdout_median_px,
            "geometry_accepted": c.score.geometry_accepted,
        }));
    } else {
        log.push(json!({
            "backend": backend,
            "seed": seed,
            "step": step,
            "sketch_score": sketch_score,
            "pairs": pairs,
            "inliers": 0,
            "geometry_accepted": false,
        }));
    }
}

pub(crate) fn raw_file_sha256(path: &Path) -> Result<[u8; 32], PipelineError> {
    let bytes = fs::read(path).map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    Ok(Sha256::digest(&bytes).into())
}

pub(crate) fn load_xfeat_model(path: &Path) -> Result<(XFeat, [u8; 32]), PipelineError> {
    let bytes = fs::read(path).map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let digest: [u8; 32] = Sha256::digest(&bytes).into();
    let weights = Weights::from_safetensors(&bytes)?;
    let (device, _) = default_burn_device();
    Ok((XFeat::load(weights, &device)?, digest))
}

/// Check a bank `encoder_contract` stamp against the loaded XFeat model.
pub(crate) fn check_encoder_contract(
    bank_kind: &str,
    stamp: &[u8; 32],
    xfeat: &XFeat,
    weights_file_sha256: &[u8; 32],
) -> Result<(), PipelineError> {
    if xfeat.accepts_encoder_contract(stamp, weights_file_sha256) {
        return Ok(());
    }
    Err(PipelineError::Invalid(format!(
        "{bank_kind} encoder_contract {} is neither the XFeat encoder contract {} nor the --model file digest {}",
        hex(stamp),
        hex(&xfeat.encoder_contract()),
        hex(weights_file_sha256),
    )))
}

/// Lowercase hex form of a SHA-256 digest for reports and error messages.
pub(crate) fn hex(digest: &[u8; 32]) -> String {
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

pub(crate) fn load_lighterglue_model(
    path: &Path,
    device: &burn::tensor::Device,
) -> Result<LighterGlue, PipelineError> {
    let bytes = fs::read(path).map_err(|source| PipelineError::Io {
        path: path.to_path_buf(),
        source,
    })?;
    let weights = Weights::from_safetensors(&bytes)?;
    Ok(LighterGlue::load(weights, device)?)
}

pub(crate) fn warmup_gpu_pipelines(
    xfeat: &XFeat,
    lighterglue: Option<&LighterGlue>,
) -> Result<f64, PipelineError> {
    #[cfg(any(feature = "wgpu", feature = "metal", feature = "cuda"))]
    {
        let t0 = std::time::Instant::now();
        let mut patch = image::GrayImage::new(32, 32);
        for (x, y, p) in patch.enumerate_pixels_mut() {
            p.0[0] = ((x * 17 + y * 31) & 0xff) as u8;
        }
        let feat = pollster::block_on(xfeat.extract(&patch, 32))?;
        if let Some(lg) = lighterglue
            && !feat.is_empty()
        {
            let cfg = FastMatchConfig::realtime_tracking();
            let _ = pollster::block_on(lg.match_features_fast(&feat, &feat, &cfg))?;
        }
        Ok(t0.elapsed().as_secs_f64() * 1000.0)
    }
    #[cfg(not(any(feature = "wgpu", feature = "metal", feature = "cuda")))]
    {
        let _ = (xfeat, lighterglue);
        Ok(0.0)
    }
}

pub(crate) fn rot_to_row_major(r: &Matrix3<f64>) -> [f64; 9] {
    [
        r[(0, 0)],
        r[(0, 1)],
        r[(0, 2)],
        r[(1, 0)],
        r[(1, 1)],
        r[(1, 2)],
        r[(2, 0)],
        r[(2, 1)],
        r[(2, 2)],
    ]
}

pub(crate) fn rot_from_row_major(m: [f64; 9]) -> Matrix3<f64> {
    Matrix3::new(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8])
}
