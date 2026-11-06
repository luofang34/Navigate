//! Single-pose orthophoto GPU or adaptive CPU terrain polish.

use super::{seeds::SeedPose, virt_mask::mask_valid_fraction};
use crate::{
    error::FlightVisualError,
    runtime::LoadedXFeat,
    surface::{Surface, SurfaceGpuRenderer},
};
use image::{DynamicImage, GrayImage};
use nalgebra::Matrix3;
use navigate_match::{FastMatchConfig, SparseFeatures, SparseMatcher};
use navigate_visual::{
    PoseCandidate,
    holdout_pnp::{candidate_priority, fit_pose_hybrid, refine_seed_eye_2pt},
};
use std::sync::Mutex;

/// Render and match the orthophoto surface at `pose` using `SurfaceGpuRenderer` or sparse CPU raycasting.
///
/// # Errors
///
/// Returns [`FlightVisualError`] when feature extraction or matcher inference fails.
#[expect(
    clippy::too_many_arguments,
    reason = "sparse pose polish borrows surface, optional GPU renderer, feature models, query keypoints, and camera geometry"
)]
pub fn polish_pose_sparse(
    surface: &Surface,
    gpu_renderer: Option<&Mutex<Option<SurfaceGpuRenderer>>>,
    xfeat: &LoadedXFeat,
    matcher: Option<&dyn SparseMatcher>,
    qry_feat: &SparseFeatures,
    pose: SeedPose,
    k_render: &Matrix3<f64>,
    render_size: [u32; 2],
) -> Result<Option<(PoseCandidate, usize, usize)>, FlightVisualError> {
    polish_pose_sparse_masked(
        surface,
        gpu_renderer,
        xfeat,
        matcher,
        qry_feat,
        pose,
        k_render,
        render_size,
        (None, false, false),
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "masked sparse pose polish borrows surface, optional GPU renderer, feature models, query keypoints, camera geometry, and polish options"
)]
pub(crate) fn polish_pose_sparse_masked(
    surface: &Surface,
    gpu_renderer: Option<&Mutex<Option<SurfaceGpuRenderer>>>,
    xfeat: &LoadedXFeat,
    matcher: Option<&dyn SparseMatcher>,
    qry_feat: &SparseFeatures,
    pose: SeedPose,
    k_render: &Matrix3<f64>,
    render_size: [u32; 2],
    (mask_small, pace_realtime, compact_nadir): (Option<&GrayImage>, bool, bool),
) -> Result<Option<(PoseCandidate, usize, usize)>, FlightVisualError> {
    if !pose.0.iter().all(|v| v.is_finite()) || pose.0[0].hypot(pose.0[1]) > 95_000.0 {
        return Ok(None);
    }
    let subwindow_masked = mask_small.is_some_and(|m| mask_valid_fraction(m) < 0.70);
    let fast_paced_nadir = pace_realtime && !subwindow_masked && -pose.1[(2, 2)] > 0.85;
    let kp_cap = if subwindow_masked {
        if compact_nadir {
            1024
        } else if pace_realtime {
            1536
        } else {
            2048
        }
    } else if fast_paced_nadir {
        if compact_nadir { 960 } else { 1408 }
    } else {
        qry_feat.len().clamp(1024, 1536)
    };
    let lg_kp_cap = if subwindow_masked {
        if compact_nadir {
            896
        } else if pace_realtime {
            1280
        } else {
            2048
        }
    } else if fast_paced_nadir && compact_nadir {
        640
    } else {
        1024
    };
    let mut gpu_view_opt = None;
    if let Some(lock) = gpu_renderer
        && let Ok(mut guard) = lock.lock()
    {
        if guard.is_none() {
            *guard = SurfaceGpuRenderer::try_new(surface, k_render, render_size);
        }
        if let Some(gpu) = guard.as_mut() {
            gpu_view_opt = gpu
                .render_view(surface, pose.0, &pose.1, k_render, render_size)
                .ok();
        }
    }

    let (ref_gray, pre_rays) = if let Some(ref gv) = gpu_view_opt {
        (gv.image.clone(), 0_usize)
    } else {
        let ((ref_rgb, _, _), render_stats) =
            surface.render_adaptive(pose.0, &pose.1, k_render, render_size, 4, 8, 0.025);
        (
            DynamicImage::ImageRgb8(ref_rgb).into_luma8(),
            render_stats.rays_cast,
        )
    };

    let raw_ref = xfeat.extract(&ref_gray, kp_cap)?;
    let ref_feat = if let Some(ref gv) = gpu_view_opt {
        let unproj_all = gv.unproject_pixels(surface, &raw_ref.pixels);
        let filtered = filter_terrain_features_only(&raw_ref, &unproj_all);
        if filtered.len() < 16 {
            return Ok(None);
        }
        filtered
    } else {
        raw_ref
    };

    let (matches, layers) = if let Some(m) = matcher {
        let pol_layers = if (fast_paced_nadir || subwindow_masked) && compact_nadir {
            4
        } else {
            6
        };
        let cfg = FastMatchConfig {
            max_keypoints: kp_cap.min(lg_kp_cap),
            min_layers: pol_layers,
            max_layers: pol_layers,
            exit_check_layers: 0,
            prune_after_layer: None,
            ..FastMatchConfig::default()
        };
        let mut batched = m.match_batch_indexed_blocking(&[&ref_feat], qry_feat, &cfg)?;
        let (idx_m, stats) = batched.pop().unwrap_or_default();
        let px_m = idx_m.into_iter().map(|im| im.pixel_match).collect();
        (px_m, stats.layers_executed)
    } else {
        (ref_feat.mutual_matches(qry_feat), 0)
    };

    let ref_pixels: Vec<[f64; 2]> = matches
        .iter()
        .map(|m| [m.reference.x, m.reference.y])
        .collect();
    let mut world_pts = Vec::with_capacity(matches.len());
    let mut img_px = Vec::with_capacity(matches.len());
    let rays_used = if let Some(gv) = gpu_view_opt {
        let unproj = gv.unproject_pixels(surface, &ref_pixels);
        for (m, w_opt) in matches.iter().zip(unproj) {
            if let Some(w) = w_opt {
                world_pts.push(w);
                img_px.push([m.query.x, m.query.y]);
            }
        }
        0_usize
    } else {
        let hits = surface.raycast_pixels(pose.0, &pose.1, k_render, &ref_pixels);
        for (m, hit) in matches.iter().zip(hits) {
            if let Some(h) = hit {
                world_pts.push(h.world_enu);
                img_px.push([m.query.x, m.query.y]);
            }
        }
        pre_rays + ref_pixels.len()
    };

    let mut cand = fit_pose_hybrid(&world_pts, &img_px, k_render, render_size, Some(pose));
    if !cand.as_ref().is_some_and(|c| c.score.geometry_accepted)
        && let Some(tri_init) = refine_seed_eye_2pt(&world_pts, &img_px, pose, k_render)
        && let Some(tri_cand) =
            fit_pose_hybrid(&world_pts, &img_px, k_render, render_size, Some(tri_init))
        && cand.as_ref().is_none_or(|prev| {
            candidate_priority(&tri_cand.score) > candidate_priority(&prev.score)
        })
    {
        cand = Some(tri_cand);
    }
    Ok(cand.map(|c| (c, rays_used, layers)))
}

fn filter_terrain_features_only(
    raw: &SparseFeatures,
    world_opts: &[Option<[f64; 3]>],
) -> SparseFeatures {
    let mut out = SparseFeatures::empty();
    for (i, w_opt) in world_opts.iter().enumerate() {
        if w_opt.is_some() {
            out.pixels.push(raw.pixels[i]);
            out.model_pixels.push(raw.model_pixels[i]);
            out.scores.push(raw.scores[i]);
            out.descriptors
                .extend_from_slice(&raw.descriptors[i * 64..(i + 1) * 64]);
        }
    }
    out
}
