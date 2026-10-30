//! Two-tier `25-30 Hz` temporal corridor tracker and [`MapCell`] EKF independence gate.

use super::{
    CorridorTrackOptions,
    admission::{
        FusionAdmission, classify_anchor_cell, seed_tracks, top_up_tracks, write_corridor_report,
    },
    anchor::acquire_corridor_anchor,
    masking::{AmortizedMasker, MaskFrameTelemetry},
};
use crate::{
    PipelineError,
    metadata::PositionClaim,
    pose::{PoseCandidate, fit_pose_hybrid, fit_pose_l1_gnc},
    realtime::helpers::{
        load_lighterglue_model, load_xfeat_model, log_attempt, raw_file_sha256, rot_to_row_major,
        warmup_gpu_pipelines,
    },
    stages::{
        estimate_intrinsics, load_case_mask, load_case_rgb, load_profile_cases, scale_k,
        scaled_size,
    },
    surface::Surface,
    testset::PhotoCase,
    util::write_json,
    video_ops::relative_rotation_ransac,
};
use image::imageops::{FilterType, colorops::grayscale};
use nalgebra::{Matrix3, Vector3};
use navigate_match::{
    ActiveTrackSet, CorridorLandmarkBank, HadamardProjector, PreparedKashinQuery, TrackMatchConfig,
};
use navigate_visual_session::MapCell;
use serde_json::{Value, json};
use std::{collections::BTreeMap, fs, time::Instant};

struct TrackState {
    active_tracks: ActiveTrackSet,
    prev_eye: Option<[f64; 3]>,
    prev_vel: [f64; 3],
    prev_rot: Option<Matrix3<f64>>,
    prev_rot_step: Matrix3<f64>,
    prev_prepared: Option<PreparedKashinQuery>,
    frames_since_anchor: usize,
    visited_cells: BTreeMap<MapCell, String>,
}

impl TrackState {
    fn new() -> Self {
        Self {
            active_tracks: ActiveTrackSet::empty(),
            prev_eye: None,
            prev_vel: [0.0; 3],
            prev_rot: None,
            prev_rot_step: Matrix3::identity(),
            prev_prepared: None,
            frames_since_anchor: 0,
            visited_cells: BTreeMap::new(),
        }
    }
}

/// Localize and track sequential frames against a [`CorridorLandmarkBank`] (`corridor-track`).
///
/// # Errors
///
/// Returns [`PipelineError`] when bank verification, masking, matching, or report writing fails.
pub fn run_corridor_track(opts: &CorridorTrackOptions) -> Result<Value, PipelineError> {
    let surface = Surface::from_catalog(&opts.catalog_path)?;
    let catalog_digest = raw_file_sha256(&opts.catalog_path)?;
    let (xfeat, encoder_digest) = load_xfeat_model(&opts.xfeat_weights_path)?;
    let lighterglue = opts
        .lighterglue_weights_path
        .as_deref()
        .map(|p| load_lighterglue_model(p, xfeat.device()))
        .transpose()?;
    let bank = CorridorLandmarkBank::load_file(&opts.corridor_bank_path)?;
    if bank.catalog_sha256 != catalog_digest {
        return Err(PipelineError::Invalid(
            "corridor bank catalog_sha256 does not match --catalog".into(),
        ));
    }
    if bank.encoder_contract != encoder_digest {
        return Err(PipelineError::Invalid(
            "corridor bank encoder_contract does not match --model".into(),
        ));
    }

    let mut masker = AmortizedMasker::new(
        opts.seg_model_dir.as_deref(),
        &opts.mask_policy,
        6,
        opts.mask_interval.max(1),
        xfeat.device(),
    )?;
    fs::create_dir_all(&opts.output_dir).map_err(|source| PipelineError::Io {
        path: opts.output_dir.clone(),
        source,
    })?;
    write_json(
        &opts.output_dir.join("run.json"),
        &json!({ "catalog_sha256": surface.catalog_sha256 }),
    )?;

    let projector = HadamardProjector::new(bank.seed);
    let shader_warmup_ms = warmup_gpu_pipelines(&xfeat, lighterglue.as_ref())?;
    let cases = load_profile_cases(&opts.profile_path)?;
    let kp_limit = opts.query_keypoints.clamp(16, 4096);
    let cell_m = opts.map_cell_m.clamp(50.0, 20_000.0);
    let min_trk_inliers = opts.min_track_inliers.max(24);
    let mut state = TrackState::new();
    let mut evaluated = 0_usize;
    let mut accepted = 0_usize;
    let mut warm_tracked_frames = 0_usize;
    let mut independent_fixes = 0_usize;
    let mut seg_refreshes = 0_usize;
    let mut telemetry = Vec::new();

    for raw_case in &cases {
        if let Some(allow) = &opts.only_cases
            && !allow.contains(&raw_case.id)
        {
            continue;
        }
        evaluated += 1;
        let case_t0 = Instant::now();
        let case = apply_case_overrides(raw_case, opts);
        let query_rgb = load_case_rgb(&opts.dataset_dir, &case.image)?;
        let disk_mask = (!case.valid_mask.is_empty())
            .then(|| load_case_mask(&opts.dataset_dir, &case.valid_mask, case.size))
            .transpose()?;

        let k = estimate_intrinsics(&case);
        let render_size = scaled_size(case.size, 800);
        let sx = f64::from(render_size[0]) / f64::from(case.size[0].max(1));
        let sy = f64::from(render_size[1]) / f64::from(case.size[1].max(1));
        let k_render = scale_k(&k, sx, sy);

        let qry_small = image::imageops::resize(
            &query_rgb,
            render_size[0],
            render_size[1],
            FilterType::Triangle,
        );
        let fallback_small = disk_mask.as_ref().map(|m| {
            image::imageops::resize(m, render_size[0], render_size[1], FilterType::Nearest)
        });
        let est_rot_for_mask = state.prev_rot.map(|r| state.prev_rot_step * r);
        let qry_gray = grayscale(&qry_small);
        let (mask_small, mask_tel): (_, MaskFrameTelemetry) = masker.mask_frame_with_luma(
            &qry_small,
            &qry_gray,
            fallback_small.as_ref(),
            est_rot_for_mask.as_ref(),
            &k_render,
        )?;
        seg_refreshes += usize::from(mask_tel.segformer_refreshed);
        let qry_kp_floor = 256_usize.min(kp_limit);
        let qry_kp_cap = (((kp_limit as f64) * mask_tel.valid_fraction.clamp(0.25, 1.0)).round()
            as usize)
            .clamp(qry_kp_floor, kp_limit);

        let t_xfeat = Instant::now();
        let qry_feat =
            pollster::block_on(xfeat.extract_masked(&qry_gray, Some(&mask_small), qry_kp_cap))?;
        let query_xfeat_ms = t_xfeat.elapsed().as_secs_f64() * 1000.0;
        let prepared = PreparedKashinQuery::from_features(&projector, &qry_feat)?;

        let t_solve = Instant::now();
        let mut attempts_log = Vec::new();
        let mut lg_layers = 0_usize;
        let mut polish_rays = 0_usize;
        let pred_pose = predict_interframe_pose(&state, &prepared, &k_render);
        let force_anchor =
            opts.anchor_interval > 0 && state.frames_since_anchor >= opts.anchor_interval;

        let mut solved_warm = None;
        if !state.active_tracks.is_empty()
            && !force_anchor
            && let Some(pred) = pred_pose
        {
            solved_warm = try_warm_track(
                &state.active_tracks,
                &prepared,
                pred,
                &k_render,
                render_size,
                min_trk_inliers,
                &mut attempts_log,
            )?;
        }

        let (best, admission) = if let Some(cand) = solved_warm {
            warm_tracked_frames += 1;
            state.frames_since_anchor += 1;
            if cand.score.inliers < 260 {
                top_up_tracks(
                    &mut state.active_tracks,
                    &surface,
                    &prepared,
                    &qry_feat,
                    &cand,
                );
            }
            (
                Some((0, 0, cand)),
                FusionAdmission {
                    tracking_mode: "warm_track",
                    evidence_kind: "surface_track_odometry",
                    independent_ekf_fix: false,
                    fusion_eligibility: "correlated_track",
                    map_cell: None,
                    earlier_anchor_id: None,
                },
            )
        } else {
            let acq = acquire_corridor_anchor(
                (&surface, &bank, &projector),
                (&xfeat, lighterglue.as_ref()),
                (&case, opts),
                (&qry_feat, &prepared),
                (&k, &k_render, render_size),
                (pred_pose, masker.horizon_attitude(&k_render)),
                &mut attempts_log,
            )?;
            lg_layers += acq.lg_layers;
            polish_rays += acq.polish_rays;
            if let Some((_, _, ref cand)) = acq.best
                && cand.score.geometry_accepted
            {
                state.frames_since_anchor = 0;
                state.active_tracks =
                    seed_tracks(&surface, &prepared, &qry_feat, cand, &acq.best_view);
                let adm = classify_anchor_cell(
                    &mut state.visited_cells,
                    &mut independent_fixes,
                    &surface.catalog_sha256,
                    cand.eye,
                    cell_m,
                    &case.id,
                );
                (acq.best, adm)
            } else {
                state.active_tracks = ActiveTrackSet::empty();
                (
                    acq.best,
                    FusionAdmission {
                        tracking_mode: "unresolved",
                        evidence_kind: "corridor_map_anchor",
                        independent_ekf_fix: false,
                        fusion_eligibility: "rejected",
                        map_cell: None,
                        earlier_anchor_id: None,
                    },
                )
            }
        };
        let solve_ms = t_solve.elapsed().as_secs_f64() * 1000.0;

        if let Some((_, _, ref cand)) = best
            && cand.score.geometry_accepted
        {
            if mask_tel.segformer_refreshed {
                masker.set_key_rotation(cand.rotation);
            }
            update_motion_state(&mut state, cand, prepared);
        }

        let total_case_ms = case_t0.elapsed().as_secs_f64() * 1000.0;
        let is_accepted = best
            .as_ref()
            .is_some_and(|(_, _, c)| c.score.geometry_accepted);
        telemetry.push(json!({
            "id": case.id,
            "accepted": is_accepted,
            "tracking_mode": admission.tracking_mode,
            "evidence_kind": admission.evidence_kind,
            "independent_ekf_fix": admission.independent_ekf_fix,
            "fusion_eligibility": admission.fusion_eligibility,
            "map_cell": admission.map_cell,
            "active_tracks": state.active_tracks.len(),
            "query_keypoints": qry_feat.len(),
            "lighterglue_layers": lg_layers,
            "polish_rays_cast": polish_rays,
            "segformer_refreshed": mask_tel.segformer_refreshed,
            "valid_mask_fraction": mask_tel.valid_fraction,
            "propeller_pixels": mask_tel.propeller_pixels,
            "mask_ms": mask_tel.mask_ms,
            "query_xfeat_ms": query_xfeat_ms,
            "solve_ms": solve_ms,
            "total_case_ms": total_case_ms,
        }));

        write_corridor_report(
            &opts.output_dir,
            &surface,
            &case.id,
            &admission,
            attempts_log,
            best,
            &mut accepted,
        )?;
    }

    Ok(json!({
        "evaluated": evaluated,
        "accepted": accepted,
        "warm_tracked_frames": warm_tracked_frames,
        "independent_ekf_fixes": independent_fixes,
        "segformer_refreshes": seg_refreshes,
        "corridor_tiles": bank.tiles.len(),
        "corridor_landmarks": bank.total_landmarks(),
        "shader_warmup_ms": shader_warmup_ms,
        "output": opts.output_dir.display().to_string(),
        "telemetry": telemetry,
    }))
}

fn update_motion_state(
    state: &mut TrackState,
    cand: &PoseCandidate,
    prepared: PreparedKashinQuery,
) {
    if let Some(pe) = state.prev_eye {
        let dv = Vector3::from(cand.eye) - Vector3::from(pe);
        if dv.norm() <= 500.0 {
            state.prev_vel = [dv.x, dv.y, dv.z];
        }
    }
    if let Some(pr) = state.prev_rot {
        state.prev_rot_step = cand.rotation * pr.transpose();
    }
    state.prev_eye = Some(cand.eye);
    state.prev_rot = Some(cand.rotation);
    state.prev_prepared = Some(prepared);
}

fn apply_case_overrides(case: &PhotoCase, opts: &CorridorTrackOptions) -> PhotoCase {
    let mut out = case.clone();
    if opts.prior_override.is_none() && opts.focal_35mm_override.is_none() {
        return out;
    }
    let mut claims = out.prior.unwrap_or_default();
    if let Some(([lat, lon, alt], heading)) = opts.prior_override {
        claims.position = Some(PositionClaim {
            latitude: lat,
            longitude: lon,
            reported_horizontal_error_m: None,
            altitude_m: Some(alt),
            unsigned_altitude_m: Some(alt.abs()),
            altitude_reference_text: None,
            altitude_datum: "override_unverified".into(),
        });
        if let Some(hdg) = heading {
            claims.direction_deg = Some(hdg.rem_euclid(360.0));
        }
    }
    if let Some(f35) = opts.focal_35mm_override {
        claims.focal_35mm_mm = Some(f35);
    }
    out.prior = Some(claims);
    out
}

fn predict_interframe_pose(
    state: &TrackState,
    cur_q: &PreparedKashinQuery,
    k: &Matrix3<f64>,
) -> Option<([f64; 3], Matrix3<f64>)> {
    let (prev_eye, prev_rot) = (state.prev_eye?, state.prev_rot?);
    let pred_eye = [
        prev_eye[0] + 0.85 * state.prev_vel[0],
        prev_eye[1] + 0.85 * state.prev_vel[1],
        prev_eye[2] + 0.85 * state.prev_vel[2],
    ];
    let mut pred_rot = state.prev_rot_step * prev_rot;
    if let Some(prev_q) = &state.prev_prepared {
        let (px_a, px_b) = match_interframe_top(prev_q, cur_q, 192);
        if px_a.len() >= 12
            && let Some((r_delta, inliers)) = relative_rotation_ransac(&px_a, &px_b, k, 4.0)
        {
            let inl_n = inliers.iter().filter(|&&b| b).count();
            let trace_cos = ((r_delta.trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
            if inl_n >= 10 && trace_cos >= 15.0_f64.to_radians().cos() {
                pred_rot = r_delta * prev_rot;
            }
        }
    }
    Some((pred_eye, pred_rot))
}

fn match_interframe_top(
    prev_q: &PreparedKashinQuery,
    cur_q: &PreparedKashinQuery,
    cap: usize,
) -> (Vec<[f64; 2]>, Vec<[f64; 2]>) {
    let na = prev_q.valid_indices().len().min(cap);
    let nb = cur_q.valid_indices().len().min(cap);
    let (sa, sb) = (prev_q.sign_bits(), cur_q.sign_bits());
    let (pa, pb) = (prev_q.pixels(), cur_q.pixels());
    let (ia, ib) = (prev_q.valid_indices(), cur_q.valid_indices());
    let mut out_a = Vec::new();
    let mut out_b = Vec::new();
    for a in 0..na {
        let [ax, ay] = pa[ia[a]];
        let mut best = (usize::MAX, 65_u32, 65_u32);
        for b in 0..nb {
            let [bx, by] = pb[ib[b]];
            if (ax - bx).abs() > 96.0 || (ay - by).abs() > 96.0 {
                continue;
            }
            let hd = (sa[a] ^ sb[b]).count_ones();
            if hd < best.1 {
                best = (b, hd, best.1);
            } else if hd < best.2 {
                best.2 = hd;
            }
        }
        if best.0 != usize::MAX && best.1 <= 14 && best.1 + 3 <= best.2 {
            out_a.push([ax, ay]);
            out_b.push(pb[ib[best.0]]);
        }
    }
    (out_a, out_b)
}

fn try_warm_track(
    tracks: &ActiveTrackSet,
    prepared: &PreparedKashinQuery,
    pred: ([f64; 3], Matrix3<f64>),
    k_render: &Matrix3<f64>,
    render_size: [u32; 2],
    min_inliers: usize,
    attempts_log: &mut Vec<Value>,
) -> Result<Option<PoseCandidate>, PipelineError> {
    let cfg = TrackMatchConfig {
        search_radius_px: 36.0,
        hamming_gate: 22,
        min_dot: 0.64,
        ratio_limit: 0.90,
    };
    let intrinsics = [
        k_render[(0, 0)],
        k_render[(1, 1)],
        k_render[(0, 2)],
        k_render[(1, 2)],
    ];
    let matches = tracks.match_radius(
        prepared,
        pred.0,
        &rot_to_row_major(&pred.1),
        intrinsics,
        render_size,
        &cfg,
    )?;
    let mut world = Vec::with_capacity(matches.len());
    let mut px = Vec::with_capacity(matches.len());
    for m in &matches {
        world.push(m.world_point_enu);
        px.push(m.query_pixel);
    }
    let cand = fit_pose_l1_gnc(&world, &px, k_render, render_size, pred)
        .or_else(|| fit_pose_hybrid(&world, &px, k_render, render_size, Some(pred)));
    log_attempt(
        attempts_log,
        "kashin-3d-radius-track+l1-gnc",
        0,
        0,
        1.0,
        world.len(),
        cand.as_ref(),
    );
    Ok(cand.filter(|c| c.score.geometry_accepted && c.score.inliers >= min_inliers))
}
