//! Two-tier `25-30 Hz` temporal corridor tracker and [`MapCell`] EKF independence gate.

use super::{
    CorridorTrackOptions,
    admission::{
        FusionAdmission, classify_anchor_cell, seed_tracks, top_up_tracks, write_corridor_report,
    },
    anchor::acquire_corridor_anchor,
    anchor_worker::{AnchorArrival, AnchorContext, AnchorJob, BackgroundAnchor},
    latency::StageLatencies,
    masking::{AmortizedMasker, MaskFrameTelemetry},
    motion::{TrackState, predict_interframe_pose, try_warm_track, update_motion_state},
};
use crate::{
    PipelineError,
    metadata::PositionClaim,
    realtime::helpers::{
        check_encoder_contract, load_lighterglue_model, load_xfeat_model, raw_file_sha256,
        warmup_gpu_pipelines,
    },
    stages::{
        estimate_intrinsics, load_case_mask, load_case_rgb, load_profile_cases, scale_k,
        scaled_size,
    },
    surface::Surface,
    testset::PhotoCase,
    util::write_json,
};
use image::imageops::{FilterType, colorops::grayscale};
use navigate_match::{
    ActiveTrackSet, CorridorLandmarkBank, HadamardProjector, PreparedKashinQuery,
};
use serde_json::{Value, json};
use std::{fs, sync::Arc, time::Instant};

/// Localize and track sequential frames against a [`CorridorLandmarkBank`] (`corridor-track`).
///
/// # Errors
///
/// Returns [`PipelineError`] when bank verification, masking, matching, or report writing fails.
pub fn run_corridor_track(opts: &CorridorTrackOptions) -> Result<Value, PipelineError> {
    let surface = Arc::new(Surface::from_catalog(&opts.catalog_path)?);
    let catalog_digest = raw_file_sha256(&opts.catalog_path)?;
    let (xfeat, encoder_digest) = load_xfeat_model(&opts.xfeat_weights_path)?;
    let lighterglue = opts
        .lighterglue_weights_path
        .as_deref()
        .map(|p| load_lighterglue_model(p, xfeat.device()))
        .transpose()?;
    let bank = Arc::new(CorridorLandmarkBank::load_file(&opts.corridor_bank_path)?);
    if bank.catalog_sha256 != catalog_digest {
        return Err(PipelineError::Invalid(
            "corridor bank catalog_sha256 does not match --catalog".into(),
        ));
    }
    check_encoder_contract(
        "corridor bank",
        &bank.encoder_contract,
        &xfeat,
        &encoder_digest,
    )?;

    let mut masker = AmortizedMasker::new(
        opts.seg_model_dir.as_deref(),
        &opts.mask_policy,
        6,
        opts.mask_interval.max(1),
        xfeat.device(),
    )?;
    if opts.background_refresh {
        masker = masker.with_background_refresh()?;
    }
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
    let mut background_anchor = opts
        .background_refresh
        .then(|| {
            BackgroundAnchor::spawn(AnchorContext {
                surface: Arc::clone(&surface),
                bank: Arc::clone(&bank),
                projector,
                xfeat: xfeat.clone(),
                lighterglue: lighterglue.clone(),
                opts: opts.clone(),
            })
        })
        .transpose()?;
    let mut latencies = StageLatencies::default();
    let mut anchor_results = Vec::new();
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
        if let Some(worker) = background_anchor.as_mut()
            && let Some(outcome) = worker.poll()?
        {
            AnchorArrival {
                state: &mut state,
                independent_fixes: &mut independent_fixes,
                latencies: &mut latencies,
                results: &mut anchor_results,
                catalog_sha256: &surface.catalog_sha256,
                cell_m,
            }
            .apply(outcome, Some(&case.id));
        }
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
        latencies.seg_refresh_ms.extend(mask_tel.segformer_ms);
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
        let mut force_anchor =
            opts.anchor_interval > 0 && state.frames_since_anchor >= opts.anchor_interval;
        if force_anchor
            && !state.active_tracks.is_empty()
            && let Some(worker) = background_anchor.as_mut()
        {
            // Warm tracking continues while the periodic re-anchor runs.
            worker.submit_if_idle(|| AnchorJob {
                case: case.clone(),
                qry_feat: qry_feat.clone(),
                prepared: prepared.clone(),
                k,
                k_render,
                render_size,
                pred_pose,
                live_attitude: masker.horizon_attitude(&k_render),
            })?;
            force_anchor = false;
        }

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
                    render_size,
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
            // A lost track cannot wait for a stale anchor: finish it, then anchor this frame.
            if let Some(worker) = background_anchor.as_mut()
                && let Some(outcome) = worker.wait()?
            {
                AnchorArrival {
                    state: &mut state,
                    independent_fixes: &mut independent_fixes,
                    latencies: &mut latencies,
                    results: &mut anchor_results,
                    catalog_sha256: &surface.catalog_sha256,
                    cell_m,
                }
                .apply(outcome, Some(&case.id));
            }
            let t_anchor = Instant::now();
            let acq = acquire_corridor_anchor(
                (&surface, &bank, &projector),
                (&xfeat, lighterglue.as_ref()),
                (&case, opts),
                (&qry_feat, &prepared),
                (&k, &k_render, render_size),
                (pred_pose, masker.horizon_attitude(&k_render)),
                &mut attempts_log,
            )?;
            latencies
                .anchor_ms
                .push(t_anchor.elapsed().as_secs_f64() * 1000.0);
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
            masker.note_solved_rotation(cand.rotation);
            update_motion_state(&mut state, cand, prepared);
        }

        let total_case_ms = case_t0.elapsed().as_secs_f64() * 1000.0;
        if admission.tracking_mode == "warm_track" {
            latencies.warm_track_ms.push(total_case_ms);
        }
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
            "segformer_submitted": mask_tel.segformer_submitted,
            "segformer_ms": mask_tel.segformer_ms,
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

    if let Some(mut worker) = background_anchor.take() {
        if let Some(outcome) = worker.wait()? {
            AnchorArrival {
                state: &mut state,
                independent_fixes: &mut independent_fixes,
                latencies: &mut latencies,
                results: &mut anchor_results,
                catalog_sha256: &surface.catalog_sha256,
                cell_m,
            }
            .apply(outcome, None);
        }
        drop(worker);
    }
    drop(masker);

    let mut summary = json!({
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
        "background_refresh": opts.background_refresh,
        "anchor_results": anchor_results,
    });
    if let Some(map) = summary.as_object_mut() {
        latencies.write_summary(map);
    }
    Ok(summary)
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
