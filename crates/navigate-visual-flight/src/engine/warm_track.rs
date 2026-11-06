//! Warm 3D tracking, `GlobalFrustumMatch` maneuver retry, and `RecentAnchorBackup` recovery.

use super::FlightVisualEngine;
use crate::{
    anchor::{
        AnchorPrior, SeedPose, is_plausible_anchor_candidate, nadir_screen::rot_to_row_major,
        record_attempt,
    },
    diagnostics::AttemptRecord,
    error::FlightVisualError,
    surface::Surface,
    tracks::{TOP_UP_FRUSTUM_MARGIN_PX, top_up_tracks_with_bank},
};
use nalgebra::Matrix3;
use navigate_locate::corridor::{
    CHI2_GATE_3DOF_99, ColdDeadReckonPrior, KinematicState, TrackingSearchStrategy,
    VehicleDynamicsProfile, plausibility::compute_fisher_geometry_covariance, try_warm_track,
};
use navigate_match::{ActiveTrackSet, PreparedKashinQuery, SparseFeatures};
use navigate_visual::PoseCandidate;

const RECENT_BACKUP_MAX_AGE_NS: u64 = 1_500_000_000;
const RECENT_BACKUP_MAX_USES: u8 = 3;

pub(super) fn resolve_anchor_query_prior(
    surface: &Surface,
    mut prior: AnchorPrior,
    cold_prior: Option<&ColdDeadReckonPrior>,
    nominal_agl_m: f64,
    pred_pose: Option<SeedPose>,
) -> AnchorPrior {
    if pred_pose.is_none()
        && nominal_agl_m > 250.0
        && let Some(cp) = cold_prior
        && let Some((lat, lon)) = prior.lat_lon()
    {
        let p_enu = surface.local(lat, lon, prior.alt_msl_m.unwrap_or(0.0));
        let ext_shift =
            (p_enu[0] - cp.anchor_prior_enu[0]).hypot(p_enu[1] - cp.anchor_prior_enu[1]);
        let dr_shift = (cp.center_enu[0] - cp.anchor_prior_enu[0])
            .hypot(cp.center_enu[1] - cp.anchor_prior_enu[1]);
        if ext_shift < 1.0 && dr_shift > 10.0 {
            let [dlat, dlon, dalt] = surface.geodetic(cp.center_enu);
            prior.lat_deg = Some(dlat);
            prior.lon_deg = Some(dlon);
            if prior.alt_msl_m.is_some() {
                prior.alt_msl_m = Some(dalt);
            }
        }
    }
    prior
}

pub(super) fn predict_step_kinematics(
    kin_opt: Option<&KinematicState>,
    pred_pose: Option<SeedPose>,
    dt_s: f64,
    profile: &VehicleDynamicsProfile,
) -> Option<KinematicState> {
    kin_opt.map(|kin| {
        let mut kp = kin.clone();
        if let Some((pe, pr)) = pred_pose {
            kp.eye_enu = pe;
            kp.rot_world_to_cam = pr;
        }
        let step_disp = profile.max_displacement(dt_s);
        let vel_norm_sq = kp.vel_enu_mps[0] * kp.vel_enu_mps[0]
            + kp.vel_enu_mps[1] * kp.vel_enu_mps[1]
            + kp.vel_enu_mps[2] * kp.vel_enu_mps[2];
        let step_var = if vel_norm_sq < 1e-6 {
            (step_disp * 0.65).powi(2).max(4.0)
        } else {
            let accel_disp = 0.5 * profile.rms_accel_mps2 * dt_s * dt_s;
            accel_disp.max(step_disp * 0.30).powi(2).max(2.25)
        };
        kp.pos_cov_enu_m2 += Matrix3::identity() * step_var;
        kp.att_sigma_rad += profile.max_angular_rate_rad_s * 0.1 * dt_s;
        kp
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "warm PnP check with GlobalFrustumMatch maneuver fallback shares surface, tracks, intrinsics, and kinematics"
)]
pub(super) fn attempt_warm_solve(
    surface: &Surface,
    tracks: &ActiveTrackSet,
    prepared: &PreparedKashinQuery,
    pred: SeedPose,
    (k_render, render_size, min_trk): (&Matrix3<f64>, [u32; 2], usize),
    predicted_kin: Option<&KinematicState>,
    strategy: TrackingSearchStrategy,
    mut attempts: Option<&mut Vec<AttemptRecord>>,
) -> Result<Option<PoseCandidate>, FlightVisualError> {
    let (att, mut solved) = try_warm_track(
        tracks,
        prepared,
        pred,
        k_render,
        render_size,
        min_trk,
        strategy,
    )?;
    if let Some(log) = attempts.as_deref_mut() {
        record_attempt(
            log,
            att.method,
            0,
            0,
            0.0,
            att.matched_landmarks,
            att.candidate.as_ref(),
            None,
        );
    }
    if let Some(ref cand) = solved
        && !passes_warm_gates(cand, surface, predicted_kin, CHI2_GATE_3DOF_99)
    {
        solved = None;
    }
    if solved.is_none() && matches!(strategy, TrackingSearchStrategy::RadiusSearch { .. }) {
        let (fb_att, fb_solved) = try_warm_track(
            tracks,
            prepared,
            pred,
            k_render,
            render_size,
            min_trk,
            TrackingSearchStrategy::GlobalFrustumMatch,
        )?;
        if let Some(log) = attempts {
            record_attempt(
                log,
                fb_att.method,
                0,
                1,
                0.0,
                fb_att.matched_landmarks,
                fb_att.candidate.as_ref(),
                None,
            );
        }
        if let Some(cand) = fb_solved
            && passes_warm_gates(&cand, surface, predicted_kin, CHI2_GATE_3DOF_99 * 2.0)
        {
            solved = Some(cand);
        }
    }
    Ok(solved)
}

fn passes_warm_gates(
    cand: &PoseCandidate,
    surface: &Surface,
    predicted_kin: Option<&KinematicState>,
    chi2_gate: f64,
) -> bool {
    if !is_plausible_anchor_candidate(cand, surface) {
        return false;
    }
    if let Some(kin) = predicted_kin {
        let mut geom_cov = compute_fisher_geometry_covariance(cand)
            .fixed_view::<3, 3>(0, 0)
            .into_owned();
        if !cand.points.is_empty() {
            let eye_v = nalgebra::Vector3::from(cand.eye);
            let mean_range_m = cand
                .points
                .iter()
                .map(|p| (nalgebra::Vector3::from(*p) - eye_v).norm())
                .sum::<f64>()
                / (cand.points.len() as f64);
            let range_sigma_m = (0.02 * mean_range_m).clamp(1.5, 60.0);
            geom_cov += Matrix3::identity() * (range_sigma_m * range_sigma_m);
        }
        let eff_gate = if (cand.score.inliers >= 48
            && cand.score.holdout_inlier_fraction >= 0.45
            && cand.score.holdout_median_px <= 7.0)
            || (cand.score.inliers >= 28
                && cand.score.holdout_inlier_fraction >= 0.40
                && cand.score.holdout_median_px <= 7.5
                && cand.score.train_rms_px <= 5.0)
        {
            chi2_gate * 3.0
        } else {
            chi2_gate
        };
        if !kin.mahalanobis_innovation_gate(cand.eye, geom_cov, eff_gate) {
            return false;
        }
    }
    true
}

pub(super) fn refresh_warm_frustum_and_top_up(
    eng: &mut FlightVisualEngine,
    cand: &PoseCandidate,
    prepared: &PreparedKashinQuery,
    qry_feat: &SparseFeatures,
    render_size: [u32; 2],
) {
    let target_tracks = 640.min(qry_feat.len());
    let is_coreml = eng.resources.xfeat.runtime_name() == "coreml";
    let check_frustum = eng.state.frames_since_anchor.is_multiple_of(6) || cand.score.inliers < 180;
    if check_frustum {
        let intrinsics = [
            cand.k[(0, 0)],
            cand.k[(1, 1)],
            cand.k[(0, 2)],
            cand.k[(1, 2)],
        ];
        eng.state.active_tracks.retain_in_frustum(
            cand.eye,
            &rot_to_row_major(&cand.rotation),
            intrinsics,
            render_size,
            TOP_UP_FRUSTUM_MARGIN_PX,
        );
    }
    let need_urgent = if is_coreml {
        eng.state.active_tracks.len() < 64
    } else {
        eng.state.active_tracks.len() < 128 || cand.score.inliers < 40
    };
    let need_periodic = !is_coreml
        && eng.state.frames_since_anchor.is_multiple_of(8)
        && cand.score.inliers < 96
        && (eng.state.active_tracks.len() < target_tracks.min(192)
            || cand.score.image_support_fraction < 0.35);
    if need_urgent || need_periodic {
        top_up_tracks_with_bank(
            &mut eng.state.active_tracks,
            &eng.resources.surface,
            &eng.resources.store,
            prepared,
            qry_feat,
            cand,
            render_size,
        );
    }
}

pub(super) fn try_recover_from_recent_backup(
    eng: &mut FlightVisualEngine,
    cur_ns: u64,
    prepared: &PreparedKashinQuery,
    qry_feat: &SparseFeatures,
    (k_render, render_size, min_trk): (&Matrix3<f64>, [u32; 2], usize),
    attempts: Option<&mut Vec<AttemptRecord>>,
) -> Result<Option<PoseCandidate>, FlightVisualError> {
    let Some(ref backup) = eng.recent_backup else {
        return Ok(None);
    };
    if backup.used_count >= RECENT_BACKUP_MAX_USES
        || cur_ns.saturating_sub(backup.arrival_ns) > RECENT_BACKUP_MAX_AGE_NS
        || backup.seeded_tracks.is_empty()
    {
        return Ok(None);
    }
    let backup_pred = eng.state.propagate_pose_from_calibrated(
        backup.arrival_frame_seq,
        backup.pose.0,
        backup.pose.1,
        k_render,
        backup.agl_m,
        &eng.config.dynamics_profile,
    );
    let (att, solved) = try_warm_track(
        &backup.seeded_tracks,
        prepared,
        backup_pred,
        k_render,
        render_size,
        min_trk,
        TrackingSearchStrategy::GlobalFrustumMatch,
    )?;
    if let Some(log) = attempts {
        record_attempt(
            log,
            att.method,
            0,
            2,
            0.0,
            att.matched_landmarks,
            att.candidate.as_ref(),
            None,
        );
    }
    let Some(cand) = solved.filter(|c| is_plausible_anchor_candidate(c, &eng.resources.surface))
    else {
        return Ok(None);
    };
    let seeded_clone = backup.seeded_tracks.clone();
    if let Some(ref mut b) = eng.recent_backup {
        b.used_count = b.used_count.saturating_add(1);
    }
    eng.state.active_tracks = seeded_clone;
    top_up_tracks_with_bank(
        &mut eng.state.active_tracks,
        &eng.resources.surface,
        &eng.resources.store,
        prepared,
        qry_feat,
        &cand,
        render_size,
    );
    Ok(Some(cand))
}
