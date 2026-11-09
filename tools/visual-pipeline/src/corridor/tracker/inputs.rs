//! CLI input mapping helpers from `PhotoCase` datasets to [`LiveFrameInput`].

use super::super::{CorridorTrackOptions, VehicleDynamicsProfile};
use crate::{
    civil_time::parse_iso8601_ms,
    metadata::PositionClaim,
    stages::{
        estimate_intrinsics_with_estimator, focal::FocalLengthEstimator,
        has_calibrated_focal_prior, load_horizon_attitude,
    },
    testset::PhotoCase,
};
use navigate_contract::{
    ClockDomainId, GeodeticPosition, MonotonicNanos, ObservationStamp, SourceEpoch, SourceId,
    WrappingSequence,
};
use navigate_visual_flight::{
    AttitudePrior, CalibrationMode, CameraIntrinsics, CameraMountProfile, FramePixels,
    LiveFrameInput, MatcherRuntimeConfig, SearchPrior, SpectralBand, anchor::AnchorSearchOpts,
};
use std::{
    path::Path,
    time::{Duration, Instant},
};

pub(crate) fn matcher_runtime_for_options(
    lg_path: Option<&Path>,
    xfeat_runtime: &navigate_visual_flight::XFeatRuntimeConfig,
) -> MatcherRuntimeConfig {
    let onnx_model_path = lg_path.and_then(|p| {
        let is_onnx = p
            .extension()
            .and_then(|s| s.to_str())
            .is_some_and(|e| e.eq_ignore_ascii_case("onnx"));
        if is_onnx {
            return Some(p.to_path_buf());
        }
        let stem_onnx = p.with_extension("onnx");
        if stem_onnx.is_file() {
            return Some(stem_onnx);
        }
        let sibling = p.parent().map(|d| d.join("lighterglue.onnx"))?;
        sibling.is_file().then_some(sibling)
    });
    MatcherRuntimeConfig {
        runtime: xfeat_runtime.runtime,
        onnx_model_path,
        onnx_runtime_path: xfeat_runtime.onnx_runtime_path.clone(),
    }
}

/// Depression below the nose of the window camera of the fixed-wing photo
/// cases. The tracker takes the camera attitude from the horizon, so the
/// mount rotation only describes the camera; the occlusion flag is what the
/// tracker reads.
const FIXED_WING_CAMERA_DEPRESSION_DEG: f64 = 45.0;

pub(crate) fn mount_profile_for_dynamics(profile: &VehicleDynamicsProfile) -> CameraMountProfile {
    if profile.max_speed_mps >= 40.0 {
        CameraMountProfile::fixed_wing_oblique(FIXED_WING_CAMERA_DEPRESSION_DEG.to_radians())
    } else {
        CameraMountProfile::rotary_gimbal_nadir()
    }
}

pub(crate) fn apply_case_overrides(case: &PhotoCase, opts: &CorridorTrackOptions) -> PhotoCase {
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

pub(crate) fn effective_case_time_s(case: &PhotoCase) -> Option<f64> {
    case.time_s.filter(|t| t.is_finite()).or_else(|| {
        case.prior
            .as_ref()?
            .capture_time
            .as_deref()
            .and_then(parse_iso8601_ms)
            .map(|ms| (ms as f64) * 1e-3)
    })
}

pub(crate) fn pace_if_realtime(
    pace_realtime: bool,
    raw_case: &PhotoCase,
    pace_origin: &mut Option<(f64, Instant)>,
) {
    if pace_realtime && let Some(t_s) = effective_case_time_s(raw_case) {
        if let Some((ref mut t0_s, ref mut wall0)) = *pace_origin {
            let target_s = (t_s - *t0_s).max(0.0);
            let elapsed_s = wall0.elapsed().as_secs_f64();
            if target_s > elapsed_s {
                std::thread::sleep(Duration::from_secs_f64(target_s - elapsed_s));
            } else if elapsed_s - target_s > 0.25 {
                *t0_s = t_s;
                *wall0 = Instant::now();
            }
        } else {
            *pace_origin = Some((t_s, Instant::now()));
        }
    }
}

pub(crate) fn advance_monotonic_clock(
    case: &PhotoCase,
    prev_raw_time_s: &mut Option<f64>,
    cur_monotonic_ns: &mut u64,
    first_frame: bool,
) -> u64 {
    let raw_s = effective_case_time_s(case);
    if first_frame {
        *prev_raw_time_s = raw_s;
        *cur_monotonic_ns = 1_000_000_000;
        return *cur_monotonic_ns;
    }
    let delta_ns = match (raw_s, *prev_raw_time_s) {
        (Some(cur), Some(prev)) if cur > prev => {
            ((cur - prev) * 1e9).round().clamp(100_000.0, 3.6e12) as u64
        }
        (Some(cur), Some(prev)) if cur < prev => 5_001_000_000,
        _ => 33_333_333,
    };
    *prev_raw_time_s = raw_s;
    *cur_monotonic_ns = cur_monotonic_ns.saturating_add(delta_ns.max(100_000));
    *cur_monotonic_ns
}

pub(crate) fn build_search_opts(opts: &CorridorTrackOptions) -> AnchorSearchOpts {
    AnchorSearchOpts {
        attempts: opts.anchor_attempts.max(1),
        heading_step_deg: opts.heading_step_deg,
        height_factors: if opts.height_factors.is_empty() {
            vec![1.0]
        } else {
            opts.height_factors.clone()
        },
        query_keypoints: opts.query_keypoints.clamp(16, 4096),
        polish_anchors: opts.polish_anchors,
        pace_realtime: opts.pace_realtime,
        anchor_uncertainty_m: opts.anchor_uncertainty_m,
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "single helper packs decoded RGB, optional mask, intrinsics, mount, and PhotoCase claims into LiveFrameInput"
)]
pub(crate) fn build_live_frame_input<'a>(
    dataset_dir: &Path,
    case: &PhotoCase,
    rgb: &'a image::RgbImage,
    static_mask: Option<&'a image::GrayImage>,
    mount: CameraMountProfile,
    seq: u32,
    monotonic_ns: u64,
    focal_est: &FocalLengthEstimator,
) -> LiveFrameInput<'a> {
    let (w, h) = rgb.dimensions();
    let k_full = estimate_intrinsics_with_estimator(case, focal_est);
    let cal_mode = if has_calibrated_focal_prior(case) {
        CalibrationMode::Calibrated
    } else {
        CalibrationMode::Uncalibrated
    };
    let intrinsics = CameraIntrinsics {
        width: w,
        height: h,
        fx: k_full[(0, 0)],
        fy: k_full[(1, 1)],
        cx: k_full[(0, 2)],
        cy: k_full[(1, 2)],
        calibration_confidence: cal_mode,
    };
    let horizon_att = load_horizon_attitude(dataset_dir, case, &k_full);
    let yaw_rad = case
        .prior
        .as_ref()
        .and_then(|p| p.direction_deg)
        .filter(|d| d.is_finite())
        .map(|d| d.rem_euclid(360.0).to_radians());
    let imu_attitude = match (horizon_att, yaw_rad) {
        (Some((down_deg, roll_deg)), y) => Some(AttitudePrior {
            roll_rad: roll_deg.to_radians(),
            pitch_rad: down_deg.to_radians(),
            yaw_rad: y,
            tilt_sigma_rad: 0.08,
            yaw_sigma_rad: 0.35,
            body_ang_vel_rad_s: None,
        }),
        (None, Some(y)) => Some(AttitudePrior {
            roll_rad: f64::NAN,
            pitch_rad: f64::NAN,
            yaw_rad: Some(y),
            tilt_sigma_rad: 3.0,
            yaw_sigma_rad: 0.35,
            body_ang_vel_rad_s: None,
        }),
        (None, None) => None,
    };
    let pos_opt = case.prior.as_ref().and_then(|p| p.position.as_ref());
    let baro_alt_msl_m = pos_opt
        .and_then(|p| p.altitude_m)
        .filter(|a| a.is_finite())
        .map(|alt| (alt, 25.0));
    let filter_prior = pos_opt
        .filter(|p| p.latitude.is_finite() && p.longitude.is_finite())
        .map(|p| SearchPrior {
            center_geodetic: GeodeticPosition::new(
                p.latitude.to_radians(),
                p.longitude.to_radians(),
                p.altitude_m.unwrap_or(f64::NAN),
            ),
            horizontal_sigma_m: p.reported_horizontal_error_m.unwrap_or(45.0).max(15.0),
            velocity_ned_mps: None,
        });

    LiveFrameInput {
        stamp: ObservationStamp::new(
            SourceId::new(1),
            SourceEpoch::new(1),
            WrappingSequence::new(seq),
            MonotonicNanos::from_nanos(monotonic_ns),
            ClockDomainId::new(1),
        ),
        pixels: FramePixels::Rgb8 {
            data: rgb.as_raw(),
            width: w,
            height: h,
            stride: (w as usize) * 3,
        },
        band: SpectralBand::Visible,
        static_valid_mask: static_mask,
        intrinsics,
        mount,
        imu_attitude,
        baro_alt_msl_m,
        filter_prior,
    }
}
