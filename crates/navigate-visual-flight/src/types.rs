//! Core input, policy, telemetry, and outcome types for [`crate::FlightVisualEngine`].

use image::GrayImage;
use nalgebra::{Matrix3, UnitQuaternion};
use navigate_camera::{CameraMount, SpectralBand};
use navigate_contract::{GeodeticPosition, ObservationStamp};
use navigate_visual::fusion_bridge::VisualFix;

mod attitude;
mod convert;

/// Scheduling policy for background map-anchor and tile-synthesis workers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionPolicy {
    /// Non-blocking real-time flight mode (`30 Hz` foreground never waits on workers).
    LiveFlight,
    /// Deterministic virtual-clock mode for repeatable CI and unpaced log replay.
    DeterministicReplay {
        /// Virtual latency budget in milliseconds before an anchor job delivers.
        anchor_budget_ms: u64,
        /// Virtual latency budget in milliseconds before a tile job delivers.
        tile_budget_ms: u64,
    },
}

/// Borrowed pixel buffer for one live camera frame without disk I/O.
#[derive(Clone, Copy, Debug)]
pub enum FramePixels<'a> {
    /// Row-strided 8-bit grayscale luma buffer.
    Gray8 {
        /// Raw grayscale bytes (`height * stride` minimum length).
        data: &'a [u8],
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
        /// Row stride in bytes (`>= width as usize`).
        stride: usize,
    },
    /// Zero-copy borrowed `Y` plane from an ISP or hardware decoder `NV12` frame.
    Nv12YPlane {
        /// Luma plane bytes (`height * y_stride` minimum length).
        y_plane: &'a [u8],
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
        /// Luma row stride in bytes (`>= width as usize`).
        y_stride: usize,
    },
    /// Row-strided 8-bit interleaved `RGB` buffer.
    Rgb8 {
        /// Raw interleaved `RGB` bytes (`height * stride` minimum length).
        data: &'a [u8],
        /// Image width in pixels.
        width: u32,
        /// Image height in pixels.
        height: u32,
        /// Row stride in bytes (`>= 3 * width as usize`).
        stride: usize,
    },
}

impl<'a> FramePixels<'a> {
    /// Return `[width, height]` of the borrowed pixel frame.
    #[must_use]
    pub const fn dimensions(self) -> [u32; 2] {
        match self {
            Self::Gray8 { width, height, .. }
            | Self::Nv12YPlane { width, height, .. }
            | Self::Rgb8 { width, height, .. } => [width, height],
        }
    }
}

/// Confidence level of the camera focal calibration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CalibrationMode {
    /// Factory or pre-flight calibrated focal length.
    Calibrated,
    /// Approximate focal prior with relative one-sigma uncertainty `sigma_rel`.
    PriorFocal {
        /// Relative focal-length one-sigma ratio (for example `0.15` for `15%`).
        sigma_rel: f32,
    },
    /// Unknown focal length that requires cold focal-multiplier estimation.
    Uncalibrated,
}

/// Pinhole camera intrinsics for a live frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraIntrinsics {
    /// Source frame width in pixels.
    pub width: u32,
    /// Source frame height in pixels.
    pub height: u32,
    /// Horizontal focal length in pixels.
    pub fx: f64,
    /// Vertical focal length in pixels.
    pub fy: f64,
    /// Horizontal principal point in pixels.
    pub cx: f64,
    /// Vertical principal point in pixels.
    pub cy: f64,
    /// Calibration confidence mode.
    pub calibration_confidence: CalibrationMode,
}

impl CameraIntrinsics {
    /// Convert intrinsics to a `3 x 3` pinhole calibration matrix `K`.
    #[must_use]
    pub fn to_matrix3(self) -> Matrix3<f64> {
        Matrix3::new(self.fx, 0.0, self.cx, 0.0, self.fy, self.cy, 0.0, 0.0, 1.0)
    }

    /// Scale intrinsics to a target `[width, height]` resolution.
    ///
    /// Pixel centres have integer coordinates, as in
    /// `navigate_camera::Intrinsics::resized` and in the frame resampling of
    /// `prepare_tracking_buffers`, so the principal point maps as
    /// `(c + 0.5)·s − 0.5`.
    #[must_use]
    pub fn scaled_to(self, target: [u32; 2]) -> Self {
        let sx = f64::from(target[0].max(1)) / f64::from(self.width.max(1));
        let sy = f64::from(target[1].max(1)) / f64::from(self.height.max(1));
        Self {
            width: target[0].max(1),
            height: target[1].max(1),
            fx: self.fx * sx,
            fy: self.fy * sy,
            cx: (self.cx + 0.5) * sx - 0.5,
            cy: (self.cy + 0.5) * sy - 0.5,
            calibration_confidence: self.calibration_confidence,
        }
    }
}

/// Airframe platform class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlatformKind {
    /// Multi-rotor or stabilised gimbal camera with unobstructed view.
    RotaryGimbal,
    /// Fixed-wing aircraft with forward or oblique mount.
    FixedWing,
    /// Compound VTOL aircraft.
    CompoundVtol,
}

/// Camera-to-airframe mount configuration and occlusion properties.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraMountProfile {
    /// Fixed rotation from body coordinates to camera coordinates.
    pub cam_from_body: UnitQuaternion<f64>,
    /// True when wings, struts, nose, or propellers can enter the camera field of view.
    pub has_airframe_occlusion: bool,
    /// Airframe platform class.
    pub platform: PlatformKind,
}

impl CameraMountProfile {
    /// Unobstructed rotary-wing gimbal mount that looks down with the
    /// image top toward the nose, as [`CameraMount::nadir`]
    /// (`has_airframe_occlusion = false`).
    #[must_use]
    pub fn rotary_gimbal_nadir() -> Self {
        Self {
            cam_from_body: CameraMount::nadir().cam_from_body,
            has_airframe_occlusion: false,
            platform: PlatformKind::RotaryGimbal,
        }
    }

    /// Fixed-wing mount that looks forward and down by `depression_rad`
    /// below the body's forward axis, as [`CameraMount::depressed`]
    /// (`has_airframe_occlusion = true`).
    #[must_use]
    pub fn fixed_wing_oblique(depression_rad: f64) -> Self {
        Self {
            cam_from_body: CameraMount::depressed(depression_rad).cam_from_body,
            has_airframe_occlusion: true,
            platform: PlatformKind::FixedWing,
        }
    }
}

/// Optional IMU or gimbal attitude prior of the camera at frame capture
/// time. The angles are those of the camera, not of the body: a host with a
/// body attitude makes the prior with [`AttitudePrior::from_body_attitude`],
/// which applies [`CameraMountProfile::cam_from_body`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttitudePrior {
    /// Camera roll angle in radians (`0.0` = level horizon).
    pub roll_rad: f64,
    /// Camera downward pitch angle in radians (`pi / 2` = straight nadir).
    pub pitch_rad: f64,
    /// Optional compass heading in radians clockwise from North (`0..2*pi`).
    pub yaw_rad: Option<f64>,
    /// Tilt (roll and pitch) one-sigma uncertainty in radians.
    pub tilt_sigma_rad: f64,
    /// Heading one-sigma uncertainty in radians.
    pub yaw_sigma_rad: f64,
    /// Optional body angular velocity `[p, q, r]` in radians per second.
    pub body_ang_vel_rad_s: Option<[f64; 3]>,
}

/// Optional coarse navigation filter prior at frame capture time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchPrior {
    /// Geodetic prior center (`latitude_rad`, `longitude_rad`, `altitude_m`).
    pub center_geodetic: GeodeticPosition,
    /// Horizontal one-sigma uncertainty in meters.
    pub horizontal_sigma_m: f64,
    /// Optional local `NED` velocity `[north, east, down]` in meters per second.
    pub velocity_ned_mps: Option<[f64; 3]>,
}

/// Borrowed per-frame input to [`crate::FlightVisualEngine::process_frame`].
#[derive(Clone, Copy, Debug)]
pub struct LiveFrameInput<'a> {
    /// Hardware monotonic capture stamp and sequence number.
    pub stamp: ObservationStamp,
    /// Borrowed image pixels (`Gray8`, `Nv12YPlane`, or `Rgb8`).
    pub pixels: FramePixels<'a>,
    /// Spectral band of the camera. Only a visible frame runs the
    /// visible-trained `Segformer` semantic mask.
    pub band: SpectralBand,
    /// Optional static binary valid-pixel mask (`255` = valid, `0` = occluded).
    pub static_valid_mask: Option<&'a GrayImage>,
    /// Camera intrinsics and calibration confidence.
    pub intrinsics: CameraIntrinsics,
    /// Camera mount geometry and airframe occlusion flag.
    pub mount: CameraMountProfile,
    /// Optional IMU or gimbal attitude prior.
    pub imu_attitude: Option<AttitudePrior>,
    /// Optional barometric altitude `(alt_msl_m, sigma_m)`.
    pub baro_alt_msl_m: Option<(f64, f64)>,
    /// Optional coarse geodetic position and velocity prior.
    pub filter_prior: Option<SearchPrior>,
}

/// Basin verification confidence for an accepted map anchor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AnchorConfidence {
    /// Tight reprojection and unambiguous multi-hypothesis basin margin.
    Confirmed,
    /// Accepted with elevated residuals or competing spatial basin; requires confirmation.
    Provisional,
}

impl AnchorConfidence {
    /// Return the stable telemetry identifier for this confidence level.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Confirmed => "confirmed",
            Self::Provisional => "provisional",
        }
    }
}

/// Inter-frame visual dead-reckoning increment reported on each frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DeadReckonStep {
    /// Estimated camera translation `[east_m, north_m, up_m]` in `ENU` meters.
    pub delta_enu_m: [f64; 3],
    /// Estimated yaw change in radians across the frame step.
    pub yaw_delta_rad: f64,
    /// Estimated height above ground level in meters.
    pub agl_est_m: f64,
}

/// Active frame localization mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackingMode {
    /// Fast foreground 3D landmark tracking (`25-30 Hz`).
    WarmTrack,
    /// Newly admitted map anchor fix.
    CorridorAnchor,
    /// Cold search in flight while visual odometry propagates search center.
    AcquiringAnchor,
    /// Unresolved frame with no accepted pose.
    Unresolved,
}

impl TrackingMode {
    /// Return the canonical telemetry string for this tracking mode.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::WarmTrack => "warm_track",
            Self::CorridorAnchor => "corridor_anchor",
            Self::AcquiringAnchor => "acquiring_anchor",
            Self::Unresolved => "unresolved",
        }
    }

    /// Return the canonical evidence-kind string for this tracking mode.
    #[must_use]
    pub const fn evidence_kind_str(self) -> &'static str {
        match self {
            Self::WarmTrack => "surface_track_odometry",
            Self::CorridorAnchor => "corridor_map_anchor",
            Self::AcquiringAnchor => "visual_odometry",
            Self::Unresolved => "corridor_map_anchor",
        }
    }
}

/// EKF fusion eligibility classification for a frame outcome.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FusionEligibility {
    /// Independent map cell fix eligible for EKF ingestion.
    Independent,
    /// Map anchor inside a previously visited map cell.
    SharedMapCell,
    /// Warm-tracked pose correlated with an earlier map anchor.
    CorrelatedTrack,
    /// Cold anchor search in progress.
    AcquiringAnchor,
    /// Pose rejected by geometry or physical plausibility gates.
    Rejected,
}

impl FusionEligibility {
    /// Return the canonical telemetry string for this fusion eligibility.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Independent => "independent",
            Self::SharedMapCell => "shared_map_cell",
            Self::CorrelatedTrack => "correlated_track",
            Self::AcquiringAnchor => "acquiring_anchor",
            Self::Rejected => "rejected",
        }
    }
}

/// Watchdog health and bounded-queue counters for background workers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkerHealth {
    /// True when the background anchor worker is alive.
    pub anchor_alive: bool,
    /// True when the background anchor worker recorded a fault.
    pub anchor_faulted: bool,
    /// True when an anchor job is currently running.
    pub anchor_in_flight: bool,
    /// Cumulative anchor jobs superseded before execution.
    pub anchor_superseded: u32,
    /// Cumulative anchor jobs dropped due to busy queue.
    pub anchor_dropped: u32,
    /// True when the background tile builder worker is alive.
    pub tile_alive: bool,
    /// True when the background tile builder worker recorded a fault.
    pub tile_faulted: bool,
    /// Cumulative tile jobs dropped due to full bounded queue.
    pub tile_dropped: u32,
    /// Cumulative `Segformer` refresh jobs superseded before execution.
    pub mask_superseded: u32,
}

/// Fixed-size zero-allocation telemetry emitted per frame on the flight thread.
#[derive(Clone, Copy, Debug)]
pub struct FrameTelemetry {
    /// True when the frame produced an accepted metric camera pose.
    pub accepted: bool,
    /// Active localization mode for this frame.
    pub tracking_mode: TrackingMode,
    /// EKF fusion eligibility classification.
    pub fusion_eligibility: FusionEligibility,
    /// True when this frame produced a new independent EKF map fix.
    pub independent_ekf_fix: bool,
    /// Map cell `[east_idx, north_idx]` when an anchor fix was classified.
    pub map_cell: Option<[i64; 2]>,
    /// Number of active 3D landmarks in the foreground track set.
    pub active_tracks: u32,
    /// Number of corridor tiles currently resident in the LRU store.
    pub cached_tiles: u32,
    /// Resident corridor tile memory in bytes.
    pub resident_window_bytes: usize,
    /// Extracted query keypoint count on this frame.
    pub query_keypoints: u32,
    /// `LighterGlue` cross-attention layers executed synchronously on this frame.
    pub lighterglue_layers: u16,
    /// CPU terrain rays cast synchronously on this frame.
    pub polish_rays_cast: u32,
    /// True when a `Segformer` refresh completed on this frame.
    pub segformer_refreshed: bool,
    /// True when a `Segformer` refresh was submitted on this frame.
    pub segformer_submitted: bool,
    /// `Segformer` inference duration in milliseconds when refreshed.
    pub segformer_ms: Option<f64>,
    /// Fraction of pixels kept unmasked (`0.0..=1.0`).
    pub valid_mask_fraction: f32,
    /// Pixels masked by the temporal propeller detector.
    pub propeller_pixels: u32,
    /// Median frame luma (`0..=255`).
    pub luma_p50: u8,
    /// Foreground masking duration in milliseconds.
    pub mask_ms: f64,
    /// Foreground `XFeat` extraction duration in milliseconds.
    pub query_xfeat_ms: f64,
    /// Foreground pose-solve duration in milliseconds.
    pub solve_ms: f64,
    /// Total foreground wall time for the frame in milliseconds.
    pub total_frame_ms: f64,
    /// Delayed-anchor capture lag (`t_arrival - t_capture`) in nanoseconds (`0` when synchronous).
    pub capture_lag_ns: u64,
    /// Watchdog worker health and bounded-queue counters.
    pub worker_health: WorkerHealth,
}

/// Output of [`crate::FlightVisualEngine::process_frame`] for flight navigation fusion.
#[derive(Clone, Debug)]
pub struct FlightFrameOutcome {
    /// Validated `VisualFix` ready for `NavigationFilter::ingest` when independent or admitted.
    pub visual_fix: Option<VisualFix>,
    /// Basin confidence when the pose originated from or holds a map anchor.
    pub anchor_confidence: Option<AnchorConfidence>,
    /// Inter-frame visual dead-reckoning delta when odometry matched.
    pub dead_reckon: Option<DeadReckonStep>,
    /// Fixed-size zero-allocation frame telemetry.
    pub telemetry: FrameTelemetry,
}

#[cfg(test)]
mod tests;
