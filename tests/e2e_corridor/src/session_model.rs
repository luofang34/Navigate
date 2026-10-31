//! Session wiring, filter fix ingestion, dual-path dead reckoning, and anchor resets.

use nalgebra::Matrix3;
use navigate_contract::{
    ClockDomainId, GeodeticPosition, MonotonicNanos, ObservationStamp, SensorClass,
    SourceComposition, SourceEpoch, SourceId, SymmetricCov3, WrappingSequence,
};
use navigate_fusion::{
    FusionConfig, IngestOutcome, NavigationFilter, Observation, ObservationValue,
};
use navigate_visual::CameraModel;
use navigate_visual_fusion::{
    VerticalDatum, VisualErrorBudget, VisualFixSource, VisualSourceIdentity,
};
use navigate_visual_session::{
    Calibration, CameraMount, CaptureTime, ContinuityEpoch, FrameKey, FrameRecord, LensModel, Pose,
    SessionConfig, StreamId, VerticalReference, VisualSession,
};

/// Active dead reckoning path mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeadReckonPath {
    /// Path A: External sensor prior projected through NavigationFilter EKF.
    PathA,
    /// Path B: Pure vision fallback with optical flow or kinematic coasting.
    PathB,
}

/// Dead reckoning estimate emitted during visual track loss.
#[derive(Debug, Clone)]
pub struct DeadReckonEstimate {
    /// Estimated position in ENU tangent plane.
    pub eye_enu: [f64; 3],
    /// Propagated position covariance matrix.
    pub pos_cov_enu_m2: Matrix3<f64>,
    /// Active dead-reckoning computation path.
    pub path: DeadReckonPath,
}

/// Dual-path dead reckoner managing EKF priors and vision-only coasting.
#[derive(Debug, Clone)]
pub struct DualPathDeadReckoner {
    /// Current kinematic state for Path B coasting.
    pub kinematic: crate::kinematics_model::KinematicState,
    /// External sensors available flag.
    pub external_sensors_available: bool,
}

impl DualPathDeadReckoner {
    /// Constructs a dead reckoner with initial kinematic state.
    pub fn new(initial_eye: [f64; 3], initial_vel: [f64; 3], external_sensors: bool) -> Self {
        Self {
            kinematic: crate::kinematics_model::KinematicState::new(initial_eye, initial_vel),
            external_sensors_available: external_sensors,
        }
    }

    /// Evaluates dead reckoning step over interval dt_s.
    pub fn step(
        &mut self,
        dt_s: f64,
        ground_flow_vectors: usize,
        a_rms_mps2: f64,
        omega_rms_rad_s: f64,
    ) -> DeadReckonEstimate {
        if self.external_sensors_available {
            self.kinematic
                .propagate_covariance(dt_s, a_rms_mps2 * 0.5, omega_rms_rad_s * 0.5);
            let pred_eye = self.kinematic.predict_position(dt_s);
            self.kinematic.eye_enu = pred_eye;
            DeadReckonEstimate {
                eye_enu: pred_eye,
                pos_cov_enu_m2: self.kinematic.pos_cov_enu_m2,
                path: DeadReckonPath::PathA,
            }
        } else {
            let dt = crate::kinematics_model::clamp_dt_s(dt_s);
            if ground_flow_vectors >= 6 {
                self.kinematic
                    .propagate_covariance(dt, a_rms_mps2 * 0.8, omega_rms_rad_s * 0.8);
            } else {
                self.kinematic
                    .propagate_covariance(dt, a_rms_mps2, omega_rms_rad_s);
            }
            let pred_eye = self.kinematic.predict_position(dt);
            self.kinematic.eye_enu = pred_eye;
            DeadReckonEstimate {
                eye_enu: pred_eye,
                pos_cov_enu_m2: self.kinematic.pos_cov_enu_m2,
                path: DeadReckonPath::PathB,
            }
        }
    }
}

/// Evaluates vertical velocity update innovation gate on coplanar rays.
pub fn can_update_vertical_velocity(holdout_count: usize, inliers: usize) -> bool {
    holdout_count >= 12 || inliers >= 48
}

/// Anchor worker tracking state tracking previous pose and recomputed velocity.
#[derive(Debug, Clone, Default)]
pub struct AnchorWorkerState {
    /// Previous accepted eye position in ENU.
    pub prev_eye: Option<[f64; 3]>,
    /// Previous accepted camera rotation.
    pub prev_rot: Option<Matrix3<f64>>,
    /// Current translational velocity in meters per second.
    pub vel_enu_mps: [f64; 3],
}

impl AnchorWorkerState {
    /// Resets state on tracking loss event.
    pub fn on_track_loss(&mut self) {
        self.prev_eye = None;
        self.prev_rot = None;
        self.vel_enu_mps = [0.0, 0.0, 0.0];
    }

    /// Resets state and updates velocity on accepted anchor arrival.
    pub fn on_anchor_accepted(&mut self, new_eye: [f64; 3], new_rot: Matrix3<f64>, dt_s: f64) {
        if let Some(prev) = self.prev_eye {
            let dt = crate::kinematics_model::clamp_dt_s(dt_s);
            self.vel_enu_mps = [
                (new_eye[0] - prev[0]) / dt,
                (new_eye[1] - prev[1]) / dt,
                (new_eye[2] - prev[2]) / dt,
            ];
        } else {
            self.vel_enu_mps = [0.0, 0.0, 0.0];
        }
        self.prev_eye = Some(new_eye);
        self.prev_rot = Some(new_rot);
    }

    /// Resets state and odometry on trajectory relocation event.
    pub fn on_relocation(&mut self, relocated_eye: [f64; 3], dt_s: f64) {
        if let Some(prev) = self.prev_eye {
            let dt = crate::kinematics_model::clamp_dt_s(dt_s);
            self.vel_enu_mps = [
                (relocated_eye[0] - prev[0]) / dt,
                (relocated_eye[1] - prev[1]) / dt,
                (relocated_eye[2] - prev[2]) / dt,
            ];
        }
        self.prev_eye = Some(relocated_eye);
    }
}

/// Visual corridor session integrating session machine, fix source, and EKF.
pub struct VisualCorridorSession {
    /// Underlying visual session state machine.
    pub session: VisualSession,
    /// Visual fix source adapter converting estimates to EKF observations.
    pub fix_source: VisualFixSource,
    /// Local tangent plane Extended Kalman Filter.
    pub filter: NavigationFilter,
    /// Next frame sequence counter.
    pub next_seq: u64,
}

impl VisualCorridorSession {
    /// Constructs and initializes a new visual corridor session.
    pub fn new() -> Result<Self, String> {
        let calib = Calibration {
            intrinsics: CameraModel {
                width: 1920,
                height: 1080,
                fx: 1000.0,
                fy: 1000.0,
                cx: 960.0,
                cy: 540.0,
            },
            lens: LensModel::Undistorted {
                residual_rms_px: 0.2,
            },
            mount: CameraMount::Fixed {
                body_from_camera: Pose::translation(0.0, 0.0, 0.0),
            },
            map_vertical: VerticalReference::Wgs84Ellipsoid,
        };
        let session =
            VisualSession::new(SessionConfig::standard(), calib).map_err(|e| format!("{e:?}"))?;
        let identity = VisualSourceIdentity {
            source: SourceId::new(1),
            epoch: SourceEpoch::new(0),
            clock: ClockDomainId::new(1),
        };
        let budget = VisualErrorBudget {
            map_horizontal_m: 1.0,
            map_vertical_m: 1.0,
            calibration_m: 0.5,
            geometry_model_m: 1.0,
            vertical_datum: VerticalDatum::Wgs84Ellipsoid,
            vertical_datum_m: 0.5,
        };
        let fix_source = VisualFixSource::new(identity, budget).map_err(|e| format!("{e:?}"))?;
        let filter = NavigationFilter::new(FusionConfig::default(), ClockDomainId::new(1));
        Ok(Self {
            session,
            fix_source,
            filter,
            next_seq: 0,
        })
    }

    /// Registers a camera frame with monotonic timestamp and image dimensions.
    pub fn register_frame(&mut self, t_ns: u64) -> Result<(), String> {
        let key = FrameKey {
            stream: StreamId(1),
            continuity: ContinuityEpoch(0),
            index: self.next_seq,
        };
        self.next_seq += 1;
        let record = FrameRecord {
            key,
            media: None,
            capture: CaptureTime {
                clock: ClockDomainId::new(1),
                at_ns: t_ns,
                error_bound_ns: 1000,
            },
            observation_sha256: format!("{:064x}", self.next_seq),
        };
        self.session
            .observe_frame(record)
            .map_err(|e| format!("{e:?}"))
    }

    /// Ingests a synthetic position fix into the navigation filter.
    pub fn ingest_test_fix(&mut self, t_ns: u64, lat_deg: f64, lon_deg: f64, alt_m: f64) -> bool {
        let pos = GeodeticPosition::new(lat_deg.to_radians(), lon_deg.to_radians(), alt_m);
        let seq = self.next_seq;
        self.next_seq += 1;
        let stamp = ObservationStamp::new(
            SourceId::new(1),
            SourceEpoch::new(0),
            WrappingSequence::new(seq as u32),
            MonotonicNanos::from_nanos(t_ns),
            ClockDomainId::new(1),
        );
        let value = ObservationValue::PositionFix {
            position: pos,
            covariance: SymmetricCov3::from_diagonal(4.0, 4.0, 9.0),
        };
        let obs = Observation::new(
            stamp,
            value,
            SourceComposition::of(SensorClass::VisualLandmark),
        );
        let outcome = self.filter.ingest(&obs, MonotonicNanos::from_nanos(t_ns));
        outcome == IngestOutcome::Accepted
    }
}
