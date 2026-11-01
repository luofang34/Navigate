//! Visual session lifecycle, filter fix admission, and dual-path dead reckoning.

pub mod bridge;
mod dead_reckon;
mod worker;

pub use super::kinematics::can_update_vertical_velocity;
pub use bridge::{
    FilterIntegritySnapshot, SessionAnchorReport, build_estimate_from_candidate,
    cv_fisher_to_camera_covariance, cv_world_to_cam_to_camera_pose,
    cv_world_to_cam_to_session_pose,
};
pub use dead_reckon::{DeadReckonEstimate, DeadReckonPath, DualPathDeadReckoner};
pub use worker::AnchorWorkerState;

use super::kinematics::{MAX_DELTA_T_S, MIN_DELTA_T_S};
use nalgebra::Matrix3;
use navigate_contract::{
    ClockDomainId, GeodeticPosition, MonotonicNanos, ObservationStamp, SensorClass,
    SourceComposition, SourceEpoch, SourceId, SymmetricCov3, WrappingSequence,
};
use navigate_fusion::{
    FusionConfig, IngestOutcome, NavigationFilter, Observation, ObservationValue,
};
use navigate_visual::{CameraModel, Estimate};
use navigate_visual_fusion::{
    EvidenceIndependence, VerticalDatum, VisualErrorBudget, VisualFixSource, VisualSourceIdentity,
};
use navigate_visual_session::{
    AnchorDecision, AnchorObservation, AnchorPolicy, Calibration, CameraMount, CaptureTime,
    ContinuityEpoch, FrameKey, FrameRecord, LensModel, MotionOutcome, Pose, RelativeMotion,
    ScaleSource, SessionConfig, SessionError, StreamId, VerticalReference, VisualSession,
};
use std::collections::{BTreeMap, HashSet};

/// Integrated visual session, fix source, and tangent-plane navigation filter.
pub struct VisualCorridorSession {
    /// Visual session state machine.
    pub session: VisualSession,
    /// Fix source that converts visual pose estimates into filter observations.
    pub fix_source: VisualFixSource,
    /// Tangent-plane navigation filter.
    pub filter: NavigationFilter,
    /// Next frame sequence index.
    pub next_seq: u64,
    /// Current visual continuity epoch.
    pub current_epoch: ContinuityEpoch,
    /// Previous registered frame key.
    pub prev_registered_key: Option<FrameKey>,
    /// Most recent registered frame key.
    pub last_registered_key: Option<FrameKey>,
    /// Evidence digest of `prev_registered_key`.
    pub prev_registered_sha256: Option<String>,
    /// Evidence digest of `last_registered_key`.
    pub last_registered_sha256: Option<String>,
    /// Newest frame with odometry in the current continuity epoch.
    pub last_odom_frame: Option<(FrameKey, String, Pose)>,
    /// Seen frame digests in this session.
    pub seen_digests: HashSet<String>,
    /// Case identifier for each registered frame key.
    pub frame_case_ids: BTreeMap<FrameKey, String>,
    /// Cumulative count of `TrajectoryRevised` events emitted by `VisualSession`.
    pub trajectory_revisions: usize,
    /// Most recent registered monotonic capture timestamp in nanoseconds.
    pub last_capture_ns: u64,
    /// Dead reckoner for track-loss intervals.
    pub dead_reckoner: DualPathDeadReckoner,
}

impl VisualCorridorSession {
    /// Create a visual corridor session with standard calibration and error budgets.
    ///
    /// # Errors
    /// Returns an error string when session or fix-source initialization fails.
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
        let dead_reckoner = DualPathDeadReckoner::new([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], false);
        Ok(Self {
            session,
            fix_source,
            filter,
            next_seq: 0,
            current_epoch: ContinuityEpoch(0),
            prev_registered_key: None,
            last_registered_key: None,
            prev_registered_sha256: None,
            last_registered_sha256: None,
            last_odom_frame: None,
            seen_digests: HashSet::new(),
            frame_case_ids: BTreeMap::new(),
            trajectory_revisions: 0,
            last_capture_ns: 0,
            dead_reckoner,
        })
    }

    /// Create a visual corridor session with a custom [`AnchorPolicy`].
    ///
    /// # Errors
    /// Returns an error string when initialization or policy validation fails.
    pub fn with_anchor_policy(policy: AnchorPolicy) -> Result<Self, String> {
        let mut session = Self::new()?;
        session
            .session
            .set_anchor_policy(policy)
            .map_err(|e| format!("{e:?}"))?;
        Ok(session)
    }

    /// Set the anchor admission policy for the session.
    ///
    /// # Errors
    /// Returns [`SessionError`] when policy parameters are invalid.
    pub fn set_anchor_policy(&mut self, policy: AnchorPolicy) -> Result<(), SessionError> {
        self.session.set_anchor_policy(policy)
    }

    /// Return the current anchor admission policy.
    #[must_use]
    pub fn anchor_policy(&self) -> &AnchorPolicy {
        self.session.anchor_policy()
    }

    /// Register a camera frame at monotonic timestamp `t_ns`.
    ///
    /// # Errors
    /// Returns an error string when the timestamp is duplicate or out of order.
    pub fn register_frame(&mut self, t_ns: u64) -> Result<(), String> {
        let sha = format!("{:064x}", self.next_seq + 1);
        self.register_frame_with_digest(t_ns, &sha)
    }

    /// Register a camera frame at monotonic timestamp `t_ns` with explicit digest `sha256`.
    ///
    /// # Errors
    /// Returns an error string when the timestamp or digest is invalid.
    pub fn register_frame_with_digest(&mut self, t_ns: u64, sha256: &str) -> Result<(), String> {
        let key = FrameKey {
            stream: StreamId(1),
            continuity: self.current_epoch,
            index: self.next_seq,
        };
        let record = FrameRecord {
            key,
            media: None,
            capture: CaptureTime {
                clock: ClockDomainId::new(1),
                at_ns: t_ns,
                error_bound_ns: 1000,
            },
            observation_sha256: sha256.to_owned(),
        };
        self.session
            .observe_frame(record)
            .map_err(|e| format!("{e:?}"))?;
        self.next_seq += 1;
        self.prev_registered_key = self.last_registered_key;
        self.last_registered_key = Some(key);
        self.prev_registered_sha256 = self.last_registered_sha256.take();
        self.last_registered_sha256 = Some(sha256.to_owned());
        self.last_capture_ns = t_ns;
        Ok(())
    }

    /// Advance the continuity epoch after track loss or unobservable scale.
    pub fn advance_epoch(&mut self) {
        self.current_epoch = ContinuityEpoch(self.current_epoch.0 + 1);
        self.last_odom_frame = None;
    }

    /// Apply relative motion between the two most recent registered frames.
    ///
    /// # Errors
    /// Returns [`SessionError`] when metric scale is unobservable or motion validation fails.
    pub fn apply_motion_step(
        &mut self,
        _dt_s: f64,
        from_to: Pose,
        scale: ScaleSource,
    ) -> Result<MotionOutcome, SessionError> {
        let (Some(from_key), Some(to_key)) = (self.prev_registered_key, self.last_registered_key)
        else {
            return Err(SessionError::Invalid {
                field: "missing registered frames",
            });
        };

        if !scale.is_metric() {
            self.advance_epoch();
            return Err(SessionError::ScaleUnobservable {
                from: from_key,
                to: to_key,
            });
        }

        let from_sha = self
            .prev_registered_sha256
            .clone()
            .unwrap_or_else(|| format!("{:064x}", from_key.index + 1));
        let to_sha = self
            .last_registered_sha256
            .clone()
            .unwrap_or_else(|| format!("{:064x}", to_key.index + 1));

        let motion = RelativeMotion {
            from: from_key,
            to: to_key,
            from_to,
            scale,
            from_observation_sha256: from_sha,
            to_observation_sha256: to_sha,
        };
        match self.session.apply_motion(motion) {
            Ok(outcome) => Ok(outcome),
            Err(e) => {
                self.advance_epoch();
                Err(e)
            }
        }
    }

    /// Submit an anchor observation to the visual session.
    ///
    /// # Errors
    /// Returns [`SessionError`] when anchor validation fails.
    pub fn submit_anchor(
        &mut self,
        anchor: AnchorObservation,
    ) -> Result<AnchorDecision, SessionError> {
        self.session.submit_anchor(anchor)
    }

    /// Apply `policy` and submit an anchor observation.
    ///
    /// # Errors
    /// Returns [`SessionError`] when policy or anchor validation fails.
    pub fn submit_anchor_with_consensus(
        &mut self,
        anchor: AnchorObservation,
        policy: AnchorPolicy,
    ) -> Result<AnchorDecision, SessionError> {
        self.session.set_anchor_policy(policy)?;
        self.session.submit_anchor(anchor)
    }

    /// Reset the navigation filter and update dead-reckoning state on relocation.
    pub fn handle_relocation(
        &mut self,
        relocated_eye: [f64; 3],
        relocated_rot: Matrix3<f64>,
        dt_s: f64,
    ) {
        self.filter.reset();
        self.dead_reckoner.clear_filter_solution();
        let prev_eye = self.dead_reckoner.kinematic.eye_enu;
        if dt_s > 0.0 {
            let dt = dt_s.clamp(MIN_DELTA_T_S, MAX_DELTA_T_S);
            self.dead_reckoner.kinematic.vel_enu_mps = [
                (relocated_eye[0] - prev_eye[0]) / dt,
                (relocated_eye[1] - prev_eye[1]) / dt,
                (relocated_eye[2] - prev_eye[2]) / dt,
            ];
        } else {
            self.dead_reckoner.kinematic.vel_enu_mps = [0.0, 0.0, 0.0];
        }
        self.dead_reckoner.kinematic.eye_enu = relocated_eye;
        self.dead_reckoner.kinematic.rot_world_to_cam = relocated_rot;
    }

    /// Submit an anchor observation and ingest an eligible fix into the filter.
    ///
    /// # Errors
    /// Returns an error string when anchor submission or fix conversion fails.
    pub fn submit_and_ingest_estimate(
        &mut self,
        t_ns: u64,
        anchor: AnchorObservation,
        estimate: &Estimate,
    ) -> Result<Option<Observation>, String> {
        let decision = self
            .session
            .submit_anchor(anchor)
            .map_err(|e| format!("{e:?}"))?;
        let _ = self.drain_session_events();
        match decision {
            AnchorDecision::Accepted { eligibility } => {
                let ind = EvidenceIndependence::ValidatedIndependent.restricted_by(&eligibility);
                if ind != EvidenceIndependence::ValidatedIndependent {
                    return Ok(None);
                }
                let fix = self
                    .fix_source
                    .convert(estimate, ind)
                    .map_err(|e| format!("{e:?}"))?;
                let outcome = self
                    .filter
                    .ingest(&fix.observation, MonotonicNanos::from_nanos(t_ns));
                if let Some(sol) = self.filter.tick(MonotonicNanos::from_nanos(t_ns)) {
                    self.dead_reckoner
                        .set_filter_solution(sol, estimate.frame, t_ns);
                }
                if outcome == IngestOutcome::Accepted {
                    Ok(Some(fix.observation))
                } else {
                    Ok(None)
                }
            }
            AnchorDecision::Relocated { .. } => {
                let eye = [
                    estimate.pose.position.x,
                    estimate.pose.position.y,
                    estimate.pose.position.z,
                ];
                let rot = *estimate.pose.orientation.to_rotation_matrix().matrix();
                self.handle_relocation(eye, rot, 0.0);
                let ind = EvidenceIndependence::ValidatedIndependent;
                let fix = self
                    .fix_source
                    .convert(estimate, ind)
                    .map_err(|e| format!("{e:?}"))?;
                let outcome = self
                    .filter
                    .ingest(&fix.observation, MonotonicNanos::from_nanos(t_ns));
                if let Some(sol) = self.filter.tick(MonotonicNanos::from_nanos(t_ns)) {
                    self.dead_reckoner
                        .set_filter_solution(sol, estimate.frame, t_ns);
                }
                if outcome == IngestOutcome::Accepted {
                    Ok(Some(fix.observation))
                } else {
                    Ok(None)
                }
            }
            _ => Ok(None),
        }
    }

    /// Ingest a synthetic visual landmark position fix into the navigation filter.
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

    /// Step the dual-path dead reckoner across interval `dt_s`.
    pub fn dead_reckon(
        &mut self,
        dt_s: f64,
        ground_flow_vectors: usize,
        a_rms_mps2: f64,
        omega_rms_rad_s: f64,
    ) -> DeadReckonEstimate {
        self.dead_reckoner
            .step(dt_s, ground_flow_vectors, a_rms_mps2, omega_rms_rad_s)
    }
}

#[cfg(test)]
mod tests;
