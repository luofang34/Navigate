//! Dual-path dead reckoning across track loss with Path A filter priors and Path B coasting.

use crate::corridor::kinematics::{KinematicState, MAX_DELTA_T_S, MIN_DELTA_T_S};
use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use navigate_contract::{
    ClockDomainId, DurationNanos, FaultDetection, GeodeticPosition, IntegrityAssessment,
    MonotonicNanos, NavigationSolution, NedVelocity, Redundancy, SensorClass, SolutionQuality,
    SolutionStamp, SourceComposition, SourceEpoch, SymmetricCov3, WrappingSequence,
};
use navigate_visual::{LocalFrame, SearchPrior};
use navigate_visual_fusion::{FrameCapture, PropagationBudget, VisualFusionError, search_prior};

/// Active dead-reckoning mode during track loss.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeadReckonPath {
    /// Path A projects a search prior from a valid navigation filter solution.
    PathA,
    /// Path B uses optical flow or kinematic coasting without external sensors.
    PathB,
}

/// Dead-reckoning position and covariance estimate during track loss.
#[derive(Debug, Clone)]
pub struct DeadReckonEstimate {
    /// Estimated camera position in `ENU` meters.
    pub eye_enu: [f64; 3],
    /// Propagated position covariance matrix in meters squared.
    pub pos_cov_enu_m2: Matrix3<f64>,
    /// Active dead-reckoning path.
    pub path: DeadReckonPath,
}

/// Dual-path dead reckoner for external sensor priors and vision fallback.
#[derive(Debug, Clone)]
pub struct DualPathDeadReckoner {
    /// Kinematic state for coasting.
    pub kinematic: KinematicState,
    /// Flag that indicates whether external sensors are available.
    pub external_sensors_available: bool,
    /// Latest navigation filter solution and local frame for Path A projection.
    pub filter_solution: Option<(NavigationSolution, LocalFrame)>,
    /// Monotonic capture timestamp in nanoseconds for Path A propagation.
    pub capture_time_ns: u64,
}

impl DualPathDeadReckoner {
    /// Create a dead reckoner from initial position and velocity.
    #[must_use]
    pub fn new(initial_eye: [f64; 3], initial_vel: [f64; 3], external_sensors: bool) -> Self {
        let mut kin = KinematicState::new(initial_eye, Matrix3::identity());
        kin.vel_enu_mps = initial_vel;
        let capture_time_ns = 1_000_000_000;
        let filter_solution = if external_sensors && let Ok(frame) = LocalFrame::wgs84_enu(0.0, 0.0)
        {
            let [lat_deg, lon_deg, alt_m] = frame.geodetic(Vector3::from(initial_eye));
            let pos = GeodeticPosition::new(lat_deg.to_radians(), lon_deg.to_radians(), alt_m);
            let vel = NedVelocity::new(initial_vel[1], initial_vel[0], -initial_vel[2]);
            let stamp = SolutionStamp::new(
                SourceEpoch::new(0),
                WrappingSequence::new(0),
                MonotonicNanos::from_nanos(capture_time_ns),
                ClockDomainId::new(1),
            );
            let comp = SourceComposition::of(SensorClass::Inertial);
            let integrity = IntegrityAssessment::new(
                SolutionQuality::Good,
                comp,
                Redundancy::None,
                1.0,
                1.0,
                FaultDetection::Unavailable,
            );
            let sol = NavigationSolution::new(
                stamp,
                pos,
                vel,
                SymmetricCov3::from_diagonal(1.0, 1.0, 1.0),
                SymmetricCov3::from_diagonal(0.25, 0.25, 0.25),
                integrity,
                comp,
            );
            Some((sol, frame))
        } else {
            None
        };
        Self {
            kinematic: kin,
            external_sensors_available: external_sensors,
            filter_solution,
            capture_time_ns,
        }
    }

    /// Update the active navigation filter solution for Path A search-prior projection.
    pub fn set_filter_solution(
        &mut self,
        solution: NavigationSolution,
        frame: LocalFrame,
        capture_time_ns: u64,
    ) {
        self.filter_solution = Some((solution, frame));
        self.capture_time_ns = capture_time_ns;
    }

    /// Clear the cached navigation filter solution.
    pub fn clear_filter_solution(&mut self) {
        self.filter_solution = None;
    }

    /// Propagate dead-reckoning position and covariance across interval `dt_s`.
    pub fn step(
        &mut self,
        dt_s: f64,
        ground_flow_vectors: usize,
        a_rms_mps2: f64,
        omega_rms_rad_s: f64,
    ) -> DeadReckonEstimate {
        let dt = dt_s.clamp(MIN_DELTA_T_S, MAX_DELTA_T_S);
        self.capture_time_ns = self
            .capture_time_ns
            .saturating_add((dt * 1_000_000_000.0).round() as u64);

        if self.external_sensors_available
            && let Some((ref sol, frame)) = self.filter_solution
        {
            let att_sigma = (self.kinematic.att_sigma_rad + omega_rms_rad_s.max(0.0) * dt)
                .clamp(1e-3, std::f64::consts::PI);
            let capture = FrameCapture {
                at: MonotonicNanos::from_nanos(self.capture_time_ns),
                clock: sol.stamp.clock,
                time_error_bound: DurationNanos::from_nanos(1_000),
                orientation: UnitQuaternion::identity(),
                attitude_sigma_rad: att_sigma,
            };
            let budget = PropagationBudget {
                maximum_age: DurationNanos::from_nanos(120_000_000_000),
                solution_time_error_bound: DurationNanos::from_nanos(1_000),
                acceleration_rms_bound_mps2: Some(a_rms_mps2.max(0.1)),
            };
            if let Ok(prior) = Self::project_search_prior(sol, frame, capture, budget) {
                let pred_eye = [
                    prior.center.position.x,
                    prior.center.position.y,
                    prior.center.position.z,
                ];
                self.kinematic.eye_enu = pred_eye;
                self.kinematic.pos_cov_enu_m2 = prior.position_covariance_m2;
                self.kinematic.att_sigma_rad = prior.attitude_sigma_rad;
                return DeadReckonEstimate {
                    eye_enu: pred_eye,
                    pos_cov_enu_m2: prior.position_covariance_m2,
                    path: DeadReckonPath::PathA,
                };
            }
        }

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

    /// Project a Path A `SearchPrior` from the navigation filter solution.
    ///
    /// # Errors
    /// Returns [`VisualFusionError`] when clock domains do not match or coordinates are invalid.
    pub fn project_search_prior(
        solution: &NavigationSolution,
        frame: LocalFrame,
        capture: FrameCapture,
        budget: PropagationBudget,
    ) -> Result<SearchPrior, VisualFusionError> {
        search_prior(solution, frame, capture, budget)
    }
}
