//! The filter state at a past time, from the history (ADR-0012,
//! ADR-0014).
//!
//! A late camera frame needs the state at its exposure time: under a
//! rotation, the attitude at the delivery time differs from the
//! attitude at the exposure time by the angular rate times the latency.
//! The query gives the state that the replay of a delayed observation
//! at the same time linearizes around. It is not a smoothed state: it is
//! conditioned on the measurements acquired at or before that time that
//! the filter has ingested. A later delayed measurement can change the
//! answer of a later query.

use navigate_contract::{MonotonicNanos, NavigationSolution, SolutionStamp};
use thiserror::Error;

use super::InertialNavigationFilter;
use super::publish::SurfaceState;
use crate::integrity;
use crate::kalman::{horizontal_1sigma_m, vertical_1sigma_m};
use crate::rejection::RejectionReason;

/// The filter state at a past time.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct PastState {
    /// The geodetic solution at the past time, with the conventions of
    /// [`InertialNavigationFilter::tick`] and `solved_at` at the past
    /// time. Its sequence is that of the last published solution: a
    /// past state is not a publication. Its integrity is the integrity
    /// of the present assessment window, judged with the past
    /// covariance.
    pub solution: NavigationSolution,
    /// Cross-covariance of the NED position and the attitude about the
    /// NED axes, in m·rad, row-major: row `i` is position axis `i`.
    pub position_attitude_cov: [[f64; 3]; 3],
    /// Angular rate of the body relative to inertial space, in rad/s, in
    /// the body axes, with the estimated gyroscope bias removed. It is
    /// the rate of the IMU interval that holds the past time.
    pub angular_rate_body_rad_s: [f64; 3],
}

/// Why the filter gives no state at a past time.
#[derive(Debug, Clone, Copy, PartialEq, Error)]
#[non_exhaustive]
pub enum PastStateRefusal {
    /// The filter has no state yet.
    #[error("the filter is not initialized")]
    NotInitialized,
    /// The regime has no reference ellipsoid, so a geodetic state has
    /// no meaning. [`InertialNavigationFilter::tick_space`] gives the
    /// present state of such a regime.
    #[error("the regime has no reference ellipsoid")]
    NoSurfaceEllipsoid,
    /// The time is after the state time. Use
    /// [`InertialNavigationFilter::tick`] for the present.
    #[error(
        "time {} ns is after the state time {} ns",
        at.as_nanos(),
        state_at.as_nanos()
    )]
    AfterStateTime {
        /// The requested time.
        at: MonotonicNanos,
        /// The state time of the filter.
        state_at: MonotonicNanos,
    },
    /// The time is before the oldest state that the history keeps. The
    /// filter does not clamp to that state and does not extrapolate.
    #[error(
        "time {} ns is before the oldest kept state at {} ns",
        at.as_nanos(),
        oldest.as_nanos()
    )]
    BeforeHistory {
        /// The requested time.
        at: MonotonicNanos,
        /// The oldest time with a state.
        oldest: MonotonicNanos,
    },
    /// The propagation from the kept state to the time failed.
    #[error("the propagation to the past time failed: {reason}")]
    Propagation {
        /// Why the propagation failed.
        reason: RejectionReason,
    },
}

impl InertialNavigationFilter {
    /// The state at `at`, which is at or before the state time and at or
    /// after the oldest kept state.
    ///
    /// The filter takes the core before the first history entry later
    /// than `at` and propagates it to `at` with the rates of the IMU
    /// interval that holds `at`. The covariance propagates with it. The
    /// call does not change the filter and does not advance the solution
    /// sequence, so it is a pure function of the retained state.
    ///
    /// The depth is the depth of the history: [`crate::inertial::MAX_INERTIAL_HISTORY`]
    /// events, each IMU sample one event.
    ///
    /// # Errors
    ///
    /// [`PastStateRefusal`] names the condition.
    pub fn state_at(&self, at: MonotonicNanos) -> Result<PastState, PastStateRefusal> {
        let current = self.core.as_ref().ok_or(PastStateRefusal::NotInitialized)?;
        if at > current.state_at {
            return Err(PastStateRefusal::AfterStateTime {
                at,
                state_at: current.state_at,
            });
        }
        let capacity = self.config.fusion.history_capacity;
        let (mut past, covering) = self
            .history
            .base_at(current, at, capacity)
            .map_err(|oldest| PastStateRefusal::BeforeHistory { at, oldest })?;
        past.propagate_with(at, covering, &self.config)
            .map_err(|reason| PastStateRefusal::Propagation { reason })?;
        let ellipsoid = past
            .regime
            .surface
            .ok_or(PastStateRefusal::NoSurfaceEllipsoid)?;
        let rates = covering
            .or(past.last_rates)
            .ok_or(PastStateRefusal::Propagation {
                reason: RejectionReason::NoInertialData,
            })?;
        let state = SurfaceState::of(&past, ellipsoid);
        let assessment = integrity::derive(
            &self.ledger,
            at,
            horizontal_1sigma_m(&state.pos_cov),
            vertical_1sigma_m(&state.pos_cov),
            &self.config.fusion,
        );
        let stamp = SolutionStamp::new(self.epoch, self.sequencer.last(), at, self.clock);
        let cross = state.pos_att_cov;
        Ok(PastState {
            solution: state.solution(
                stamp,
                assessment,
                self.ledger.epoch_composition(),
                ellipsoid,
            ),
            position_attitude_cov: core::array::from_fn(|i| {
                core::array::from_fn(|j| cross[(i, j)])
            }),
            angular_rate_body_rad_s: (rates.angular_rate - past.nominal.b_g).into(),
        })
    }
}
