//! Space derivations: a Cartesian solution plus a trajectory target
//! gives one setpoint or one typed refusal (ADR-0019).
//!
//! Each derivation admits the solution through
//! [`crate::admit_space_solution`], then evaluates one law of
//! `navigate-trajectory`. A corridor violation of the law becomes
//! [`crate::GuidanceRefusal::CorridorViolated`]. Every other error of the
//! law becomes [`crate::GuidanceRefusal::TrajectoryInvalid`].

use navigate_contract::{CartesianSolution, DurationNanos, MonotonicNanos};
use navigate_trajectory::StateVector;

use crate::refusal::GuidanceRefusal;

mod burn;
mod descent;
mod entry;
mod rendezvous;

#[cfg(test)]
mod tests;

pub use burn::guide_burn;
pub use descent::{PoweredDescentPlan, guide_powered_descent};
pub use entry::{EntryGuidance, guide_entry};
pub use rendezvous::{RendezvousPlan, guide_rendezvous};

/// Nanoseconds in one second.
const NANOS_PER_SECOND: f64 = 1e9;

/// The signed time in seconds from `now` to `later`. It is negative when
/// `later` is in the past of `now`.
pub(crate) fn seconds_until(now: MonotonicNanos, later: MonotonicNanos) -> f64 {
    let seconds = |span: DurationNanos| span.as_nanos() as f64 / NANOS_PER_SECOND;
    later.elapsed_since(now).map_or_else(
        || {
            now.elapsed_since(later)
                .map_or(0.0, |behind| -seconds(behind))
        },
        seconds,
    )
}

/// The state vector of an admitted solution.
pub(crate) fn state_of(solution: &CartesianSolution) -> Result<StateVector, GuidanceRefusal> {
    StateVector::from_solution(solution).map_err(GuidanceRefusal::from_trajectory)
}
