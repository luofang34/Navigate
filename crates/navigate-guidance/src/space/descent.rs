//! The powered descent derivation.

use navigate_astrodynamics::PlanetaryBody;
use navigate_contract::{
    CartesianSolution, ClockDomainId, GuidanceCommand, GuidanceSetpoint, MonotonicNanos,
};
use navigate_trajectory::{PoweredDescentTarget, TrajectoryError, evaluate_powered_descent};

use crate::admission::admit_space_solution;
use crate::config::SpaceGuidanceConfig;
use crate::refusal::GuidanceRefusal;

use super::{seconds_until, state_of};

/// One powered descent to a touchdown target on one body.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct PoweredDescentPlan {
    /// The body. Its body-fixed frame is the frame of the target and of
    /// the solution, and its gravity field gives `g(p)`.
    pub body: PlanetaryBody,
    /// The touchdown target in the body-fixed frame.
    pub target: PoweredDescentTarget,
    /// The powered descent initiation (PDI) time.
    pub ignition_at: MonotonicNanos,
    /// The planned touchdown time. The time to go is `touchdown_at − now`.
    pub touchdown_at: MonotonicNanos,
}

impl PoweredDescentPlan {
    /// Builds a plan from its parts.
    #[must_use]
    pub const fn new(
        body: PlanetaryBody,
        target: PoweredDescentTarget,
        ignition_at: MonotonicNanos,
        touchdown_at: MonotonicNanos,
    ) -> Self {
        Self {
            body,
            target,
            ignition_at,
            touchdown_at,
        }
    }
}

/// Derives one thrust-vector command of the powered descent `plan` from
/// a solution in the body-fixed frame of `plan.body`, or refuses with a
/// typed reason.
///
/// The command is [`GuidanceSetpoint::ThrustVector`] in the body-fixed
/// frame. The gravity is the apparent gravity of `plan.body` at the
/// position of the solution. The command has `cutoff = true` inside the
/// capture box of the target.
///
/// # Errors
///
/// [`GuidanceRefusal::TrajectoryInvalid`] when the body is not
/// plausible, the refusals of [`crate::admit_space_solution`] with the
/// body-fixed frame, [`GuidanceRefusal::CorridorViolated`] below the
/// glide slope, and [`GuidanceRefusal::TrajectoryInvalid`] for every
/// other error of the law.
pub fn guide_powered_descent(
    solution: &CartesianSolution,
    plan: &PoweredDescentPlan,
    now: MonotonicNanos,
    now_clock: ClockDomainId,
    config: &SpaceGuidanceConfig,
) -> Result<GuidanceCommand, GuidanceRefusal> {
    if !plan.body.is_plausible() {
        return Err(GuidanceRefusal::from_trajectory(
            TrajectoryError::InvalidParameter { name: "body" },
        ));
    }
    let frame = plan.body.ellipsoid.frame();
    admit_space_solution(solution, frame, now, now_clock, config)?;
    let state = state_of(solution)?;
    let (gravity, _) = plan.body.apparent_gravity_body_fixed(&state.position_m);
    let command = evaluate_powered_descent(
        &plan.target,
        &state,
        seconds_until(now, plan.touchdown_at),
        &gravity,
    )
    .map_err(GuidanceRefusal::from_trajectory)?;
    Ok(GuidanceCommand::new(
        now,
        GuidanceSetpoint::ThrustVector {
            frame,
            direction_unit: command.direction_unit.into(),
            specific_force_mps2: command.specific_force_mps2,
            velocity_to_go_mps: command.velocity_to_go_mps,
            cutoff: command.cutoff,
        },
        solution.stamp,
    ))
}
