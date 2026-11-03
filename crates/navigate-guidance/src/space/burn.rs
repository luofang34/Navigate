//! The orbital burn derivation.

use navigate_contract::{
    CartesianSolution, ClockDomainId, GuidanceCommand, GuidanceSetpoint, MonotonicNanos,
};
use navigate_trajectory::{BurnArc, evaluate_burn};

use crate::admission::admit_space_solution;
use crate::config::SpaceGuidanceConfig;
use crate::refusal::GuidanceRefusal;

use super::state_of;

/// Derives one thrust-vector command of the burn `arc` from a solution in
/// `arc.frame`, or refuses with a typed reason.
///
/// The command is [`GuidanceSetpoint::ThrustVector`] in `arc.frame`.
/// This function does not check the ignition window. The mission
/// orchestrator [`crate::step_mission`] starts the burn only inside the
/// window.
///
/// # Errors
///
/// The refusals of [`crate::admit_space_solution`] with
/// `expected_frame = arc.frame`, and
/// [`GuidanceRefusal::TrajectoryInvalid`] when the arc is invalid or the
/// state is outside the reach of the target.
pub fn guide_burn(
    solution: &CartesianSolution,
    arc: &BurnArc,
    now: MonotonicNanos,
    now_clock: ClockDomainId,
    config: &SpaceGuidanceConfig,
) -> Result<GuidanceCommand, GuidanceRefusal> {
    admit_space_solution(solution, arc.frame, now, now_clock, config)?;
    let state = state_of(solution)?;
    let command = evaluate_burn(arc, &state).map_err(GuidanceRefusal::from_trajectory)?;
    Ok(GuidanceCommand::new(
        now,
        GuidanceSetpoint::ThrustVector {
            frame: arc.frame,
            direction_unit: command.direction_unit.into(),
            specific_force_mps2: command.specific_force_mps2,
            velocity_to_go_mps: command.velocity_to_go_mps.norm(),
            cutoff: command.cutoff,
        },
        solution.stamp,
    ))
}
