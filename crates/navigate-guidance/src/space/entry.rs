//! The atmospheric entry derivation.

use navigate_contract::{
    CartesianSolution, ClockDomainId, GuidanceCommand, GuidanceSetpoint, MonotonicNanos,
};
use navigate_trajectory::{EntryCorridor, EntryState, evaluate_entry};

use crate::admission::admit_space_solution;
use crate::config::SpaceGuidanceConfig;
use crate::refusal::GuidanceRefusal;

use super::state_of;

/// The output of one step of [`guide_entry`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct EntryGuidance {
    /// The bank command.
    pub command: GuidanceCommand,
    /// The inline state for the next step.
    pub state: EntryState,
    /// `true` when the energy is at the TAEM energy of the corridor or
    /// less, so that the atmospheric route can start.
    pub taem_reached: bool,
}

/// Derives one bank command of the entry `corridor` from a solution in
/// the body-fixed frame of the corridor and the sensed drag, or refuses
/// with a typed reason.
///
/// The command is [`GuidanceSetpoint::EntryBank`]. The caller keeps the
/// returned [`EntryState`] and gives it to the next step.
///
/// # Errors
///
/// The refusals of [`crate::admit_space_solution`] with
/// `expected_frame = corridor.frame()`, and
/// [`GuidanceRefusal::TrajectoryInvalid`] for an invalid corridor, an
/// invalid sensed drag, or an undefined heading.
pub fn guide_entry(
    solution: &CartesianSolution,
    corridor: &EntryCorridor<'_>,
    state: EntryState,
    sensed_drag_mps2: f64,
    now: MonotonicNanos,
    now_clock: ClockDomainId,
    config: &SpaceGuidanceConfig,
) -> Result<EntryGuidance, GuidanceRefusal> {
    admit_space_solution(solution, corridor.frame(), now, now_clock, config)?;
    let (command, state) = evaluate_entry(corridor, state, &state_of(solution)?, sensed_drag_mps2)
        .map_err(GuidanceRefusal::from_trajectory)?;
    Ok(EntryGuidance {
        command: GuidanceCommand::new(
            now,
            GuidanceSetpoint::EntryBank {
                bank_rad: command.bank_rad,
                reference_drag_mps2: command.reference_drag_mps2,
                drag_error_mps2: command.drag_error_mps2,
                heading_course_rad: command.heading_course_rad,
            },
            solution.stamp,
        ),
        state,
        taem_reached: command.taem_reached,
    })
}
