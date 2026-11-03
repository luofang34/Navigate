//! Mission-phase orchestration over the surface and the space
//! derivations (ADR-0019).
//!
//! A [`MissionProfile`] names the phases of one mission:
//!
//! - [`MissionProfile::Reentry`]: [`MissionPhase::OrbitalCoast`] →
//!   [`MissionPhase::Burn`] → [`MissionPhase::EntryCorridor`] →
//!   [`MissionPhase::AtmosphericRoute`]. This is the sequence of a
//!   winged vehicle from orbit to a runway.
//! - [`MissionProfile::Landing`]: [`MissionPhase::OrbitalCoast`] →
//!   [`MissionPhase::PoweredDescent`] → [`MissionPhase::Landed`]. This
//!   is the sequence of a lander from a low orbit to touchdown.
//!
//! [`step_mission`] reads the [`MissionState`], the profile, and one
//! solution, and gives the next state and the command of the phase. The
//! state is `Copy` and holds the inline state of the entry law. Thus the
//! orchestrator does not allocate and keeps no hidden state. A phase
//! changes in the step that meets its trigger, and that step also gives
//! the command of the new phase when the new phase reads the same
//! solution.

use navigate_contract::{
    CartesianSolution, ClockDomainId, FrameId, GuidanceCommand, GuidanceSetpoint, MonotonicNanos,
    NavigationSolution,
};
use navigate_route::Leg;
use navigate_trajectory::{BurnArc, EntryCorridor, EntryState, TrajectoryError};

use crate::admission::admit_space_solution;
use crate::config::{GuidanceConfig, SpaceGuidanceConfig};
use crate::derive::guide;
use crate::refusal::GuidanceRefusal;
use crate::space::{PoweredDescentPlan, guide_burn, guide_entry, guide_powered_descent};

#[cfg(test)]
mod tests;

/// One phase of a mission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum MissionPhase {
    /// The vehicle coasts in orbit and waits for ignition. No command.
    OrbitalCoast,
    /// An orbital burn, for example a deorbit burn.
    Burn,
    /// The coast to the atmosphere and the entry, under bank control.
    EntryCorridor,
    /// The atmospheric flight on the legs of a route.
    AtmosphericRoute,
    /// A powered descent to a touchdown target.
    PoweredDescent,
    /// The vehicle is on the surface. No command.
    Landed,
}

/// The state of a mission between two steps.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct MissionState {
    /// The current phase.
    pub phase: MissionPhase,
    /// The inline state of the entry law.
    pub entry: EntryState,
}

impl MissionState {
    /// The start of every profile: an orbital coast with a new entry
    /// state.
    pub const START: Self = Self::new(MissionPhase::OrbitalCoast);

    /// A state in `phase` with a new entry state.
    #[must_use]
    pub const fn new(phase: MissionPhase) -> Self {
        Self {
            phase,
            entry: EntryState::INITIAL,
        }
    }

    const fn with_phase(self, phase: MissionPhase) -> Self {
        Self { phase, ..self }
    }
}

/// The phases and the targets of one mission.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum MissionProfile<'a> {
    /// From orbit through a burn and an entry corridor to a route.
    Reentry {
        /// The deorbit burn.
        burn: BurnArc,
        /// The entry corridor.
        entry: EntryCorridor<'a>,
    },
    /// From orbit through a powered descent to touchdown.
    Landing {
        /// The powered descent.
        descent: PoweredDescentPlan,
    },
}

impl MissionProfile<'_> {
    /// The frame of the solution that `phase` reads, or `None` for a
    /// phase that reads a surface solution or no solution. The orbital
    /// coast reads the frame of the next phase.
    #[must_use]
    pub const fn expected_frame(&self, phase: MissionPhase) -> Option<FrameId> {
        match (self, phase) {
            (Self::Reentry { burn, .. }, MissionPhase::OrbitalCoast | MissionPhase::Burn) => {
                Some(burn.frame)
            }
            (Self::Reentry { entry, .. }, MissionPhase::EntryCorridor) => Some(entry.frame()),
            (
                Self::Landing { descent },
                MissionPhase::OrbitalCoast | MissionPhase::PoweredDescent,
            ) => Some(descent.body.ellipsoid.frame()),
            _ => None,
        }
    }
}

/// The solution of one mission step.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum MissionInput<'s, 'p> {
    /// A solution in a Cartesian frame, for the space phases.
    Space {
        /// The solution.
        solution: &'s CartesianSolution,
        /// The sensed drag in m/s², from the accelerometers. The entry
        /// phase reads it. Other phases ignore it.
        sensed_drag_mps2: f64,
    },
    /// A solution on the surface frame of a body and the active leg, for
    /// the atmospheric route.
    Surface {
        /// The solution.
        solution: &'s NavigationSolution,
        /// The active leg of the route execution.
        leg: Leg<'p>,
    },
}

/// The admission floors of the space phases and of the route phase.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
#[non_exhaustive]
pub struct MissionConfig {
    /// The floors of the space phases.
    pub space: SpaceGuidanceConfig,
    /// The floors of the atmospheric route.
    pub surface: GuidanceConfig,
}

impl MissionConfig {
    /// Builds a config from its parts.
    #[must_use]
    pub const fn new(space: SpaceGuidanceConfig, surface: GuidanceConfig) -> Self {
        Self { space, surface }
    }
}

/// The result of one [`step_mission`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub struct MissionStep {
    /// The state for the next step.
    pub state: MissionState,
    /// The command of this step, or `None` in a coast or after landing.
    pub command: Option<GuidanceCommand>,
}

/// The time of one step: `now` and the clock domain of `now`.
#[derive(Clone, Copy)]
struct StepTime {
    now: MonotonicNanos,
    clock: ClockDomainId,
}

/// Steps the mission once.
///
/// # Errors
///
/// [`GuidanceRefusal::PhaseNotInProfile`] when the phase is not part of
/// the profile, [`GuidanceRefusal::PhaseInputMismatch`] when the input is
/// not the kind that the phase reads,
/// [`GuidanceRefusal::TrajectoryInvalid`] when the ignition window of the
/// burn closed during the coast, and the refusals of the derivation of
/// the phase: [`crate::admit_space_solution`], [`crate::guide_burn`],
/// [`crate::guide_entry`], [`crate::guide_powered_descent`], or
/// [`crate::guide`].
pub fn step_mission(
    state: MissionState,
    profile: &MissionProfile<'_>,
    input: MissionInput<'_, '_>,
    now: MonotonicNanos,
    now_clock: ClockDomainId,
    config: &MissionConfig,
) -> Result<MissionStep, GuidanceRefusal> {
    let time = StepTime {
        now,
        clock: now_clock,
    };
    match (profile, state.phase) {
        (_, MissionPhase::Landed) => Ok(MissionStep {
            state,
            command: None,
        }),
        (MissionProfile::Reentry { burn, entry }, _) => {
            reentry_step(state, burn, entry, input, time, config)
        }
        (MissionProfile::Landing { descent }, _) => {
            landing_step(state, descent, input, time, &config.space)
        }
    }
}

fn reentry_step(
    state: MissionState,
    burn: &BurnArc,
    entry: &EntryCorridor<'_>,
    input: MissionInput<'_, '_>,
    time: StepTime,
    config: &MissionConfig,
) -> Result<MissionStep, GuidanceRefusal> {
    let phase = state.phase;
    if phase == MissionPhase::AtmosphericRoute {
        let MissionInput::Surface { solution, leg } = input else {
            return Err(GuidanceRefusal::PhaseInputMismatch { phase });
        };
        let command = guide(
            solution,
            leg.lateral_reference(),
            leg.to,
            time.now,
            time.clock,
            &config.surface,
        )?;
        return Ok(MissionStep {
            state,
            command: Some(command),
        });
    }
    let (solution, sensed_drag_mps2) = space_input(phase, input)?;
    match phase {
        MissionPhase::OrbitalCoast => {
            admit_space_solution(solution, burn.frame, time.now, time.clock, &config.space)?;
            if burn.window_closed(time.now) {
                return Err(GuidanceRefusal::from_trajectory(
                    TrajectoryError::IgnitionWindowClosed {
                        close: burn.ignition_close,
                    },
                ));
            }
            if !burn.window_contains(time.now) {
                return Ok(MissionStep {
                    state,
                    command: None,
                });
            }
            burn_step(state, burn, solution, time, &config.space)
        }
        MissionPhase::Burn => burn_step(state, burn, solution, time, &config.space),
        MissionPhase::EntryCorridor => entry_step(
            state,
            entry,
            solution,
            sensed_drag_mps2,
            time,
            &config.space,
        ),
        _ => Err(GuidanceRefusal::PhaseNotInProfile { phase }),
    }
}

fn entry_step(
    state: MissionState,
    entry: &EntryCorridor<'_>,
    solution: &CartesianSolution,
    sensed_drag_mps2: f64,
    time: StepTime,
    config: &SpaceGuidanceConfig,
) -> Result<MissionStep, GuidanceRefusal> {
    let step = guide_entry(
        solution,
        entry,
        state.entry,
        sensed_drag_mps2,
        time.now,
        time.clock,
        config,
    )?;
    let phase = if step.taem_reached {
        MissionPhase::AtmosphericRoute
    } else {
        MissionPhase::EntryCorridor
    };
    Ok(MissionStep {
        state: MissionState {
            phase,
            entry: step.state,
        },
        command: Some(step.command),
    })
}

fn burn_step(
    state: MissionState,
    burn: &BurnArc,
    solution: &CartesianSolution,
    time: StepTime,
    config: &SpaceGuidanceConfig,
) -> Result<MissionStep, GuidanceRefusal> {
    let command = guide_burn(solution, burn, time.now, time.clock, config)?;
    let next = if is_cutoff(&command) {
        MissionPhase::EntryCorridor
    } else {
        MissionPhase::Burn
    };
    Ok(MissionStep {
        state: state.with_phase(next),
        command: Some(command),
    })
}

fn landing_step(
    state: MissionState,
    descent: &PoweredDescentPlan,
    input: MissionInput<'_, '_>,
    time: StepTime,
    config: &SpaceGuidanceConfig,
) -> Result<MissionStep, GuidanceRefusal> {
    let phase = state.phase;
    if !matches!(
        phase,
        MissionPhase::OrbitalCoast | MissionPhase::PoweredDescent
    ) {
        return Err(GuidanceRefusal::PhaseNotInProfile { phase });
    }
    let (solution, _) = space_input(phase, input)?;
    if phase == MissionPhase::OrbitalCoast {
        let frame = descent.body.ellipsoid.frame();
        admit_space_solution(solution, frame, time.now, time.clock, config)?;
        if time.now < descent.ignition_at {
            return Ok(MissionStep {
                state,
                command: None,
            });
        }
    }
    let command = guide_powered_descent(solution, descent, time.now, time.clock, config)?;
    let next = if is_cutoff(&command) {
        MissionPhase::Landed
    } else {
        MissionPhase::PoweredDescent
    };
    Ok(MissionStep {
        state: state.with_phase(next),
        command: Some(command),
    })
}

fn space_input<'s>(
    phase: MissionPhase,
    input: MissionInput<'s, '_>,
) -> Result<(&'s CartesianSolution, f64), GuidanceRefusal> {
    match input {
        MissionInput::Space {
            solution,
            sensed_drag_mps2,
        } => Ok((solution, sensed_drag_mps2)),
        MissionInput::Surface { .. } => Err(GuidanceRefusal::PhaseInputMismatch { phase }),
    }
}

const fn is_cutoff(command: &GuidanceCommand) -> bool {
    matches!(
        command.setpoint,
        GuidanceSetpoint::ThrustVector { cutoff: true, .. }
    )
}
