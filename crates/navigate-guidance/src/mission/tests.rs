//! Mission-phase handoffs, with closed-loop reentry and landing
//! simulations in [`shuttle`] and [`lunar`].

use nalgebra::Vector3;
use navigate_astrodynamics::PlanetaryBody;
use navigate_contract::{FrameId, GuidanceCommand, GuidanceSetpoint};
use navigate_trajectory::{
    BurnArc, BurnTarget, EntryCorridor, EntryLaw, EntryNode, PoweredDescentTarget, RadialBranch,
    StateVector, TrajectoryError,
};

use super::{
    MissionConfig, MissionInput, MissionPhase, MissionProfile, MissionState, MissionStep,
    step_mission,
};
use crate::refusal::GuidanceRefusal;
use crate::scenario::CLOCK;
use crate::scenario::space::{at_millis, point_mass, rk4, space_solution};
use crate::space::PoweredDescentPlan;

mod lunar;
mod shuttle;

const EARTH: PlanetaryBody = PlanetaryBody::EARTH_WGS84;
const EARTH_RADIUS_M: f64 = 6_378_137.0;
const ORBIT_RADIUS_M: f64 = EARTH_RADIUS_M + 400e3;

const NODES: [EntryNode; 2] = [
    EntryNode::new(-2.0e7, 1.0, 0.0),
    EntryNode::new(-7.0e7, 12.0, 0.0),
];

fn deorbit_arc() -> BurnArc {
    BurnArc::new(
        FrameId::EARTH_J2000,
        BurnTarget::Apsides {
            periapsis_radius_m: EARTH_RADIUS_M + 30e3,
            apoapsis_radius_m: ORBIT_RADIUS_M,
            gm_m3_per_s2: EARTH.gm_m3_per_s2,
            branch: RadialBranch::Descending,
        },
        at_millis(60_000),
        at_millis(120_000),
        0.3,
        0.6,
    )
}

fn reentry(nodes: &[EntryNode]) -> MissionProfile<'_> {
    MissionProfile::Reentry {
        burn: deorbit_arc(),
        entry: EntryCorridor::new(
            EARTH,
            nodes,
            Vector3::new(0.0, EARTH_RADIUS_M, 0.0),
            -6.2e7,
            EntryLaw::new(1.0, 7_200.0),
        ),
    }
}

fn leo() -> StateVector {
    StateVector::new(
        Vector3::new(ORBIT_RADIUS_M, 0.0, 0.0),
        Vector3::new(0.0, (EARTH.gm_m3_per_s2 / ORBIT_RADIUS_M).sqrt(), 0.0),
    )
}

fn step_at(
    state: MissionState,
    profile: &MissionProfile<'_>,
    frame: FrameId,
    millis: u64,
) -> Result<MissionStep, GuidanceRefusal> {
    let solution = space_solution(frame, &leo(), at_millis(millis), 5.0);
    let input = MissionInput::Space {
        solution: &solution,
        sensed_drag_mps2: 0.0,
    };
    let config = MissionConfig::default();
    step_mission(state, profile, input, at_millis(millis), CLOCK, &config)
}

fn thrust_of(command: Option<GuidanceCommand>) -> Option<(Vector3<f64>, f64, bool)> {
    match command?.setpoint {
        GuidanceSetpoint::ThrustVector {
            direction_unit,
            specific_force_mps2,
            cutoff,
            ..
        } => Some((Vector3::from(direction_unit), specific_force_mps2, cutoff)),
        _ => None,
    }
}

#[test]
fn the_coast_waits_for_the_ignition_window_and_then_burns() {
    let profile = reentry(&NODES);
    let coast = step_at(MissionState::START, &profile, FrameId::EARTH_J2000, 30_000).unwrap();
    assert_eq!(coast.state, MissionState::START);
    assert_eq!(coast.command, None);
    let ignition = step_at(MissionState::START, &profile, FrameId::EARTH_J2000, 60_000).unwrap();
    assert_eq!(ignition.state.phase, MissionPhase::Burn);
    let (direction, force, cutoff) = thrust_of(ignition.command).unwrap();
    assert!(!cutoff);
    assert!((force - 0.6).abs() < 1e-12);
    assert!(
        direction.y < -0.9,
        "a deorbit burn thrusts against the velocity"
    );
}

#[test]
fn a_closed_window_is_refused_in_the_coast() {
    let profile = reentry(&NODES);
    assert_eq!(
        step_at(MissionState::START, &profile, FrameId::EARTH_J2000, 121_000),
        Err(GuidanceRefusal::TrajectoryInvalid {
            error: TrajectoryError::IgnitionWindowClosed {
                close: at_millis(120_000)
            }
        })
    );
}

#[test]
fn each_phase_reads_its_own_frame_and_input() {
    let profile = reentry(&NODES);
    assert_eq!(
        profile.expected_frame(MissionPhase::EntryCorridor),
        Some(FrameId::EARTH_ITRF93)
    );
    assert_eq!(profile.expected_frame(MissionPhase::AtmosphericRoute), None);
    let entry = MissionState::new(MissionPhase::EntryCorridor);
    assert_eq!(
        step_at(entry, &profile, FrameId::EARTH_J2000, 1_000),
        Err(GuidanceRefusal::FrameMismatch {
            expected: FrameId::EARTH_ITRF93,
            got: FrameId::EARTH_J2000,
        })
    );
    let route = MissionState::new(MissionPhase::AtmosphericRoute);
    assert_eq!(
        step_at(route, &profile, FrameId::EARTH_ITRF93, 1_000),
        Err(GuidanceRefusal::PhaseInputMismatch {
            phase: MissionPhase::AtmosphericRoute
        })
    );
    let descent = MissionState::new(MissionPhase::PoweredDescent);
    assert_eq!(
        step_at(descent, &profile, FrameId::EARTH_J2000, 1_000),
        Err(GuidanceRefusal::PhaseNotInProfile {
            phase: MissionPhase::PoweredDescent
        })
    );
}

#[test]
fn a_landed_vehicle_gets_no_command() {
    let target = PoweredDescentTarget::new(
        Vector3::new(0.0, 0.0, 1_737_400.0),
        Vector3::zeros(),
        Vector3::zeros(),
        0.5,
        5.0,
    );
    let plan = PoweredDescentPlan::new(
        PlanetaryBody::MOON_PA,
        target,
        at_millis(0),
        at_millis(600_000),
    );
    let profile = MissionProfile::Landing { descent: plan };
    let landed = MissionState::new(MissionPhase::Landed);
    let step = step_at(landed, &profile, FrameId::EARTH_J2000, 1_000).unwrap();
    assert_eq!(step.state, landed);
    assert_eq!(step.command, None);
    let burn = MissionState::new(MissionPhase::Burn);
    assert_eq!(
        step_at(burn, &profile, FrameId::MOON_PA, 1_000),
        Err(GuidanceRefusal::PhaseNotInProfile {
            phase: MissionPhase::Burn
        })
    );
}
