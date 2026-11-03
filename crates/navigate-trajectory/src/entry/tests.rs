//! The reference profile, the tracking law, the bank reversal, and the
//! TAEM flag.

use nalgebra::Vector3;
use navigate_astrodynamics::PlanetaryBody;

use super::{
    EntryCorridor, EntryLaw, EntryNode, EntryState, evaluate_entry, specific_energy_j_per_kg,
};
use crate::error::TrajectoryError;
use crate::state::StateVector;

const RADIUS_M: f64 = 6_378_137.0;

const NODES: [EntryNode; 3] = [
    EntryNode::new(-3.0e7, 2.0, 0.0),
    EntryNode::new(-4.0e7, 8.0, 0.02),
    EntryNode::new(-6.0e7, 12.0, 0.0),
];

fn corridor(nodes: &[EntryNode]) -> EntryCorridor<'_> {
    let site = Vector3::new(RADIUS_M, 0.0, 0.0);
    EntryCorridor::new(
        PlanetaryBody::EARTH_WGS84,
        nodes,
        site,
        -6.1e7,
        EntryLaw::new(1.0, 7_200.0),
    )
}

/// An eastbound vehicle at 60 km over the equator, `angle_deg` west of
/// the site, flying at `speed` m/s.
fn eastbound(angle_deg: f64, speed: f64) -> StateVector {
    let angle = angle_deg.to_radians();
    let r = RADIUS_M + 60e3;
    StateVector::new(
        Vector3::new(r * angle.cos(), -r * angle.sin(), 0.0),
        Vector3::new(speed * angle.sin(), speed * angle.cos(), 0.0),
    )
}

#[test]
fn the_reference_is_linear_between_knots_and_flat_outside() {
    let corridor = corridor(&NODES);
    assert_eq!(corridor.reference_at(-1.0e7), (2.0, 0.0));
    assert_eq!(corridor.reference_at(-9.0e7), (12.0, 0.0));
    let (drag, rate) = corridor.reference_at(-3.5e7);
    assert!((drag - 5.0).abs() < 1e-12);
    assert!((rate - 0.01).abs() < 1e-12);
}

#[test]
fn knots_out_of_order_are_refused() {
    let nodes = [NODES[1], NODES[0]];
    assert_eq!(
        corridor(&nodes).validate(),
        Err(TrajectoryError::InvalidCorridor)
    );
    assert_eq!(
        corridor(&NODES[..1]).validate(),
        Err(TrajectoryError::InvalidCorridor)
    );
}

#[test]
fn below_the_tracking_threshold_the_lift_points_up() {
    let corridor = corridor(&NODES);
    let (command, next) = evaluate_entry(
        &corridor,
        EntryState::INITIAL,
        &eastbound(20.0, 7_000.0),
        0.1,
    )
    .unwrap();
    assert!(!command.tracking);
    assert_eq!(command.bank_rad, 0.0);
    assert_eq!(next, EntryState::INITIAL);
    assert!((command.heading_course_rad - core::f64::consts::FRAC_PI_2).abs() < 1e-9);
    assert!(command.heading_error_rad.abs() < 1e-9);
}

#[test]
fn more_drag_than_the_reference_commands_more_lift_up() {
    let corridor = corridor(&NODES);
    let vehicle = eastbound(20.0, 6_500.0);
    let energy = specific_energy_j_per_kg(&vehicle, PlanetaryBody::EARTH_WGS84.gm_m3_per_s2);
    let (reference, _) = corridor.reference_at(energy);
    let bank = |drag: f64| {
        evaluate_entry(&corridor, EntryState::INITIAL, &vehicle, drag)
            .unwrap()
            .0
            .bank_rad
    };
    assert!(bank(reference + 1.0).cos() > bank(reference).cos());
    assert!(bank(reference - 1.0).cos() < bank(reference).cos());
}

#[test]
fn a_heading_error_past_the_deadband_reverses_the_bank() {
    let corridor = corridor(&NODES);
    let mut vehicle = eastbound(20.0, 6_500.0);
    let turn = 15f64.to_radians();
    let v = vehicle.velocity_mps;
    vehicle.velocity_mps = Vector3::new(v.x, v.y * turn.cos(), -v.y * turn.sin());
    let (command, next) = evaluate_entry(&corridor, EntryState::INITIAL, &vehicle, 6.0).unwrap();
    assert!(command.heading_error_rad > corridor.law.heading_deadband_rad);
    assert_eq!(next.bank_sign, -1.0);
    assert_eq!(next.reversals, 1);
    assert!(command.bank_rad <= 0.0);
    let (_, again) = evaluate_entry(&corridor, next, &vehicle, 6.0).unwrap();
    assert_eq!(again, next, "the bank holds while it turns toward the site");
}

#[test]
fn the_taem_flag_rises_at_the_handoff_energy() {
    let corridor = corridor(&NODES);
    let (command, _) =
        evaluate_entry(&corridor, EntryState::INITIAL, &eastbound(2.0, 700.0), 12.0).unwrap();
    assert!(command.taem_reached);
    let (command, _) = evaluate_entry(
        &corridor,
        EntryState::INITIAL,
        &eastbound(20.0, 6_500.0),
        6.0,
    )
    .unwrap();
    assert!(!command.taem_reached);
}

#[test]
fn the_heading_is_undefined_on_the_polar_axis() {
    let corridor = corridor(&NODES);
    let vehicle = StateVector::new(
        Vector3::new(0.0, 0.0, RADIUS_M + 60e3),
        Vector3::new(7_000.0, 0.0, 0.0),
    );
    assert_eq!(
        evaluate_entry(&corridor, EntryState::INITIAL, &vehicle, 6.0),
        Err(TrajectoryError::UndefinedHeading)
    );
    assert!(matches!(
        evaluate_entry(
            &corridor,
            EntryState::INITIAL,
            &eastbound(20.0, 6_500.0),
            -1.0
        ),
        Err(TrajectoryError::InvalidParameter {
            name: "sensed_drag_mps2"
        })
    ));
}

#[test]
fn the_reversal_counter_wraps() {
    let state = EntryState {
        bank_sign: -1.0,
        reversals: u32::MAX,
    };
    let next = state.reversed();
    assert_eq!(next.reversals, 0);
    assert_eq!(next.bank_sign, 1.0);
}
