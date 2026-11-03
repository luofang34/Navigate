//! Space admission and the rendezvous derivation (ADR-0019).

use nalgebra::Vector3;
use navigate_astrodynamics::PlanetaryBody;
use navigate_contract::{
    ClockDomainId, DurationNanos, FrameId, GuidanceSetpoint, MonotonicNanos, SolutionQuality,
};
use navigate_trajectory::{
    ApproachCorridor, BurnArc, BurnTarget, LvlhFrame, RendezvousTarget, StateVector,
    TrajectoryError,
};

use super::{RendezvousPlan, guide_burn, guide_rendezvous, seconds_until};
use crate::admission::admit_space_solution;
use crate::config::SpaceGuidanceConfig;
use crate::refusal::GuidanceRefusal;
use crate::scenario::CLOCK;
use crate::scenario::space::{at_millis, point_mass, rk4, space_solution};

mod rendezvous;

const EARTH_GM: f64 = PlanetaryBody::EARTH_WGS84.gm_m3_per_s2;
const ORBIT_RADIUS_M: f64 = 6_378_137.0 + 400e3;

fn leo() -> StateVector {
    StateVector::new(
        Vector3::new(ORBIT_RADIUS_M, 0.0, 0.0),
        Vector3::new(0.0, (EARTH_GM / ORBIT_RADIUS_M).sqrt(), 0.0),
    )
}

#[test]
fn space_admission_enforces_its_floors_in_order() {
    let config = SpaceGuidanceConfig::default().with_max_position_1sigma_m(10.0);
    let now = at_millis(2_000);
    let good = space_solution(FrameId::EARTH_J2000, &leo(), at_millis(1_900), 5.0);
    assert_eq!(
        admit_space_solution(&good, FrameId::EARTH_J2000, now, CLOCK, &config),
        Ok(())
    );
    let other_clock = ClockDomainId::new(9);
    assert!(matches!(
        admit_space_solution(&good, FrameId::MOON_J2000, now, other_clock, &config),
        Err(GuidanceRefusal::ClockDomainMismatch { .. })
    ));
    let mut unusable = good;
    unusable.integrity.quality = SolutionQuality::Unusable;
    assert!(matches!(
        admit_space_solution(&unusable, FrameId::MOON_J2000, now, CLOCK, &config),
        Err(GuidanceRefusal::IntegrityBelowFloor { .. })
    ));
    assert!(matches!(
        admit_space_solution(&good, FrameId::MOON_J2000, at_millis(1_000), CLOCK, &config),
        Err(GuidanceRefusal::ClockInversion { .. })
    ));
    assert_eq!(
        admit_space_solution(&good, FrameId::MOON_J2000, at_millis(3_000), CLOCK, &config),
        Err(GuidanceRefusal::SolutionStale {
            age: DurationNanos::from_millis(1_100),
            bound: DurationNanos::from_millis(1_000),
        })
    );
    assert_eq!(
        admit_space_solution(&good, FrameId::MOON_J2000, now, CLOCK, &config),
        Err(GuidanceRefusal::FrameMismatch {
            expected: FrameId::MOON_J2000,
            got: FrameId::EARTH_J2000,
        })
    );
    let wide = space_solution(FrameId::EARTH_J2000, &leo(), at_millis(1_900), 12.0);
    assert_eq!(
        admit_space_solution(&wide, FrameId::EARTH_J2000, now, CLOCK, &config),
        Err(GuidanceRefusal::UncertaintyExceeded {
            max_axis_1sigma_m: 12.0,
            bound_m: 10.0,
        })
    );
    let nan = space_solution(FrameId::EARTH_J2000, &leo(), at_millis(1_900), f64::NAN);
    assert!(matches!(
        admit_space_solution(&nan, FrameId::EARTH_J2000, now, CLOCK, &config),
        Err(GuidanceRefusal::UncertaintyExceeded { .. })
    ));
}

#[test]
fn a_burn_gives_a_thrust_vector_in_its_frame() {
    let state = leo();
    let arc = BurnArc::new(
        FrameId::EARTH_J2000,
        BurnTarget::DeltaV {
            ignition_velocity_mps: state.velocity_mps,
            delta_v_mps: Vector3::new(0.0, -100.0, 0.0),
        },
        at_millis(0),
        at_millis(10_000),
        0.3,
        0.6,
    );
    let solution = space_solution(FrameId::EARTH_J2000, &state, at_millis(1_000), 5.0);
    let command = guide_burn(
        &solution,
        &arc,
        at_millis(1_000),
        CLOCK,
        &SpaceGuidanceConfig::default(),
    )
    .unwrap();
    assert_eq!(command.basis, solution.stamp);
    let GuidanceSetpoint::ThrustVector {
        frame,
        direction_unit,
        specific_force_mps2,
        velocity_to_go_mps,
        cutoff,
    } = command.setpoint
    else {
        panic!("expected a thrust vector, got {:?}", command.setpoint);
    };
    assert_eq!(frame, FrameId::EARTH_J2000);
    assert!((Vector3::from(direction_unit) + Vector3::y()).norm() < 1e-9);
    assert!((specific_force_mps2 - 0.6).abs() < 1e-12);
    assert!((velocity_to_go_mps - 100.0).abs() < 1e-6);
    assert!(!cutoff);
    let mut bad = arc;
    bad.max_specific_force_mps2 = 0.0;
    assert_eq!(
        guide_burn(
            &solution,
            &bad,
            at_millis(1_000),
            CLOCK,
            &SpaceGuidanceConfig::default()
        ),
        Err(GuidanceRefusal::TrajectoryInvalid {
            error: TrajectoryError::InvalidParameter {
                name: "max_specific_force_mps2"
            }
        })
    );
}

#[test]
fn signed_seconds_run_both_ways() {
    let now = MonotonicNanos::from_nanos(5_000_000_000);
    assert!((seconds_until(now, at_millis(7_500)) - 2.5).abs() < 1e-12);
    assert!((seconds_until(now, at_millis(4_000)) + 1.0).abs() < 1e-12);
}
