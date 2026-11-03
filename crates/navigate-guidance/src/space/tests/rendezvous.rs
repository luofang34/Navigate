//! The rendezvous derivation: a closed-loop approach in the corridor,
//! and the refusals for a frame mismatch and a corridor violation.

use super::*;

fn corridor() -> ApproachCorridor {
    ApproachCorridor::new(Vector3::new(0.0, -1.0, 0.0), 15f64.to_radians(), 200.0, 3.0)
}

fn plan(arrival_ms: u64) -> RendezvousPlan {
    RendezvousPlan::new(
        FrameId::EARTH_J2000,
        RendezvousTarget::new(Vector3::new(0.0, -30.0, 0.0), corridor(), 60.0),
        at_millis(arrival_ms),
    )
}

/// A chaser at the LVLH position `rho`, at rest in LVLH.
fn chaser_at(target: &StateVector, rho: Vector3<f64>) -> StateVector {
    let frame = LvlhFrame::from_target(target).unwrap();
    let [radial, along, normal] = frame.axes();
    StateVector::new(
        target.position_m + radial * rho.x + along * rho.y + normal * rho.z,
        frame.inertial_velocity(&rho, &Vector3::zeros()),
    )
}

#[test]
fn a_closed_loop_approach_reaches_the_hold_point_in_the_corridor() {
    let config = SpaceGuidanceConfig::default();
    let mut target = leo();
    let mut chaser = chaser_at(&target, Vector3::new(0.0, -250.0, 0.0));
    let plan = plan(300_000);
    for step in 0..600_u64 {
        let now = at_millis(step * 1_000);
        let own = space_solution(FrameId::EARTH_J2000, &chaser, now, 0.5);
        let other = space_solution(FrameId::EARTH_J2000, &target, now, 0.5);
        let command = guide_rendezvous(&own, &other, &plan, now, CLOCK, &config).unwrap();
        let GuidanceSetpoint::CartesianVelocity {
            frame,
            velocity_mps,
        } = command.setpoint
        else {
            panic!("expected a Cartesian velocity, got {:?}", command.setpoint);
        };
        assert_eq!(frame, FrameId::EARTH_J2000);
        chaser.velocity_mps = Vector3::from(velocity_mps);
        chaser = rk4(&chaser, 1.0, |p, _| point_mass(EARTH_GM, p));
        target = rk4(&target, 1.0, |p, _| point_mass(EARTH_GM, p));
    }
    let relative = LvlhFrame::from_target(&target)
        .unwrap()
        .relative_state(&chaser);
    assert!((relative.position_m - Vector3::new(0.0, -30.0, 0.0)).norm() < 0.5);
    assert!(relative.velocity_mps.norm() < 0.01);
}

#[test]
fn a_target_in_another_frame_is_refused() {
    let target = leo();
    let now = at_millis(1_000);
    let own = space_solution(
        FrameId::EARTH_J2000,
        &chaser_at(&target, Vector3::new(0.0, -250.0, 0.0)),
        now,
        0.5,
    );
    let other = space_solution(FrameId::MOON_J2000, &target, now, 0.5);
    let config = SpaceGuidanceConfig::default();
    assert_eq!(
        guide_rendezvous(&own, &other, &plan(300_000), now, CLOCK, &config),
        Err(GuidanceRefusal::FrameMismatch {
            expected: FrameId::EARTH_J2000,
            got: FrameId::MOON_J2000,
        })
    );
    let mut rotating = plan(300_000);
    rotating.frame = FrameId::EARTH_ITRF93;
    assert_eq!(
        guide_rendezvous(&own, &other, &rotating, now, CLOCK, &config),
        Err(GuidanceRefusal::TrajectoryInvalid {
            error: TrajectoryError::NonInertialFrame {
                frame: FrameId::EARTH_ITRF93
            }
        })
    );
}

#[test]
fn a_chaser_outside_the_cone_or_too_uncertain_is_refused() {
    let target = leo();
    let now = at_millis(1_000);
    let config = SpaceGuidanceConfig::default();
    let other = space_solution(FrameId::EARTH_J2000, &target, now, 0.5);
    let off_axis = chaser_at(&target, Vector3::new(80.0, -100.0, 0.0));
    let own = space_solution(FrameId::EARTH_J2000, &off_axis, now, 0.5);
    assert!(matches!(
        guide_rendezvous(&own, &other, &plan(300_000), now, CLOCK, &config),
        Err(GuidanceRefusal::CorridorViolated {
            violation: TrajectoryError::OutsideApproachCorridor { .. }
        })
    ));
    let on_axis = chaser_at(&target, Vector3::new(0.0, -100.0, 0.0));
    let own = space_solution(FrameId::EARTH_J2000, &on_axis, now, 9.0);
    assert!(matches!(
        guide_rendezvous(&own, &other, &plan(300_000), now, CLOCK, &config),
        Err(GuidanceRefusal::CorridorViolated {
            violation: TrajectoryError::UncertaintyExceedsCorridorMargin { .. }
        })
    ));
}
