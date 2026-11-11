//! A vehicle that is not at rest, and a magnetometer whose error turns
//! with the vehicle: what the rest tests let through must be inside what
//! the state declares.

use nalgebra::Vector3;

use super::rig::{Rig, at};
use super::{ATTITUDE_DEG, feed, normalized_errors, waiting};
use crate::{AlignmentNeed, AlignmentPhase, Navigator};

/// The velocity error of the state of a navigator aligned at `step`, each
/// axis as a multiple of its stated one-sigma.
fn velocity_sigmas(
    navigator: &mut crate::ReferenceNavigator,
    rig: &Rig,
    step: u64,
) -> Option<[f64; 3]> {
    let solution = navigator.solution(at(step))?;
    let v = solution.velocity;
    let error = Vector3::new(v.north_mps, v.east_mps, v.down_mps) - rig.velocity(step);
    let (n, e, d) = solution.velocity_cov.diagonal();
    Some([error.x / n.sqrt(), error.y / e.sqrt(), error.z / d.sqrt()])
}

/// A vehicle towed or drifting at a steady speed has the specific force
/// and the body rate of rest. Each speed either waits for rest or gives a
/// state whose velocity error is inside its stated three sigma; a speed
/// far past the receiver noise never aligns, and a vehicle at rest does.
#[test]
fn a_steady_drift_waits_or_aligns_within_its_stated_velocity() {
    let speeds = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.6, 1.0, 2.0];
    for (index, speed) in speeds.into_iter().enumerate() {
        let mut aligned = 0;
        for seed in 0..8_u64 {
            let mut rig = Rig::new(ATTITUDE_DEG, 1_000 * index as u64 + seed, 1.0);
            rig.velocity_ned = Vector3::new(speed * 0.6, speed * 0.8, 0.0);
            let mut navigator = rig.navigator();
            feed(&mut navigator, &mut rig, 1..=200, &mut Vec::new());
            match velocity_sigmas(&mut navigator, &rig, 200) {
                Some(sigmas) => {
                    aligned += 1;
                    assert!(
                        sigmas.iter().all(|s| s.abs() < 3.0),
                        "{speed} m/s, seed {seed}: {sigmas:?}"
                    );
                }
                None => assert_eq!(
                    waiting(&navigator),
                    (AlignmentPhase::Levelling, AlignmentNeed::Rest),
                    "{speed} m/s, seed {seed}"
                ),
            }
        }
        if speed == 0.0 {
            assert_eq!(aligned, 8, "every vehicle at rest aligns");
        }
        if speed >= 1.0 {
            assert_eq!(aligned, 0, "{speed} m/s is not rest");
        }
        // Below the gate of one fix of 0.1 m/s a moving vehicle aligns,
        // so the velocity it states is tested.
        if (0.1..=0.3).contains(&speed) {
            assert!(aligned >= 4, "{speed} m/s: {aligned} of 8 aligned");
        }
    }
}

/// A vehicle that starts to roll in the middle of a window, after the
/// only GNSS fix of the window said it was at rest: the inertial data of
/// the window are not those of rest, and no state comes from it.
#[test]
fn the_start_of_a_roll_is_not_rest() {
    let mut rig = Rig::new(ATTITUDE_DEG, 21, 1.0);
    rig.roll = Some((120, Vector3::new(0.3, 0.0, 0.0)));
    let mut navigator = rig.navigator();
    let mut sink = Vec::new();
    for step in 1..=200 {
        navigator.ingest(&rig.increment(step), &mut sink).unwrap();
        navigator.ingest(&rig.magnetic(step), &mut sink).unwrap();
        if step == 100 {
            navigator
                .ingest(&rig.gnss(step, true).0, &mut sink)
                .unwrap();
        }
    }
    assert!(navigator.solution(at(200)).is_none());
    assert_eq!(
        waiting(&navigator),
        (AlignmentPhase::Levelling, AlignmentNeed::Rest)
    );
}

/// Hard iron is a field fixed in the body: the heading error that it
/// makes changes sign when the vehicle turns half round, and it stays
/// inside the stated heading one-sigma. Near magnetic north a hard iron
/// across the body is across the field, and turns the heading by its
/// ratio to the horizontal field.
#[test]
fn a_body_fixed_hard_iron_turns_with_the_vehicle() {
    let heading_error = |heading_deg: f64| {
        let mut rig = Rig::new([2.0, -3.0, heading_deg], 31, 0.0);
        let sigma = rig.budgets.mag_disturbance_rad;
        rig.hard_iron_t = Vector3::new(0.0, 45e-6 * 0.544_639_035 * sigma, 0.0);
        let mut navigator = rig.navigator();
        feed(&mut navigator, &mut rig, 1..=200, &mut Vec::new());
        let solution = navigator.solution(at(200)).unwrap();
        let (_, _, _, heading) = normalized_errors(&rig, &solution);
        let (_, _, stated) = solution.attitude_cov.unwrap().diagonal();
        (heading, stated.sqrt(), sigma)
    };
    let (ahead, stated, sigma) = heading_error(10.0);
    let (behind, _, _) = heading_error(190.0);
    assert!((ahead.abs() - sigma).abs() < 0.3 * sigma, "{ahead} {sigma}");
    assert!((ahead + behind).abs() < 0.3 * sigma, "{ahead} {behind}");
    assert!(ahead.abs() < 3.0 * stated && behind.abs() < 3.0 * stated);
}
