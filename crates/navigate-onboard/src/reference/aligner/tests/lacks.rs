//! What an alignment in motion lacks, by name, and the data that it
//! refuses.

use nalgebra::Vector3;
use navigate_contract::{NedVelocity, SymmetricCov3};

use super::moving::{CRAB_SIGMA_RAD, RUNWAY_RAD, WHEELS_SIGMA_RAD, fly, navigator, turn};
use super::path::{Profile, Vehicle};
use super::rig::{self, Rig, at};
use super::waiting;
use crate::{
    AlignmentConfig, AlignmentPhase, AlignmentStatus, ErrorBudget, Navigator, ReferenceNavigator,
    SensorData, SensorInput, SensorKind, SensorSpec, SensorSuite,
};

/// What a moving vehicle lacks has a name, and no state comes without it.
#[test]
fn an_alignment_in_motion_names_what_it_lacks() {
    // The default configuration does not take a course as a heading: a
    // moving vehicle waits for rest, as a multirotor must.
    let mut vehicle = Vehicle::new(turn(0.0), [0.0, 0.0], 1, 1.0);
    let mut navigator = navigator(None);
    fly(&mut navigator, &mut vehicle, 400);
    assert_eq!(
        waiting(&navigator),
        (AlignmentPhase::Levelling, crate::AlignmentNeed::Rest)
    );
    // A walk at 0.8 m/s is faster than the rest gate of a receiver of
    // 0.1 m/s, and too slow for its course to be known to a heading limit
    // of 0.05 rad.
    let taxi = |_: f64| Profile {
        speed_m_s: 0.8,
        heading_rad: RUNWAY_RAD,
        roll_rad: 0.0,
        pitch_rad: 0.0,
        climb_m_s: 0.0,
    };
    let mut vehicle = Vehicle::new(taxi, [0.0, 0.0], 2, 1.0);
    let mut navigator = self::navigator(Some(WHEELS_SIGMA_RAD));
    navigator
        .set_alignment_config(AlignmentConfig {
            course_heading_sigma_rad: Some(WHEELS_SIGMA_RAD),
            max_heading_sigma_rad: 0.05,
            ..AlignmentConfig::default()
        })
        .unwrap();
    fly(&mut navigator, &mut vehicle, 400);
    assert_eq!(
        waiting(&navigator),
        (AlignmentPhase::Heading, crate::AlignmentNeed::CourseSpeed)
    );
    // One fix is not a change of velocity.
    let mut vehicle = Vehicle::new(turn(0.0), [0.0, 0.0], 3, 1.0);
    let mut navigator = self::navigator(Some(CRAB_SIGMA_RAD));
    fly(&mut navigator, &mut vehicle, 150);
    assert_eq!(
        waiting(&navigator),
        (
            AlignmentPhase::Levelling,
            crate::AlignmentNeed::VelocityFixes
        )
    );
    // An allowance for the crab angle above the limit of the heading.
    let mut navigator = self::navigator(Some(0.3));
    fly(&mut navigator, &mut vehicle, 400);
    assert_eq!(
        waiting(&navigator),
        (
            AlignmentPhase::Heading,
            crate::AlignmentNeed::HeadingUncertain
        )
    );
    assert!(navigator.solution(at(vehicle.step)).is_none());
}

/// Whether the navigator has a state.
fn aligned(navigator: &ReferenceNavigator) -> bool {
    matches!(
        navigator.alignment_status(),
        Some(AlignmentStatus::Aligned { .. })
    )
}

/// Changes the velocity of a fix by `change`, north, east, down, m/s.
fn with_velocity_change(mut fix: SensorInput, change: Vector3<f64>) -> SensorInput {
    if let SensorData::GnssFix {
        velocity: Some(v), ..
    } = &mut fix.data
    {
        *v = NedVelocity::new(
            v.north_mps + change.x,
            v.east_mps + change.y,
            v.down_mps + change.z,
        );
    }
    fix
}

/// A velocity that the inertial data do not agree with gives no state:
/// a fix whose down velocity is wrong by 1.5 m/s makes a specific force
/// that is 1.5 m/s² from that of the accelerometers in the span that ends
/// at it. The next fix aligns from the span that skips it.
#[test]
fn a_velocity_that_the_inertial_data_do_not_agree_with_gives_no_state() {
    let mut vehicle = Vehicle::new(turn(0.5), [0.0, 0.0], 4, 1.0);
    let mut navigator = navigator(Some(CRAB_SIGMA_RAD));
    let mut sink = Vec::new();
    while vehicle.step < 300 {
        navigator.ingest(&vehicle.increment(), &mut sink).unwrap();
        if !vehicle.step.is_multiple_of(100) {
            continue;
        }
        let wrong = if vehicle.step == 200 { 1.5 } else { 0.0 };
        let fix = with_velocity_change(vehicle.gnss(0), Vector3::new(0.0, 0.0, wrong));
        navigator.ingest(&fix, &mut sink).unwrap();
        if vehicle.step < 300 {
            assert!(!aligned(&navigator), "step {}", vehicle.step);
        }
        if vehicle.step == 200 {
            assert_eq!(
                waiting(&navigator),
                (
                    AlignmentPhase::Levelling,
                    crate::AlignmentNeed::ForceDisagreement
                )
            );
        }
    }
    assert!(aligned(&navigator));
}

/// Inertial increments that the navigator did not get between two fixes
/// give no state from that span. The next span aligns.
#[test]
fn lost_increments_between_two_fixes_give_no_state() {
    let mut vehicle = Vehicle::new(turn(0.5), [0.0, 0.0], 5, 1.0);
    let mut navigator = navigator(Some(CRAB_SIGMA_RAD));
    let mut sink = Vec::new();
    while vehicle.step < 300 {
        let increment = vehicle.increment();
        if !(150..160).contains(&vehicle.step) {
            navigator.ingest(&increment, &mut sink).unwrap();
        }
        if vehicle.step.is_multiple_of(100) {
            navigator.ingest(&vehicle.gnss(0), &mut sink).unwrap();
        }
        if vehicle.step == 200 {
            assert_eq!(
                waiting(&navigator),
                (AlignmentPhase::Levelling, crate::AlignmentNeed::InertialGap)
            );
        }
    }
    assert!(aligned(&navigator));
}

/// A receiver without a declared budget gives the errors in each fix. A
/// moving fix without a position covariance is no position, and a
/// velocity without a covariance has no known accuracy.
#[test]
fn a_fix_of_a_receiver_without_a_budget_states_its_errors() {
    let declared = Rig::new([0.0; 3], 0, 1.0).suite(None);
    let of = |kind| declared.of_kind(kind).next().unwrap().clone();
    let suite = SensorSuite::new(vec![
        of(SensorKind::Imu),
        SensorSpec {
            budget: ErrorBudget::Declared,
            ..of(SensorKind::Gnss)
        },
    ])
    .unwrap();
    let mut navigator = ReferenceNavigator::new(suite, rig::CLOCK, None).unwrap();
    navigator
        .set_alignment_config(AlignmentConfig {
            course_heading_sigma_rad: Some(CRAB_SIGMA_RAD),
            ..AlignmentConfig::default()
        })
        .unwrap();
    let mut vehicle = Vehicle::new(turn(0.5), [0.0, 0.0], 6, 1.0);
    let mut sink = Vec::new();
    let covariances = |step: u64| match step {
        100 => (None, None),
        200 => (Some(SymmetricCov3::from_diagonal(2.25, 2.25, 9.0)), None),
        _ => (
            Some(SymmetricCov3::from_diagonal(2.25, 2.25, 9.0)),
            Some(SymmetricCov3::from_diagonal(0.01, 0.01, 0.01)),
        ),
    };
    while vehicle.step < 400 {
        navigator.ingest(&vehicle.increment(), &mut sink).unwrap();
        if !vehicle.step.is_multiple_of(100) {
            continue;
        }
        let mut fix = vehicle.gnss(0);
        if let SensorData::GnssFix {
            position_cov,
            velocity_cov,
            ..
        } = &mut fix.data
        {
            (*position_cov, *velocity_cov) = covariances(vehicle.step);
        }
        navigator.ingest(&fix, &mut sink).unwrap();
        match vehicle.step {
            100 => assert_eq!(
                waiting(&navigator),
                (
                    AlignmentPhase::Position,
                    crate::AlignmentNeed::PositionSource
                )
            ),
            200 => assert_eq!(
                waiting(&navigator),
                (
                    AlignmentPhase::Levelling,
                    crate::AlignmentNeed::VelocityAccuracy
                )
            ),
            // The fix at 300 is the first with a known velocity error.
            300 => assert_eq!(
                waiting(&navigator),
                (
                    AlignmentPhase::Levelling,
                    crate::AlignmentNeed::VelocityFixes
                )
            ),
            _ => assert!(aligned(&navigator)),
        }
    }
}
