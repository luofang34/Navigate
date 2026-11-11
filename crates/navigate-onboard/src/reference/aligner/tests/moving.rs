//! The alignment of a moving vehicle: a takeoff roll, a banked turn, and
//! a climb in wind. The navigator gets sensor data only. The truth of
//! [`super::path`] judges the result.

use nalgebra::{UnitQuaternion, Vector3};
use navigate_contract::{NavigationSolution, SymmetricCov3};

use super::path::{Manoeuvre, Profile, Truth, Vehicle, ramp};
use super::rig::{self, Rig, at};
use super::{matrix, waiting};
use crate::{
    Alignment, AlignmentConfig, AlignmentPhase, AlignmentStatus, HeadingSource, Navigator,
    PositionSource, ReferenceNavigator,
};

/// The crab allowance of a vehicle on its wheels, radians.
pub(super) const WHEELS_SIGMA_RAD: f64 = 0.02;
/// The crab allowance of the flights of the tests, radians: a cross wind
/// of 5 m/s at 55 m/s.
pub(super) const CRAB_SIGMA_RAD: f64 = 0.09;
/// The heading of the takeoff roll, radians.
pub(super) const RUNWAY_RAD: f64 = 2.4;

/// A navigator of the suite of the rig, without a magnetometer, whose
/// course is its heading to `crab_sigma`.
pub(super) fn navigator(crab_sigma: Option<f64>) -> ReferenceNavigator {
    let suite = Rig::new([0.0; 3], 0, 1.0).suite(None);
    let mut navigator = ReferenceNavigator::new(suite, rig::CLOCK, None).unwrap();
    navigator
        .set_alignment_config(AlignmentConfig {
            course_heading_sigma_rad: crab_sigma,
            ..AlignmentConfig::default()
        })
        .unwrap();
    navigator
}

/// A takeoff roll from rest to 2.5 m/s², with the rotation of the nose to
/// 8° from 12 s to 15 s.
fn takeoff(t: f64) -> Profile {
    let t = t.max(0.0);
    Profile {
        speed_m_s: 2.5 * (t - 0.5 * (1.0 - (-t / 0.5).exp())),
        heading_rad: RUNWAY_RAD,
        roll_rad: 0.0,
        pitch_rad: 0.02 + 0.12 * ramp(t, 12.0, 15.0),
        climb_m_s: 0.0,
    }
}

/// A level turn at 55 m/s with 20° of bank, which starts on `heading`.
pub(super) fn turn(heading: f64) -> impl Fn(f64) -> Profile {
    let roll: f64 = 20.0_f64.to_radians();
    move |t| Profile {
        speed_m_s: 55.0,
        heading_rad: heading + 9.806_65 * roll.tan() / 55.0 * t,
        roll_rad: roll,
        pitch_rad: 0.05,
        climb_m_s: 0.0,
    }
}

/// A climb on `heading` at 3 m/s that gains speed and lifts its nose.
fn climb(heading: f64) -> impl Fn(f64) -> Profile {
    move |t| Profile {
        speed_m_s: 45.0 + 0.8 * t,
        heading_rad: heading,
        roll_rad: 0.03,
        pitch_rad: 0.08 + 0.05 * ramp(t, 0.5, 2.5),
        climb_m_s: 3.0,
    }
}

/// The wind, north and east, that gives a vehicle at 55 m/s on `heading`
/// the crab angle `crab`.
pub(super) fn cross_wind(heading: f64, crab: f64) -> [f64; 2] {
    let w = 55.0 * crab.tan();
    [w * heading.sin(), -w * heading.cos()]
}

/// Feeds inertial steps, with a GNSS fix each second, until the navigator
/// has a state or `steps` are done. Gives the truth at the last step.
pub(super) fn fly<F: Fn(f64) -> Profile>(
    navigator: &mut ReferenceNavigator,
    vehicle: &mut Vehicle<F>,
    steps: u64,
) -> Truth {
    let mut sink = Vec::new();
    for _ in 0..steps {
        navigator.ingest(&vehicle.increment(), &mut sink).unwrap();
        if vehicle.step.is_multiple_of(100) {
            navigator.ingest(&vehicle.gnss(0), &mut sink).unwrap();
        }
        if matches!(
            navigator.alignment_status(),
            Some(AlignmentStatus::Aligned { .. })
        ) {
            break;
        }
    }
    vehicle.truth()
}

/// The rotation from the true attitude to that of a solution, about the
/// north, east, down axes, radians.
pub(super) fn attitude_error(truth: &Truth, solution: &NavigationSolution) -> Vector3<f64> {
    let q = solution.attitude.unwrap();
    let estimated = UnitQuaternion::new_normalize(nalgebra::Quaternion::new(q.w, q.x, q.y, q.z));
    (estimated * truth.body_to_ned.inverse()).scaled_axis()
}

/// The squared error `error`, normalized by the covariance `cov`.
pub(super) fn normalized(error: Vector3<f64>, cov: &SymmetricCov3) -> f64 {
    (error.transpose() * matrix(cov).try_inverse().unwrap() * error)[(0, 0)]
}

/// The inertial data of the truth agree with the mechanization of the
/// filter: a navigator that starts from the true state and gets only
/// inertial increments without error stays on the truth through a takeoff
/// roll with a rotation, and through a turn. The other tests of this file
/// stand on this one. A build without optimization flies 4 s of each.
#[test]
fn the_inertial_data_of_the_truth_keep_a_true_state_on_the_truth() {
    let steps: u64 = if cfg!(debug_assertions) { 400 } else { 3_000 };
    let paths: [Manoeuvre; 2] = [Box::new(takeoff), Box::new(turn(0.7))];
    for profile in paths {
        let mut vehicle = Vehicle::new(profile, [3.0, -4.0], 0, 0.0);
        let mut navigator = navigator(None);
        let start = vehicle.truth();
        let [lat, lon, h] = start.position;
        let q = start.body_to_ned.quaternion();
        let small =
            |sigma: f64| SymmetricCov3::from_diagonal(sigma * sigma, sigma * sigma, sigma * sigma);
        navigator
            .align_from_truth(&Alignment {
                position: navigate_contract::GeodeticPosition::new(lat, lon, h),
                velocity: navigate_contract::NedVelocity::new(
                    start.velocity.x,
                    start.velocity.y,
                    start.velocity.z,
                ),
                attitude: navigate_contract::AttitudeQuaternion::new(q.w, q.i, q.j, q.k),
                position_cov: small(0.01),
                velocity_cov: small(0.001),
                attitude_cov: small(1e-5),
                at: at(0),
            })
            .unwrap();
        let mut sink = Vec::new();
        for _ in 0..steps {
            navigator.ingest(&vehicle.increment(), &mut sink).unwrap();
        }
        let solution = navigator.solution(at(steps)).unwrap();
        let truth = vehicle.truth();
        let position = truth.position_error_m(&solution.position).norm();
        let attitude = attitude_error(&truth, &solution).norm();
        tracing::info!(steps, position, attitude, "inertial only from the truth");
        assert!(position < 0.5, "{position} m after {steps} steps");
        assert!(attitude < 1e-4, "{attitude} rad after {steps} steps");
    }
}

/// With sensors without error the state of each manoeuvre is the truth:
/// the roll in a turn, the pitch in a takeoff roll and in a climb with a
/// rotation of the nose. A solve that takes the vehicle as not
/// accelerated is wrong by 14° in the takeoff roll and by 20° in the
/// turn.
#[test]
fn sensors_without_error_give_the_true_attitude_in_each_manoeuvre() {
    rig::show_events();
    let paths: [(&str, Manoeuvre); 3] = [
        ("takeoff roll", Box::new(takeoff)),
        ("turn", Box::new(turn(-2.0))),
        ("climb", Box::new(climb(1.1))),
    ];
    for (name, profile) in paths {
        let mut vehicle = Vehicle::new(profile, [0.0, 0.0], 0, 0.0);
        let mut navigator = navigator(Some(WHEELS_SIGMA_RAD));
        let truth = fly(&mut navigator, &mut vehicle, 1_000);
        let Some(AlignmentStatus::Aligned { record, .. }) = navigator.alignment_status() else {
            panic!("{name}: {:?}", navigator.alignment_status());
        };
        assert_eq!(record.at, at(vehicle.step), "{name}");
        assert_eq!(record.heading, Some(HeadingSource::GnssCourse), "{name}");
        assert_eq!(record.position, Some(PositionSource::Gnss), "{name}");
        assert_eq!(record.gyro_bias_rad_s, None, "{name}");
        let solution = navigator.solution(at(vehicle.step)).unwrap();
        let error = attitude_error(&truth, &solution);
        let position = truth.position_error_m(&solution.position).norm();
        tracing::info!(name, step = vehicle.step, ?error, position, "without error");
        assert!(error.norm() < 2e-3, "{name}: {error:?} rad");
        assert!(position < 1e-6, "{name}: {position} m");
    }
}

/// A fix that reaches the navigator 150 ms after the receiver took it
/// gives the state of the instant of the fix, and the increments since
/// then carry it to the present. Without them the position is late by
/// 8 m at 55 m/s.
#[test]
fn a_late_fix_gives_a_state_that_the_later_increments_carry_to_the_present() {
    let mut vehicle = Vehicle::new(turn(0.3), [0.0, 0.0], 0, 0.0);
    let mut navigator = navigator(Some(CRAB_SIGMA_RAD));
    let mut sink = Vec::new();
    let mut late = None;
    for _ in 0..215 {
        navigator.ingest(&vehicle.increment(), &mut sink).unwrap();
        if vehicle.step.is_multiple_of(100) {
            late = Some((vehicle.gnss(15), vehicle.step + 15));
        }
        if let Some((fix, _)) = late.take_if(|(_, due)| *due == vehicle.step) {
            navigator.ingest(&fix, &mut sink).unwrap();
        }
    }
    let Some(AlignmentStatus::Aligned { record, .. }) = navigator.alignment_status() else {
        panic!("{:?}", navigator.alignment_status());
    };
    assert_eq!(record.at, at(200));
    let solution = navigator.solution(at(215)).unwrap();
    let truth = vehicle.truth();
    let position = truth.position_error_m(&solution.position).norm();
    let attitude = attitude_error(&truth, &solution).norm();
    assert!(position < 0.05, "{position} m");
    assert!(attitude < 2e-3, "{attitude} rad");
}

/// A receiver and an inertial unit do not share a clock: with increments
/// of 30 ms each fix is inside an increment. That increment counts in
/// the span by its share of the time. The vehicle rolls at 0.5 rad/s at
/// the fix: a sum that ends at the last whole increment gives the
/// attitude of 20 ms before, which is wrong by 10 mrad.
#[test]
fn a_fix_inside_an_increment_divides_it() {
    let rolling = |t: f64| Profile {
        speed_m_s: 55.0,
        heading_rad: 0.3 + 0.05 * t,
        roll_rad: 0.35 * (1.5 * t).sin(),
        pitch_rad: 0.05,
        climb_m_s: 0.0,
    };
    // One vehicle gives the increments, and a second on the same path
    // gives the fixes and the truth at their own steps.
    let mut unit = Vehicle::new(rolling, [0.0, 0.0], 0, 0.0);
    let mut receiver = Vehicle::new(rolling, [0.0, 0.0], 0, 0.0);
    let mut navigator = navigator(Some(CRAB_SIGMA_RAD));
    let mut sink = Vec::new();
    let mut record = None;
    while record.is_none() && unit.step < 300 {
        navigator
            .ingest(&unit.increment_over(3), &mut sink)
            .unwrap();
        while receiver.step < unit.step {
            receiver.increment();
            if receiver.step.is_multiple_of(100) {
                let mut fix = receiver.gnss(0);
                fix.delivered_at = at(unit.step);
                navigator.ingest(&fix, &mut sink).unwrap();
            }
        }
        if let Some(AlignmentStatus::Aligned { record: r, .. }) = navigator.alignment_status() {
            record = Some(r);
        }
    }
    // The state is for the fix at step 200, and the present is the end
    // of the increment that holds that fix.
    assert_eq!(record.map(|r| r.at), Some(at(200)));
    assert_eq!(unit.step, 201);
    let solution = navigator.solution(at(unit.step)).unwrap();
    let truth = receiver.truth();
    let position = truth.position_error_m(&solution.position).norm();
    let attitude = attitude_error(&truth, &solution).norm();
    assert!(position < 0.05, "{position} m");
    assert!(attitude < 1e-3, "{attitude} rad");
}

/// The state of an alignment in motion is as good as it states, when the
/// crab angle of each flight comes from the declared allowance: turns and
/// climbs on each heading, with the errors of the sensors.
#[test]
fn the_state_of_a_moving_vehicle_is_as_good_as_it_states() {
    rig::show_events();
    let (attitude, position) = mean_normalized_errors(1.0);
    tracing::info!(
        attitude,
        position,
        "96 alignments in motion, three degrees of freedom each"
    );
    // The band of the alignment at rest: 3.3 sigma of a chi-square sum.
    assert!((2.18..=3.82).contains(&attitude), "{attitude}");
    assert!((2.18..=3.82).contains(&position), "{position}");
}

/// A crab angle of three times the declared allowance makes a heading
/// that is worse than the state declares: the normalized error shows it.
#[test]
fn a_crab_angle_larger_than_declared_shows_in_the_normalized_error() {
    let (attitude, _) = mean_normalized_errors(3.0);
    assert!(attitude > 3.82, "{attitude}");
}

/// The mean normalized attitude and position errors of 96 alignments in
/// motion. The crab angle of each is a draw of the declared allowance
/// times `crab_scale`.
fn mean_normalized_errors(crab_scale: f64) -> (f64, f64) {
    const SEEDS: u64 = 96;
    let (mut attitude, mut position) = (0.0, 0.0);
    for seed in 0..SEEDS {
        let heading = 0.4 * seed as f64;
        let crab = rig::Noise::new(seed).normal() * CRAB_SIGMA_RAD * crab_scale;
        // The turn goes 0.065 rad each second: it is on `heading` plus
        // 0.13 rad at the fix that aligns, 2 s after the start.
        let (profile, wind): (Manoeuvre, [f64; 2]) = if seed % 2 == 0 {
            (Box::new(turn(heading)), cross_wind(heading + 0.13, crab))
        } else {
            (Box::new(climb(heading)), cross_wind(heading, crab))
        };
        let mut vehicle = Vehicle::new(profile, wind, seed, 1.0);
        let mut navigator = navigator(Some(CRAB_SIGMA_RAD));
        let truth = fly(&mut navigator, &mut vehicle, 1_000);
        let solution = navigator.solution(at(vehicle.step)).unwrap();
        attitude += normalized(
            attitude_error(&truth, &solution),
            &solution.attitude_cov.unwrap(),
        );
        position += normalized(
            truth.position_error_m(&solution.position),
            &solution.position_cov,
        );
    }
    (attitude / SEEDS as f64, position / SEEDS as f64)
}

/// A takeoff roll with the errors of the sensors: no state from the first
/// fix alone, then the state at the second, from the change of velocity
/// between the two. The
/// pitch, which a solve without the acceleration has wrong by 14°, and
/// the heading are inside what the state declares.
#[test]
fn a_takeoff_roll_aligns_at_its_second_fix() {
    const SEEDS: u64 = 24;
    let mut nees = 0.0;
    for seed in 0..SEEDS {
        let mut vehicle = Vehicle::new(takeoff, [0.0, 0.0], seed, 1.0);
        let mut navigator = navigator(Some(WHEELS_SIGMA_RAD));
        fly(&mut navigator, &mut vehicle, 150);
        assert_eq!(
            waiting(&navigator),
            (
                AlignmentPhase::Levelling,
                crate::AlignmentNeed::VelocityFixes
            )
        );
        let truth = fly(&mut navigator, &mut vehicle, 1_000);
        assert_eq!(vehicle.step, 200);
        let solution = navigator.solution(at(vehicle.step)).unwrap();
        let error = attitude_error(&truth, &solution);
        let Some(AlignmentStatus::Aligned { record, .. }) = navigator.alignment_status() else {
            panic!("{:?}", navigator.alignment_status());
        };
        let tilt = error.xy().norm() / record.tilt_sigma_rad;
        let heading = error.z.abs() / record.heading_sigma_rad;
        assert!(tilt < 4.0 && heading < 4.0, "seed {seed}: {tilt} {heading}");
        nees += normalized(error, &solution.attitude_cov.unwrap());
    }
    // The wheels hold the crab angle at zero, so the state can be better
    // than it declares, and not worse.
    let nees = nees / SEEDS as f64;
    assert!(nees < 3.82, "{nees}");
}

/// With the course taken as a heading, a vehicle at rest still aligns by
/// its rest window: the fixes near rest are the rest window's evidence,
/// and the magnetometer gives the heading.
#[test]
fn a_vehicle_at_rest_aligns_by_rest_when_the_course_is_a_heading() {
    let mut rig = Rig::new([3.0, -4.0, 137.0], 83, 1.0);
    let mut navigator = rig.navigator();
    navigator
        .set_alignment_config(AlignmentConfig {
            course_heading_sigma_rad: Some(CRAB_SIGMA_RAD),
            ..AlignmentConfig::default()
        })
        .unwrap();
    super::feed(&mut navigator, &mut rig, 1..=200, &mut Vec::new());
    let Some(AlignmentStatus::Aligned { record, .. }) = navigator.alignment_status() else {
        panic!("aligned at rest: {:?}", navigator.alignment_status());
    };
    assert_eq!(record.heading, Some(HeadingSource::Magnetometer));
    assert_eq!(record.at, at(200));
}
