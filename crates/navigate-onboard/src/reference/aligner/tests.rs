//! The alignment from sensors only: no test gives the navigator a true
//! value. The truth is used to judge the result.

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use navigate_contract::{NavigationSolution, SymmetricCov3};

use super::names;
use crate::{
    AcceptanceRecord, AlignmentNeed, AlignmentPhase, AlignmentSource, AlignmentStatus,
    HeadingSource, Navigator, Outcome, PositionSource, ReferenceNavigator,
};

mod calibration;
mod declared;
mod envelope;
mod forward;
mod inputs;
mod lacks;
mod motion;
mod moving;
mod path;
mod rig;
mod window;
use rig::{GNSS, HEIGHT_M, LAT_DEG, LON_DEG, MAGNETOMETER, Rig, at};

/// Roll, pitch, and heading of the vehicle of the tests, degrees: not
/// level and not to the north, so that no sign is hidden.
const ATTITUDE_DEG: [f64; 3] = [3.0, -4.0, 137.0];

#[cfg(test)]
fn matrix(c: &SymmetricCov3) -> Matrix3<f64> {
    let [xx, xy, xz, yy, yz, zz] = c.upper_triangle();
    Matrix3::new(xx, xy, xz, xy, yy, yz, xz, yz, zz)
}

/// Feeds the steps `steps`: each inertial increment, the magnetometer at
/// 50 Hz, and a GNSS fix with a velocity each second.
#[cfg(test)]
fn feed(
    navigator: &mut ReferenceNavigator,
    rig: &mut Rig,
    steps: core::ops::RangeInclusive<u64>,
    sink: &mut Vec<AcceptanceRecord>,
) {
    for step in steps {
        navigator.ingest(&rig.increment(step), sink).unwrap();
        if step.is_multiple_of(2) {
            navigator.ingest(&rig.magnetic(step), sink).unwrap();
        }
        if step.is_multiple_of(100) {
            navigator.ingest(&rig.gnss(step, true).0, sink).unwrap();
        }
    }
}

/// The squared error of the attitude and of the position of a solution,
/// each normalized by the covariance that the solution states (three
/// degrees of freedom each), and the tilt and heading errors in radians.
#[cfg(test)]
fn normalized_errors(rig: &Rig, solution: &NavigationSolution) -> (f64, f64, f64, f64) {
    let q = solution.attitude.unwrap();
    let estimated = UnitQuaternion::new_normalize(nalgebra::Quaternion::new(q.w, q.x, q.y, q.z));
    // The rotation from the true attitude to the estimate, about the
    // north, east, down axes.
    let error = (estimated * rig.body_to_ned.inverse()).scaled_axis();
    let attitude_cov = matrix(&solution.attitude_cov.unwrap());
    let attitude = (error.transpose() * attitude_cov.try_inverse().unwrap() * error)[(0, 0)];
    let lat = LAT_DEG.to_radians();
    let position_error = Vector3::new(
        (solution.position.latitude_rad - lat) * 6_364_000.0,
        (solution.position.longitude_rad - LON_DEG.to_radians()) * 6_378_137.0 * lat.cos(),
        HEIGHT_M - solution.position.altitude_m,
    );
    let position_cov = matrix(&solution.position_cov);
    let position =
        (position_error.transpose() * position_cov.try_inverse().unwrap() * position_error)[(0, 0)];
    (attitude, position, error.xy().norm(), error.z)
}

#[test]
fn a_vehicle_at_rest_aligns_itself_from_its_sensors() {
    let mut rig = Rig::new(ATTITUDE_DEG, 7, 1.0);
    let mut navigator = rig.navigator();
    let mut sink = Vec::new();
    assert_eq!(
        navigator.alignment_status(),
        Some(AlignmentStatus::Aligning {
            phase: AlignmentPhase::Levelling,
            waiting_for: AlignmentNeed::InertialIncrements
        })
    );
    feed(&mut navigator, &mut rig, 1..=150, &mut sink);
    assert!(
        navigator.solution(at(150)).is_none(),
        "no state in the first window"
    );
    assert_eq!(
        navigator.alignment_status(),
        Some(AlignmentStatus::Aligning {
            phase: AlignmentPhase::Levelling,
            waiting_for: AlignmentNeed::RestWindow
        })
    );
    // The inputs before the state are recorded as taken by the alignment.
    assert!(
        sink.iter()
            .all(|r| r.outcome == Outcome::Unused(names::ALIGNMENT))
    );
    assert!(
        sink.iter()
            .any(|r| r.sensor == GNSS && r.measurement == "gnss_position")
    );
    assert!(sink.iter().any(|r| r.sensor == MAGNETOMETER));
    feed(&mut navigator, &mut rig, 151..=200, &mut sink);
    let Some(AlignmentStatus::Aligned { record, suspect }) = navigator.alignment_status() else {
        panic!(
            "aligned after one rest window: {:?}",
            navigator.alignment_status()
        );
    };
    assert!(!suspect);
    assert_eq!(record.at, at(200));
    assert_eq!(record.source, AlignmentSource::Sensors);
    assert_eq!(record.heading, Some(HeadingSource::Magnetometer));
    assert_eq!(record.position, Some(PositionSource::Gnss));
    // 2 mg of bias is 2 mrad of tilt; the heading holds the disturbance,
    // the declination, and the tilt through the dip of the field.
    assert!((record.tilt_sigma_rad - 0.002).abs() < 2e-4, "{record:?}");
    assert!(
        (record.heading_sigma_rad - 0.0363).abs() < 1e-3,
        "{record:?}"
    );
    let solution = navigator.solution(at(200)).unwrap();
    let (_, _, tilt, heading) = normalized_errors(&rig, &solution);
    assert!(tilt < 3.0 * 1.5 * record.tilt_sigma_rad, "{tilt}");
    assert!(heading.abs() < 3.0 * record.heading_sigma_rad, "{heading}");
    // The mean rate of the window less the Earth rate is the gyroscope
    // bias, to the noise of the mean (0.07 mrad/s).
    let bias = Vector3::from(record.gyro_bias_rad_s.unwrap());
    assert!(
        (bias - rig.gyro_bias).norm() < 4e-4,
        "{bias} {}",
        rig.gyro_bias
    );
    // After the state, the magnetometer is recorded and not fused.
    let before = sink.len();
    navigator.ingest(&rig.magnetic(202), &mut sink).unwrap();
    assert_eq!(sink[before].outcome, Outcome::Unused(names::ALIGNMENT_ONLY));
}

/// Over many vehicles whose sensor errors come from the declared budgets,
/// the error of the aligned state has the size that the state declares.
#[test]
fn the_aligned_state_is_as_good_as_it_states() {
    rig::show_events();
    const SEEDS: u64 = 96;
    let (mut attitude, mut position) = (0.0, 0.0);
    for seed in 0..SEEDS {
        let mut rig = Rig::new(
            [(seed % 9) as f64 - 4.0, -5.0, 15.0 * seed as f64],
            seed,
            1.0,
        );
        let mut navigator = rig.navigator();
        feed(&mut navigator, &mut rig, 1..=200, &mut Vec::new());
        let (a, p, _, _) = normalized_errors(&rig, &navigator.solution(at(200)).unwrap());
        attitude += a;
        position += p;
    }
    let (attitude, position) = (attitude / SEEDS as f64, position / SEEDS as f64);
    tracing::info!(
        attitude_nees = attitude,
        position_nees = position,
        "96 alignments, three degrees of freedom each"
    );
    // The sum of 96 values of three degrees of freedom is chi-square with
    // 288, which has the mean 288 and the one-sigma 24. A band of 3.3
    // sigma, which an exact covariance leaves once in a thousand sets of
    // seeds, is 209 to 367: a mean between 2.18 and 3.82.
    assert!((2.18..=3.82).contains(&attitude), "{attitude}");
    assert!((2.18..=3.82).contains(&position), "{position}");
}

/// The aligned state after 60 s at rest with a GNSS fix each second. A
/// levelled tilt and the accelerometer bias are one error, and the filter
/// takes them as two: this test guards that the filter, working on them,
/// does not turn that into an attitude more certain than it is. A step of
/// the filter is slow without optimization, so the test runs in an
/// optimized build only (`ci.sh` runs it).
#[test]
#[cfg_attr(
    debug_assertions,
    ignore = "needs an optimized build: cargo test --release -p navigate-onboard --lib aligner"
)]
fn the_aligned_state_stays_as_good_as_it_states_at_rest() {
    rig::show_events();
    const SEEDS: u64 = 24;
    const STEPS: u64 = 6_000;
    let mut attitude = 0.0;
    for seed in 0..SEEDS {
        let mut rig = Rig::new(
            [(seed % 9) as f64 - 4.0, -5.0, 15.0 * seed as f64],
            seed,
            1.0,
        );
        let mut navigator = rig.navigator();
        let mut sink = Vec::new();
        feed(&mut navigator, &mut rig, 1..=200 + STEPS, &mut sink);
        let solution = navigator.solution(at(200 + STEPS)).unwrap();
        attitude += normalized_errors(&rig, &solution).0;
        let fused = sink
            .iter()
            .filter(|r| r.measurement == "gnss_position" && r.outcome == Outcome::Accepted)
            .count();
        assert!(fused as u64 >= STEPS / 100 - 1, "{fused} fixes fused");
    }
    let attitude = attitude / SEEDS as f64;
    tracing::info!(
        seeds = SEEDS,
        seconds = STEPS / 100,
        attitude_nees = attitude,
        "at rest after the alignment"
    );
    // The sum of 24 values of three degrees of freedom is chi-square with
    // 72, of mean 72 and one-sigma 12. A band of 3.3 sigma is 32 to 112:
    // a mean between 1.35 and 4.65.
    assert!((1.35..=4.65).contains(&attitude), "{attitude}");
}

/// A declination that is wrong by ten times its declared one-sigma makes
/// a heading that is worse than the state declares: the normalized error
/// shows it.
#[test]
fn a_declination_worse_than_declared_shows_in_the_normalized_error() {
    const SEEDS: u64 = 24;
    let mut attitude = 0.0;
    for seed in 0..SEEDS {
        let mut rig = Rig::new(ATTITUDE_DEG, seed, 10.0);
        let mut navigator = rig.navigator();
        feed(&mut navigator, &mut rig, 1..=200, &mut Vec::new());
        attitude += normalized_errors(&rig, &navigator.solution(at(200)).unwrap()).0;
    }
    let mean = attitude / SEEDS as f64;
    assert!(mean > 3.82, "{mean}");
}

#[cfg(test)]
fn waiting(navigator: &ReferenceNavigator) -> (AlignmentPhase, AlignmentNeed) {
    match navigator.alignment_status() {
        Some(AlignmentStatus::Aligning { phase, waiting_for }) => (phase, waiting_for),
        other => panic!("not aligned: {other:?}"),
    }
}

#[test]
fn an_alignment_names_what_it_lacks_and_gives_no_state_without_it() {
    let mut sink = Vec::new();
    // A receiver that gives no velocity is no evidence of rest.
    let mut rig = Rig::new(ATTITUDE_DEG, 1, 1.0);
    let mut navigator = rig.navigator();
    for step in 1..=200 {
        navigator.ingest(&rig.increment(step), &mut sink).unwrap();
        navigator.ingest(&rig.magnetic(step), &mut sink).unwrap();
        if step == 100 {
            navigator
                .ingest(&rig.gnss(step, false).0, &mut sink)
                .unwrap();
        }
    }
    assert_eq!(
        waiting(&navigator),
        (AlignmentPhase::Levelling, AlignmentNeed::RestEvidence)
    );
    // No magnetometer in the suite: no heading source at rest.
    let suite = rig.suite(None);
    let mut navigator = ReferenceNavigator::new(suite, rig::CLOCK, None).unwrap();
    for step in 1..=200 {
        navigator.ingest(&rig.increment(step), &mut sink).unwrap();
        if step == 100 {
            navigator
                .ingest(&rig.gnss(step, true).0, &mut sink)
                .unwrap();
        }
    }
    assert_eq!(
        waiting(&navigator),
        (AlignmentPhase::Heading, AlignmentNeed::HeadingSource)
    );
    // A magnetometer with 17° of body disturbance is not a heading.
    let mut navigator =
        rig.told(ReferenceNavigator::new(rig.suite(Some(0.3)), rig::CLOCK, None).unwrap());
    feed(&mut navigator, &mut rig, 1..=200, &mut sink);
    assert_eq!(
        waiting(&navigator),
        (AlignmentPhase::Heading, AlignmentNeed::HeadingUncertain)
    );
    assert!(navigator.solution(at(200)).is_none());
}

#[test]
fn a_surveyed_point_aligns_a_vehicle_without_gnss() {
    let mut rig = Rig::new(ATTITUDE_DEG, 3, 1.0);
    let mut navigator = rig.navigator();
    let mut sink = Vec::new();
    for step in 1..=200 {
        navigator.ingest(&rig.increment(step), &mut sink).unwrap();
        navigator.ingest(&rig.magnetic(step), &mut sink).unwrap();
        if step.is_multiple_of(50) {
            navigator.ingest(&Rig::surveyed(step), &mut sink).unwrap();
        }
    }
    let Some(AlignmentStatus::Aligned { record, .. }) = navigator.alignment_status() else {
        panic!("aligned from the surveyed point");
    };
    assert_eq!(record.position, Some(PositionSource::SurveyedPoint));
    let solution = navigator.solution(at(200)).unwrap();
    assert!((solution.position.altitude_m - HEIGHT_M).abs() < 1e-9);
    let (_, _, sigma_d) = solution.position_cov.diagonal();
    assert!((sigma_d.sqrt() - 0.5).abs() < 1e-9);
    // No sensor measures the velocity on the point: its one-sigma is what
    // a mean specific force at the gate of the noise makes over the
    // window, 2 mm/s/√Hz over 2 s, spread to the gate of four sigma.
    let (north, _, _) = solution.velocity_cov.diagonal();
    let expected = 2e-3 * 2.0_f64.sqrt() * (1.0 + 16.0 / 3.0_f64).sqrt();
    assert!((north.sqrt() - expected).abs() < 1e-6, "{}", north.sqrt());
    let v = solution.velocity;
    assert!(v.north_mps.hypot(v.east_mps).hypot(v.down_mps) < 3.0 * expected);
}

#[test]
fn a_transfer_alignment_and_a_realignment_replace_the_state() {
    let mut rig = Rig::new(ATTITUDE_DEG, 5, 1.0);
    let mut navigator = rig.navigator();
    let mut sink = Vec::new();
    feed(&mut navigator, &mut rig, 1..=200, &mut sink);
    let own = navigator.solution(at(200)).unwrap();
    // The host gives the state of another estimator: it replaces the
    // state that the navigator found.
    let transfer = crate::Alignment {
        position: own.position,
        velocity: own.velocity,
        attitude: navigate_contract::AttitudeQuaternion::new(1.0, 0.0, 0.0, 0.0),
        position_cov: SymmetricCov3::from_diagonal(4.0, 4.0, 9.0),
        velocity_cov: SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
        attitude_cov: SymmetricCov3::from_diagonal(1e-4, 1e-4, 4e-4),
        at: at(200),
        imu_bias_prior: None,
    };
    navigator.align(&transfer).unwrap();
    let Some(AlignmentStatus::Aligned { record, .. }) = navigator.alignment_status() else {
        panic!("aligned by transfer");
    };
    assert_eq!(record.source, AlignmentSource::Transfer);
    assert_eq!((record.heading, record.position), (None, None));
    assert!((record.tilt_sigma_rad - 0.01).abs() < 1e-12);
    assert!((record.heading_sigma_rad - 0.02).abs() < 1e-12);
    // An evaluation can give the true state through its own entry, and
    // the record says so.
    navigator.align_from_truth(&transfer).unwrap();
    assert!(matches!(
        navigator.alignment_status(),
        Some(AlignmentStatus::Aligned { record, .. }) if record.source == AlignmentSource::Truth
    ));
    navigator.align(&transfer).unwrap();
    // Five positions in a row 400 m away are refused at the gate: the
    // alignment is suspect, and the navigator does not reset itself.
    for step in 201..=205 {
        navigator.ingest(&rig.increment(step), &mut sink).unwrap();
        let mut far = rig.gnss(step, true).0;
        if let crate::SensorData::GnssFix { position, .. } = &mut far.data {
            position.latitude_rad += 400.0 / 6_364_000.0;
        }
        navigator.ingest(&far, &mut sink).unwrap();
    }
    assert!(matches!(
        navigator.alignment_status(),
        Some(AlignmentStatus::Aligned { suspect: true, .. })
    ));
    // The host orders a new alignment: the state goes, and the sensors
    // give the next one.
    navigator.realign().unwrap();
    assert!(navigator.solution(at(205)).is_none());
    assert_eq!(waiting(&navigator).0, AlignmentPhase::Levelling);
    feed(&mut navigator, &mut rig, 206..=406, &mut sink);
    let Some(AlignmentStatus::Aligned { record, suspect }) = navigator.alignment_status() else {
        panic!("aligned again: {:?}", navigator.alignment_status());
    };
    assert_eq!((record.source, suspect), (AlignmentSource::Sensors, false));
}

#[test]
fn a_fork_in_a_rest_window_aligns_as_the_original_does() {
    let mut rig = Rig::new(ATTITUDE_DEG, 11, 1.0);
    let mut original = rig.navigator();
    let mut sink = Vec::new();
    feed(&mut original, &mut rig, 1..=120, &mut sink);
    let mut fork = original.snapshot();
    let mut inputs = Vec::new();
    for step in 121..=210 {
        inputs.push(rig.increment(step));
        inputs.push(rig.magnetic(step));
        if step == 200 {
            inputs.push(rig.gnss(step, true).0);
        }
    }
    let run = |navigator: &mut dyn Navigator| {
        let mut records = Vec::new();
        for input in &inputs {
            navigator.ingest(input, &mut records).unwrap();
        }
        (
            records,
            navigator.alignment_status(),
            navigator.solution(at(210)),
        )
    };
    let (records, status, solution) = run(&mut original);
    assert!(matches!(status, Some(AlignmentStatus::Aligned { .. })));
    assert_eq!(run(&mut *fork), (records, status, solution));
}

/// The gyroscope bias of the rest window starts the filter's bias: a
/// vehicle at rest with 2 mrad/s of bias about the vertical keeps its
/// heading after the state. A transfer alignment that carries the bias in
/// its prior, as a ground calibration before an in-air start gives it,
/// keeps the heading too, to the prior's one-sigma times the time; one
/// without a prior turns the heading by the bias times the time.
#[test]
fn the_gyro_bias_of_a_window_or_a_prior_starts_the_filter() {
    // The prior of a ground calibration: the true bias, to 0.1 mrad/s.
    const PRIOR_SIGMA: f64 = 1e-4;
    let heading_drift = |transfer: Option<bool>| {
        let mut rig = Rig::new(ATTITUDE_DEG, 71, 1.0);
        rig.gyro_bias = Vector3::new(2e-4, -3e-4, 2e-3);
        let mut navigator = rig.navigator();
        let mut sink = Vec::new();
        feed(&mut navigator, &mut rig, 1..=200, &mut sink);
        let aligned = navigator.solution(at(200)).unwrap();
        if let Some(with_prior) = transfer {
            let accel = rig.budgets.accel_bias.powi(2);
            let gyro = PRIOR_SIGMA * PRIOR_SIGMA;
            let prior = crate::ImuBiasPrior::new(
                [0.0; 3],
                SymmetricCov3::from_diagonal(accel, accel, accel),
                (rig.gyro_bias + Vector3::new(0.0, 0.0, 0.5 * PRIOR_SIGMA)).into(),
                SymmetricCov3::from_diagonal(gyro, gyro, gyro),
            );
            navigator
                .align(&crate::Alignment {
                    position: aligned.position,
                    velocity: aligned.velocity,
                    attitude: aligned.attitude.unwrap(),
                    position_cov: aligned.position_cov,
                    velocity_cov: aligned.velocity_cov,
                    attitude_cov: aligned.attitude_cov.unwrap(),
                    at: at(200),
                    imu_bias_prior: with_prior.then_some(prior),
                })
                .unwrap();
            let Some(AlignmentStatus::Aligned { record, .. }) = navigator.alignment_status() else {
                panic!("aligned by transfer");
            };
            assert_eq!(record.gyro_bias_rad_s.is_some(), with_prior);
        }
        let start = normalized_errors(&rig, &aligned).3;
        feed(&mut navigator, &mut rig, 201..=500, &mut sink);
        let end = normalized_errors(&rig, &navigator.solution(at(500)).unwrap()).3;
        (end - start).abs()
    };
    let own = heading_drift(None);
    let with_prior = heading_drift(Some(true));
    let without = heading_drift(Some(false));
    // 2 mrad/s over 3 s is 6 mrad. The bias of the window is known to
    // the gyroscope noise over 2 s, 0.07 mrad/s; that of the prior to
    // 0.1 mrad/s, 0.3 mrad over 3 s.
    assert!(own < 1e-3, "{own}");
    assert!(with_prior < 3.0 * PRIOR_SIGMA * 3.0, "{with_prior}");
    assert!(without > 4e-3, "{without}");
}
