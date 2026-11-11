//! A declared start without GNSS. The true start is the declared one plus
//! a dispersion drawn from the declared one-sigmas, and the wind and the
//! vertical speed, which the statement does not say, are drawn from their
//! allowances: the normalized error of the first state is then testable.

use nalgebra::Vector3;
use navigate_contract::{NavigationSolution, SourceId};
use navigate_data::VerticalDatum;

use super::moving::{attitude_error, cross_wind, normalized};
use super::path::{Profile, Truth, Vehicle, ramp};
use super::rig::{self, Noise, Rig, at};
use super::waiting;
use crate::{
    AlignmentConfig, AlignmentNeed, AlignmentPhase, AlignmentSource, AlignmentStatus,
    DeclaredHeight, DeclaredSpeed, DeclaredStart, ErrorBudget, Navigator, ReferenceNavigator,
    SensorData, SensorInput, SensorKind, SensorSpec, SensorSuite, SpeedKind,
};

const OPERATOR: SourceId = SourceId::new(5);
/// The step of the statement: three seconds of flight.
const STATED_AT: u64 = 300;
/// The crab allowance of a fixed wing in the air, radians.
const CRAB_SIGMA_RAD: f64 = 0.09;

/// A navigator with an inertial unit and an operator, whose course is its
/// heading to `crab`.
fn navigator(crab: Option<f64>) -> ReferenceNavigator {
    let declared = Rig::new([0.0; 3], 0, 1.0).suite(None);
    let imu = declared.of_kind(SensorKind::Imu).next().unwrap().clone();
    let operator = SensorSpec {
        id: OPERATOR,
        kind: SensorKind::Operator,
        budget: ErrorBudget::Declared,
        ..imu.clone()
    };
    let suite = SensorSuite::new(vec![imu, operator]).unwrap();
    let mut navigator = ReferenceNavigator::new(suite, rig::CLOCK, None).unwrap();
    navigator
        .set_alignment_config(AlignmentConfig {
            course_heading_sigma_rad: crab,
            ..AlignmentConfig::default()
        })
        .unwrap();
    navigator
}

/// The statement of the true state `truth`, each part moved by a draw of
/// its declared one-sigma times `honesty`; zero gives the truth.
fn statement(truth: &Truth, airspeed: f64, noise: &mut Noise, honesty: f64) -> DeclaredStart {
    let [lat, lon, height] = truth.position;
    let heading = truth.body_to_ned.euler_angles().2;
    let (h, v, psi, speed) = (30.0, 15.0, 0.05, 2.0);
    DeclaredStart {
        latitude_rad: lat + noise.normal() * h * honesty / 6_364_000.0,
        longitude_rad: lon + noise.normal() * h * honesty / (6_378_137.0 * lat.cos()),
        horizontal_sigma_m: h,
        height: DeclaredHeight {
            height_m: height + noise.normal() * v * honesty,
            datum: VerticalDatum::Ellipsoidal,
            sigma_m: v,
        },
        true_heading_rad: heading + noise.normal() * psi * honesty,
        heading_sigma_rad: psi,
        speed: DeclaredSpeed {
            kind: SpeedKind::Airspeed,
            speed_m_s: airspeed + noise.normal() * speed * honesty,
            sigma_m_s: speed,
        },
    }
}

/// Flies to the statement, delivers it `late` steps after its stamp, and
/// gives the truth at the stamp and the solution at the delivery.
fn declare<F: Fn(f64) -> Profile>(
    navigator: &mut ReferenceNavigator,
    vehicle: &mut Vehicle<F>,
    start: impl Fn(&Truth) -> DeclaredStart,
    late: u64,
) -> (Truth, Option<NavigationSolution>) {
    let mut sink = Vec::new();
    while vehicle.step < STATED_AT {
        navigator.ingest(&vehicle.increment(), &mut sink).unwrap();
    }
    let truth = vehicle.truth();
    let stated = start(&truth);
    while vehicle.step < STATED_AT + late {
        navigator.ingest(&vehicle.increment(), &mut sink).unwrap();
    }
    let input = SensorInput {
        sensor: OPERATOR,
        acquired_at: at(STATED_AT),
        delivered_at: at(vehicle.step),
        data: SensorData::DeclaredStart(stated),
    };
    navigator.ingest(&input, &mut sink).unwrap();
    (truth, navigator.solution(at(vehicle.step)))
}

/// Straight flight at 55 m/s through the air on `heading`, climbing at
/// `climb` m/s.
fn cruise(heading: f64, climb: f64) -> impl Fn(f64) -> Profile {
    move |_| Profile {
        speed_m_s: 55.0,
        heading_rad: heading,
        roll_rad: 0.0,
        pitch_rad: 0.04 + climb / 55.0,
        climb_m_s: climb,
    }
}

/// The normalized errors of attitude, velocity and position of the state
/// at its stamp, and that of the heading alone.
fn errors(truth: &Truth, solution: &NavigationSolution) -> [f64; 4] {
    let attitude = attitude_error(truth, solution);
    let attitude_cov = solution.attitude_cov.unwrap();
    let v = solution.velocity;
    let velocity = Vector3::new(v.north_mps, v.east_mps, v.down_mps) - truth.velocity;
    let position = truth.position_error_m(&solution.position);
    let (_, _, heading_var) = attitude_cov.diagonal();
    [
        normalized(attitude, &attitude_cov),
        normalized(velocity, &solution.velocity_cov),
        normalized(position, &solution.position_cov),
        attitude.z * attitude.z / heading_var,
    ]
}

/// How a declared start states its speed, and what the navigator knows
/// of the direction of the velocity.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Case {
    /// Airspeed of a vehicle that moves along its nose, in a wind drawn
    /// from the wind allowance.
    Air,
    /// Ground speed of a vehicle that moves along its nose, with a crab
    /// angle drawn from the crab allowance.
    Ground,
    /// Airspeed of a vehicle whose course is not its heading.
    Undirected,
}

/// The mean normalized errors of 96 declared starts in cruise, with the
/// unstated wind and climb drawn from their allowances and the statement
/// drawn from its one-sigmas times `honesty`; and the largest stated tilt
/// one-sigma.
fn mean_errors(case: Case, honesty: f64) -> ([f64; 4], f64) {
    const SEEDS: u64 = 96;
    let config = AlignmentConfig::default();
    let mut sum = [0.0; 4];
    let mut widest_tilt: f64 = 0.0;
    for seed in 0..SEEDS {
        let mut noise = Noise::new(seed.wrapping_add(7_000));
        let heading = noise.normal() * 3.0;
        let wind = match case {
            Case::Ground => cross_wind(heading, noise.normal() * CRAB_SIGMA_RAD),
            Case::Air | Case::Undirected => [
                noise.normal() * config.declared_wind_sigma_m_s,
                noise.normal() * config.declared_wind_sigma_m_s,
            ],
        };
        let climb = noise.normal() * config.declared_vertical_speed_sigma_m_s;
        let mut vehicle = Vehicle::new(cruise(heading, climb), wind, seed, 1.0);
        let crab = (case != Case::Undirected).then_some(CRAB_SIGMA_RAD);
        let mut navigator = navigator(crab);
        let (truth, solution) = declare(
            &mut navigator,
            &mut vehicle,
            |truth| {
                let mut start = statement(truth, 55.0, &mut Noise::new(seed), honesty);
                if case == Case::Ground {
                    start.speed.kind = SpeedKind::GroundSpeed;
                    start.speed.speed_m_s += truth.velocity.xy().norm() - 55.0;
                }
                start
            },
            0,
        );
        let Some(solution) = solution else {
            panic!("{case:?} seed {seed}: {:?}", navigator.alignment_status());
        };
        let (tilt_var, _, _) = solution.attitude_cov.unwrap().diagonal();
        widest_tilt = widest_tilt.max(tilt_var.sqrt());
        let each = errors(&truth, &solution);
        for (total, value) in sum.iter_mut().zip(each) {
            *total += value / SEEDS as f64;
        }
    }
    (sum, widest_tilt)
}

/// The first state of a declared start is as good as it states. The
/// tilt one-sigma holds every horizontal acceleration that the size of the
/// specific force cannot show, so the attitude and the velocity may be
/// better than they state and not worse; the position and the heading
/// come from the statement alone and have the size that it states.
#[test]
fn a_declared_start_is_as_good_as_it_states() {
    rig::show_events();
    for case in [Case::Air, Case::Ground, Case::Undirected] {
        let ([attitude, velocity, position, heading], widest_tilt) = mean_errors(case, 1.0);
        tracing::info!(
            ?case,
            attitude,
            velocity,
            position,
            heading,
            widest_tilt,
            "96 declared starts: mean normalized errors, widest tilt one-sigma"
        );
        // Three degrees of freedom: a band of 3.3 sigma about three. One:
        // about one.
        assert!(
            attitude < 3.82 && velocity < 3.82,
            "{case:?}: {attitude} {velocity}"
        );
        assert!((2.18..=3.82).contains(&position), "{case:?}: {position}");
        assert!((0.52..=1.48).contains(&heading), "{case:?}: {heading}");
        // The tilt one-sigma of the size test, at 55 m/s.
        assert!(widest_tilt < 0.25, "{case:?}: {widest_tilt}");
    }
    // A statement five times worse than it says shows.
    let ([_, _, position, heading], _) = mean_errors(Case::Air, 5.0);
    assert!(position > 3.82 && heading > 1.48, "{position} {heading}");
}

/// A vehicle that accelerates along its heading: each acceleration waits
/// for steady motion or gives a tilt inside its stated three sigma, and a
/// hard acceleration waits.
#[test]
fn an_acceleration_waits_or_stays_inside_the_stated_tilt() {
    for (index, acceleration) in [0.0, 0.3, 0.6, 1.0, 1.5, 2.0, 3.0].into_iter().enumerate() {
        let profile = move |t: f64| Profile {
            speed_m_s: 40.0 + acceleration * t,
            heading_rad: 1.0,
            roll_rad: 0.0,
            pitch_rad: 0.04,
            climb_m_s: 0.0,
        };
        let mut vehicle = Vehicle::new(profile, [0.0, 0.0], index as u64, 1.0);
        let mut navigator = navigator(Some(CRAB_SIGMA_RAD));
        let speed = 40.0 + acceleration * STATED_AT as f64 * rig::STEP_S;
        let (truth, solution) = declare(
            &mut navigator,
            &mut vehicle,
            |truth| statement(truth, speed, &mut Noise::new(1), 0.0),
            0,
        );
        match solution {
            Some(solution) => {
                let error = attitude_error(&truth, &solution);
                let (tilt_var, _, _) = solution.attitude_cov.unwrap().diagonal();
                let tilt = error.xy().norm() / tilt_var.sqrt();
                assert!(tilt < 3.0, "{acceleration} m/s²: {tilt} sigma");
            }
            None => assert_eq!(
                waiting(&navigator),
                (AlignmentPhase::Levelling, AlignmentNeed::SteadyMotion),
                "{acceleration} m/s²"
            ),
        }
        if acceleration == 0.0 || acceleration >= 3.0 {
            assert_eq!(solution.is_some(), acceleration == 0.0, "{acceleration}");
        }
    }
}

/// In a turn the declared speed and the body rate give the centripetal
/// acceleration: a vehicle that moves along its nose takes it out and is
/// inside its stated tilt. Another vehicle cannot, and a turn of 20° of
/// bank waits for steady motion.
#[test]
fn a_turn_is_inside_the_stated_tilt() {
    let turn = |t: f64| Profile {
        speed_m_s: 55.0,
        heading_rad: 0.5 + 9.806_65 * 20.0_f64.to_radians().tan() / 55.0 * t,
        roll_rad: 20.0_f64.to_radians() * ramp(t, 0.0, 0.5),
        pitch_rad: 0.05,
        climb_m_s: 0.0,
    };
    for crab in [Some(CRAB_SIGMA_RAD), None] {
        let mut vehicle = Vehicle::new(turn, [0.0, 0.0], 11, 1.0);
        let mut navigator = navigator(crab);
        let (truth, solution) = declare(
            &mut navigator,
            &mut vehicle,
            |truth| statement(truth, 55.0, &mut Noise::new(2), 0.0),
            0,
        );
        // Without the turn taken out, 20° of bank is 0.6 m/s² of specific
        // force more than gravity: not steady.
        let Some(solution) = solution else {
            assert_eq!(crab, None);
            assert_eq!(
                waiting(&navigator),
                (AlignmentPhase::Levelling, AlignmentNeed::SteadyMotion)
            );
            continue;
        };
        let error = attitude_error(&truth, &solution);
        let (tilt_var, _, _) = solution.attitude_cov.unwrap().diagonal();
        // Taken out, the turn leaves less than 2° of the 20° bank: what is
        // left is the angle of attack, the velocity off the body axis.
        assert!(error.xy().norm() < 0.035, "{}", error.xy().norm());
        assert!(
            error.xy().norm() < 3.0 * tilt_var.sqrt(),
            "{crab:?}: {} of {}",
            error.xy().norm(),
            tilt_var.sqrt()
        );
    }
}

/// A statement that comes late gives the state at its stamp, and the
/// increments after the stamp carry it to the present.
#[test]
fn a_late_statement_is_the_state_at_its_stamp() {
    let mut vehicle = Vehicle::new(cruise(2.0, 0.0), [3.0, -2.0], 13, 1.0);
    let mut navigator = navigator(Some(CRAB_SIGMA_RAD));
    let (_, solution) = declare(
        &mut navigator,
        &mut vehicle,
        |truth| statement(truth, 55.0, &mut Noise::new(3), 0.0),
        50,
    );
    let Some(AlignmentStatus::Aligned { record, .. }) = navigator.alignment_status() else {
        panic!("aligned: {:?}", navigator.alignment_status());
    };
    assert_eq!(
        (record.at, record.source),
        (at(STATED_AT), AlignmentSource::Declared)
    );
    let solution = solution.unwrap();
    let [attitude, velocity, position, _] = errors(&vehicle.truth(), &solution);
    assert!(attitude < 16.0 && velocity < 16.0 && position < 16.0);
}

/// A height above a geoid takes the bound of the undulation in its
/// error; a height of unknown reference gives no state.
#[test]
fn a_height_needs_its_datum() {
    for (datum, aligned) in [
        (
            VerticalDatum::Orthometric {
                geoid: navigate_data::GeoidModel::Egm96,
            },
            true,
        ),
        (VerticalDatum::Unknown, false),
    ] {
        let mut vehicle = Vehicle::new(cruise(0.3, 0.0), [0.0, 0.0], 17, 1.0);
        let mut navigator = navigator(Some(CRAB_SIGMA_RAD));
        let (_, solution) = declare(
            &mut navigator,
            &mut vehicle,
            |truth| {
                let mut start = statement(truth, 55.0, &mut Noise::new(4), 0.0);
                start.height.datum = datum;
                start
            },
            0,
        );
        assert_eq!(solution.is_some(), aligned, "{datum:?}");
        match solution {
            Some(solution) => {
                let (_, _, vertical) = solution.position_cov.diagonal();
                assert!(vertical.sqrt() > 107.0);
            }
            None => assert_eq!(
                waiting(&navigator),
                (AlignmentPhase::Position, AlignmentNeed::HeightDatum)
            ),
        }
    }
}

/// A fork taken in flight before a late statement aligns as the original
/// does: the kept increments and the configuration go with the fork.
#[test]
fn a_fork_before_a_late_statement_aligns_as_the_original_does() {
    let mut vehicle = Vehicle::new(cruise(1.2, 1.0), [2.0, 4.0], 19, 1.0);
    let mut original = navigator(Some(CRAB_SIGMA_RAD));
    let mut sink = Vec::new();
    while vehicle.step < STATED_AT {
        original.ingest(&vehicle.increment(), &mut sink).unwrap();
    }
    let stated = statement(&vehicle.truth(), 55.0, &mut Noise::new(5), 1.0);
    let mut inputs = Vec::new();
    while vehicle.step < STATED_AT + 40 {
        inputs.push(vehicle.increment());
    }
    inputs.push(SensorInput {
        sensor: OPERATOR,
        acquired_at: at(STATED_AT),
        delivered_at: at(vehicle.step),
        data: SensorData::DeclaredStart(stated),
    });
    let mut fork = original.snapshot();
    let run = |navigator: &mut dyn Navigator| {
        let mut records = Vec::new();
        for input in &inputs {
            navigator.ingest(input, &mut records).unwrap();
        }
        (
            records,
            navigator.alignment_status(),
            navigator.solution(at(vehicle.step)),
        )
    };
    let (records, status, solution) = run(&mut original);
    assert!(matches!(status, Some(AlignmentStatus::Aligned { .. })));
    assert_eq!(run(&mut *fork), (records, status, solution));
}

/// An increment of no or negative length is not kept: a declared start
/// after it aligns from the good increments and does not fail.
#[test]
fn an_increment_of_no_length_is_not_kept() {
    let mut vehicle = Vehicle::new(cruise(0.7, 0.0), [0.0, 0.0], 23, 1.0);
    let mut navigator = navigator(Some(CRAB_SIGMA_RAD));
    let mut sink = Vec::new();
    while vehicle.step < STATED_AT {
        let input = vehicle.increment();
        if vehicle.step == STATED_AT - 50 {
            for dt_s in [-0.01, f64::NAN, 0.0] {
                let bad = SensorInput {
                    data: SensorData::ImuIncrement {
                        dt_s,
                        delta_theta_rad: [0.0; 3],
                        delta_velocity_m_s: [0.0; 3],
                    },
                    ..input.clone()
                };
                navigator.ingest(&bad, &mut sink).unwrap();
            }
        }
        navigator.ingest(&input, &mut sink).unwrap();
    }
    let stated = statement(&vehicle.truth(), 55.0, &mut Noise::new(6), 0.0);
    let input = SensorInput {
        sensor: OPERATOR,
        acquired_at: at(STATED_AT),
        delivered_at: at(STATED_AT),
        data: SensorData::DeclaredStart(stated),
    };
    navigator.ingest(&input, &mut sink).unwrap();
    assert!(navigator.solution(at(STATED_AT)).is_some());
}
