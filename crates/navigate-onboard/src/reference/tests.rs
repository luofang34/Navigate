mod correlated_gnss;

use super::*;
use crate::sensor::SensorSpec;
use crate::{GuidanceLaw, Onboard, RouteDirector};
use nalgebra::{Matrix2, UnitQuaternion, Vector2, Vector3};
use navigate_camera::{CameraMount, ImagerModel, Intrinsics};
use navigate_contract::{AttitudeQuaternion, GeodeticPosition, NedVelocity};
use navigate_startrack::{StarCentroid, catalogue_ecef};

pub(crate) const CLOCK: ClockDomainId = ClockDomainId::new(7);
const IMU: SourceId = SourceId::new(1);
const GNSS: SourceId = SourceId::new(2);
const BARO: SourceId = SourceId::new(3);
const STARS: SourceId = SourceId::new(4);
const PASS: SourceId = SourceId::new(5);
const LAT: f64 = 41.7;
const LON: f64 = -1.0;
const HEIGHT: f64 = 1_000.0;
/// 2026-06-15T00:00:00Z, at local midnight in summer: the sky is dark.
const UTC_AT_ZERO: f64 = 1_781_481_600.0;
const STEP_NS: u64 = 20_000_000;
/// IMU steps in one second.
const RATE: u64 = 50;

#[cfg(test)]
fn spec(id: SourceId, kind: SensorKind, budget: ErrorBudget) -> SensorSpec {
    SensorSpec {
        id,
        kind,
        imager: None,
        mount: None,
        rate_hz: 1.0,
        budget,
    }
}

#[cfg(test)]
fn star_imager() -> ImagerModel {
    ImagerModel::pinhole(Intrinsics::centred(512, 512, 20.0_f64.to_radians()).unwrap())
}

#[cfg(test)]
pub(crate) fn suite(extra: &[SensorKind]) -> SensorSuite {
    let mut sensors = vec![spec(
        IMU,
        SensorKind::Imu,
        ErrorBudget::Imu {
            accel_noise_mps2_sqrt_hz: 6e-4,
            gyro_noise_rad_s_sqrt_hz: 3e-6,
            accel_bias: FogmBias::new(3_600.0, 5e-4),
            gyro_bias: FogmBias::new(3_600.0, 5e-6),
        },
    )];
    for kind in extra {
        sensors.push(match kind {
            SensorKind::Gnss => spec(
                GNSS,
                *kind,
                ErrorBudget::Gnss {
                    horizontal_sigma_m: 1.0,
                    vertical_sigma_m: 2.0,
                    velocity_sigma_m_s: 0.1,
                    correlated: None,
                },
            ),
            SensorKind::Barometer => spec(
                BARO,
                *kind,
                ErrorBudget::Barometer {
                    white_sigma_m: 0.5,
                    bias: FogmBias::new(3_600.0, 3.0),
                },
            ),
            SensorKind::StarCamera => SensorSpec {
                imager: Some(star_imager()),
                mount: Some(CameraMount::zenith()),
                ..spec(
                    STARS,
                    *kind,
                    ErrorBudget::StarCamera {
                        mount_alignment_sigma_rad: 2e-5,
                    },
                )
            },
            _ => spec(PASS, *kind, ErrorBudget::Declared),
        });
    }
    SensorSuite::new(sensors).unwrap()
}

#[cfg(test)]
fn at(step: u64) -> MonotonicNanos {
    MonotonicNanos::from_nanos(1_000_000_000 + step * STEP_NS)
}

#[cfg(test)]
pub(crate) fn aligned(extra: &[SensorKind]) -> ReferenceNavigator {
    aligned_off(extra, 0.0, 4.0)
}

/// Aligned `north_m` north of the true position with a north and east
/// variance of `variance_m2`.
#[cfg(test)]
fn aligned_off(extra: &[SensorKind], north_m: f64, variance_m2: f64) -> ReferenceNavigator {
    let stars = StarCatalogueConfig::with_utc_at_zero(UTC_AT_ZERO);
    let mut navigator = ReferenceNavigator::new(suite(extra), CLOCK, Some(stars)).unwrap();
    navigator
        .align(&Alignment {
            position: GeodeticPosition::new(
                LAT.to_radians() + north_m / 6_364_000.0,
                LON.to_radians(),
                HEIGHT,
            ),
            velocity: NedVelocity::new(0.0, 0.0, 0.0),
            attitude: AttitudeQuaternion::new(1.0, 0.0, 0.0, 0.0),
            position_cov: SymmetricCov3::from_diagonal(variance_m2, variance_m2, 9.0),
            velocity_cov: SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
            attitude_cov: SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-5),
            at: at(0),
        })
        .unwrap();
    navigator
}

/// The increment of a vehicle at rest with level body axes along NED:
/// the Earth rate and the reaction to normal gravity (Somigliana, with the
/// free-air gradient).
#[cfg(test)]
fn rest_increment(step: u64) -> SensorInput {
    let dt = STEP_NS as f64 * 1e-9;
    let (s, c) = LAT.to_radians().sin_cos();
    let g = 9.780_325_335_9 * (1.0 + 0.001_931_852_652_41 * s * s)
        / (1.0 - 0.006_694_379_990_13 * s * s).sqrt()
        - 3.086e-6 * HEIGHT;
    let omega = 7.292_115e-5;
    SensorInput {
        sensor: IMU,
        acquired_at: at(step),
        delivered_at: at(step),
        data: SensorData::ImuIncrement {
            dt_s: dt,
            delta_theta_rad: [omega * c * dt, 0.0, -omega * s * dt],
            delta_velocity_m_s: [0.0, 0.0, -g * dt],
        },
    }
}

#[cfg(test)]
fn gnss_fix(step: u64, north_m: f64) -> SensorInput {
    let lat = LAT.to_radians() + north_m / 6_364_000.0;
    SensorInput {
        sensor: GNSS,
        acquired_at: at(step),
        delivered_at: at(step),
        data: SensorData::GnssFix {
            position: GeodeticPosition::new(lat, LON.to_radians(), HEIGHT),
            velocity: Some(NedVelocity::new(0.0, 0.0, 0.0)),
            position_cov: None,
            velocity_cov: None,
        },
    }
}

/// Runs `seconds` at rest, with the GNSS fix of `gnss` once a second.
#[cfg(test)]
fn run(
    navigator: &mut ReferenceNavigator,
    seconds: u64,
    gnss: Option<f64>,
    sink: &mut Vec<AcceptanceRecord>,
) {
    for step in 1..=seconds * RATE {
        navigator.ingest(&rest_increment(step), sink).unwrap();
        if let Some(north) = gnss.filter(|_| step.is_multiple_of(RATE)) {
            navigator.ingest(&gnss_fix(step, north), sink).unwrap();
        }
    }
}

#[cfg(test)]
fn north_error_m(navigator: &mut ReferenceNavigator, seconds: u64) -> f64 {
    let solution = navigator.solution(at(seconds * RATE)).unwrap();
    (solution.position.latitude_rad - LAT.to_radians()) * 6_364_000.0
}

#[test]
fn a_vehicle_at_rest_stays_where_it_was_aligned() {
    let mut navigator = aligned(&[]);
    let mut sink = Vec::new();
    run(&mut navigator, 30, None, &mut sink);
    let error = north_error_m(&mut navigator, 30);
    // A one-sigma attitude of 1 mrad tilts gravity into 9.8 mm/s² of
    // acceleration: half of that over 30 s is 4.4 m.
    assert!(error.abs() < 4.4, "{error}");
    assert!(sink.is_empty());
}

#[test]
fn gnss_fixes_pull_the_solution_and_each_is_recorded() {
    // Aligned 15 m north of the truth with a 20 m one-sigma; the fixes are
    // at the truth with a 1 m one-sigma.
    let mut navigator = aligned_off(&[SensorKind::Gnss], 15.0, 400.0);
    let mut sink = Vec::new();
    run(&mut navigator, 30, Some(0.0), &mut sink);
    let error = north_error_m(&mut navigator, 30);
    assert!(error.abs() < 1.0, "{error}");
    let positions: Vec<_> = sink
        .iter()
        .filter(|r| r.measurement == "gnss_position")
        .collect();
    assert_eq!(positions.len(), 30);
    let all_accepted = positions
        .iter()
        .all(|r| r.sensor == GNSS && r.outcome == Outcome::Accepted);
    assert!(all_accepted, "{positions:?}");
    let velocities = sink
        .iter()
        .filter(|r| r.measurement == "gnss_velocity")
        .count();
    assert_eq!(velocities, 30);
}

#[test]
fn a_declared_sensor_without_inputs_changes_nothing() {
    let mut bare = aligned(&[]);
    let mut declared = aligned(&[SensorKind::Gnss, SensorKind::Barometer]);
    let mut sink = Vec::new();
    run(&mut bare, 10, None, &mut sink);
    run(&mut declared, 10, None, &mut sink);
    assert!(sink.is_empty());
    let (a, b) = (
        bare.solution(at(10 * RATE)).unwrap(),
        declared.solution(at(10 * RATE)).unwrap(),
    );
    assert_eq!(a.position, b.position);
    assert_eq!(a.velocity, b.velocity);
}

#[test]
fn baro_and_pass_through_observations_are_recorded() {
    let mut navigator = aligned(&[SensorKind::Barometer, SensorKind::ObservationSource]);
    let mut sink = Vec::new();
    run(&mut navigator, 1, None, &mut sink);
    let baro = SensorInput {
        sensor: BARO,
        acquired_at: at(RATE),
        delivered_at: at(RATE),
        data: SensorData::BaroAltitude { altitude_m: HEIGHT },
    };
    navigator.ingest(&baro, &mut sink).unwrap();
    let stamp = ObservationStamp::new(
        PASS,
        SourceEpoch::new(0),
        WrappingSequence::new(0),
        at(RATE),
        CLOCK,
    );
    let value = ObservationValue::Altitude {
        altitude_m: HEIGHT + 1.0,
        variance_m2: 4.0,
    };
    let observation =
        Observation::new(stamp, value, SourceComposition::of(SensorClass::Barometric));
    let pass = SensorInput {
        sensor: PASS,
        acquired_at: at(RATE),
        delivered_at: at(RATE),
        data: SensorData::Observation(observation),
    };
    navigator.ingest(&pass, &mut sink).unwrap();
    let kinds: Vec<_> = sink
        .iter()
        .map(|r| (r.measurement, r.outcome.clone()))
        .collect();
    assert_eq!(
        kinds,
        [
            ("baro_altitude", Outcome::Accepted),
            ("observation", Outcome::Accepted)
        ]
    );
}

#[test]
fn undeclared_sensors_wrong_data_and_bad_suites_are_refused() {
    let mut navigator = aligned(&[]);
    let mut sink = Vec::new();
    assert_eq!(
        navigator.ingest(&gnss_fix(1, 0.0), &mut sink),
        Err(OnboardError::UnknownSensor(GNSS))
    );
    let wrong = SensorInput {
        data: SensorData::BaroAltitude { altitude_m: 1.0 },
        ..rest_increment(1)
    };
    assert!(matches!(
        navigator.ingest(&wrong, &mut sink),
        Err(OnboardError::WrongData { .. })
    ));
    let imu = suite(&[]).sensors()[0].clone();
    assert!(SensorSuite::new(vec![imu.clone(), imu.clone()]).is_err());
    assert!(SensorSuite::new(vec![]).is_err());
    let camera = spec(
        STARS,
        SensorKind::StarCamera,
        ErrorBudget::StarCamera {
            mount_alignment_sigma_rad: 1e-5,
        },
    );
    assert!(SensorSuite::new(vec![imu, camera]).is_err());
    assert!(ReferenceNavigator::new(suite(&[SensorKind::StarCamera]), CLOCK, None).is_err());
}

/// The catalogue stars that a zenith camera on a level vehicle at the
/// alignment sees at `acquired_at`, as ideal centroids.
#[cfg(test)]
fn zenith_centroids(acquired_at: MonotonicNanos) -> Vec<StarCentroid> {
    let config = StarCatalogueConfig::with_utc_at_zero(UTC_AT_ZERO);
    let catalogue = catalogue_ecef(config.epoch(acquired_at), 5.5).unwrap();
    let ecef_from_body = UnitQuaternion::from_matrix(
        &navigate_fusion::inertial::ned_to_body_fixed_dcm(LAT.to_radians(), LON.to_radians()),
    );
    let cam_from_ecef = CameraMount::zenith().cam_from_body * ecef_from_body.inverse();
    let k = star_imager().intrinsics;
    catalogue
        .iter()
        .filter_map(|s| k.project(&(cam_from_ecef * Vector3::from(s.direction))))
        .filter(|p| (5.0..506.0).contains(&p.x) && (5.0..506.0).contains(&p.y))
        .map(|pixel: Vector2<f64>| StarCentroid {
            pixel,
            flux: 50_000.0,
            pixel_count: 9,
            peak: 5_000.0,
            saturated: false,
            length_px: 0.0,
            covariance_px2: Matrix2::identity() * 0.01,
        })
        .collect()
}

#[test]
fn a_star_frame_is_identified_and_offered_as_lines_of_sight() {
    let mut navigator = aligned(&[SensorKind::StarCamera]);
    let mut sink = Vec::new();
    run(&mut navigator, 1, None, &mut sink);
    let mid = at(RATE);
    let centroids = zenith_centroids(mid);
    assert!(centroids.len() >= 6, "{}", centroids.len());
    let exposure = SensorInput {
        sensor: STARS,
        acquired_at: mid,
        delivered_at: mid,
        data: SensorData::StarExposureMid,
    };
    navigator.ingest(&exposure, &mut sink).unwrap();
    let frame = SensorInput {
        sensor: STARS,
        acquired_at: mid,
        delivered_at: mid,
        data: SensorData::StarCentroids { centroids },
    };
    navigator.ingest(&frame, &mut sink).unwrap();
    let accepted = sink
        .iter()
        .filter(|r| r.measurement == "star_line_of_sight" && r.outcome == Outcome::Accepted)
        .count();
    assert!(accepted >= 4, "{sink:?}");
    let empty = SensorInput {
        sensor: STARS,
        acquired_at: mid,
        delivered_at: mid,
        data: SensorData::StarCentroids { centroids: vec![] },
    };
    sink.clear();
    navigator.ingest(&empty, &mut sink).unwrap();
    assert!(
        matches!(
            sink.as_slice(),
            [AcceptanceRecord {
                outcome: Outcome::Unused(_),
                ..
            }]
        ),
        "{sink:?}"
    );
}

#[test]
fn the_bundle_links_a_navigator_and_a_director_as_trait_objects() {
    use navigate_contract::{AltitudeConstraint, Ident, Waypoint};
    let waypoint = |ident: &str, lon: f64| {
        Waypoint::new(
            Ident::try_new(ident).unwrap(),
            GeodeticPosition::new(LAT.to_radians(), lon.to_radians(), HEIGHT),
        )
        .with_altitude(AltitudeConstraint::At(HEIGHT))
    };
    let waypoints = [waypoint("A", LON + 0.01), waypoint("B", LON + 0.05)];
    let config = navigate_route::ExecutionConfig::new(100.0).with_surface_gravity_mps2(9.806_65);
    let law = GuidanceLaw::DeviationTracking(
        navigate_guidance::GuidanceConfig::default()
            .with_expected_ellipsoid(navigate_contract::Ellipsoid::WGS84),
    );
    let director =
        RouteDirector::new("TEST", waypoints.to_vec(), config, law, 40.0, CLOCK).unwrap();
    let mut onboard = Onboard::new(Box::new(aligned(&[])), Box::new(director));
    let (solution, output) = onboard.direct(at(0));
    assert!(solution.is_some());
    assert_eq!(output.active_index, Some(0));
    assert_eq!(output.commanded_speed_mps, 40.0);
    assert!(output.setpoint.is_some() || output.refusal.is_some());
    assert_eq!(onboard.director.active_index(), Some(0));
}

#[test]
fn a_snapshot_forks_the_run_and_restores_it() {
    let mut navigator = aligned(&[SensorKind::Gnss]);
    let mut sink = Vec::new();
    run(&mut navigator, 5, Some(0.0), &mut sink);
    let snapshot = navigator.snapshot();
    let mut fork = snapshot.snapshot();
    for step in 5 * RATE + 1..=8 * RATE {
        navigator.ingest(&rest_increment(step), &mut sink).unwrap();
        fork.ingest(&rest_increment(step), &mut sink).unwrap();
    }
    let now = at(8 * RATE);
    assert_eq!(navigator.solution(now), fork.solution(now));
    navigator
        .ingest(&gnss_fix(8 * RATE, 3.0), &mut sink)
        .unwrap();
    assert_ne!(navigator.solution(now), fork.solution(now));
    navigator.restore(&*snapshot).unwrap();
    let mut from_snapshot = snapshot.snapshot();
    let then = at(5 * RATE);
    assert_eq!(navigator.solution(then), from_snapshot.solution(then));
    assert!(navigator.requirements().is_empty() && navigator.load_report().is_empty());
}

#[test]
fn every_line_of_sight_of_a_second_frame_is_accepted() {
    let mut navigator = aligned(&[SensorKind::StarCamera]);
    let mut sink = Vec::new();
    run(&mut navigator, 1, None, &mut sink);
    let mut per_frame = Vec::new();
    for second in [1_u64, 2] {
        if second == 2 {
            for step in RATE + 1..=2 * RATE {
                navigator.ingest(&rest_increment(step), &mut sink).unwrap();
            }
        }
        let mid = at(second * RATE);
        let exposure = SensorInput {
            sensor: STARS,
            acquired_at: mid,
            delivered_at: mid,
            data: SensorData::StarExposureMid,
        };
        navigator.ingest(&exposure, &mut sink).unwrap();
        let centroids = zenith_centroids(mid);
        let frame = SensorInput {
            sensor: STARS,
            acquired_at: mid,
            delivered_at: mid,
            data: SensorData::StarCentroids { centroids },
        };
        sink.clear();
        navigator.ingest(&frame, &mut sink).unwrap();
        per_frame.push(sink.clone());
    }
    let [first, second] = [&per_frame[0], &per_frame[1]];
    assert!(first.len() >= 4, "{first:?}");
    assert!(
        first.iter().all(|r| r.context == Some("lost_in_space")),
        "{first:?}"
    );
    assert!(second.len() >= 4, "{second:?}");
    let accepted = second
        .iter()
        .filter(|r| r.outcome == Outcome::Accepted)
        .count();
    assert_eq!(accepted, second.len(), "{second:?}");
    assert!(
        second.iter().all(|r| r.context == Some("tracking")),
        "{second:?}"
    );
}
