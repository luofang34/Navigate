//! One filter from a launch site on the Earth to the lunar south pole:
//! the rotating frame of the Earth, the inertial frame of the Earth, the
//! inertial frame of the Moon, and the rotating frame of the Moon.

use core::f64::consts::FRAC_PI_2;

use nalgebra::{UnitQuaternion, Vector3};
use navigate_contract::{CartesianPosition, DurationNanos, Ellipsoid};
use navigate_geodesy::BodyEllipsoid;

use super::*;
use crate::inertial::FrameChange;
use crate::inertial::local_level::body_fixed;
use crate::inertial::nominal::to_contract;
use crate::observation::{Observation, ObservationValue};
use navigate_astrodynamics::{
    CentralBodyGravity, PointMassBody, SpaceEnvironment, SpaceGravityModel, ThirdBodySet,
};

const EARTH_ANGLE_RAD: f64 = 0.8;
const MOON_ANGLE_RAD: f64 = -2.1;
const MOON_DISTANCE_M: f64 = 384_400_000.0;
/// The time of the delayed fix in the rotating frame of the Earth.
const DELAYED_FIX_MS: u64 = 3_500;

fn cape_canaveral() -> GeodeticPosition {
    GeodeticPosition::new(
        28.396_837_f64.to_radians(),
        (-80.605_659_f64).to_radians(),
        3.0,
    )
}

/// The lunar landing site at 89.9°S.
fn landing_site() -> GeodeticPosition {
    GeodeticPosition::new((-89.9_f64).to_radians(), 0.5, 2_000.0)
}

fn about_z(angle_rad: f64) -> UnitQuaternion<f64> {
    UnitQuaternion::from_axis_angle(&Vector3::z_axis(), angle_rad)
}

fn earth_inertial() -> SpaceEnvironment {
    let central = PlanetaryBody::EARTH_WGS84.central_gravity();
    SpaceEnvironment::inertial(
        FrameId::EARTH_J2000,
        SpaceGravityModel::CentralBody(central),
    )
}

fn moon_inertial() -> SpaceEnvironment {
    let moon = PlanetaryBody::MOON_PA;
    SpaceEnvironment::inertial(
        FrameId::MOON_J2000,
        SpaceGravityModel::MultiBody {
            central: CentralBodyGravity {
                pole_unit_in_frame: [0.0, 0.0, 1.0],
                ..moon.central_gravity()
            },
            third_bodies: ThirdBodySet::from_slice(&[PointMassBody {
                naif_id: 399,
                gm_m3_per_s2: PlanetaryBody::EARTH_WGS84.gm_m3_per_s2,
                position_in_frame_m: [-MOON_DISTANCE_M, 0.0, 0.0],
            }])
            .expect("one body"),
        },
    )
}

fn config() -> InertialConfig {
    InertialConfig {
        fusion: crate::config::FusionConfig {
            staleness_bound: DurationNanos::from_millis(30_000),
            ..InertialConfig::default().fusion
        },
        ..InertialConfig::default()
    }
}

/// Propagates with the rates `rates` in `frame` from `from_ms` to
/// `to_ms` in steps of 500 ms. `sequence` counts the IMU samples.
fn coast(
    filter: &mut InertialNavigationFilter,
    sequence: &mut u32,
    rates: ([f64; 3], [f64; 3]),
    frame: FrameId,
    (from_ms, to_ms): (u64, u64),
) {
    for at_ms in (from_ms + 500..=to_ms).step_by(500) {
        *sequence = sequence.wrapping_add(1);
        let sample = imu_in(*sequence, at_ms, rates, 0.5, frame);
        assert_accepted(filter.ingest_imu(&sample, t(at_ms)));
    }
}

/// The change into the inertial frame of the Moon. The test puts the
/// Moon where the vehicle is 2 km above the landing site and descends
/// at 3 m/s in the rotating frame of the Moon.
fn to_moon(filter: &InertialNavigationFilter) -> FrameChange {
    let moon = PlanetaryBody::MOON_PA;
    let state = core(filter);
    let r_pa = body_fixed(&landing_site(), moon.ellipsoid);
    let v_pa = -3.0 * r_pa.normalize();
    let pa_to_inertial = about_z(MOON_ANGLE_RAD).inverse();
    let r_rel = pa_to_inertial * r_pa;
    let v_rel = pa_to_inertial * (v_pa + moon.rotation_body_fixed().cross(&r_pa));
    FrameChange::between_environments(
        &filter.environment(),
        moon_inertial(),
        None,
        (state.nominal.p - r_rel).into(),
        (state.nominal.v - v_rel).into(),
        to_contract(&UnitQuaternion::identity()),
    )
}

/// Runs the flight. `delayed_fix` offers a fix in the rotating frame of
/// the Earth, acquired at 3.5 s, after the last change of frame.
fn fly(delayed_fix: bool) -> InertialNavigationFilter {
    let mut filter = surface_filter(config(), PlanetaryBody::EARTH_WGS84, cape_canaveral());
    let pad = rest_rates(&filter);
    let mut sequence = 0_u32;
    coast(
        &mut filter,
        &mut sequence,
        pad,
        FrameId::EARTH_ITRF93,
        (0, 2_000),
    );
    assert!(filter.tick(t(2_000)).is_some());
    let space = filter.tick_space(t(2_000)).expect("each regime");
    assert_eq!(space.frame, FrameId::EARTH_ITRF93);
    // Ascent: 2 g of thrust along the body `−z` axis, which is up.
    let ascent = ([pad.0[0], pad.0[1], pad.0[2] - 19.6], pad.1);
    coast(
        &mut filter,
        &mut sequence,
        ascent,
        FrameId::EARTH_ITRF93,
        (2_000, 3_500),
    );
    let fix_position = core(&filter).nominal.p + Vector3::new(3.0, -2.0, 1.0);
    coast(
        &mut filter,
        &mut sequence,
        ascent,
        FrameId::EARTH_ITRF93,
        (3_500, 6_000),
    );
    let rotating = filter.environment();
    let to_inertial = FrameChange::between_environments(
        &rotating,
        earth_inertial(),
        None,
        [0.0; 3],
        [0.0; 3],
        to_contract(&about_z(EARTH_ANGLE_RAD)),
    );
    filter.change_frame(to_inertial, t(6_000)).expect("valid");
    assert!(filter.tick(t(6_000)).is_none());
    let burn = ([0.0, 0.0, -20.0], [0.0; 3]);
    coast(
        &mut filter,
        &mut sequence,
        burn,
        FrameId::EARTH_J2000,
        (6_000, 8_000),
    );
    filter
        .change_frame(to_moon(&filter), t(8_000))
        .expect("valid");
    assert_eq!(filter.frame(), FrameId::MOON_J2000);
    let to_surface = FrameChange::between_environments(
        &filter.environment(),
        SpaceEnvironment::from_body(PlanetaryBody::MOON_PA),
        Some(BodyEllipsoid::MOON_PA),
        [0.0; 3],
        [0.0; 3],
        to_contract(&about_z(MOON_ANGLE_RAD)),
    );
    filter.change_frame(to_surface, t(8_000)).expect("valid");
    if delayed_fix {
        let fix = Observation::in_frame(
            stamp(2, 1, DELAYED_FIX_MS),
            ObservationValue::CartesianPositionFix {
                position: CartesianPosition::new(fix_position.x, fix_position.y, fix_position.z),
                covariance: SymmetricCov3::from_diagonal(4.0, 4.0, 4.0),
            },
            gnss(),
            FrameId::EARTH_ITRF93,
        );
        assert_accepted(filter.ingest(&fix, t(8_000)));
    }
    filter
}

#[test]
fn one_filter_flies_from_the_earth_to_the_lunar_south_pole() {
    let mut filter = fly(false);
    assert_eq!(filter.frame(), FrameId::MOON_PA);
    assert_eq!(filter.surface_ellipsoid(), Some(BodyEllipsoid::MOON_PA));
    let solution = filter.tick(t(8_000)).expect("a lunar surface solution");
    assert_eq!(solution.ellipsoid, Ellipsoid::MOON_PA);
    let site = landing_site();
    assert!((solution.position.latitude_rad - site.latitude_rad).abs() < 1e-9);
    assert!((solution.position.longitude_rad - site.longitude_rad).abs() < 1e-6);
    assert!((solution.position.altitude_m - site.altitude_m).abs() < 1e-3);
    assert!(
        (solution.velocity.down_mps - 3.0).abs() < 1e-6,
        "{:?}",
        solution.velocity
    );
    assert!(solution.position.latitude_rad + FRAC_PI_2 < 2e-3);
    let space = filter.tick_space(t(8_000)).expect("each regime");
    assert_eq!(space.frame, FrameId::MOON_PA);
    // Hover in the rotating frame of the Moon with a lunar fix.
    let mut lunar = filter.clone();
    let hover = rest_rates(&lunar);
    let mut sequence = 1_000_u32;
    coast(
        &mut lunar,
        &mut sequence,
        hover,
        FrameId::MOON_PA,
        (8_000, 9_000),
    );
    let fix = Observation::in_frame(
        stamp(3, 1, 9_000),
        ObservationValue::PositionFix {
            position: lunar.tick(t(9_000)).expect("published").position,
            covariance: SymmetricCov3::from_diagonal(25.0, 25.0, 25.0),
        },
        SourceComposition::of(SensorClass::VisualLandmark),
        FrameId::MOON_PA,
    );
    assert_accepted(lunar.ingest(&fix, t(9_000)));
    assert!(lunar.tick(t(9_000)).is_some());
}

#[test]
fn a_delayed_fix_replays_through_every_change_of_frame() {
    let without = fly(false);
    let with = fly(true);
    assert_eq!(with.frame(), FrameId::MOON_PA);
    let moved = (core(&with).nominal.p - core(&without).nominal.p).norm();
    assert!(moved > 1.0 && moved < 10.0, "{moved}");
    let sigma = |f: &InertialNavigationFilter| core(f).p.fixed_view::<3, 3>(0, 0).trace();
    assert!(sigma(&with) < sigma(&without));
}

#[test]
fn an_initialization_after_the_changes_stays_in_the_active_frame() {
    let mut filter = fly(false);
    filter
        .initialize_surface(
            landing_site(),
            NedVelocity::new(0.0, 0.0, 0.0),
            identity_attitude(),
            SymmetricCov3::from_diagonal(25.0, 25.0, 25.0),
            SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
            SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-6),
            t(8_000),
            gnss(),
        )
        .expect("plausible");
    assert_eq!(filter.frame(), FrameId::MOON_PA);
    assert_eq!(filter.surface_ellipsoid(), Some(BodyEllipsoid::MOON_PA));
    assert_eq!(
        core(&filter).regime.env,
        SpaceEnvironment::from_body(PlanetaryBody::MOON_PA)
    );
    let solution = filter.tick(t(8_000)).expect("a lunar surface solution");
    assert_eq!(solution.ellipsoid, Ellipsoid::MOON_PA);
    assert!((solution.position.latitude_rad - landing_site().latitude_rad).abs() < 1e-12);
    assert!((solution.position.altitude_m - landing_site().altitude_m).abs() < 1e-6);
}
