use nalgebra::{Matrix3, Vector3};
use navigate_contract::FrameId;

use super::{ForceEvaluation, OrbitalForceModel, OrbitalGravity};
use crate::{
    ASTRONOMICAL_UNIT_M, CannonballSrp, ExponentialDrag, GravityField, HarmonicNormalization,
    PlanetaryBody, SOLAR_PRESSURE_AT_1_AU_N_PER_M2, SpaceEnvironment, SpaceGravityModel,
    SphericalHarmonicField, ThirdBodySet, harmonic_index,
};

const EARTH: PlanetaryBody = PlanetaryBody::EARTH_WGS84;

/// The point mass and `J2` of the Earth, unnormalized, for a field that
/// lives for the whole test run.
static J2_C: [f64; 6] = [1.0, 0.0, 0.0, -1.082_626_68e-3, 0.0, 0.0];
static J2_S: [f64; 6] = [0.0; 6];

fn srp() -> CannonballSrp {
    CannonballSrp {
        reflectivity_cr: 1.5,
        area_to_mass_m2_per_kg: 0.01,
        sun_position_in_frame_m: [ASTRONOMICAL_UNIT_M, 0.0, 0.0],
        central_body_radius_m: 6_378_137.0,
    }
}

fn drag() -> ExponentialDrag {
    ExponentialDrag {
        cd: 2.2,
        area_to_mass_m2_per_kg: 0.02,
        reference_density_kg_m3: 3.0e-12,
        reference_altitude_m: 400e3,
        scale_height_m: 60e3,
        equatorial_radius_m: 6_378_137.0,
        atmosphere_rotation_rad_s: [0.0, 0.0, EARTH.rotation_rate_rad_per_s],
    }
}

fn leo() -> (Vector3<f64>, Vector3<f64>) {
    (
        Vector3::new(4.0e6, -3.5e6, 3.6e6),
        Vector3::new(4_500.0, 5_200.0, -0.8e3),
    )
}

/// Central differences of the acceleration on the position and on the
/// velocity.
fn numeric(
    f: impl Fn(&Vector3<f64>, &Vector3<f64>) -> Vector3<f64>,
    r: &Vector3<f64>,
    v: &Vector3<f64>,
) -> (Matrix3<f64>, Matrix3<f64>) {
    let mut position = Matrix3::zeros();
    let mut velocity = Matrix3::zeros();
    for axis in 0..3 {
        let mut dr = Vector3::zeros();
        dr[axis] = 10.0;
        position.set_column(axis, &((f(&(r + dr), v) - f(&(r - dr), v)) / 20.0));
        let mut dv = Vector3::zeros();
        dv[axis] = 0.1;
        velocity.set_column(axis, &((f(r, &(v + dv)) - f(r, &(v - dv))) / 0.2));
    }
    (position, velocity)
}

fn assert_jacobians(evaluation: &ForceEvaluation, numeric: &(Matrix3<f64>, Matrix3<f64>)) {
    let scale_r = evaluation.position_jacobian.abs().max().max(1e-30);
    let scale_v = evaluation.velocity_jacobian.abs().max().max(1e-30);
    let error_r = (evaluation.position_jacobian - numeric.0).abs().max() / scale_r;
    let error_v = (evaluation.velocity_jacobian - numeric.1).abs().max() / scale_v;
    assert!(
        error_r < 1e-6,
        "{error_r}\n{}",
        evaluation.position_jacobian
    );
    assert!(
        error_v < 1e-6,
        "{error_v}\n{}",
        evaluation.velocity_jacobian
    );
}

#[test]
fn the_pressure_decreases_with_the_square_of_the_distance_and_vanishes_in_the_shadow() {
    let model = srp();
    let at_one_au = model.evaluate(&Vector3::new(0.0, 7.0e6, 0.0));
    let expected = SOLAR_PRESSURE_AT_1_AU_N_PER_M2 * 1.5 * 0.01;
    let magnitude = at_one_au.acceleration_mps2.norm();
    assert!((magnitude / expected - 1.0).abs() < 1e-3, "{magnitude}");
    assert!(at_one_au.acceleration_mps2.x < 0.0, "away from the Sun");
    let far = CannonballSrp {
        sun_position_in_frame_m: [2.0 * ASTRONOMICAL_UNIT_M, 0.0, 0.0],
        ..model
    };
    let quarter = far.evaluate(&Vector3::new(0.0, 7.0e6, 0.0));
    assert!((quarter.acceleration_mps2.norm() / magnitude - 0.25).abs() < 1e-3);
    // Behind the Earth inside the cylinder, and just outside of it.
    let umbra = Vector3::new(-7.0e6, 6.0e6, 0.0);
    assert!(model.in_shadow(&umbra));
    assert_eq!(model.evaluate(&umbra), ForceEvaluation::ZERO);
    let lit = Vector3::new(-7.0e6, 6.5e6, 0.0);
    assert!(!model.in_shadow(&lit));
    assert!(model.evaluate(&lit).acceleration_mps2.norm() > 0.0);
    assert!(!model.in_shadow(&Vector3::new(7.0e6, 0.0, 0.0)));
}

#[test]
fn the_pressure_gradient_matches_central_differences() {
    let model = CannonballSrp {
        sun_position_in_frame_m: [1.0e9, 2.0e8, -3.0e8],
        ..srp()
    };
    let r = Vector3::new(-2.0e7, 4.0e7, 1.0e7);
    let evaluation = model.evaluate(&r);
    let mut numeric_r = Matrix3::zeros();
    for axis in 0..3 {
        let mut d = Vector3::zeros();
        d[axis] = 1.0e4;
        numeric_r.set_column(
            axis,
            &((model.evaluate(&(r + d)).acceleration_mps2
                - model.evaluate(&(r - d)).acceleration_mps2)
                / 2.0e4),
        );
    }
    assert_jacobians(&evaluation, &(numeric_r, Matrix3::zeros()));
}

#[test]
fn drag_opposes_the_air_relative_velocity_and_has_exact_jacobians() {
    let model = drag();
    let (r, v) = leo();
    let evaluation = model.evaluate(&r, &v);
    let w = v - model.air_velocity(&r);
    let a = evaluation.acceleration_mps2;
    assert!((a.normalize() + w.normalize()).norm() < 1e-12);
    let expected = 0.5 * model.density(&r) * 2.2 * 0.02 * w.norm_squared();
    assert!((a.norm() / expected - 1.0).abs() < 1e-12);
    let numeric = numeric(|r, v| model.evaluate(r, v).acceleration_mps2, &r, &v);
    assert_jacobians(&evaluation, &numeric);
    // A body at rest in the atmosphere has no drag.
    let co_rotating = model.air_velocity(&r);
    assert_eq!(model.evaluate(&r, &co_rotating), ForceEvaluation::ZERO);
    assert!(
        !ExponentialDrag {
            scale_height_m: 0.0,
            ..model
        }
        .is_plausible()
    );
}

#[test]
fn the_force_model_adds_every_term_with_exact_jacobians_in_a_rotating_frame() {
    let env = SpaceEnvironment::from_body(EARTH)
        .with_drag(ExponentialDrag {
            atmosphere_rotation_rad_s: [0.0; 3],
            ..drag()
        })
        .with_srp(srp());
    let model = env.force_model();
    let (r, v) = leo();
    let evaluation = model.evaluate(&r, &v);
    let omega = Vector3::new(0.0, 0.0, EARTH.rotation_rate_rad_per_s);
    let (g_app, _) = env.gravity(&r);
    let extra = srp().evaluate(&r).acceleration_mps2
        + env
            .drag
            .map_or(Vector3::zeros(), |d| d.evaluate(&r, &v).acceleration_mps2);
    let expected = g_app - 2.0 * omega.cross(&v) + extra;
    assert!((evaluation.acceleration_mps2 - expected).norm() < 1e-15);
    let numeric = numeric(|r, v| model.evaluate(r, v).acceleration_mps2, &r, &v);
    assert_jacobians(&evaluation, &numeric);
    assert!(env.is_plausible());
    let mut broken = env;
    broken.drag = Some(ExponentialDrag {
        cd: f64::NAN,
        ..drag()
    });
    assert!(!broken.is_plausible());
    assert_eq!(SpaceEnvironment::from_body(EARTH).srp, None);
}

#[test]
fn a_borrowed_or_static_harmonic_field_gives_the_central_body_gravity() {
    let field = SphericalHarmonicField::new(
        EARTH.gm_m3_per_s2,
        EARTH.ellipsoid.semi_major_axis_m(),
        2,
        HarmonicNormalization::Unnormalized,
        &J2_C,
        &J2_S,
    )
    .expect("plausible");
    let env = SpaceEnvironment::inertial(
        FrameId::EARTH_J2000,
        SpaceGravityModel::SphericalHarmonics {
            field,
            third_bodies: ThirdBodySet::EMPTY,
        },
    );
    let central = EARTH.central_gravity();
    let (r, _) = leo();
    let (g, _) = env.gravity(&r);
    assert!((g - central.acceleration(&r)).norm() < 1e-13 * g.norm());
    // A field that borrows coefficients on the stack.
    let mut c = [0.0; 6];
    c[0] = 1.0;
    c[harmonic_index(2, 0)] = -EARTH.j2;
    let s = [0.0; 6];
    let local = SphericalHarmonicField::new(
        EARTH.gm_m3_per_s2,
        EARTH.ellipsoid.semi_major_axis_m(),
        2,
        HarmonicNormalization::Unnormalized,
        &c,
        &s,
    )
    .expect("plausible");
    let model = OrbitalForceModel {
        gravity: OrbitalGravity::Harmonics {
            field: local,
            third_bodies: ThirdBodySet::EMPTY,
        },
        frame_rotation_rad_per_s: [0.0; 3],
        srp: None,
        drag: None,
    };
    assert!(model.is_plausible());
    let a = model.evaluate(&r, &Vector3::zeros()).acceleration_mps2;
    assert!((a - g).norm() < 1e-15 * g.norm());
}

#[test]
fn a_runge_kutta_orbit_returns_to_its_start() {
    let env = SpaceEnvironment::inertial(
        FrameId::EARTH_J2000,
        SpaceGravityModel::CentralBody(crate::CentralBodyGravity {
            j2: 0.0,
            ..EARTH.central_gravity()
        }),
    );
    let model = env.force_model();
    let radius = 6_778_137.0;
    let speed = (EARTH.gm_m3_per_s2 / radius).sqrt();
    let period = core::f64::consts::TAU * radius / speed;
    let steps = 600_u32;
    let dt_s = period / f64::from(steps);
    let start = (
        Vector3::new(radius, 0.0, 0.0),
        Vector3::new(0.0, speed, 0.0),
    );
    let (mut r, mut v) = start;
    for _ in 0..steps {
        (r, v) = model.rk4_step(&r, &v, dt_s);
    }
    assert!((r - start.0).norm() < 0.1, "{}", (r - start.0).norm());
    assert!((v - start.1).norm() < 1e-4);
    // A constant extra acceleration adds `½·a·t²` to the position.
    let push = Vector3::new(0.0, 0.0, 1e-3);
    let free = SpaceEnvironment::inertial(FrameId::EARTH_J2000, SpaceGravityModel::Zero);
    let (r1, v1) =
        free.force_model()
            .rk4_step_with(&Vector3::zeros(), &Vector3::zeros(), 10.0, |_| push);
    assert!((r1.z - 0.05).abs() < 1e-15 && (v1.z - 0.01).abs() < 1e-15);
}
