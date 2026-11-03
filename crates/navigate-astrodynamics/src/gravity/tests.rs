use std::format;

use nalgebra::{Matrix3, Vector3};
use navigate_contract::{FrameId, GeodeticPosition};
use navigate_geodesy::{BodyEllipsoid, LocalTangentPlane, NedOffset, geodetic_to_body_fixed};

use super::*;
use crate::PlanetaryBody;

/// Central differences of `f` with step `h` in each axis.
fn finite_difference(
    f: impl Fn(&Vector3<f64>) -> Vector3<f64>,
    r: &Vector3<f64>,
    h: f64,
) -> Matrix3<f64> {
    let mut jacobian = Matrix3::zeros();
    for axis in 0..3 {
        let mut step = Vector3::zeros();
        step[axis] = h;
        let column = (f(&(r + step)) - f(&(r - step))) / (2.0 * h);
        jacobian.set_column(axis, &column);
    }
    jacobian
}

#[test]
fn the_surface_frame_reads_the_body_centered_position() {
    let body = PlanetaryBody::EARTH_WGS84;
    let origin = GeodeticPosition::new(47.0_f64.to_radians(), 8.0_f64.to_radians(), 500.0);
    let plane = LocalTangentPlane::new(origin, BodyEllipsoid::WGS84).expect("plausible origin");
    let frame = SurfaceFrame::new(plane, body);
    let (g0, _) = frame.gravity(&Vector3::zeros());
    // At the origin the apparent gravity points almost straight down.
    assert!(g0.z > 9.8 && g0.x.abs() < 0.03 && g0.y.abs() < 1e-9, "{g0}");
    // 100 km north the local vertical tilts by 100 km / R in the plane.
    let p = Vector3::new(100_000.0, 0.0, 0.0);
    let (g, gradient) = frame.gravity(&p);
    let tilt = -g.x / g.z;
    assert!((tilt - 100_000.0 / 6_371_000.0).abs() < 1e-3, "{tilt}");
    let at = frame.plane().from_ned(&NedOffset::new(p.x, p.y, p.z));
    let r = geodetic_to_body_fixed(&at, body.ellipsoid);
    assert!((frame.body_fixed_position(&p) - Vector3::new(r.x_m, r.y_m, r.z_m)).norm() < 1e-6);
    let numeric = finite_difference(|q| frame.gravity(q).0, &p, 10.0);
    assert!((gradient - numeric).abs().max() < 1e-6 * gradient.abs().max());
    let omega = frame.frame_rotation();
    assert!((omega.norm() - body.rotation_rate_rad_per_s).abs() < 1e-18);
    // The rotation vector in NED: north `ω·cos φ`, down `−ω·sin φ`.
    assert!((omega.x - body.rotation_rate_rad_per_s * 47.0_f64.to_radians().cos()).abs() < 1e-15);
    assert!((omega.z + body.rotation_rate_rad_per_s * 47.0_f64.to_radians().sin()).abs() < 1e-15);
    // A reference reads the same field.
    let by_reference = &frame;
    assert_eq!(by_reference.gravity(&p), frame.gravity(&p));
    assert_eq!(by_reference.frame_rotation(), omega);
}

#[test]
fn the_environment_of_a_body_is_its_rotating_body_fixed_frame() {
    for body in [
        PlanetaryBody::EARTH_WGS84,
        PlanetaryBody::MOON_PA,
        PlanetaryBody::MARS_IAU,
    ] {
        let env = SpaceEnvironment::from_body(body);
        assert!(env.is_plausible());
        assert_eq!(env.frame, body.ellipsoid.frame());
        assert_eq!(env.frame_rotation(), body.rotation_body_fixed());
        for lat_deg in [-90.0_f64, -30.0, 0.0, 47.0, 90.0] {
            let at = GeodeticPosition::new(lat_deg.to_radians(), 1.1, 2_000.0);
            let r = geodetic_to_body_fixed(&at, body.ellipsoid);
            let p = Vector3::new(r.x_m, r.y_m, r.z_m);
            let (g, gradient) = env.gravity(&p);
            let (g_body, gradient_body) = body.apparent_gravity_body_fixed(&p);
            assert!((g - g_body).norm() <= 1e-15 * g_body.norm(), "{lat_deg}");
            assert!((gradient - gradient_body).abs().max() <= 1e-15 * gradient_body.abs().max());
        }
    }
}

fn earth_j2000_central() -> CentralBodyGravity {
    // The J2000 axes are equatorial, so the pole of the Earth is close
    // to `z`. A pole tilted by the obliquity exercises the
    // coordinate-free form.
    let (sin_obliquity, cos_obliquity) = 23.439_281_f64.to_radians().sin_cos();
    CentralBodyGravity {
        gm_m3_per_s2: 3.986_004_418e14,
        j2: 1.082_626_68e-3,
        equatorial_radius_m: 6_378_137.0,
        pole_unit_in_frame: [0.0, sin_obliquity, cos_obliquity],
    }
}

fn moon_body() -> PointMassBody {
    PointMassBody {
        naif_id: 301,
        gm_m3_per_s2: 4.904_869_5e12,
        position_in_frame_m: [384_400_000.0, 0.0, 0.0],
    }
}

fn sun_body() -> PointMassBody {
    PointMassBody {
        naif_id: 10,
        gm_m3_per_s2: 1.327_124_400_18e20,
        position_in_frame_m: [-1.2e11, 8.0e10, 3.4e10],
    }
}

#[test]
fn the_tidal_acceleration_is_zero_at_the_frame_center() {
    let moon = moon_body();
    assert_eq!(moon.tidal_acceleration(&Vector3::zeros()), Vector3::zeros());
    // At the Moon side of a 7000 km orbit the tide pulls toward the Moon.
    let near = moon.tidal_acceleration(&Vector3::new(7_000_000.0, 0.0, 0.0));
    let expected = 2.0 * moon.gm_m3_per_s2 * 7_000_000.0 / 384_400_000.0_f64.powi(3);
    assert!(
        near.x > 0.0 && (near.x / expected - 1.0).abs() < 0.03,
        "{near}"
    );
}

#[test]
fn cislunar_gravity_gradient_matches_finite_differences() {
    let model = SpaceGravityModel::MultiBody {
        central: earth_j2000_central(),
        third_bodies: ThirdBodySet::from_slice(&[moon_body(), sun_body()]).expect("two bodies"),
    };
    let env = SpaceEnvironment::inertial(FrameId::EARTH_J2000, model);
    assert!(env.is_plausible());
    for r in [
        Vector3::new(380_000_000.0, 4_000_000.0, -2_000_000.0),
        Vector3::new(6_778_000.0, 1_000_000.0, 2_500_000.0),
    ] {
        let (_, analytic) = env.gravity(&r);
        let numeric = finite_difference(|p| env.gravity(p).0, &r, 50.0);
        let error = (analytic - numeric).abs().max();
        assert!(error < 1e-9, "{error}");
        assert!(
            error < 1e-6 * analytic.abs().max(),
            "{error} vs {}",
            analytic.abs().max()
        );
    }
}

#[test]
fn the_third_body_set_holds_a_fixed_number_of_bodies_inline() {
    let mut set = ThirdBodySet::EMPTY;
    assert!(set.is_empty());
    for _ in 0..MAX_THIRD_BODIES {
        set.push(moon_body()).expect("room");
    }
    assert_eq!(set.len(), MAX_THIRD_BODIES);
    let full = set;
    assert_eq!(
        set.push(sun_body()),
        Err(ThirdBodyCapacityError {
            capacity: MAX_THIRD_BODIES
        })
    );
    assert_eq!(set, full, "a refused body changes nothing");
    let too_many = [moon_body(); MAX_THIRD_BODIES + 1];
    assert!(ThirdBodySet::from_slice(&too_many).is_err());
    // Equality reads the bodies in the set, not the unused slots.
    let one = ThirdBodySet::from_slice(&[moon_body()]).expect("one body");
    let mut grown = ThirdBodySet::EMPTY;
    grown.push(moon_body()).expect("room");
    assert_eq!(one, grown);
    assert_ne!(one, ThirdBodySet::EMPTY);
    assert_eq!(format!("{one:?}"), format!("{:?}", [moon_body()]));
}

#[test]
fn a_rotating_frame_adds_the_centrifugal_term() {
    let env = SpaceEnvironment {
        frame: FrameId::MOON_PA,
        frame_rotation_rad_per_s: [0.0, 0.0, 2.661_699_5e-6],
        gravity: SpaceGravityModel::Zero,
    };
    let r = Vector3::new(2_000_000.0, 0.0, 500_000.0);
    let (g, gradient) = env.gravity(&r);
    let w2 = 2.661_699_5e-6_f64.powi(2);
    assert!((g - Vector3::new(w2 * 2_000_000.0, 0.0, 0.0)).norm() < 1e-15);
    assert!((gradient - Matrix3::from_diagonal(&Vector3::new(w2, w2, 0.0))).norm() < 1e-25);
}

#[test]
fn implausible_constants_are_refused() {
    let mut central = earth_j2000_central();
    central.pole_unit_in_frame = [0.0, 0.0, 2.0];
    assert!(!central.is_plausible());
    let env = SpaceEnvironment::inertial(
        FrameId::EARTH_J2000,
        SpaceGravityModel::CentralBody(central),
    );
    assert!(!env.is_plausible());
    let at_center = PointMassBody {
        position_in_frame_m: [0.0; 3],
        ..moon_body()
    };
    assert!(!at_center.is_plausible());
    let model = SpaceGravityModel::MultiBody {
        central: earth_j2000_central(),
        third_bodies: ThirdBodySet::from_slice(&[at_center]).expect("one body"),
    };
    assert!(!model.is_plausible());
}
