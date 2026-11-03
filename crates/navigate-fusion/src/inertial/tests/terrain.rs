//! Terrain clearance over sloped terrain with an off-nadir beam, and
//! the separation of the barometric bias from geometric heights.

use std::vec;
use std::vec::Vec;

use nalgebra::{SMatrix, UnitQuaternion, Vector3};
use navigate_contract::{ObservationStamp, SensorClass, SourceComposition};
use navigate_geodesy::{
    BodyEllipsoid, BodyFixedPosition, LocalTangentPlane, NedOffset, body_fixed_to_geodetic,
};

use super::*;
use crate::inertial::estimate::InertialCore;
use crate::inertial::local_level::LocalLevel;
use crate::inertial::models::terrain_clearance;
use crate::inertial::nominal::{
    ALT_BIAS, ATT, InertialNominal, Mat17, N, POS, Vec17, rotation_vector,
};
use crate::inertial::regime::Regime;
use crate::observation::{Observation, ObservationValue};

const SLOPE: [f64; 2] = [0.08, -0.12];
const TERRAIN_HEIGHT_M: f64 = 350.0;

fn beam() -> [f64; 3] {
    let b = Vector3::new(0.12, -0.07, 1.0).normalize();
    [b.x, b.y, b.z]
}

fn sloped_core() -> InertialCore {
    let plane = LocalTangentPlane::new(earth_origin(), BodyEllipsoid::WGS84).expect("plausible");
    let vehicle = plane.from_ned(&NedOffset::new(4_000.0, -2_500.0, -900.0));
    let q_local = UnitQuaternion::from_euler_angles(0.15, -0.2, 0.7);
    InertialCore {
        regime: Regime::surface_of(&PlanetaryBody::EARTH_WGS84),
        nominal: InertialNominal {
            p: earth_body_fixed(&vehicle),
            v: Vector3::zeros(),
            q: LocalLevel::at(vehicle).bf_to_ned_quaternion().inverse() * q_local,
            b_a: Vector3::zeros(),
            b_g: Vector3::zeros(),
            b_alt: 7.0,
            b_mag: 0.0,
        },
        p: Mat17::identity(),
        state_at: t(0),
        last_rates: None,
        mag_key: None,
    }
}

fn clearance_value(clearance_m: f64) -> ObservationValue {
    ObservationValue::TerrainClearance {
        clearance_m,
        terrain_height_m: TERRAIN_HEIGHT_M,
        terrain_slope_north_east: SLOPE,
        beam_body_unit: beam(),
        variance_m2: 1.0,
    }
}

/// The exact slant range for the state `nominal ⊕ dx` against the
/// terrain plane through the nominal horizontal position. The local axes
/// stay the axes at the nominal vehicle.
fn true_range(core: &InertialCore, dx: &Vec17) -> f64 {
    let nominal = core.nominal;
    let perturbed = nominal.inject(dx);
    let geodetic = |p: Vector3<f64>| {
        body_fixed_to_geodetic(&BodyFixedPosition::new(p.x, p.y, p.z), BodyEllipsoid::WGS84)
    };
    let vehicle = geodetic(nominal.p);
    let local = LocalTangentPlane::new(vehicle, BodyEllipsoid::WGS84).expect("plausible");
    let moved = geodetic(perturbed.p);
    let offset = local.to_ned(&moved);
    let terrain = TERRAIN_HEIGHT_M + SLOPE[0] * offset.north_m + SLOPE[1] * offset.east_m;
    let frame_to_local = LocalLevel::at(vehicle).bf_to_ned();
    let u = frame_to_local * perturbed.c_b_to_f() * Vector3::from(beam());
    let m = Vector3::new(SLOPE[0], SLOPE[1], 1.0);
    (moved.altitude_m - terrain) / m.dot(&u)
}

#[test]
fn terrain_clearance_jacobian_matches_finite_differences() {
    let core = sloped_core();
    let rho = true_range(&core, &Vec17::zeros());
    let model = terrain_clearance(&core, BodyEllipsoid::WGS84, &clearance_value(rho))
        .expect("valid geometry");
    assert!(model.residual[0].abs() < 1e-6, "{}", model.residual[0]);
    let mut numeric = SMatrix::<f64, 1, N>::zeros();
    for j in (POS..POS + 3).chain(ATT..ATT + 3).chain([ALT_BIAS]) {
        let eps = if j < ATT { 0.5 } else { 1e-6 };
        let mut dx = Vec17::zeros();
        dx[j] = eps;
        numeric[(0, j)] = (true_range(&core, &dx) - true_range(&core, &(-dx))) / (2.0 * eps);
    }
    let error = (model.h - numeric).abs().max();
    assert!(
        error < 1e-5 * model.h.abs().max(),
        "{error}\n{}\n{}",
        model.h,
        numeric
    );
    // The slope and the attitude both matter: a flat nadir model would
    // give `∂ρ/∂p = up` and `∂ρ/∂δθ = 0`.
    assert!(model.h.fixed_view::<1, 3>(0, ATT).abs().max() > 10.0);
    assert!(model.h[(0, ALT_BIAS)] == 0.0);
}

#[test]
fn a_grazing_beam_or_a_vehicle_below_the_terrain_is_refused() {
    let mut core = sloped_core();
    let grazing = ObservationValue::TerrainClearance {
        clearance_m: 500.0,
        terrain_height_m: TERRAIN_HEIGHT_M,
        terrain_slope_north_east: SLOPE,
        beam_body_unit: [0.0, 0.0, -1.0],
        variance_m2: 1.0,
    };
    // A beam that points away from the terrain has `κ < 0.05`.
    assert!(terrain_clearance(&core, BodyEllipsoid::WGS84, &grazing).is_err());
    let below = earth_body_fixed(&GeodeticPosition::new(
        earth_origin().latitude_rad,
        earth_origin().longitude_rad,
        TERRAIN_HEIGHT_M - 50.0,
    ));
    core.nominal.p = below;
    assert!(terrain_clearance(&core, BodyEllipsoid::WGS84, &clearance_value(10.0)).is_err());
}

fn observation(
    stamp: ObservationStamp,
    value: ObservationValue,
    class: SensorClass,
) -> Observation {
    Observation::new(stamp, value, SourceComposition::of(class))
}

/// Runs ten seconds at rest with the aiding observations of each second.
fn run(aiding: impl Fn(u32, u64) -> Vec<Observation>) -> InertialNavigationFilter {
    let mut filter = surface_filter(
        InertialConfig::default(),
        PlanetaryBody::EARTH_WGS84,
        earth_origin(),
    );
    let rates = rest_rates(&filter);
    for k in 1..=10_u32 {
        let at_ms = u64::from(k) * 1_000;
        assert_accepted(filter.ingest_imu(&imu(k, at_ms, rates, 1.0), t(at_ms)));
        for obs in aiding(k, at_ms) {
            assert_accepted(filter.ingest(&obs, t(at_ms)));
        }
    }
    filter
}

fn gnss_altitude(k: u32, at_ms: u64) -> Observation {
    let value = ObservationValue::Altitude {
        altitude_m: earth_origin().altitude_m,
        variance_m2: 1.0,
    };
    observation(stamp(2, k, at_ms), value, SensorClass::Gnss)
}

#[test]
fn only_a_barometric_altitude_moves_the_barometric_bias() {
    let truth_altitude_m = earth_origin().altitude_m;
    let free = run(|_, _| Vec::new());
    let altimeter = run(|k, at_ms| {
        let value = ObservationValue::TerrainClearance {
            clearance_m: truth_altitude_m - 100.0,
            terrain_height_m: 100.0,
            terrain_slope_north_east: [0.0, 0.0],
            beam_body_unit: [0.0, 0.0, 1.0],
            variance_m2: 1.0,
        };
        vec![observation(
            stamp(1, k, at_ms),
            value,
            SensorClass::SurfaceAltimeter,
        )]
    });
    let gnss_height = run(|k, at_ms| vec![gnss_altitude(k, at_ms)]);
    // A barometric altitude alone cannot split its offset between the
    // height and the bias. A geometric height makes the bias observable.
    let baro = run(|k, at_ms| {
        let value = ObservationValue::Altitude {
            altitude_m: truth_altitude_m + 6.0,
            variance_m2: 1.0,
        };
        vec![
            gnss_altitude(k, at_ms),
            observation(stamp(3, k, at_ms), value, SensorClass::Barometric),
        ]
    });
    let bias =
        |f: &InertialNavigationFilter| (core(f).nominal.b_alt, core(f).p[(ALT_BIAS, ALT_BIAS)]);
    let (_, free_variance) = bias(&free);
    for geometric in [&altimeter, &gnss_height] {
        let (b_alt, variance) = bias(geometric);
        assert_eq!(b_alt, 0.0);
        assert!((variance - free_variance).abs() <= 1e-12 * free_variance);
        // The geometric height does reach the vertical position.
        assert!(core(geometric).p[(2, 2)] < 0.5 * core(&free).p[(2, 2)]);
    }
    let (b_alt, variance) = bias(&baro);
    assert!((b_alt - 6.0).abs() < 1.0, "{b_alt}");
    assert!(
        variance < 0.5 * free_variance,
        "{variance} vs {free_variance}"
    );
    // The rotation-vector helper agrees with the injected attitude.
    let q = core(&baro).nominal.q;
    assert!(rotation_vector(&(q.inverse() * q)).norm() < 1e-15);
}
