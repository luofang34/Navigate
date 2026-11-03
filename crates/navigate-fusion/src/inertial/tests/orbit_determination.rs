//! Reduced-dynamic orbit determination in low Earth orbit: a
//! force-model coast with `J2`, drag, and the empirical acceleration,
//! two-way range and range rate from three ground stations, star
//! tracker directions, and an impulsive burn.

use nalgebra::{UnitQuaternion, Vector3};
use navigate_contract::{
    CartesianPosition, CartesianVelocity, DurationNanos, SensorClass, SpaceUncertainty,
    SymmetricCov3,
};

use super::*;
use crate::inertial::{ForceModelCoast, PropagationMode};
use crate::kalman::FogmBias;
use crate::observation::{LineOfSightReference, Observation, ObservationValue};
use navigate_astrodynamics::{ExponentialDrag, SpaceEnvironment, SpaceGravityModel};

const EARTH: PlanetaryBody = PlanetaryBody::EARTH_WGS84;
const RANGE_SIGMA_M: f64 = 1.0;
const RATE_SIGMA_MPS: f64 = 1e-3;
const STAR_SIGMA_RAD: f64 = 1e-5;
const STEP_MS: u64 = 20_000;
const BURN_MS: u64 = 600_000;
const END_MS: u64 = 1_200_000;
const RANGE_SOURCE: u32 = 1;
const RATE_SOURCE: u32 = 2;
const STAR_SOURCE: u32 = 3;

fn drag(cd: f64) -> ExponentialDrag {
    ExponentialDrag {
        cd,
        area_to_mass_m2_per_kg: 0.02,
        reference_density_kg_m3: 3.0e-12,
        reference_altitude_m: 400e3,
        scale_height_m: 60e3,
        equatorial_radius_m: EARTH.ellipsoid.semi_major_axis_m(),
        atmosphere_rotation_rad_s: [0.0, 0.0, EARTH.rotation_rate_rad_per_s],
    }
}

/// The environment of the filter. The true drag coefficient is larger,
/// and the empirical acceleration absorbs the difference.
fn env(cd: f64) -> SpaceEnvironment {
    SpaceEnvironment::inertial(
        FrameId::EARTH_J2000,
        SpaceGravityModel::CentralBody(EARTH.central_gravity()),
    )
    .with_drag(drag(cd))
}

/// The body-fixed position of three stations, turned into the inertial
/// axes at the time `at_s`.
fn station(index: u32, at_s: f64) -> Vector3<f64> {
    let (lat, lon) = match index % 3 {
        0 => (0.6_f64, 0.2_f64),
        1 => (0.1, 1.4),
        _ => (-0.5, 0.9),
    };
    let r = EARTH.ellipsoid.semi_major_axis_m();
    let body_fixed = Vector3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()) * r;
    let rotation =
        UnitQuaternion::from_axis_angle(&Vector3::z_axis(), EARTH.rotation_rate_rad_per_s * at_s);
    rotation * body_fixed
}

fn station_velocity(index: u32, at_s: f64) -> Vector3<f64> {
    Vector3::new(0.0, 0.0, EARTH.rotation_rate_rad_per_s).cross(&station(index, at_s))
}

fn stars() -> [Vector3<f64>; 3] {
    [
        Vector3::new(0.2, 0.9, 0.3).normalize(),
        Vector3::new(-0.7, 0.1, 0.7).normalize(),
        Vector3::new(0.5, -0.6, -0.6).normalize(),
    ]
}

/// The true trajectory: position and velocity every second.
struct Truth {
    states: std::vec::Vec<(Vector3<f64>, Vector3<f64>)>,
    attitude: UnitQuaternion<f64>,
}

fn start() -> (Vector3<f64>, Vector3<f64>) {
    let radius = EARTH.ellipsoid.semi_major_axis_m() + 400e3;
    let speed = (EARTH.gm_m3_per_s2 / radius).sqrt();
    let inclination = 0.9_f64;
    (
        Vector3::new(radius, 0.0, 0.0),
        Vector3::new(0.0, speed * inclination.cos(), speed * inclination.sin()),
    )
}

/// The true burn: the commanded change plus an execution error.
fn true_burn() -> Vector3<f64> {
    commanded_burn() + Vector3::new(0.01, -0.008, 0.005)
}

fn commanded_burn() -> Vector3<f64> {
    Vector3::new(0.0, 0.8, -0.3)
}

fn truth() -> Truth {
    let model = env(2.4).force_model();
    let (mut r, mut v) = start();
    let mut states = std::vec![(r, v)];
    for second in 1..=(END_MS / 1_000) {
        (r, v) = model.rk4_step(&r, &v, 1.0);
        if second * 1_000 == BURN_MS {
            v += true_burn();
        }
        states.push((r, v));
    }
    Truth {
        states,
        attitude: UnitQuaternion::from_euler_angles(0.3, -0.2, 1.1),
    }
}

fn config() -> InertialConfig {
    InertialConfig {
        fusion: crate::config::FusionConfig {
            staleness_bound: DurationNanos::from_millis(60_000),
            ..InertialConfig::default().fusion
        },
        dmc: Some(FogmBias::new(900.0, 1e-6)),
        ..InertialConfig::default()
    }
}

fn radio() -> SourceComposition {
    SourceComposition::of(SensorClass::RadioNavigation)
}

fn observations(truth: &Truth, k: u32, at_ms: u64) -> [Observation; 3] {
    let at_s = at_ms as f64 * 1e-3;
    let (r, v) = truth.states[(at_ms / 1_000) as usize];
    let index = k % 3;
    let tx = station(index, at_s);
    let rx = station(index, at_s);
    let range = (r - tx).norm() + (r - rx).norm() + 25.0;
    let w = v - station_velocity(index, at_s);
    let rate = 2.0 * (r - tx).normalize().dot(&w);
    let star = stars()[(k % 3) as usize];
    let tilt = UnitQuaternion::from_scaled_axis(
        Vector3::new(noise(3 * k), noise(3 * k + 1), noise(3 * k + 2)) * STAR_SIGMA_RAD,
    );
    let measured = tilt * (truth.attitude.inverse() * star);
    let values = [
        (
            RANGE_SOURCE,
            radio(),
            ObservationValue::TwoWayRange {
                tx_station_pos_m: tx.into(),
                rx_station_pos_m: rx.into(),
                gravitation_gm_m3_s2: None,
                round_trip_range_m: range + RANGE_SIGMA_M * noise(10_000 + k),
                turnaround_delay_m: 25.0,
                variance_m2: RANGE_SIGMA_M * RANGE_SIGMA_M,
            },
        ),
        (
            RATE_SOURCE,
            radio(),
            ObservationValue::TwoWayRangeRate {
                tx_station_pos_m: tx.into(),
                tx_station_vel_mps: station_velocity(index, at_s).into(),
                rx_station_pos_m: rx.into(),
                rx_station_vel_mps: station_velocity(index, at_s).into(),
                round_trip_range_rate_mps: rate + RATE_SIGMA_MPS * noise(20_000 + k),
                variance_m2_per_s2: RATE_SIGMA_MPS * RATE_SIGMA_MPS,
            },
        ),
        (
            STAR_SOURCE,
            SourceComposition::of(SensorClass::Celestial),
            ObservationValue::LineOfSight {
                measured_body_unit: measured.into(),
                reference: LineOfSightReference::InertialDirection {
                    direction_in_frame_unit: star.into(),
                },
                angular_variance_rad2: STAR_SIGMA_RAD * STAR_SIGMA_RAD,
            },
        ),
    ];
    values.map(|(source, composition, value)| {
        Observation::in_frame(
            stamp(source, k, at_ms),
            value,
            composition,
            FrameId::EARTH_J2000,
        )
    })
}

/// The radial, along-track, and cross-track 1-sigma values. A state
/// without an orbit plane gives `NaN`, which each bound refuses.
fn rtn(filter: &mut InertialNavigationFilter, at_ms: u64) -> [f64; 3] {
    let solution = filter.tick_space(t(at_ms)).expect("initialized");
    let SpaceUncertainty::OrbitalRtn {
        radial_1sigma_m,
        along_track_1sigma_m,
        cross_track_1sigma_m,
        ..
    } = solution.integrity.uncertainty
    else {
        return [f64::NAN; 3];
    };
    [radial_1sigma_m, along_track_1sigma_m, cross_track_1sigma_m]
}

#[test]
fn a_leo_coast_with_a_burn_converges_in_radial_along_track_and_cross_track_axes() {
    let truth = truth();
    let (r0, v0) = truth.states[0];
    let mut filter = InertialNavigationFilter::new_space(config(), CLOCK, env(2.2));
    let q0 = truth.attitude * UnitQuaternion::from_scaled_axis(Vector3::new(1e-3, -5e-4, 8e-4));
    filter
        .initialize_space(
            CartesianPosition::new(r0.x + 120.0, r0.y - 80.0, r0.z + 60.0),
            CartesianVelocity::new(v0.x - 0.08, v0.y + 0.05, v0.z + 0.1),
            crate::inertial::nominal::to_contract(&q0),
            SymmetricCov3::from_diagonal(9e4, 9e4, 9e4),
            SymmetricCov3::from_diagonal(0.09, 0.09, 0.09),
            SymmetricCov3::from_diagonal(4e-6, 4e-6, 4e-6),
            t(0),
            gnss(),
        )
        .expect("plausible");
    let coast = ForceModelCoast {
        acceleration_noise_sigma_mps2_sqrt_hz: 1e-7,
        body_rate_noise_sigma_rad_s_sqrt_hz: 1e-8,
        ..ForceModelCoast::new(10.0)
    };
    filter
        .set_propagation_mode(PropagationMode::ForceModel(coast), t(0))
        .expect("plausible");
    let initial = rtn(&mut filter, 0);
    assert!(initial.iter().all(|sigma| *sigma > 250.0), "{initial:?}");
    let mut before_burn = [0.0; 3];
    let mut k = 0_u32;
    for at_ms in (STEP_MS..=END_MS).step_by(STEP_MS as usize) {
        if at_ms == BURN_MS {
            before_burn = rtn(&mut filter, at_ms);
            let burn_cov = SymmetricCov3::from_diagonal(4e-4, 4e-4, 4e-4);
            filter
                .apply_impulsive_burn(commanded_burn().into(), burn_cov, t(at_ms))
                .expect("burn in the force-model mode");
        }
        k = k.wrapping_add(1);
        for obs in observations(&truth, k, at_ms) {
            assert_accepted(filter.ingest(&obs, t(at_ms)));
        }
    }
    let last = rtn(&mut filter, END_MS);
    for (sigma, axis) in before_burn.iter().zip(["radial", "along", "cross"]) {
        assert!(*sigma < 5.0, "{axis} before the burn: {before_burn:?}");
    }
    for (sigma, axis) in last.iter().zip(["radial", "along", "cross"]) {
        assert!(*sigma < 5.0, "{axis} at the end: {last:?}");
    }
    let state = core(&filter);
    let (r, v) = truth.states[(END_MS / 1_000) as usize];
    let position_error = (state.nominal.p - r).norm();
    let velocity_error = (state.nominal.v - v).norm();
    assert!(position_error < 10.0, "{position_error}");
    assert!(velocity_error < 0.01, "{velocity_error}");
    let attitude_error = state.nominal.q.angle_to(&truth.attitude);
    assert!(attitude_error < 1e-4, "{attitude_error}");
    assert_eq!(filter.rejections().innovation_gate, 0);
}
