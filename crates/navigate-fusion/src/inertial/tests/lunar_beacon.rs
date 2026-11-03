//! One-way ranging to lunar surface beacons: four beacons near the south
//! pole of the Moon give pseudoranges and pseudorange rates to a
//! spacecraft in low lunar orbit. The receiver clock starts with a bias
//! of 100 m and a drift of 0.5 m/s. The filter estimates the position,
//! the clock bias, and the clock drift together in the rotating
//! body-fixed frame of the Moon.

use nalgebra::Vector3;
use navigate_contract::{
    CartesianPosition, CartesianVelocity, DurationNanos, SensorClass, SymmetricCov3,
};

use super::*;
use crate::inertial::nominal::{CLOCK_BIAS, CLOCK_DRIFT};
use crate::inertial::{ClockNoiseModel, ForceModelCoast, PropagationMode};
use crate::observation::{Observation, ObservationValue};
use navigate_astrodynamics::SpaceEnvironment;

const MOON: PlanetaryBody = PlanetaryBody::MOON_ME;
const RANGE_SIGMA_M: f64 = 1.0;
const RATE_SIGMA_MPS: f64 = 5e-3;
const CLOCK_BIAS_M: f64 = 100.0;
const CLOCK_DRIFT_MPS: f64 = 0.5;
const STEP_MS: u64 = 10_000;
const END_MS: u64 = 300_000;

fn moon_radius() -> f64 {
    MOON.ellipsoid.semi_major_axis_m()
}

fn surface(latitude_deg: f64, longitude_deg: f64) -> Vector3<f64> {
    let (lat, lon) = (latitude_deg.to_radians(), longitude_deg.to_radians());
    Vector3::new(lat.cos() * lon.cos(), lat.cos() * lon.sin(), lat.sin()) * moon_radius()
}

/// Four beacons at different elevation angles, so that the vertical
/// position and the clock bias separate.
fn beacons() -> [Vector3<f64>; 4] {
    [
        surface(-90.0, 0.0),
        surface(-84.0, 0.0),
        surface(-82.0, 120.0),
        surface(-80.0, 240.0),
    ]
}

/// The true states every second in the rotating frame of the Moon: a
/// polar orbit 100 km high that crosses the south pole.
fn truth() -> std::vec::Vec<(Vector3<f64>, Vector3<f64>)> {
    let env = SpaceEnvironment::from_body(MOON);
    let model = env.force_model();
    let radius = moon_radius() + 100e3;
    let latitude = (-80.0_f64).to_radians();
    let r = Vector3::new(latitude.cos(), 0.0, latitude.sin()) * radius;
    let speed = (MOON.gm_m3_per_s2 / radius).sqrt();
    let inertial = Vector3::new(latitude.sin(), 0.0, -latitude.cos()) * speed;
    let v = inertial - Vector3::new(0.0, 0.0, MOON.rotation_rate_rad_per_s).cross(&r);
    let mut states = std::vec![(r, v)];
    let (mut r, mut v) = (r, v);
    for _ in 0..(END_MS / 1_000) {
        (r, v) = model.rk4_step(&r, &v, 1.0);
        states.push((r, v));
    }
    states
}

fn config() -> InertialConfig {
    InertialConfig {
        fusion: crate::config::FusionConfig {
            staleness_bound: DurationNanos::from_millis(60_000),
            ..InertialConfig::default().fusion
        },
        clock_model: Some(ClockNoiseModel {
            white_frequency_psd_m2_per_s: 1e-3,
            random_walk_frequency_psd_m2_per_s3: 1e-6,
            initial_bias_sigma_m: 300.0,
            initial_drift_sigma_mps: 2.0,
        }),
        ..InertialConfig::default()
    }
}

fn observation(source: u32, k: u32, at_ms: u64, value: ObservationValue) -> Observation {
    Observation::in_frame(
        stamp(source, k, at_ms),
        value,
        SourceComposition::of(SensorClass::RadioNavigation),
        FrameId::MOON_ME,
    )
}

fn epoch(truth: &[(Vector3<f64>, Vector3<f64>)], k: u32, at_ms: u64) -> std::vec::Vec<Observation> {
    let at_s = at_ms as f64 * 1e-3;
    let (r, v) = truth[(at_ms / 1_000) as usize];
    let bias = CLOCK_BIAS_M + CLOCK_DRIFT_MPS * at_s;
    let mut out = std::vec::Vec::new();
    for (index, beacon) in (0_u32..).zip(beacons()) {
        let unit = (r - beacon).normalize();
        let seed = 100 * k + index;
        let pseudorange = (r - beacon).norm() + bias + 2.0 + RANGE_SIGMA_M * noise(seed);
        let rate = unit.dot(&v) + CLOCK_DRIFT_MPS + RATE_SIGMA_MPS * noise(50_000 + seed);
        out.push(observation(
            10 + index,
            k,
            at_ms,
            ObservationValue::Pseudorange {
                emitter_position_m: beacon.into(),
                pseudorange_m: pseudorange,
                sagnac_and_media_correction_m: 2.0,
                variance_m2: RANGE_SIGMA_M * RANGE_SIGMA_M,
            },
        ));
        out.push(observation(
            20 + index,
            k,
            at_ms,
            ObservationValue::PseudorangeRate {
                emitter_position_m: beacon.into(),
                emitter_velocity_mps: [0.0; 3],
                pseudorange_rate_mps: rate,
                variance_m2_per_s2: RATE_SIGMA_MPS * RATE_SIGMA_MPS,
            },
        ));
    }
    out
}

#[test]
fn pseudoranges_and_rates_estimate_the_position_and_the_clock_together() {
    let truth = truth();
    let (r0, v0) = truth[0];
    let mut filter =
        InertialNavigationFilter::new_space(config(), CLOCK, SpaceEnvironment::from_body(MOON));
    filter
        .initialize_space(
            CartesianPosition::new(r0.x + 40.0, r0.y - 30.0, r0.z + 25.0),
            CartesianVelocity::new(v0.x + 0.03, v0.y - 0.02, v0.z + 0.04),
            identity_attitude(),
            SymmetricCov3::from_diagonal(1e4, 1e4, 1e4),
            SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
            SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-6),
            t(0),
            gnss(),
        )
        .expect("plausible");
    let coast = ForceModelCoast {
        acceleration_noise_sigma_mps2_sqrt_hz: 1e-8,
        ..ForceModelCoast::new(10.0)
    };
    filter
        .set_propagation_mode(PropagationMode::ForceModel(coast), t(0))
        .expect("plausible");
    let mut k = 0_u32;
    for at_ms in (STEP_MS..=END_MS).step_by(STEP_MS as usize) {
        k = k.wrapping_add(1);
        for obs in epoch(&truth, k, at_ms) {
            assert_accepted(filter.ingest(&obs, t(at_ms)));
        }
    }
    let state = core(&filter);
    let (r, v) = truth[(END_MS / 1_000) as usize];
    let true_bias = CLOCK_BIAS_M + CLOCK_DRIFT_MPS * END_MS as f64 * 1e-3;
    let position_error = (state.nominal.p - r).norm();
    let velocity_error = (state.nominal.v - v).norm();
    let bias_error = (state.nominal.clock_bias_m - true_bias).abs();
    let drift_error = (state.nominal.clock_drift_mps - CLOCK_DRIFT_MPS).abs();
    assert!(position_error < 3.0, "{position_error}");
    assert!(velocity_error < 0.01, "{velocity_error}");
    assert!(bias_error < 3.0, "{bias_error}");
    assert!(drift_error < 5e-3, "{drift_error}");
    let bias_sigma = state.p[(CLOCK_BIAS, CLOCK_BIAS)].sqrt();
    let drift_sigma = state.p[(CLOCK_DRIFT, CLOCK_DRIFT)].sqrt();
    assert!(bias_sigma < 3.0, "{bias_sigma}");
    assert!(drift_sigma < 5e-3, "{drift_sigma}");
    assert_eq!(filter.rejections().innovation_gate, 0);
    // Without the clock states the same pseudorange is refused by name.
    let mut no_clock = InertialNavigationFilter::new_space(
        InertialConfig {
            clock_model: None,
            ..config()
        },
        CLOCK,
        SpaceEnvironment::from_body(MOON),
    );
    no_clock
        .initialize_space(
            CartesianPosition::new(r0.x, r0.y, r0.z),
            CartesianVelocity::new(v0.x, v0.y, v0.z),
            identity_attitude(),
            SymmetricCov3::from_diagonal(1e4, 1e4, 1e4),
            SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
            SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-6),
            t(0),
            gnss(),
        )
        .expect("plausible");
    let first = epoch(&truth, 1, 0);
    assert_eq!(
        no_clock.ingest(&first[0], t(0)),
        IngestOutcome::Rejected(crate::rejection::RejectionReason::ClockStateDisabled {
            kind: crate::observation::MeasurementKind::Pseudorange
        })
    );
    assert_eq!(no_clock.rejections().clock_state_disabled, 1);
}
