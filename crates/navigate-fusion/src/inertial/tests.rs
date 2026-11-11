mod bias_prior;
mod checkpoints;
mod coast;
mod deep_space;
mod frame_change;
mod increment;
mod lunar_beacon;
mod magnetic;
mod measurements;
mod orbit_determination;
mod past;
mod polar;
mod regimes;
mod schuler;
mod sequence;
mod space;
mod stochastic;
mod surface;
mod terrain;
mod transition;

use nalgebra::Vector3;
use navigate_contract::{
    AttitudeQuaternion, ClockDomainId, FrameId, GeodeticPosition, MonotonicNanos, NedVelocity,
    ObservationStamp, SensorClass, SourceComposition, SourceEpoch, SourceId, SymmetricCov3,
    WrappingSequence,
};

use super::estimate::InertialCore;
use super::nominal::{ErrorMat, ErrorVec, InertialNominal, rotation_vector};
use super::{ImuSample, InertialConfig, InertialNavigationFilter};
use crate::rejection::IngestOutcome;
use navigate_astrodynamics::{GravityField, PlanetaryBody};

const CLOCK: ClockDomainId = ClockDomainId::new(11);
/// The source of the IMU stream in every test.
const IMU_SOURCE: u32 = 100;

fn t(ms: u64) -> MonotonicNanos {
    MonotonicNanos::from_nanos(ms.saturating_mul(1_000_000))
}

fn stamp(source: u32, sequence: u32, at_ms: u64) -> ObservationStamp {
    ObservationStamp::new(
        SourceId::new(source),
        SourceEpoch::new(1),
        WrappingSequence::new(sequence),
        t(at_ms),
        CLOCK,
    )
}

fn earth_origin() -> GeodeticPosition {
    GeodeticPosition::new(47.0_f64.to_radians(), 8.0_f64.to_radians(), 500.0)
}

fn identity_attitude() -> AttitudeQuaternion {
    AttitudeQuaternion::new(1.0, 0.0, 0.0, 0.0)
}

fn gnss() -> SourceComposition {
    SourceComposition::of(SensorClass::Gnss)
}

fn inertial() -> SourceComposition {
    SourceComposition::of(SensorClass::Inertial)
}

/// A surface filter initialized at rest at `origin` at time 0 with the
/// body axes along the local NED axes.
fn surface_filter(
    config: InertialConfig,
    body: PlanetaryBody,
    origin: GeodeticPosition,
) -> InertialNavigationFilter {
    let mut filter = InertialNavigationFilter::new_surface(config, CLOCK, body);
    filter
        .initialize_surface(
            origin,
            NedVelocity::new(0.0, 0.0, 0.0),
            identity_attitude(),
            SymmetricCov3::from_diagonal(25.0, 25.0, 25.0),
            SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
            SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-6),
            t(0),
            gnss(),
        )
        .expect("plausible initial state");
    filter
}

fn core(filter: &InertialNavigationFilter) -> &InertialCore {
    filter.core.as_ref().expect("initialized")
}

/// The IMU rates of a vehicle at rest in the body-fixed frame at the
/// nominal position and attitude: `f = −Cᵀ·g_app(p)` and `ω_ib = Cᵀ·ω`.
fn rest_rates(filter: &InertialNavigationFilter) -> ([f64; 3], [f64; 3]) {
    let state = core(filter);
    let (g, _) = state.regime.gravity(&state.nominal.p);
    let omega = state.regime.frame_rotation();
    let c = state.nominal.c_b_to_f();
    ((c.transpose() * -g).into(), (c.transpose() * omega).into())
}

/// The body-fixed position of `position` on WGS84.
fn earth_body_fixed(position: &GeodeticPosition) -> Vector3<f64> {
    super::local_level::body_fixed(position, navigate_geodesy::BodyEllipsoid::WGS84)
}

fn imu(sequence: u32, at_ms: u64, rates: ([f64; 3], [f64; 3]), dt_s: f64) -> ImuSample {
    ImuSample::new(
        stamp(IMU_SOURCE, sequence, at_ms),
        rates.0,
        rates.1,
        dt_s,
        inertial(),
    )
}

fn imu_in(
    sequence: u32,
    at_ms: u64,
    rates: ([f64; 3], [f64; 3]),
    dt_s: f64,
    frame: FrameId,
) -> ImuSample {
    imu(sequence, at_ms, rates, dt_s).with_frame(frame)
}

fn assert_accepted(outcome: IngestOutcome) {
    assert!(
        outcome.is_accepted(),
        "expected acceptance, got {outcome:?}"
    );
}

/// The error state of `x` relative to `reference`, with the attitude
/// as the body-axis rotation vector.
fn error_between(x: &InertialNominal, reference: &InertialNominal) -> ErrorVec {
    let mut dx = ErrorVec::zeros();
    dx.fixed_rows_mut::<3>(0).copy_from(&(x.p - reference.p));
    dx.fixed_rows_mut::<3>(3).copy_from(&(x.v - reference.v));
    dx.fixed_rows_mut::<3>(6)
        .copy_from(&rotation_vector(&(reference.q.inverse() * x.q)));
    dx.fixed_rows_mut::<3>(9)
        .copy_from(&(x.b_a - reference.b_a));
    dx.fixed_rows_mut::<3>(12)
        .copy_from(&(x.b_g - reference.b_g));
    dx[15] = x.b_alt - reference.b_alt;
    dx[16] = x.b_mag - reference.b_mag;
    dx.fixed_rows_mut::<3>(17)
        .copy_from(&(x.a_dmc - reference.a_dmc));
    dx[20] = x.clock_bias_m - reference.clock_bias_m;
    dx[21] = x.clock_drift_mps - reference.clock_drift_mps;
    dx
}

/// A deterministic noise sample with zero mean and unit variance: a
/// uniform value from a fixed integer hash, scaled by √12.
fn noise(k: u32) -> f64 {
    let hash = k.wrapping_mul(2_654_435_761).wrapping_add(0x9E37_79B9) >> 8;
    let uniform = f64::from(hash) / f64::from(1_u32 << 24) - 0.5;
    uniform * 12.0_f64.sqrt()
}
