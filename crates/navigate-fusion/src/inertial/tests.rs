mod magnetic;
mod schuler;
mod space;
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
use super::{ImuSample, InertialConfig, InertialNavigationFilter};
use crate::environment::{GravityField, PlanetaryBody};
use crate::rejection::IngestOutcome;

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

/// The IMU rates of a vehicle at rest at the plane origin with the body
/// axes along the plane axes: `f = −g_app(0)` and `ω_ib = ω_plane`.
fn rest_rates(filter: &InertialNavigationFilter) -> ([f64; 3], [f64; 3]) {
    let regime = &core(filter).regime;
    let (g, _) = regime.gravity(&Vector3::zeros());
    let omega = regime.frame_rotation();
    ((-g).into(), omega.into())
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
