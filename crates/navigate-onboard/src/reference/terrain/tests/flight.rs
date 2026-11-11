//! The vehicle of the tests: a level vehicle that heads north along a
//! meridian at a constant speed, its inertial increments, and its camera.

use std::sync::Arc;

use nalgebra::{UnitQuaternion, Vector3};
use navigate_camera::{CameraMount, ImagerModel, Intrinsics};
use navigate_contract::{
    AttitudeQuaternion, ClockDomainId, GeodeticPosition, MonotonicNanos, NavigationSolution,
    NedVelocity, SourceId, SymmetricCov3,
};
use navigate_fusion::FogmBias;

use super::world::{TODAY, World, camera_model, frame_at};
use crate::{
    AcceptanceRecord, Alignment, ErrorBudget, MapReport, Navigator, OnboardError,
    ReferenceNavigator, SensorData, SensorInput, SensorKind, SensorSpec, SensorSuite,
    TerrainConfig, TerrainMapRequest,
};

pub(crate) const CLOCK: ClockDomainId = ClockDomainId::new(7);
pub(crate) const IMU: SourceId = SourceId::new(1);
pub(crate) const CAMERA: SourceId = SourceId::new(6);
/// One inertial step: 50 Hz.
pub(crate) const STEP_NS: u64 = 20_000_000;
/// Inertial steps in one second.
pub(crate) const RATE: u64 = 50;

const EARTH_RATE: f64 = 7.292_115e-5;

/// The time of an inertial step.
pub(crate) fn at(step: u64) -> MonotonicNanos {
    MonotonicNanos::from_nanos(1_000_000_000 + step * STEP_NS)
}

/// The meridian radius of curvature of WGS84 at a latitude, metres.
fn meridian_radius_m(lat_rad: f64) -> f64 {
    let (a, e2) = (6_378_137.0, 6.694_379_990_141_316e-3);
    a * (1.0 - e2) / (1.0 - e2 * lat_rad.sin().powi(2)).powf(1.5)
}

/// A level vehicle that heads north from `start` at `speed_m_s`. Its body
/// axes are the north, east, down axes.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Flight {
    /// `[lat, lon]` in degrees and the height in metres at step zero.
    pub start: [f64; 3],
    pub speed_m_s: f64,
}

impl Flight {
    /// The true position of the body reference point at a step.
    pub(crate) fn body(&self, step: u64) -> [f64; 3] {
        let [lat, lon, h] = self.start;
        let t = (step * STEP_NS) as f64 * 1e-9;
        let north_rad = self.speed_m_s * t / (meridian_radius_m(lat.to_radians()) + h);
        [lat + north_rad.to_degrees(), lon, h]
    }

    /// The inertial increment of the step that ends at `step`: the Earth
    /// rate and the transport rate, and the reaction to normal gravity
    /// (Somigliana, with the free-air gradient) with the Coriolis and
    /// centripetal terms of the motion.
    pub(crate) fn increment(&self, step: u64) -> SensorInput {
        let dt = STEP_NS as f64 * 1e-9;
        let [lat, _, h] = self.body(step);
        let (s, c) = lat.to_radians().sin_cos();
        let g = 9.780_325_335_9 * (1.0 + 0.001_931_852_652_41 * s * s)
            / (1.0 - 0.006_694_379_990_13 * s * s).sqrt()
            - 3.086e-6 * h;
        let v = self.speed_m_s;
        let transport = v / (meridian_radius_m(lat.to_radians()) + h);
        SensorInput {
            sensor: IMU,
            acquired_at: at(step),
            delivered_at: at(step),
            data: SensorData::ImuIncrement {
                dt_s: dt,
                delta_theta_rad: [EARTH_RATE * c * dt, -transport * dt, -EARTH_RATE * s * dt],
                delta_velocity_m_s: [
                    0.0,
                    -2.0 * EARTH_RATE * s * v * dt,
                    (v * transport - g) * dt,
                ],
            },
        }
    }

    /// The frame that the camera took at `step`, delivered `lag` steps
    /// later.
    pub(crate) fn frame(
        &self,
        world: &World,
        mount: &CameraMount,
        step: u64,
        lag: u64,
    ) -> SensorInput {
        let level = UnitQuaternion::identity();
        let camera = camera_model();
        SensorInput {
            sensor: CAMERA,
            acquired_at: at(step),
            delivered_at: at(step + lag),
            data: SensorData::TerrainFrame {
                width: camera.width,
                height: camera.height,
                luma: frame_at(world, &camera, self.body(step), &level, mount, step),
            },
        }
    }

    /// The error of a solution against the truth at `step`: north, east,
    /// and up, in metres.
    pub(crate) fn error_m(&self, solution: &NavigationSolution, step: u64) -> Vector3<f64> {
        let [lat, lon, h] = self.body(step);
        let p = &solution.position;
        let radius = meridian_radius_m(lat.to_radians()) + h;
        Vector3::new(
            (p.latitude_rad - lat.to_radians()) * radius,
            (p.longitude_rad - lon.to_radians()) * 6_378_137.0 * lat.to_radians().cos(),
            p.altitude_m - h,
        )
    }
}

/// Shows the `tracing` events of a test with `--nocapture`.
pub(crate) fn show_events() {
    tracing_subscriber::fmt()
        .with_test_writer()
        .with_max_level(tracing::Level::INFO)
        .try_init()
        .ok();
}

/// The suite of the tests: an IMU and one terrain camera on `mount`.
pub(crate) fn suite(imager: ImagerModel, mount: CameraMount) -> SensorSuite {
    let imu = SensorSpec {
        id: IMU,
        kind: SensorKind::Imu,
        imager: None,
        mount: None,
        rate_hz: RATE as f64,
        budget: ErrorBudget::Imu {
            accel_noise_mps2_sqrt_hz: 6e-4,
            gyro_noise_rad_s_sqrt_hz: 3e-6,
            accel_bias: FogmBias::new(3_600.0, 5e-4),
            gyro_bias: FogmBias::new(3_600.0, 5e-6),
        },
    };
    let camera = SensorSpec {
        id: CAMERA,
        kind: SensorKind::TerrainCamera,
        imager: Some(imager),
        mount: Some(mount),
        rate_hz: 1.0,
        budget: ErrorBudget::TerrainCamera {
            calibration_sigma_rad: 2e-4,
        },
    };
    SensorSuite::new(vec![imu, camera]).unwrap()
}

/// The imager of [`camera_model`].
pub(crate) fn imager() -> ImagerModel {
    let k = camera_model();
    ImagerModel::pinhole(Intrinsics {
        width: k.width,
        height: k.height,
        fx: k.fx,
        fy: k.fy,
        cx: k.cx,
        cy: k.cy,
    })
}

/// A tuning for fast tests: a short ray march over the gentle relief of
/// the test world, and references that interpolate between rays four
/// pixels apart.
pub(crate) fn fast() -> TerrainConfig {
    TerrainConfig {
        far_m: 2_500.0,
        steps: 16,
        stride: 4,
        luma_edge: u8::MAX,
        ..TerrainConfig::default()
    }
}

/// The request of the tests for the package of `world`.
pub(crate) fn request(world: &World) -> TerrainMapRequest {
    TerrainMapRequest {
        config: fast(),
        ..TerrainMapRequest::new(world.package, TODAY)
    }
}

/// A navigator for `flight` with the camera on `mount`, aligned `off`
/// metres (north, east, up) from the truth at step zero, with a horizontal
/// one-sigma of `sigma_m` on each axis.
pub(crate) fn navigator(
    flight: &Flight,
    mount: CameraMount,
    off: [f64; 3],
    sigma_m: f64,
) -> ReferenceNavigator {
    let mut navigator = ReferenceNavigator::new(suite(imager(), mount), CLOCK, None).unwrap();
    let [lat, lon, h] = flight.start;
    let radius = meridian_radius_m(lat.to_radians()) + h;
    let east_radius = 6_378_137.0 * lat.to_radians().cos();
    navigator
        .align(&Alignment {
            position: GeodeticPosition::new(
                lat.to_radians() + off[0] / radius,
                lon.to_radians() + off[1] / east_radius,
                h + off[2],
            ),
            velocity: NedVelocity::new(flight.speed_m_s, 0.0, 0.0),
            attitude: AttitudeQuaternion::new(1.0, 0.0, 0.0, 0.0),
            position_cov: SymmetricCov3::from_diagonal(
                sigma_m * sigma_m,
                sigma_m * sigma_m,
                sigma_m * sigma_m,
            ),
            velocity_cov: SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
            attitude_cov: SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-5),
            at: at(0),
        })
        .unwrap();
    navigator
}

/// Loads the package of `world` with the request of the tests.
pub(crate) fn load(navigator: &mut ReferenceNavigator, world: &World) -> MapReport {
    navigator
        .load_terrain_map(Arc::clone(&world.store), &request(world), None)
        .unwrap()
        .clone()
}

/// Flies the steps `from..=to`: each inertial increment, and a frame at
/// each step that `frames` selects, delivered at once.
pub(crate) fn fly(
    navigator: &mut dyn Navigator,
    flight: &Flight,
    world: &World,
    mount: &CameraMount,
    steps: core::ops::RangeInclusive<u64>,
    frames: impl Fn(u64) -> bool,
    sink: &mut Vec<AcceptanceRecord>,
) -> Result<(), OnboardError> {
    for step in steps {
        navigator.ingest(&flight.increment(step), sink)?;
        if frames(step) {
            navigator.ingest(&flight.frame(world, mount, step, 0), sink)?;
        }
    }
    Ok(())
}
