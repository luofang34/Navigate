//! The Cargo feature `terrain-camera` adds the terrain means and changes
//! nothing else.

use navigate_camera::{CameraMount, ImagerModel, Intrinsics};
use navigate_contract::{ClockDomainId, SourceId};
use navigate_fusion::FogmBias;
use navigate_onboard::{ErrorBudget, ReferenceNavigator, SensorKind, SensorSpec, SensorSuite};

const IMU: SourceId = SourceId::new(1);
const CAMERA: SourceId = SourceId::new(2);

/// A suite with an IMU and, if `camera`, a nadir terrain camera.
#[cfg(test)]
fn suite(camera: bool) -> SensorSuite {
    let mut sensors = vec![SensorSpec {
        id: IMU,
        kind: SensorKind::Imu,
        imager: None,
        mount: None,
        rate_hz: 50.0,
        budget: ErrorBudget::Imu {
            accel_noise_mps2_sqrt_hz: 6e-4,
            gyro_noise_rad_s_sqrt_hz: 3e-6,
            accel_bias: FogmBias::new(3_600.0, 5e-4),
            gyro_bias: FogmBias::new(3_600.0, 5e-6),
        },
    }];
    if camera {
        let intrinsics = Intrinsics::centred(160, 120, 1.0).unwrap();
        sensors.push(SensorSpec {
            id: CAMERA,
            kind: SensorKind::TerrainCamera,
            imager: Some(ImagerModel::pinhole(intrinsics)),
            mount: Some(CameraMount::nadir()),
            rate_hz: 1.0,
            budget: ErrorBudget::TerrainCamera {
                calibration_sigma_rad: 2e-4,
            },
        });
    }
    SensorSuite::new(sensors).unwrap()
}

#[cfg(not(feature = "terrain-camera"))]
#[test]
fn a_terrain_camera_without_the_feature_is_refused_by_name() {
    use navigate_onboard::OnboardError;
    assert_eq!(
        ReferenceNavigator::new(suite(true), ClockDomainId::new(3), None).err(),
        Some(OnboardError::FeatureDisabled {
            sensor: CAMERA,
            kind: "terrain camera",
            feature: "terrain-camera",
        })
    );
    assert!(ReferenceNavigator::new(suite(false), ClockDomainId::new(3), None).is_ok());
}

/// The same alignment and the same inertial increments give the same
/// solution with a declared terrain camera that gives no frame and
/// without one: the means changes no value by its presence.
#[cfg(feature = "terrain-camera")]
#[test]
fn a_declared_terrain_camera_without_frames_changes_no_value() {
    use navigate_contract::{
        AttitudeQuaternion, GeodeticPosition, MonotonicNanos, NedVelocity, SymmetricCov3,
    };
    use navigate_onboard::{Alignment, Navigator, SensorData, SensorInput};
    let at = |step: u64| MonotonicNanos::from_nanos(1_000_000_000 + step * 20_000_000);
    let run = |camera: bool| {
        let mut navigator =
            ReferenceNavigator::new(suite(camera), ClockDomainId::new(3), None).unwrap();
        navigator
            .align(&Alignment {
                position: GeodeticPosition::new(0.7, -0.02, 1_000.0),
                velocity: NedVelocity::new(30.0, 5.0, 0.0),
                attitude: AttitudeQuaternion::new(1.0, 0.0, 0.0, 0.0),
                position_cov: SymmetricCov3::from_diagonal(25.0, 25.0, 9.0),
                velocity_cov: SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
                attitude_cov: SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-5),
                at: at(0),
                imu_bias_prior: None,
            })
            .unwrap();
        for step in 1..=100 {
            let increment = SensorInput {
                sensor: IMU,
                acquired_at: at(step),
                delivered_at: at(step),
                data: SensorData::ImuIncrement {
                    dt_s: 0.02,
                    delta_theta_rad: [1e-6, -2e-6, 3e-6],
                    delta_velocity_m_s: [0.001, -0.002, -0.196],
                },
            };
            navigator.ingest(&increment, &mut ()).unwrap();
        }
        navigator.solution(at(100)).unwrap()
    };
    assert_eq!(run(true), run(false));
}
