//! Per-frame heights: values only with metric evidence, reasons otherwise.

use crate::support::*;
use nalgebra::Vector3;
use navigate_visual_session::{
    AxisRange, CameraMount, Height, HeightBasis, HeightInputs, HeightSample, HeightSensor,
    RangeSource, TerrainSample, TerrainSource, Unlocated, Unobservable, VerticalReference,
};

fn terrain(elevation_m: f64, vertical: VerticalReference) -> TerrainSample {
    TerrainSample {
        elevation_m,
        sigma_m: 1.0,
        vertical,
        source: TerrainSource::MapElevation { map: map() },
    }
}

fn value(height: &Height) -> f64 {
    match height {
        Height::Estimated { value_m, .. } => *value_m,
        Height::Unobservable(reason) => panic!("expected a value, found {reason:?}"),
    }
}

#[test]
fn no_anchor_and_no_sensor_gives_reasons_and_no_value() {
    let mut flight = Flight::standard();
    let k = flight.fly_to(nadir(Vector3::new(0.0, 0.0, 120.0), 0.0));
    let inputs = HeightInputs {
        terrain: Some(terrain(10.0, VerticalReference::Wgs84Ellipsoid)),
        ..HeightInputs::default()
    };
    let height = flight.session.height_at(&k, &inputs).unwrap();
    assert_eq!(
        height.camera_height,
        Height::Unobservable(Unobservable::NoHeight(Unlocated::NoAnchor))
    );
    assert_eq!(
        height.camera_agl,
        Height::Unobservable(Unobservable::NoHeight(Unlocated::NoAnchor))
    );
    assert_eq!(
        height.axis_range,
        Height::Unobservable(Unobservable::NoRange)
    );
}

#[test]
fn anchored_frames_separate_camera_body_vertical_and_slant_heights() {
    let mut flight = Flight::standard();
    let k = flight.fly_to(nadir(Vector3::new(0.0, 0.0, 120.0), 0.7));
    flight
        .session
        .submit_anchor(anchor(&flight, &k, Vector3::zeros()))
        .unwrap();
    let inputs = HeightInputs {
        terrain: Some(terrain(15.0, VerticalReference::Wgs84Ellipsoid)),
        axis_range: Some(AxisRange {
            range_m: 105.0,
            sigma_m: 1.0,
            source: RangeSource::RenderedDepth { map: map() },
        }),
        ..HeightInputs::default()
    };
    let height = flight.session.height_at(&k, &inputs).unwrap();
    assert!((value(&height.camera_height) - 120.0).abs() < 0.5);
    assert!(
        (value(&height.body_height) - 120.2).abs() < 0.5,
        "the body origin is 0.2 m above the camera"
    );
    assert!((value(&height.camera_agl) - 105.0).abs() < 0.5);
    assert!(matches!(
        height.camera_agl,
        Height::Estimated {
            basis: HeightBasis::TerrainDifference,
            ..
        }
    ));
    assert_eq!(value(&height.axis_range), 105.0);

    let other = HeightInputs {
        terrain: Some(terrain(
            15.0,
            VerticalReference::Orthometric {
                geoid: "EGM2008".into(),
            },
        )),
        ..HeightInputs::default()
    };
    let shared_map = flight.session.height_at(&k, &other).unwrap();
    assert!(
        matches!(shared_map.camera_agl, Height::Estimated { .. }),
        "one map package shares its datum"
    );
    let mut host_terrain = terrain(
        15.0,
        VerticalReference::Orthometric {
            geoid: "EGM2008".into(),
        },
    );
    host_terrain.source = TerrainSource::Host {
        name: "survey".into(),
    };
    let mismatch = flight
        .session
        .height_at(
            &k,
            &HeightInputs {
                terrain: Some(host_terrain),
                ..HeightInputs::default()
            },
        )
        .unwrap();
    assert_eq!(
        mismatch.camera_agl,
        Height::Unobservable(Unobservable::VerticalMismatch)
    );
    assert_eq!(height.vertical, VerticalReference::Wgs84Ellipsoid);
}

#[test]
fn a_sensor_height_gives_body_values_and_a_gimbal_needs_its_attitude() {
    let mut config_flight = Flight::standard();
    let k = config_flight.fly_to(nadir(Vector3::new(0.0, 0.0, 80.0), 0.0));
    let inputs = HeightInputs {
        terrain: Some(TerrainSample {
            elevation_m: 30.0,
            sigma_m: 2.0,
            vertical: VerticalReference::Wgs84Ellipsoid,
            source: TerrainSource::Host { name: "dem".into() },
        }),
        sensor_height: Some(HeightSample {
            height_m: 110.0,
            sigma_m: 3.0,
            vertical: VerticalReference::Wgs84Ellipsoid,
            sensor: HeightSensor::Gnss,
        }),
        ..HeightInputs::default()
    };
    let height = config_flight.session.height_at(&k, &inputs).unwrap();
    assert_eq!(value(&height.body_agl), 80.0);
    assert!(matches!(
        height.camera_height,
        Height::Unobservable(Unobservable::NoHeight(Unlocated::NoAnchor))
    ));

    let mut calibration = calibration();
    calibration.mount = CameraMount::Gimbal {
        body_from_base: navigate_visual_session::Pose::identity(),
        clock: navigate_contract::ClockDomainId::new(7),
    };
    let mut gimbal = Flight::new(navigate_visual_session::SessionConfig::standard());
    gimbal.session = navigate_visual_session::VisualSession::new(
        navigate_visual_session::SessionConfig::standard(),
        calibration,
    )
    .unwrap();
    let g = gimbal.fly_to(nadir(Vector3::new(0.0, 0.0, 80.0), 0.0));
    gimbal
        .session
        .submit_anchor(anchor(&gimbal, &g, Vector3::zeros()))
        .unwrap();
    let without = gimbal
        .session
        .height_at(&g, &HeightInputs::default())
        .unwrap();
    assert_eq!(
        without.body_height,
        Height::Unobservable(Unobservable::GimbalMissing)
    );
    assert!(matches!(without.camera_height, Height::Estimated { .. }));
}
