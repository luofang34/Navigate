//! The terrain camera through the exported items only, as a host uses it:
//! declare the camera, open a store, load the map, give a frame.

use std::collections::BTreeSet;
use std::io::Cursor;
use std::sync::Arc;

use image::{ImageFormat, Rgb, RgbImage};
use nalgebra::Vector2;
use navigate_camera::{CameraMount, ImagerModel, Intrinsics};
use navigate_contract::{
    AttitudeQuaternion, ClockDomainId, GeodeticPosition, MonotonicNanos, NedVelocity, SourceId,
    SymmetricCov3,
};
use navigate_data::corridor::corridor_tiles;
use navigate_data::{
    Accuracy, DayNumber, Descriptor, DirectoryStore, Encoding, HorizontalDatum, LayerKey,
    LayerKind, LayerSpec, Package, PackageWriter, Revision, Sha256Digest, SourceKey, VerticalDatum,
};
use navigate_fusion::FogmBias;
use navigate_onboard::{
    AcceptanceRecord, Alignment, AssumedAccuracy, ErrorBudget, Navigator, Outcome,
    ReferenceNavigator, SensorData, SensorInput, SensorKind, SensorSpec, SensorSuite,
    TerrainMapRequest,
};
use navigate_visual::{CameraModel, CameraPose, LocalFrame};
use navigate_visual_render::keyframes::camera_orientation;
use navigate_visual_render::{SceneRequest, SparseSceneSampler, scene_around};

const CENTRE: [f64; 2] = [41.70, -1.00];
const HEIGHT_M: f64 = 1_050.0;
const IMU: SourceId = SourceId::new(1);
const CAMERA: SourceId = SourceId::new(2);

#[cfg(test)]
fn layer(kind: LayerKind) -> LayerKey {
    LayerKey {
        kind,
        source: SourceKey::Synthetic,
    }
}

#[cfg(test)]
fn png(image: &RgbImage) -> Vec<u8> {
    let mut out = Cursor::new(Vec::new());
    image.write_to(&mut out, ImageFormat::Png).unwrap();
    out.into_inner()
}

/// Ground of smooth fields, 80 m to 250 m across at zoom 15.
#[cfg(test)]
fn luma(x: u32, y: u32) -> u8 {
    let wave = |v: u32, period: f64| (f64::from(v) / period * core::f64::consts::TAU).sin();
    let field = wave(x, 23.0) * wave(y, 31.0) + 0.7 * wave(x + 2 * y, 57.0) * wave(3 * x + y, 71.0);
    (128.0 + 70.0 * field).clamp(0.0, 255.0) as u8
}

/// A package of synthetic imagery over flat ground at 250 m, with no
/// stated accuracy, in a directory store.
#[cfg(test)]
fn install(directory: &std::path::Path) -> Sha256Digest {
    let mut store = DirectoryStore::new(directory);
    let mut writer = PackageWriter::new();
    let spec = |kind, encoding, vertical_datum| LayerSpec {
        key: layer(kind),
        revision: Revision::Unversioned,
        horizontal_datum: HorizontalDatum::Wgs84,
        vertical_datum,
        encoding,
        tile_pixels: 256,
        accuracy: Accuracy::unknown(),
        correlation_zoom: 16,
    };
    writer
        .add_layer(spec(LayerKind::Imagery, Encoding::Png, None))
        .unwrap();
    let ellipsoidal = Some(VerticalDatum::Ellipsoidal);
    writer
        .add_layer(spec(
            LayerKind::Elevation,
            Encoding::TerrariumPng,
            ellipsoidal,
        ))
        .unwrap();
    let mut tiles = BTreeSet::new();
    corridor_tiles(&[CENTRE], 2_500.0, 15, &mut tiles).unwrap();
    for a in tiles {
        let tile = RgbImage::from_fn(256, 256, |x, y| {
            let v = luma(a.x * 256 + x, a.y * 256 + y);
            Rgb([v, v, v])
        });
        let key = layer(LayerKind::Imagery);
        pollster::block_on(writer.add_tile(&mut store, key, a, &png(&tile))).unwrap();
    }
    let mut tiles = BTreeSet::new();
    corridor_tiles(&[CENTRE], 5_000.0, 12, &mut tiles).unwrap();
    // Terrarium: 250 m is (128, 250, 0).
    let flat = png(&RgbImage::from_pixel(256, 256, Rgb([128, 250, 0])));
    for a in tiles {
        let key = layer(LayerKind::Elevation);
        pollster::block_on(writer.add_tile(&mut store, key, a, &flat)).unwrap();
    }
    pollster::block_on(writer.seal(&mut store, &Descriptor::default())).unwrap()
}

#[cfg(test)]
fn intrinsics() -> Intrinsics {
    Intrinsics {
        width: 160,
        height: 120,
        fx: 150.0,
        fy: 150.0,
        cx: 79.5,
        cy: 59.5,
    }
}

/// The frame of a nadir camera above the centre, with the nose to the
/// north.
#[cfg(test)]
fn frame(store: &DirectoryStore, package: Sha256Digest) -> Vec<u8> {
    let k = intrinsics();
    let camera = CameraModel {
        width: k.width,
        height: k.height,
        fx: k.fx,
        fy: k.fy,
        cx: k.cx,
        cy: k.cy,
    };
    let request = SceneRequest {
        imagery_layer: layer(LayerKind::Imagery),
        imagery_zoom: 15,
        elevation_layer: layer(LayerKind::Elevation),
        elevation_zoom: 12,
        centre: CENTRE,
        half_side_m: 1_500.0,
    };
    let opened = pollster::block_on(Package::open(store, package)).unwrap();
    let window = pollster::block_on(scene_around(&opened, store, &request, None)).unwrap();
    let pose = CameraPose {
        position: LocalFrame::wgs84_enu(CENTRE[0], CENTRE[1])
            .unwrap()
            .local([CENTRE[0], CENTRE[1], HEIGHT_M])
            .unwrap(),
        orientation: camera_orientation(0.0, core::f64::consts::FRAC_PI_2),
    };
    let (view, _) = SparseSceneSampler::new(&window.scene, 2_500.0, 16, 10)
        .unwrap()
        .render_adaptive(&camera, &pose, 2, u8::MAX, 0.05)
        .unwrap();
    view.image.into_raw()
}

/// The suite of a host: an IMU and a nadir terrain camera, given as a
/// forward camera pitched down by a right angle.
#[cfg(test)]
fn suite() -> SensorSuite {
    let sensor = |id, kind, budget| SensorSpec {
        id,
        kind,
        imager: None,
        mount: None,
        rate_hz: 50.0,
        budget,
    };
    let imu = ErrorBudget::Imu {
        accel_noise_mps2_sqrt_hz: 6e-4,
        gyro_noise_rad_s_sqrt_hz: 3e-6,
        accel_bias: FogmBias::new(3_600.0, 5e-4),
        gyro_bias: FogmBias::new(3_600.0, 5e-6),
    };
    let budget = ErrorBudget::TerrainCamera {
        calibration_sigma_rad: 2e-4,
    };
    let camera = SensorSpec {
        imager: Some(ImagerModel::pinhole(intrinsics())),
        mount: Some(CameraMount::rotated_forward(
            0.0,
            -core::f64::consts::FRAC_PI_2,
            0.0,
        )),
        ..sensor(CAMERA, SensorKind::TerrainCamera, budget)
    };
    SensorSuite::new(vec![sensor(IMU, SensorKind::Imu, imu), camera]).unwrap()
}

#[test]
fn a_host_declares_the_camera_loads_the_map_and_gives_a_frame() {
    let directory = tempfile::tempdir().unwrap();
    let package = install(directory.path());
    let mut navigator = ReferenceNavigator::new(suite(), ClockDomainId::new(3), None).unwrap();

    // The host opens the store, and states the day and the accuracy that
    // the package does not state.
    let store = Arc::new(DirectoryStore::new(directory.path()));
    let mut request = TerrainMapRequest::new(package, DayNumber(9_800));
    request.assumed_accuracy = Some(AssumedAccuracy {
        horizontal_sigma_m: 3.0,
        vertical_sigma_m: 5.0,
    });
    let report = navigator
        .load_terrain_map(Arc::clone(&store), &request, None)
        .unwrap();
    assert_eq!(report.package, package);
    assert_eq!(report.imagery.zooms, [15]);

    // Aligned 25 m north and 20 m east of the truth.
    let at = MonotonicNanos::from_nanos(1_000_000_000);
    let (north_radius_m, east_radius_m) = (6_364_000.0, 4_762_000.0);
    navigator
        .align(&Alignment {
            position: GeodeticPosition::new(
                CENTRE[0].to_radians() + 25.0 / north_radius_m,
                CENTRE[1].to_radians() + 20.0 / east_radius_m,
                HEIGHT_M,
            ),
            velocity: NedVelocity::new(0.0, 0.0, 0.0),
            attitude: AttitudeQuaternion::new(1.0, 0.0, 0.0, 0.0),
            position_cov: SymmetricCov3::from_diagonal(900.0, 900.0, 100.0),
            velocity_cov: SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
            attitude_cov: SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-5),
            at,
            imu_bias_prior: None,
        })
        .unwrap();
    let input = SensorInput {
        sensor: CAMERA,
        acquired_at: at,
        delivered_at: at,
        data: SensorData::TerrainFrame {
            width: 160,
            height: 120,
            luma: frame(&store, package),
        },
    };
    let mut records: Vec<AcceptanceRecord> = Vec::new();
    navigator.ingest(&input, &mut records).unwrap();
    assert_eq!(records.len(), 1);
    assert_eq!(records[0].measurement, "terrain_position");
    assert_eq!(records[0].outcome, Outcome::Accepted);
    assert_eq!(records[0].context, Some("render"));
    let solution = navigator.solution(at).unwrap();
    let error_m = Vector2::new(
        (solution.position.latitude_rad - CENTRE[0].to_radians()) * north_radius_m,
        (solution.position.longitude_rad - CENTRE[1].to_radians()) * east_radius_m,
    );
    // The alignment was 32 m off. Three pixels of this camera are 16 m.
    assert!(error_m.norm() < 16.0, "{error_m}");
}
