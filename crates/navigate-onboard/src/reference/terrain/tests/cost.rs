//! The cost of one frame, with and without keyframes, and where the time
//! goes. The test states each time through `tracing` and asserts none: run
//! it with `--release -- --nocapture cost`. An optimized build measures a
//! camera of 800 by 600 pixels, which the navigator matches at 320 by 240.
//! A build without optimization measures the small camera of the other
//! tests, so that the test stays short there.

use std::sync::Arc;
use std::time::Instant;

use image::GrayImage;
use image::imageops::{FilterType, resize};
use navigate_camera::{CameraMount, ImagerModel, Intrinsics};
use navigate_data::Sha256Digest;
use navigate_visual::{CameraModel, Frame, FrameStamp, PosePrior};

use super::fixes::{HEIGHT_M, belly, terrain};
use super::flight::{CAMERA, CLOCK, Flight, at, show_events, suite};
use super::world::{
    CENTRE, TODAY, World, WorldSpec, camera_model, correlation_cell, frame_at, frame_with_rays,
};
use crate::reference::terrain::belief::Belief;
use crate::reference::terrain::view::{Sight, Views, model};
use crate::{
    Alignment, Navigator, Outcome, ReferenceNavigator, SensorData, SensorInput, TerrainConfig,
    TerrainMapRequest,
};

/// The declared camera of the cost measurement, 56° across: 800 by 600
/// pixels in an optimized build, so that the working resolution of 320 by
/// 240 applies.
#[cfg(test)]
fn camera() -> CameraModel {
    if cfg!(debug_assertions) {
        return camera_model();
    }
    CameraModel {
        width: 800,
        height: 600,
        fx: 750.0,
        fy: 750.0,
        cx: 399.5,
        cy: 299.5,
    }
}

#[cfg(test)]
fn milliseconds<T>(work: impl FnOnce() -> T) -> (T, f64) {
    let start = Instant::now();
    let value = work();
    (value, start.elapsed().as_secs_f64() * 1e3)
}

/// A hovering navigator with the cost camera, aligned 20 m north of the
/// centre, with the default tuning of the means.
#[cfg(test)]
fn aligned(world: &World, products: Option<Sha256Digest>) -> ReferenceNavigator {
    let hover = Flight {
        start: [CENTRE[0], CENTRE[1], HEIGHT_M],
        speed_m_s: 0.0,
    };
    aligned_on(world, products, &hover, [20.0, 0.0, 0.0])
}

/// A navigator with the cost camera for `flight`, aligned `off` metres
/// (north, east, up) from its start, with the default tuning of the means.
#[cfg(test)]
fn aligned_on(
    world: &World,
    products: Option<Sha256Digest>,
    flight: &Flight,
    off: [f64; 3],
) -> ReferenceNavigator {
    let k = camera();
    let imager = ImagerModel::pinhole(Intrinsics {
        width: k.width,
        height: k.height,
        fx: k.fx,
        fy: k.fy,
        cx: k.cx,
        cy: k.cy,
    });
    let mut navigator = ReferenceNavigator::new(suite(imager, belly()), CLOCK, None).unwrap();
    let (lat, lon) = (flight.start[0].to_radians(), flight.start[1].to_radians());
    navigator
        .align(&Alignment {
            position: navigate_contract::GeodeticPosition::new(
                lat + off[0] / 6_364_000.0,
                lon + off[1] / (6_378_137.0 * lat.cos()),
                flight.start[2] + off[2],
            ),
            velocity: navigate_contract::NedVelocity::new(flight.speed_m_s, 0.0, 0.0),
            attitude: navigate_contract::AttitudeQuaternion::new(1.0, 0.0, 0.0, 0.0),
            position_cov: navigate_contract::SymmetricCov3::from_diagonal(400.0, 400.0, 100.0),
            velocity_cov: navigate_contract::SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
            attitude_cov: navigate_contract::SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-5),
            at: at(0),
            imu_bias_prior: None,
        })
        .unwrap();
    let request = TerrainMapRequest {
        products,
        ..TerrainMapRequest::new(world.package, TODAY)
    };
    navigator
        .load_terrain_map(Arc::clone(&world.store), &request, None)
        .unwrap();
    navigator
}

/// The world, the product list of its keyframes, and the pixels of the
/// frame that the hovering camera sees.
#[cfg(test)]
fn scene() -> (World, Sha256Digest, Flight, Vec<u8>) {
    let world = World::build(&WorldSpec::default());
    // Keyframes at the working resolution of the camera at this height
    // over this map: 800 m of range and zoom 15 pixels of 3.57 m.
    let k = camera();
    let declared = Intrinsics {
        width: k.width,
        height: k.height,
        fx: k.fx,
        fy: k.fy,
        cx: k.cx,
        cy: k.cy,
    };
    let working = TerrainConfig::default()
        .working_camera(&declared, 800.0, 3.567)
        .unwrap();
    let keyframes = world.keyframes(model(&working.intrinsics), HEIGHT_M);
    let hover = Flight {
        start: [CENTRE[0], CENTRE[1], HEIGHT_M],
        speed_m_s: 0.0,
    };
    let level = nalgebra::UnitQuaternion::identity();
    let luma = frame_at(&world, &camera(), hover.body(0), &level, &belly(), 1);
    (world, keyframes.list, hover, luma)
}

/// The whole step of the navigator for one frame, on each path.
#[test]
fn the_cost_of_one_frame_with_and_without_keyframes() {
    show_events();
    let (world, list, hover, luma) = scene();
    let frame = |step: u64| SensorInput {
        sensor: CAMERA,
        acquired_at: at(step),
        delivered_at: at(step),
        data: SensorData::TerrainFrame {
            width: camera().width,
            height: camera().height,
            luma: luma.clone(),
        },
    };
    for (products, path) in [(None, "render"), (Some(list), "keyframe")] {
        let mut navigator = aligned(&world, products);
        let mut sink = Vec::new();
        navigator.ingest(&hover.increment(1), &mut sink).unwrap();
        let ((), first_ms) = milliseconds(|| navigator.ingest(&frame(1), &mut sink).unwrap());
        navigator.ingest(&hover.increment(2), &mut sink).unwrap();
        let ((), next_ms) = milliseconds(|| navigator.ingest(&frame(2), &mut sink).unwrap());
        assert_eq!(terrain(&sink)[0], (Outcome::Accepted, Some(path)));
        let pixels = (camera().width, camera().height);
        tracing::info!(
            path,
            ?pixels,
            first_ms,
            next_ms,
            "one frame through the navigator"
        );
    }
}

/// The parts of the step: the scene window, the rendered reference, the
/// stored keyframe, and the match on each.
#[test]
fn where_the_time_of_one_frame_goes() {
    show_events();
    let (world, list, hover, luma) = scene();
    let mut navigator = aligned(&world, Some(list));
    let mut sink = Vec::new();
    navigator.ingest(&hover.increment(1), &mut sink).unwrap();
    let belief = Belief::before(&navigator.solution(at(1)).unwrap(), 0.0).unwrap();
    let map = navigator.terrain.map.clone().unwrap();
    let state = navigator.terrain.cameras[&CAMERA.get()].clone();
    let sight = Sight {
        declared: &state.declared,
        mount: &state.mount,
        belief: &belief,
        radius_m: 60.0,
        ground_height_m: None,
    };
    let mut views = Views::default();
    let (first, window_and_render_ms) = milliseconds(|| views.render(&map, &sight));
    let (rendered, render_ms) = milliseconds(|| views.render(&map, &sight));
    assert!(first.is_ok());
    let rendered = rendered.unwrap();
    let (stored, keyframe_read_ms) = milliseconds(|| views.reference(&map, &sight));
    let (_, keyframe_held_ms) = milliseconds(|| views.reference(&map, &sight));
    let stored = stored.unwrap();
    assert_eq!(stored.path, "keyframe");
    let native = GrayImage::from_raw(camera().width, camera().height, luma).unwrap();
    let mut localizer = state.localizer.clone();
    let mut match_ms = [0.0; 2];
    let mut resample_ms = 0.0;
    for (slot, (reference, sequence)) in match_ms.iter_mut().zip([(&rendered, 1), (&stored, 2)]) {
        let k = reference.working.intrinsics;
        let (image, ms) = milliseconds(|| resize(&native, k.width, k.height, FilterType::Triangle));
        resample_ms = ms;
        let query = Frame {
            stamp: FrameStamp {
                sequence,
                capture_time_ns: sequence,
            },
            camera: model(&k),
            image,
        };
        let prior = PosePrior {
            pose: belief
                .camera_in(reference.view.frame, &state.mount)
                .unwrap(),
            position_radius_m: 60.0,
            attitude_radius_rad: 0.05,
        };
        let (estimate, ms) =
            milliseconds(|| localizer.estimate_blocking(&query, &reference.view, &prior));
        assert!(estimate.is_ok());
        *slot = ms;
    }
    let working = rendered.working;
    tracing::info!(
        declared = ?(camera().width, camera().height),
        working = ?(working.intrinsics.width, working.intrinsics.height),
        ground_pixel_m = working.ground_pixel_m,
        limit = ?working.limit,
        resample_ms,
        scene_window_ms = window_and_render_ms - render_ms,
        render_ms,
        keyframe_read_ms,
        keyframe_held_ms,
        match_on_render_ms = match_ms[0],
        match_on_keyframe_ms = match_ms[1],
        "the parts of one frame"
    );
}

/// The hover, the camera mount, and the alignment error (north, east, up,
/// metres) of the pose `i`: a belly camera for an even pose and an oblique
/// camera with a lever arm for an odd pose.
#[cfg(test)]
fn pose(i: u32) -> (Flight, CameraMount, [f64; 3]) {
    let a = f64::from(i);
    let hover = Flight {
        start: [
            CENTRE[0] + 0.004 * (a * 1.7).sin(),
            CENTRE[1] + 0.006 * (a * 2.3).cos(),
            HEIGHT_M + 60.0 * (a * 0.9).sin(),
        ],
        speed_m_s: 0.0,
    };
    let mount = if i.is_multiple_of(2) {
        belly()
    } else {
        CameraMount::rotated_forward(0.1, -70.0_f64.to_radians(), 0.4 + a)
            .with_lever_arm(nalgebra::Vector3::new(2.0, -0.7, 0.9))
    };
    let off = [
        30.0 * (a * 1.1).sin(),
        30.0 * (a * 0.7).cos(),
        10.0 * (a * 1.9).sin(),
    ];
    (hover, mount, off)
}

/// The fixes of `poses` hovering vehicles over `world`, alternately through
/// the belly camera and an oblique camera of 320 by 240 pixels, with the
/// tuning `config`: the count of fused fixes, the root mean square of the
/// horizontal and of the vertical error in metres, and the mean time of a
/// frame in milliseconds.
#[cfg(test)]
fn fixes_with(world: &World, config: TerrainConfig, poses: u32) -> (u32, f64, f64, f64) {
    let k = Intrinsics {
        width: 320,
        height: 240,
        fx: 300.0,
        fy: 300.0,
        cx: 159.5,
        cy: 119.5,
    };
    let request = TerrainMapRequest {
        config: TerrainConfig {
            max_search_radius_m: 80.0,
            // The search stays about the belief: these poses hold the
            // covariance of a fix, not the acquisition of a wide belief.
            max_acquisition_radius_m: 80.0,
            ..config
        },
        ..TerrainMapRequest::new(world.package, TODAY)
    };
    let rays = (1, if cfg!(debug_assertions) { 2 } else { 1 });
    let (mut fused, mut horizontal_m2, mut vertical_m2, mut ms) = (0, 0.0, 0.0, 0.0);
    for i in 0..poses {
        let (hover, mount, off) = pose(i);
        let suite = suite(ImagerModel::pinhole(k), mount);
        let mut navigator = ReferenceNavigator::new(suite, CLOCK, None).unwrap();
        let (lat, lon) = (hover.start[0].to_radians(), hover.start[1].to_radians());
        navigator
            .align(&Alignment {
                position: navigate_contract::GeodeticPosition::new(
                    lat + off[0] / 6_364_000.0,
                    lon + off[1] / (6_378_137.0 * lat.cos()),
                    hover.start[2] + off[2],
                ),
                velocity: navigate_contract::NedVelocity::new(0.0, 0.0, 0.0),
                attitude: navigate_contract::AttitudeQuaternion::new(1.0, 0.0, 0.0, 0.0),
                position_cov: navigate_contract::SymmetricCov3::from_diagonal(1e6, 1e6, 1e6),
                velocity_cov: navigate_contract::SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
                attitude_cov: navigate_contract::SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-5),
                at: at(0),
                imu_bias_prior: None,
            })
            .unwrap();
        navigator
            .load_terrain_map(Arc::clone(&world.store), &request, None)
            .unwrap();
        let mut sink = Vec::new();
        navigator.ingest(&hover.increment(1), &mut sink).unwrap();
        let level = nalgebra::UnitQuaternion::identity();
        let frame = SensorInput {
            sensor: CAMERA,
            acquired_at: at(1),
            delivered_at: at(1),
            data: SensorData::TerrainFrame {
                width: k.width,
                height: k.height,
                // An optimized build casts a ray for each pixel of the
                // frame, so that neither setting has the sampling of the
                // frame on its side.
                luma: frame_with_rays(world, &model(&k), hover.body(1), &level, &mount, rays),
            },
        };
        let ((), frame_ms) = milliseconds(|| navigator.ingest(&frame, &mut sink).unwrap());
        ms += frame_ms;
        if terrain(&sink) == [(Outcome::Accepted, Some("render"))] {
            let error = hover.error_m(&navigator.solution(at(1)).unwrap(), 1);
            fused += 1;
            horizontal_m2 += error.xy().norm_squared();
            vertical_m2 += error.z * error.z;
        }
    }
    let n = f64::from(fused.max(1));
    (
        fused,
        (horizontal_m2 / n).sqrt(),
        (vertical_m2 / n).sqrt(),
        ms / f64::from(poses.max(1)),
    )
}

/// The error and the cost of the reference renderer with one ray for each
/// pixel of textured ground (a grid of 8 pixels that is refined at each
/// luma step of 8 counts) and with one ray for each block of 4 by 4 pixels
/// and interpolation between them, over gentle and over steep ground.
#[test]
fn the_error_and_the_cost_of_each_render_setting() {
    show_events();
    let poses = if cfg!(debug_assertions) { 2 } else { 8 };
    let dense = TerrainConfig {
        stride: 8,
        luma_edge: 8,
        ..TerrainConfig::default()
    };
    let sparse = TerrainConfig {
        stride: 4,
        luma_edge: u8::MAX,
        ..TerrainConfig::default()
    };
    for (ground, relief_m) in [("hills of 80 m", 80.0), ("hills of 400 m", 400.0)] {
        let world = World::build(&WorldSpec {
            relief_m,
            ..WorldSpec::default()
        });
        let settings = [
            ("a ray for each pixel", dense),
            ("a ray for each 4x4 block", sparse),
        ];
        for (setting, config) in settings {
            let (fused, horizontal_rms_m, vertical_rms_m, frame_ms) =
                fixes_with(&world, config, poses);
            tracing::info!(
                ground,
                setting,
                poses,
                fused,
                horizontal_rms_m,
                vertical_rms_m,
                frame_ms,
                "320x240 fixes"
            );
            assert!(fused > 0, "{ground}, {setting}");
        }
    }
}

/// A cruise to the north at 60 m/s with one frame each second over
/// correlation cells of zoom 16 (457 m): how many frames are fused,
/// matched and withheld, and not matched, and what a frame costs. An
/// optimized build flies 20 s with the camera of 800 by 600 pixels.
#[test]
fn the_cost_of_a_cruise_with_one_frame_each_second() {
    show_events();
    let seconds: u64 = if cfg!(debug_assertions) { 6 } else { 20 };
    let world = World::build(&WorldSpec {
        correlation_zoom: 16,
        ..WorldSpec::default()
    });
    // The track is along the middle of a column of cells: a track within
    // the search radius of a cell edge has a cell without a fix beside it
    // on each frame, and then each frame is matched.
    let ([_, lon], _) = correlation_cell(CENTRE, 16);
    let south = (700.0 / 6_364_000.0_f64).to_degrees();
    let flight = Flight {
        start: [CENTRE[0] - south, lon, HEIGHT_M],
        speed_m_s: 60.0,
    };
    let mut navigator = aligned_on(&world, None, &flight, [20.0, -15.0, 5.0]);
    let level = nalgebra::UnitQuaternion::identity();
    let mut sink = Vec::new();
    let (mut total_ms, mut matched_ms) = (0.0, 0.0_f64);
    for step in 1..=seconds * 50 {
        navigator
            .ingest(&flight.increment(step), &mut sink)
            .unwrap();
        if !step.is_multiple_of(50) {
            continue;
        }
        let frame = SensorInput {
            sensor: CAMERA,
            acquired_at: at(step),
            delivered_at: at(step),
            data: SensorData::TerrainFrame {
                width: camera().width,
                height: camera().height,
                luma: frame_at(&world, &camera(), flight.body(step), &level, &belly(), step),
            },
        };
        let before = sink.len();
        let ((), ms) = milliseconds(|| navigator.ingest(&frame, &mut sink).unwrap());
        total_ms += ms;
        if sink[before..].iter().any(|r| r.context.is_some()) {
            matched_ms = matched_ms.max(ms);
        }
    }
    let records = terrain(&sink);
    let fused = records.iter().filter(|r| r.0 == Outcome::Accepted).count();
    let unmatched = records.iter().filter(|r| r.1.is_none()).count();
    let error = flight.error_m(&navigator.solution(at(seconds * 50)).unwrap(), seconds * 50);
    tracing::info!(
        declared = ?(camera().width, camera().height),
        working = ?navigator.terrain_working_camera(CAMERA),
        frames = records.len(),
        fused,
        matched_and_withheld = records.len() - fused - unmatched,
        unmatched,
        mean_frame_ms = total_ms / records.len() as f64,
        slowest_matched_frame_ms = matched_ms,
        horizontal_error_m = error.xy().norm(),
        "a cruise at 60 m/s over 457 m cells"
    );
    assert!(
        fused >= 1 && fused + unmatched <= records.len(),
        "{records:?}"
    );
}
