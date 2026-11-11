//! The cost of one frame, with and without keyframes, and where the time
//! goes. The test states each time through `tracing` and asserts none: run
//! it with `--release -- --nocapture cost`. An optimized build measures a
//! camera of 320 by 240 pixels. A build without optimization measures the
//! small camera of the other tests, so that the test stays short there.

use std::sync::Arc;
use std::time::Instant;

use image::GrayImage;
use navigate_camera::{ImagerModel, Intrinsics};
use navigate_data::Sha256Digest;
use navigate_visual::{CameraModel, Frame, FrameStamp, PosePrior};

use super::fixes::{HEIGHT_M, belly, terrain};
use super::flight::{CAMERA, CLOCK, Flight, at, show_events, suite};
use super::world::{CENTRE, TODAY, World, WorldSpec, camera_model, frame_at};
use crate::reference::terrain::belief::Belief;
use crate::reference::terrain::view::Views;
use crate::{
    Alignment, Navigator, Outcome, ReferenceNavigator, SensorData, SensorInput, TerrainMapRequest,
};

/// The camera of the cost measurement, 56° across.
#[cfg(test)]
fn camera() -> CameraModel {
    if cfg!(debug_assertions) {
        return camera_model();
    }
    CameraModel {
        width: 320,
        height: 240,
        fx: 300.0,
        fy: 300.0,
        cx: 159.5,
        cy: 119.5,
    }
}

#[cfg(test)]
fn milliseconds<T>(work: impl FnOnce() -> T) -> (T, f64) {
    let start = Instant::now();
    let value = work();
    (value, start.elapsed().as_secs_f64() * 1e3)
}

/// A hovering navigator with the cost camera, aligned 20 m off, with the
/// default tuning of the means.
#[cfg(test)]
fn aligned(world: &World, products: Option<Sha256Digest>) -> ReferenceNavigator {
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
    let lat = CENTRE[0].to_radians() + 20.0 / 6_364_000.0;
    navigator
        .align(&Alignment {
            position: navigate_contract::GeodeticPosition::new(
                lat,
                CENTRE[1].to_radians(),
                HEIGHT_M,
            ),
            velocity: navigate_contract::NedVelocity::new(0.0, 0.0, 0.0),
            attitude: navigate_contract::AttitudeQuaternion::new(1.0, 0.0, 0.0, 0.0),
            position_cov: navigate_contract::SymmetricCov3::from_diagonal(400.0, 400.0, 100.0),
            velocity_cov: navigate_contract::SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
            attitude_cov: navigate_contract::SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-5),
            at: at(0),
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
    let keyframes = world.keyframes(camera(), HEIGHT_M);
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
    let (camera, mount, calibration) = (state.camera, state.mount, state.calibration);
    let mut views = Views::default();
    let (first, window_and_render_ms) =
        milliseconds(|| views.render(&map, &camera, &mount, &belief, 60.0));
    let (rendered, render_ms) = milliseconds(|| views.render(&map, &camera, &mount, &belief, 60.0));
    assert!(first.is_ok());
    let rendered = rendered.unwrap();
    let (stored, keyframe_read_ms) =
        milliseconds(|| views.reference(&map, &camera, calibration, &mount, &belief, 60.0));
    let (_, keyframe_held_ms) =
        milliseconds(|| views.reference(&map, &camera, calibration, &mount, &belief, 60.0));
    let stored = stored.unwrap();
    assert_eq!(stored.path, "keyframe");
    let mut localizer = state.localizer.clone();
    let mut match_ms = [0.0; 2];
    for (slot, (reference, sequence)) in match_ms.iter_mut().zip([(&rendered, 1), (&stored, 2)]) {
        let query = Frame {
            stamp: FrameStamp {
                sequence,
                capture_time_ns: sequence,
            },
            camera,
            image: GrayImage::from_raw(camera.width, camera.height, luma.clone()).unwrap(),
        };
        let prior = PosePrior {
            pose: belief.camera_in(reference.view.frame, &mount).unwrap(),
            position_radius_m: 60.0,
            attitude_radius_rad: 0.05,
        };
        let (estimate, ms) =
            milliseconds(|| localizer.estimate_blocking(&query, &reference.view, &prior));
        assert!(estimate.is_ok());
        *slot = ms;
    }
    tracing::info!(
        pixels = ?(camera.width, camera.height),
        scene_window_ms = window_and_render_ms - render_ms,
        render_ms,
        keyframe_read_ms,
        keyframe_held_ms,
        match_on_render_ms = match_ms[0],
        match_on_keyframe_ms = match_ms[1],
        "the parts of one frame"
    );
}
