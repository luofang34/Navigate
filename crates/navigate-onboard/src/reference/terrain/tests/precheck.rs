//! The check before the match and the working resolution: what a frame
//! costs is bounded, and no fix of a new map cell is lost.

use std::sync::Arc;

use navigate_camera::{ImagerModel, Intrinsics};
use navigate_contract::{AttitudeQuaternion, GeodeticPosition, NedVelocity, SymmetricCov3};

use super::fixes::{HEIGHT_M, belly, north_through_cells, terrain};
use super::flight::{CAMERA, CLOCK, Flight, RATE, at, fly, load, navigator, request, suite};
use super::world::GROUND_M;
use super::world::{CENTRE, World, WorldSpec, frame_at};
use crate::{
    Alignment, Navigator, Outcome, ReferenceNavigator, SensorData, SensorInput, SensorSuite,
    TerrainConfig, TerrainMapRequest, WorkingLimit,
};
use navigate_camera::CameraMount;
use navigate_contract::SourceId;

/// Correlation cells of 228 m: twelve frames of the flight for each cell.
const ZOOM: u32 = 17;

/// A second terrain camera that looks 20° forward of the nadir: from the
/// height of the flights it sees the ground 290 m ahead, more than one
/// cell of [`ZOOM`].
pub(super) const FORWARD: SourceId = SourceId::new(7);

pub(super) fn forward() -> CameraMount {
    CameraMount::rotated_forward(
        0.0,
        20.0_f64.to_radians() - core::f64::consts::FRAC_PI_2,
        0.0,
    )
}

#[test]
fn frames_that_can_only_show_cells_whose_block_a_newer_cell_took_are_not_matched() {
    let world = World::build(&WorldSpec {
        correlation_zoom: ZOOM,
        ..WorldSpec::default()
    });
    let flight = north_through_cells(ZOOM, 12.0, HEIGHT_M, -1.0);
    // The belly camera and the forward camera: the forward camera reaches
    // each cell first, and its fix takes the one map block from the cell
    // under the vehicle.
    let mut sensors = suite(super::flight::imager(), belly()).sensors().to_vec();
    let mut second = sensors.last().cloned().unwrap();
    second.id = FORWARD;
    second.mount = Some(forward());
    sensors.push(second);
    let mut navigator = navigator(&flight, belly(), [-20.0, 30.0, -8.0], 40.0);
    let aligned = navigator.clone();
    navigator = ReferenceNavigator::new(SensorSuite::new(sensors).unwrap(), CLOCK, None).unwrap();
    navigator.align(&aligned_state(&aligned)).unwrap();
    load(&mut navigator, &world);
    let mut sink = Vec::new();
    for step in 1..=375 {
        navigator
            .ingest(&flight.increment(step), &mut sink)
            .unwrap();
        if step.is_multiple_of(25) {
            navigator
                .ingest(&flight.frame(&world, &belly(), step, 0), &mut sink)
                .unwrap();
            let mut ahead = flight.frame(&world, &forward(), step, 0);
            ahead.sensor = FORWARD;
            navigator.ingest(&ahead, &mut sink).unwrap();
        }
    }
    let ahead: Vec<_> = sink
        .iter()
        .filter(|r| r.sensor == FORWARD && r.measurement == crate::reference::terrain::MEASUREMENT)
        .map(|r| r.outcome.clone())
        .collect();
    let below = terrain(&sink);
    assert_eq!((ahead.len(), below.len()), (15, 15));
    // The forward camera fuses each frame: its ground is in the cell whose
    // block it holds, or in a new cell.
    assert!(ahead.iter().all(|o| *o == Outcome::Accepted), "{ahead:?}");
    // Once the forward camera took the block for the next cell, the cell
    // under the vehicle has a fix and no block: its frames are withheld,
    // and a frame whose search disk holds only such cells is not matched.
    let shared = Outcome::Unused("shared_map_error");
    let withheld = below.iter().filter(|r| r.0 == shared).count();
    let unmatched = below
        .iter()
        .filter(|r| r.0 == shared && r.1.is_none())
        .count();
    tracing::info!(
        withheld,
        unmatched,
        "the belly camera behind the forward camera"
    );
    assert!(withheld >= 10 && unmatched >= 6, "{below:?}");
    let error = flight.error_m(&navigator.solution(at(375)).unwrap(), 375);
    assert!(error.xy().norm() < 16.0, "{error}");
}

/// The alignment of `navigator` at step zero.
pub(super) fn aligned_state(navigator: &ReferenceNavigator) -> Alignment {
    let mut copy = navigator.clone();
    let solution = copy.solution(at(0)).unwrap();
    Alignment {
        position: solution.position,
        velocity: solution.velocity,
        attitude: solution.attitude.unwrap(),
        position_cov: solution.position_cov,
        velocity_cov: solution.velocity_cov,
        attitude_cov: solution.attitude_cov.unwrap(),
        at: at(0),
        imu_bias_prior: None,
    }
}

/// A navigator with the belly camera `imager`, aligned 30 m north, 20 m
/// west, and 10 m above the centre at the height of the flights.
#[cfg(test)]
fn aligned_hover(imager: ImagerModel) -> ReferenceNavigator {
    let mut navigator = ReferenceNavigator::new(suite(imager, belly()), CLOCK, None).unwrap();
    navigator
        .align(&Alignment {
            position: GeodeticPosition::new(
                CENTRE[0].to_radians() + 30.0 / 6_364_000.0,
                CENTRE[1].to_radians() - 20.0 / 4_762_000.0,
                HEIGHT_M + 10.0,
            ),
            velocity: NedVelocity::new(0.0, 0.0, 0.0),
            attitude: AttitudeQuaternion::new(1.0, 0.0, 0.0, 0.0),
            position_cov: SymmetricCov3::from_diagonal(1_600.0, 1_600.0, 400.0),
            velocity_cov: SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
            attitude_cov: SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-5),
            at: at(0),
            imu_bias_prior: None,
        })
        .unwrap();
    navigator
}

#[test]
fn a_large_frame_is_matched_at_the_working_resolution() {
    let world = World::build(&WorldSpec::default());
    // A camera of 320 by 240 pixels with the field of view of the test
    // camera, and a working resolution of a quarter of its pixels.
    let declared = Intrinsics {
        width: 320,
        height: 240,
        fx: 300.0,
        fy: 300.0,
        cx: 159.5,
        cy: 119.5,
    };
    let config = TerrainConfig {
        max_match_pixels: 160 * 120,
        ..request(&world).config
    };

    let hover = Flight {
        start: [CENTRE[0], CENTRE[1], HEIGHT_M],
        speed_m_s: 0.0,
    };
    let mut navigator = aligned_hover(ImagerModel::pinhole(declared));
    let scaled = TerrainMapRequest {
        config,
        ..request(&world)
    };
    navigator
        .load_terrain_map(Arc::clone(&world.store), &scaled, None)
        .unwrap();
    let mut sink = Vec::new();
    for step in 1..=RATE {
        navigator.ingest(&hover.increment(step), &mut sink).unwrap();
    }
    let camera = navigate_visual::CameraModel {
        width: 320,
        height: 240,
        fx: 300.0,
        fy: 300.0,
        cx: 159.5,
        cy: 119.5,
    };
    let level = nalgebra::UnitQuaternion::identity();
    let frame = SensorInput {
        sensor: CAMERA,
        acquired_at: at(RATE),
        delivered_at: at(RATE),
        data: SensorData::TerrainFrame {
            width: 320,
            height: 240,
            luma: frame_at(&world, &camera, hover.body(RATE), &level, &belly(), 3),
        },
    };
    assert_eq!(navigator.terrain_working_camera(CAMERA), None);
    navigator.ingest(&frame, &mut sink).unwrap();
    assert_eq!(terrain(&sink), [(Outcome::Accepted, Some("render"))]);
    // The navigator states the camera of the match and what set it.
    let working = navigator.terrain_working_camera(CAMERA).unwrap();
    assert_eq!(
        (working.intrinsics.width, working.intrinsics.height),
        (160, 120)
    );
    assert!((working.intrinsics.fx - 150.0).abs() < 1e-9);
    assert!((working.intrinsics.cx - 79.5).abs() < 1e-9);
    assert_eq!(working.limit, WorkingLimit::PixelBound);
    // The working pixel is 5.3 m on the ground, as that of the test
    // camera: the same bound of three pixels holds.
    let error = hover.error_m(&navigator.solution(at(RATE)).unwrap(), RATE);
    tracing::info!(
        north_m = error.x,
        east_m = error.y,
        "a 320x240 frame matched at 160x120"
    );
    assert!(error.xy().norm() < 16.0 && error.z.abs() < 16.0, "{error}");
}

/// The camera of the first trial: 800 by 600 pixels, 67.4° across, 1 700 m
/// above the ground, over zoom 15 imagery of 3.57 m.
#[test]
fn the_working_resolution_follows_the_map_texel_inside_its_bounds() {
    let declared = Intrinsics {
        width: 800,
        height: 600,
        fx: 600.0,
        fy: 600.0,
        cx: 399.5,
        cy: 299.5,
    };
    let (range_m, texel_m) = (1_700.0, 3.567);
    // The native pixel is 2.83 m, finer than the map. A match at the map
    // texel needs 635 by 477 pixels; the default bound holds it to 320 by
    // 240 with pixels of 7.1 m, and the limit says so.
    let bounded = TerrainConfig::default()
        .working_camera(&declared, range_m, texel_m)
        .unwrap();
    assert_eq!(
        (bounded.intrinsics.width, bounded.intrinsics.height),
        (320, 240)
    );
    assert_eq!(bounded.limit, WorkingLimit::PixelBound);
    assert!((bounded.ground_pixel_m - 7.083).abs() < 0.01, "{bounded:?}");
    let roomy = TerrainConfig {
        max_match_pixels: 800 * 600,
        ..TerrainConfig::default()
    };
    let at_texel = roomy.working_camera(&declared, range_m, texel_m).unwrap();
    assert_eq!(
        (at_texel.intrinsics.width, at_texel.intrinsics.height),
        (635, 477)
    );
    assert_eq!(at_texel.limit, WorkingLimit::MapTexel);
    assert!(
        (at_texel.ground_pixel_m - texel_m).abs() < 0.01,
        "{at_texel:?}"
    );
    // A map finer than the camera leaves the camera as it is.
    let native = roomy.working_camera(&declared, range_m, 1.0).unwrap();
    assert_eq!(
        (native.intrinsics, native.limit),
        (declared, WorkingLimit::Native)
    );
    // Low over a coarse map, the texel rule asks for 34 by 25 pixels; the
    // floor keeps enough of the image for the matcher.
    let low = roomy.working_camera(&declared, 150.0, texel_m).unwrap();
    assert_eq!((low.intrinsics.width, low.intrinsics.height), (160, 120));
    assert_eq!(low.limit, WorkingLimit::PixelFloor);
    assert!(roomy.working_camera(&declared, 0.0, texel_m).is_none());
}

/// A camera 25° off the nadir, 150 m above the ground: the ground that it
/// sees is 70 m ahead of the vehicle, more than half a correlation cell
/// of 114 m. The cell of a fix is that of this ground, not of the place
/// of the vehicle, so the first frame of a new cell is a fix while the
/// vehicle is still above the cell before.
#[test]
fn a_tilted_low_camera_fuses_each_frame_of_the_cells_that_it_sees() {
    let world = World::build(&WorldSpec {
        correlation_zoom: 18,
        imagery_zoom: 17,
        radius_m: 700.0,
        relief_m: 20.0,
        ..WorldSpec::default()
    });
    let tilt = 25.0_f64.to_radians();
    let mount = CameraMount::rotated_forward(0.0, tilt - core::f64::consts::FRAC_PI_2, 0.0);
    let height_m = GROUND_M + 150.0;
    let flight = north_through_cells(18, 6.0, height_m, 150.0 * tilt.tan());
    let mut navigator = navigator(&flight, mount, [6.0, -5.0, 3.0], 12.0);
    load(&mut navigator, &world);
    let mut sink = Vec::new();
    let each_half_second = |step: u64| step.is_multiple_of(25);
    // Thirteen frames: six for each of two cells, 19 m apart, and the
    // first frame of a third cell.
    fly(
        &mut navigator,
        &flight,
        &world,
        &mount,
        1..=325,
        each_half_second,
        &mut sink,
    )
    .unwrap();
    let records = terrain(&sink);
    assert_eq!(records.len(), 13);
    // The ground enters a new cell at frames 6 and 12, while the vehicle
    // is still above the cell before: each new cell takes the block, and
    // each frame of the cell that holds the block is fused.
    assert!(
        records
            .iter()
            .all(|r| *r == (Outcome::Accepted, Some("render"))),
        "{records:?}"
    );
}
