//! Golden case: the Oryol rooftop photo, located end to end on a GPU.
//!
//! The expected pose is the published one (Sokoly-Demos `missions/9c6b2c`): camera at
//! 52.98224275 N, 36.06007134 E, 291.32 m, heading 275°, 39° below the horizon. The
//! renderer needs a hardware GPU, so CI cannot run this case; run it before a
//! release of the matcher or the pipeline:
//!
//! ```sh
//! NAVIGATE_LOFTR_WEIGHTS=loftr-f16.safetensors NAVIGATE_TILE_CACHE=tiles \
//!   cargo test --release -p navigate-locate --features http --test oryol -- --ignored
//! ```
use burn::tensor::Device;
use navigate_locate::{CachedHttpSource, LocateRequest, LookAt, Outcome, SceneRequest, locate};
use navigate_match::{Loftr, LoftrMatcher, Weights};
use navigate_visual::CameraModel;
use std::path::Path;

const EXPECTED: [f64; 3] = [
    52.982_242_754_858_81,
    36.060_071_337_921_784,
    291.323_347_706_665_5,
];
const MAX_POSITION_ERROR_M: f64 = 5.0;
const MAX_ANGLE_ERROR_DEG: f64 = 1.5;

#[cfg(test)]
fn request(target: [f64; 2], heading: f64) -> LocateRequest {
    let photo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../navigate-match/tests/parity/oryol-query.png");
    let image = image::open(photo).unwrap().to_luma8();
    let (width, height) = image.dimensions();
    // The published pose assumed 710 px at 1024 px width.
    let focal = 710.0 * f64::from(width) / 1024.0;
    LocateRequest {
        image,
        camera: CameraModel {
            width,
            height,
            fx: focal,
            fy: focal,
            cx: (f64::from(width) - 1.0) / 2.0,
            cy: (f64::from(height) - 1.0) / 2.0,
        },
        target,
        look_at: LookAt::standard(heading),
        scene: SceneRequest {
            anchor: target,
            radius_m: 800.0,
            imagery_zoom: 18,
            terrain_zoom: 14,
            imagery_tile_pixels: 256,
            max_bytes: 512 << 20,
            concurrency: 16,
        },
        position_radius_m: 400.0,
        attitude_radius_rad: 0.9,
    }
}

#[cfg(test)]
fn check(target: [f64; 2], heading: f64) {
    let weights = std::env::var("NAVIGATE_LOFTR_WEIGHTS").expect("set NAVIGATE_LOFTR_WEIGHTS");
    let cache = std::env::var("NAVIGATE_TILE_CACHE").expect("set NAVIGATE_TILE_CACHE");
    let device = Device::wgpu(Default::default());
    let model = Loftr::load(
        Weights::from_safetensors(&std::fs::read(weights).unwrap()).unwrap(),
        &device,
    )
    .unwrap();
    let source = CachedHttpSource::new(cache);
    let outcome = pollster::block_on(locate(
        &source,
        LoftrMatcher::new(model, "golden"),
        request(target, heading),
        |_| {},
    ))
    .unwrap();
    let Outcome::Found(located) = outcome else {
        panic!("the Oryol photo was not located");
    };
    let [lat, lon, h] = located.geodetic;
    let north = (lat - EXPECTED[0]).to_radians() * 6_371_000.0;
    let east = (lon - EXPECTED[1]).to_radians() * 6_371_000.0 * EXPECTED[0].to_radians().cos();
    let error = (north * north + east * east + (h - EXPECTED[2]).powi(2)).sqrt();
    assert!(
        error <= MAX_POSITION_ERROR_M,
        "position error {error:.2} m: {:?}",
        located.geodetic
    );
    let [heading_deg, pitch_deg, _] = located.angles_deg;
    assert!(
        (heading_deg - 275.0).abs() <= MAX_ANGLE_ERROR_DEG,
        "heading {heading_deg}"
    );
    assert!(
        (pitch_deg - 39.0).abs() <= MAX_ANGLE_ERROR_DEG,
        "pitch {pitch_deg}"
    );
    assert!(located.refined, "the refinement view must confirm the pose");
}

#[test]
#[ignore = "needs a GPU, the weight file and the tile cache"]
fn the_oryol_photo_is_located_from_the_centre_point() {
    check([52.982_336, 36.058_259], 275.0);
}

#[test]
#[ignore = "needs a GPU, the weight file and the tile cache"]
fn the_oryol_photo_is_located_from_a_rough_pin_and_heading() {
    // About 30 m from the photo centre and 10° from the true heading.
    check([52.982_45, 36.058_45], 285.0);
}
