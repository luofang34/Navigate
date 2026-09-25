#![allow(clippy::expect_used, clippy::panic)]
use super::*;
use crate::{
    CameraModel, CameraPose, Frame, FrameStamp, LocalFrame, LocalizerConfig, MapRevision,
    PosePrior, PoseVerifier, ReferenceView, TrackingReference,
};
use nalgebra::{UnitQuaternion, Vector3};

fn camera() -> CameraModel {
    CameraModel {
        width: 320,
        height: 180,
        fx: 230.0,
        fy: 230.0,
        cx: 159.5,
        cy: 89.5,
    }
}
fn pose(t: f64) -> CameraPose {
    CameraPose {
        position: Vector3::new(15.0 * t.sin(), 8.0 * (t * 0.5).sin(), 100.0 + 3.0 * t.sin()),
        orientation: UnitQuaternion::from_euler_angles(
            0.08 * t.sin(),
            0.06 * (t * 0.7).sin(),
            0.6 * t.sin(),
        ),
    }
}
fn texture(world: Vector3<f64>) -> u8 {
    let x = world.x / 3.1;
    let y = world.y / 3.7;
    let ix = x.floor() as i32;
    let iy = y.floor() as i32;
    let hash = |x: i32, y: i32| {
        let h = (x as u32).wrapping_mul(374761393) ^ (y as u32).wrapping_mul(668265263);
        20.0 + f64::from((h ^ (h >> 13)).wrapping_mul(1274126177) % 210)
    };
    let mix = |a: f64, b: f64, t: f64| a + (b - a) * t;
    mix(
        mix(hash(ix, iy), hash(ix.wrapping_add(1), iy), x - x.floor()),
        mix(
            hash(ix, iy.wrapping_add(1)),
            hash(ix.wrapping_add(1), iy.wrapping_add(1)),
            x - x.floor(),
        ),
        y - y.floor(),
    ) as u8
}
fn surface(pose: CameraPose) -> ReferenceView {
    let camera = camera();
    let mut depth_m = Vec::new();
    let image = GrayImage::from_fn(camera.width, camera.height, |x, y| {
        let pixel = Vector2::new(f64::from(x), f64::from(y));
        let ray = camera.unproject(&pose, pixel, 1.0) - pose.position;
        let depth = -pose.position.z / ray.z;
        depth_m.push(depth as f32);
        image::Luma([texture(camera.unproject(&pose, pixel, depth))])
    });
    ReferenceView {
        pose,
        image,
        depth_m,
        map: MapRevision {
            release_id: "synthetic-plane".into(),
            manifest_sha256: "a".repeat(64),
        },
        frame: LocalFrame::anchor_mercator(40.0, -74.0).expect("local frame"),
    }
}
fn frame(sequence: u64, pose: CameraPose) -> Frame {
    Frame {
        camera: camera(),
        stamp: FrameStamp {
            sequence,
            capture_time_ns: sequence * 200_000_000,
        },
        image: surface(pose).image,
    }
}

#[test]
#[ignore = "requires a hardware compute adapter; run explicitly on the target device"]
fn rendered_sequence_tracks_free_camera_attitude_without_systematic_drift() {
    let mut matcher = pollster::block_on(GpuPyramidalMatcher::new()).expect("GPU");
    let verifier = PoseVerifier::new(LocalizerConfig::default()).expect("geometry policy");
    let prior = PosePrior {
        pose: pose(0.0),
        position_radius_m: 100.0,
        attitude_radius_rad: 2.0,
    };
    let mut estimate = prior.pose;
    let mut previous = frame(0, estimate);
    let mut max_position = 0.0_f64;
    let mut max_rotation = 0.0_f64;
    for sequence in 1..=100 {
        let truth = pose(sequence as f64 * std::f64::consts::TAU / 100.0);
        let current = frame(sequence, truth);
        let pairs = matcher
            .match_images_blocking(&previous.image, &current.image)
            .expect("matches");
        let surface = surface(estimate);
        let update = verifier
            .track(
                &current,
                TrackingReference {
                    observation: &previous,
                    surface: &surface,
                },
                &prior,
                &pairs,
                matcher.identity(),
            )
            .unwrap_or_else(|e| panic!("frame {sequence}: {e}; {} matches", pairs.len()));
        estimate = update.pose;
        max_position = max_position.max((estimate.position - truth.position).norm());
        max_rotation = max_rotation.max(estimate.orientation.angle_to(&truth.orientation));
        previous = current;
    }
    assert!(
        max_position < 1.0,
        "maximum position error: {max_position} m"
    );
    assert!(
        max_rotation < 0.02,
        "maximum rotation error: {max_rotation} rad"
    );
}
