//! A sloped bare-earth terrain with roofs that the elevation model does not contain.

use crate::support::{camera, local_frame, map, nadir};
use image::GrayImage;
use nalgebra::{UnitQuaternion, Vector2, Vector3};
use navigate_visual::{PixelMatch, ReferenceView};
use navigate_visual_session::{Pose, to_camera};

/// Elevation model: `z = SLOPE_X x + SLOPE_Y y`.
pub const SLOPE_X: f64 = 0.03;
pub const SLOPE_Y: f64 = -0.015;
/// Roof height above the elevation model.
pub const ROOF_M: f64 = 15.0;
/// The true focal length differs from the declared calibration, as with an
/// assumed field of view and an unremoved lens distortion.
pub const TRUE_FOCAL_SCALE: f64 = 1.04;

fn true_camera() -> navigate_visual::CameraModel {
    let c = camera();
    navigate_visual::CameraModel {
        fx: c.fx * TRUE_FOCAL_SCALE,
        fy: c.fy * TRUE_FOCAL_SCALE,
        ..c
    }
}

pub fn dem(x: f64, y: f64) -> f64 {
    SLOPE_X * x + SLOPE_Y * y
}

/// Houses: 12 m squares on a 30 m grid, about a third of the ground.
fn roof(x: f64, y: f64) -> bool {
    x.rem_euclid(30.0) < 12.0 && y.rem_euclid(30.0) < 17.0
}

fn ray(c: &navigate_visual::CameraModel, pose: &Pose, pixel: Vector2<f64>) -> Vector3<f64> {
    pose.rotation * Vector3::new((pixel.x - c.cx) / c.fx, (c.cy - pixel.y) / c.fy, -1.0)
}

/// Optical-axis depth to the plane `z = dem(x, y) + lift`.
fn plane_depth(pose: &Pose, d: &Vector3<f64>, lift: f64) -> Option<f64> {
    let p = pose.translation.vector;
    let denominator = d.z - SLOPE_X * d.x - SLOPE_Y * d.y;
    let t = (dem(p.x, p.y) + lift - p.z) / denominator;
    (t.is_finite() && t > 0.0).then_some(t)
}

/// The scene point that the true camera sees through `pixel`: a roof when
/// the ray meets one first.
fn scene_point(pose: &Pose, pixel: Vector2<f64>) -> Option<Vector3<f64>> {
    let d = ray(&true_camera(), pose, pixel);
    let p = pose.translation.vector;
    if let Some(t) = plane_depth(pose, &d, ROOF_M) {
        let top = p + d * t;
        if roof(top.x, top.y) {
            return Some(top);
        }
    }
    plane_depth(pose, &d, 0.0).map(|t| p + d * t)
}

/// Bare-earth depth rendered at `pose`, as a host renders from the elevation model.
pub fn dem_view(pose: &Pose) -> ReferenceView {
    let c = camera();
    let mut depth = Vec::with_capacity((c.width * c.height) as usize);
    for v in 0..c.height {
        for u in 0..c.width {
            let d = ray(&c, pose, Vector2::new(f64::from(u), f64::from(v)));
            depth.push(plane_depth(pose, &d, 0.0).map_or(0.0, |t| t as f32));
        }
    }
    ReferenceView {
        map: map(),
        frame: local_frame(),
        pose: to_camera(pose),
        image: GrayImage::new(c.width, c.height),
        depth_m: depth,
    }
}

/// Matches of true scene points between two true cameras, with deterministic pixel noise.
pub fn scene_matches(earlier: &Pose, later: &Pose, seed: &mut u64) -> Vec<PixelMatch> {
    let c = true_camera();
    let mut pairs = Vec::new();
    for row in 0..9 {
        for col in 0..12 {
            let pixel = Vector2::new(14.0 + 26.0 * f64::from(col), 12.0 + 26.0 * f64::from(row));
            let Some(world) = scene_point(earlier, pixel) else {
                continue;
            };
            let Some(q) = c.project(&to_camera(later), world) else {
                continue;
            };
            if q.x > 2.0 && q.y > 2.0 && q.x < 317.0 && q.y < 237.0 {
                pairs.push(PixelMatch {
                    reference: pixel,
                    query: q + noise(seed),
                });
            }
        }
    }
    pairs
}

/// Matches of ground points between a map reference and the true camera: map
/// imagery shows the ground, so map matches use elevation-model points.
pub fn map_matches(reference: &Pose, truth: &Pose) -> Vec<PixelMatch> {
    let (c, seen) = (camera(), true_camera());
    let mut pairs = Vec::new();
    for row in 0..8 {
        for col in 0..10 {
            let pixel = Vector2::new(20.0 + 31.0 * f64::from(col), 15.0 + 30.0 * f64::from(row));
            let d = ray(&c, reference, pixel);
            let Some(t) = plane_depth(reference, &d, 0.0) else {
                continue;
            };
            let world = reference.translation.vector + d * t;
            if let Some(q) = seen.project(&to_camera(truth), world)
                && q.x > 2.0
                && q.y > 2.0
                && q.x < 317.0
                && q.y < 237.0
            {
                pairs.push(PixelMatch {
                    reference: pixel,
                    query: q,
                });
            }
        }
    }
    pairs
}

fn noise(seed: &mut u64) -> Vector2<f64> {
    let mut next = || {
        *seed = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        ((*seed >> 11) as f64 / (1_u64 << 53) as f64 - 0.5) * 0.6
    };
    Vector2::new(next(), next())
}

/// The true camera at frame `i`: about 2 m per frame, a climb, a turn, and
/// slow real tilt changes of a few degrees.
pub fn truth(i: usize) -> Pose {
    let t = i as f64;
    let heading = 0.3 + if i > 120 { (t - 120.0) * 0.012 } else { 0.0 };
    let along = 2.0 * t;
    let base = Vector3::new(along * heading.cos(), along * heading.sin(), 0.0);
    let height = 100.0 + 30.0 * (t / 300.0 * std::f64::consts::PI).sin();
    let position = Vector3::new(base.x, base.y, dem(base.x, base.y) + height);
    let tilt =
        UnitQuaternion::from_euler_angles(0.06 * (t / 37.0).sin(), 0.05 * (t / 23.0).cos(), 0.0);
    let level = nadir(position, heading);
    Pose::from_parts(level.translation, tilt * level.rotation)
}

/// Angle between the true and the estimated camera view axes, in degrees.
pub fn view_error_deg(estimate: &Pose, truth: &Pose) -> f64 {
    let axis = Vector3::new(0.0, 0.0, -1.0);
    (estimate.rotation * axis)
        .angle(&(truth.rotation * axis))
        .to_degrees()
}

/// Matches between a map reference and the true camera where the map shows
/// objects above the elevation model. The host drapes map imagery over the
/// bare earth, so a roof in the map gets the ground depth, while the camera
/// sees the roof top. The elevation model also has a local slope error
/// `slope_error` (east, north) about the ground under the reference centre.
/// Pixel noise is deterministic.
pub fn object_map_matches(
    reference: &Pose,
    truth: &Pose,
    slope_error: Vector2<f64>,
    seed: &mut u64,
) -> Vec<PixelMatch> {
    let (c, seen) = (camera(), true_camera());
    let Some(centre) = centre_ground(reference) else {
        return Vec::new();
    };
    let mut pairs = Vec::new();
    for row in 0..8 {
        for col in 0..10 {
            let pixel = Vector2::new(20.0 + 31.0 * f64::from(col), 15.0 + 30.0 * f64::from(row));
            let d = ray(&c, reference, pixel);
            let Some(t) = plane_depth(reference, &d, 0.0) else {
                continue;
            };
            let ground = reference.translation.vector + d * t;
            let lift = if roof(ground.x, ground.y) {
                ROOF_M
            } else {
                0.0
            };
            let slope = slope_error.dot(&(ground - centre).xy());
            let world = ground + Vector3::new(0.0, 0.0, lift + slope);
            if let Some(q) = seen.project(&to_camera(truth), world)
                && q.x > 2.0
                && q.y > 2.0
                && q.x < 317.0
                && q.y < 237.0
            {
                pairs.push(PixelMatch {
                    reference: pixel,
                    query: q + noise(seed),
                });
            }
        }
    }
    pairs
}

/// The bare-earth point under the image centre: the ground that a mosaic
/// puts at the centre of the frame.
pub fn centre_ground(pose: &Pose) -> Option<Vector3<f64>> {
    let c = camera();
    let d = ray(&c, pose, Vector2::new(c.cx, c.cy));
    plane_depth(pose, &d, 0.0).map(|t| pose.translation.vector + d * t)
}
