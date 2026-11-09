//! Measured and predicted star directions, and the nearest-star match.

use nalgebra::{UnitQuaternion, Vector2, Vector3};
use navigate_camera::{CameraMount, ImagerModel};
use navigate_celestial::StarSight;

use crate::{StarCentroid, StarMatch};

/// A catalogue star with its direction in the body axes for one attitude.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Candidate {
    pub(crate) hr: u16,
    pub(crate) ecef_unit: Vector3<f64>,
    pub(crate) body_unit: Vector3<f64>,
}

/// The measured unit direction of each centroid in the body axes. A
/// centroid where the lens distortion does not invert gives `None`, so the
/// index of each direction stays the index of its centroid.
pub(crate) fn body_rays(
    centroids: &[StarCentroid],
    imager: &ImagerModel,
    mount: &CameraMount,
) -> Vec<Option<Vector3<f64>>> {
    centroids
        .iter()
        .map(|c| {
            let distorted = imager.intrinsics.normalize(&c.pixel);
            let ideal = imager.distortion.undistort(&distorted).ok()?;
            let ray = Vector3::new(ideal.x, ideal.y, 1.0).normalize();
            Some(mount.camera_to_body(&ray))
        })
        .collect()
}

/// One standard deviation of a measured direction in radians for a
/// centroid deviation of `centroid_sigma_px` on each image axis.
pub(crate) fn ray_sigma_rad(imager: &ImagerModel, centroid_sigma_px: f64) -> f64 {
    centroid_sigma_px / imager.intrinsics.fx.min(imager.intrinsics.fy)
}

/// The catalogue stars that the attitude `ecef_from_body` puts inside the
/// image, or less than `margin_rad` outside its edge.
pub(crate) fn predicted_in_view(
    catalogue: &[StarSight],
    imager: &ImagerModel,
    mount: &CameraMount,
    ecef_from_body: &UnitQuaternion<f64>,
    margin_rad: f64,
) -> Vec<Candidate> {
    let body_from_ecef = ecef_from_body.inverse();
    let k = &imager.intrinsics;
    let margin_px = margin_rad * k.fx.max(k.fy);
    let (low, high_x, high_y) = (
        -0.5 - margin_px,
        f64::from(k.width) - 0.5 + margin_px,
        f64::from(k.height) - 0.5 + margin_px,
    );
    catalogue
        .iter()
        .filter_map(|star| {
            let ecef_unit = Vector3::from(star.direction).normalize();
            let body_unit = body_from_ecef * ecef_unit;
            let camera = mount.cam_from_body * body_unit;
            if camera.z <= 0.0 {
                return None;
            }
            let ideal = Vector2::new(camera.x / camera.z, camera.y / camera.z);
            let pixel = k.denormalize(&imager.distortion.distort(&ideal));
            let inside = (low..high_x).contains(&pixel.x) && (low..high_y).contains(&pixel.y);
            inside.then_some(Candidate {
                hr: star.hr,
                ecef_unit,
                body_unit,
            })
        })
        .collect()
}

/// The nearest and the second-nearest distance from `target` to the items
/// of `others`, with the index of the nearest.
fn two_nearest<'a>(
    target: &Vector3<f64>,
    others: impl Iterator<Item = (usize, &'a Vector3<f64>)>,
) -> Option<(usize, f64, f64)> {
    let mut best: Option<(usize, f64, f64)> = None;
    for (index, other) in others {
        let angle = target.angle(other);
        best = Some(match best {
            None => (index, angle, f64::INFINITY),
            Some((_, first, second)) if angle < first => (index, angle, first.min(second)),
            Some((i, first, second)) => (i, first, second.min(angle)),
        });
    }
    best
}

/// Each measured direction with the nearest candidate inside `gate_rad`.
/// The match must be unique in both directions. When a second candidate is
/// inside the gate, the nearest distance must be at most `ratio` times the
/// second distance.
pub(crate) fn match_nearest(
    measured: &[Option<Vector3<f64>>],
    candidates: &[Candidate],
    gate_rad: f64,
    ratio: f64,
) -> Vec<StarMatch> {
    let accept = |first: f64, second: f64| {
        first < gate_rad && (second >= gate_rad || first <= ratio * second)
    };
    let rays = || {
        measured
            .iter()
            .enumerate()
            .filter_map(|(i, r)| r.as_ref().map(|r| (i, r)))
    };
    rays()
        .filter_map(|(index, ray)| {
            let star_directions = candidates.iter().map(|c| &c.body_unit).enumerate();
            let (star, first, second) = two_nearest(ray, star_directions)?;
            let candidate = candidates.get(star)?;
            if !accept(first, second) {
                return None;
            }
            let (back, back_first, back_second) = two_nearest(&candidate.body_unit, rays())?;
            (back == index && accept(back_first, back_second)).then_some(StarMatch {
                centroid_index: index,
                hr: candidate.hr,
                body_unit: *ray,
                ecef_unit: candidate.ecef_unit,
                separation_rad: first,
            })
        })
        .collect()
}
