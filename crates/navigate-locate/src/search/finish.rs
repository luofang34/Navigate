//! The locate step at a checked place.
use super::{SearchProgress, View};
use crate::{LocateError, LocateRequest, LookAt, Outcome, SceneRequest, TileSource, locate};
use navigate_visual::{AsyncImageMatcher, CameraModel};

/// Long edge of the photo in the locate step, pixels: the matcher works at about this
/// size, and a portrait photo must not exceed it in height.
pub(crate) const LOCATE_EDGE: u32 = 640;

/// The photo reduced so that its long edge is at most [`LOCATE_EDGE`], and the focal
/// length at that size.
pub(super) fn working_size(gray: &image::GrayImage, focal_px: f64) -> (image::GrayImage, f64) {
    let (w, h) = gray.dimensions();
    let scale = f64::from(LOCATE_EDGE) / f64::from(w.max(h));
    if scale >= 1.0 {
        return (gray.clone(), focal_px);
    }
    let size = |v: u32| (f64::from(v) * scale).round().max(1.0) as u32;
    let small = image::imageops::resize(
        gray,
        size(w),
        size(h),
        image::imageops::FilterType::Triangle,
    );
    (small, focal_px * scale)
}

/// Where the locate step looks: a view hypothesis and the ground at its footprint
/// center.
pub(super) struct Aim {
    /// The view hypothesis.
    pub view: View,
    /// Latitude and longitude of the ground at the footprint center.
    pub ground: [f64; 2],
    /// Distance from the point below the camera to the footprint center, meters.
    pub centre_back_m: f64,
    /// How far the true ground can be from `ground`, meters: a ranked crop center is
    /// up to about three quarters of a crop side from the footprint center.
    pub slack_m: f64,
}

/// A fast locate step aimed where the view's optical axis meets the ground.
pub(super) async fn locate_at<T: TileSource, M: AsyncImageMatcher>(
    tiles: &T,
    matcher: M,
    gray: &image::GrayImage,
    aim: &Aim,
    focal_px: f64,
    progress: &mut impl FnMut(SearchProgress),
) -> Result<Outcome, LocateError> {
    let view = &aim.view;
    let axis_m = view.height_m / view.pitch_deg.clamp(5.0, 90.0).to_radians().tan();
    let heading = view.heading_deg.to_radians();
    let back = axis_m - aim.centre_back_m;
    let target = offset(aim.ground, [back * heading.sin(), back * heading.cos()]);
    let (w, h) = gray.dimensions();
    let camera = CameraModel {
        width: w,
        height: h,
        fx: focal_px,
        fy: focal_px,
        cx: (f64::from(w) - 1.0) / 2.0,
        cy: (f64::from(h) - 1.0) / 2.0,
    };
    let reach = axis_m.hypot(view.height_m);
    let radius_m = (reach * 1.3 + 300.0).clamp(300.0, 4_000.0);
    // A fast test of one ranked place: the horizon gives the camera angle within a
    // few degrees, the height prior within its spread, the heading within its step.
    let look_at = LookAt {
        heading_deg: view.heading_deg,
        heading_offsets_deg: vec![-10.0, 0.0, 10.0],
        pitches_deg: vec![view.pitch_deg],
        heights_m: vec![view.height_m * 0.85, view.height_m * 1.15],
    };
    let request = LocateRequest {
        image: gray.clone(),
        camera,
        target,
        look_at,
        scene: SceneRequest {
            anchor: target,
            radius_m,
            imagery_zoom: if radius_m <= 1_000.0 {
                18
            } else if radius_m <= 2_500.0 {
                17
            } else {
                16
            },
            terrain_zoom: 14,
            imagery_tile_pixels: 256,
            max_bytes: 512 << 20,
            concurrency: 16,
        },
        position_radius_m: (0.35 * reach).max(400.0) + aim.slack_m,
        attitude_radius_rad: 0.9,
    };
    let outcome = locate(tiles, matcher, request, |p| {
        progress(SearchProgress::Locating(p))
    })
    .await?;
    if let Outcome::NotFound {
        candidates,
        best_matches,
    } = &outcome
    {
        tracing::debug!(
            ?target,
            heading = view.heading_deg,
            candidates,
            best_matches,
            "no pose at this place"
        );
    }
    Ok(outcome)
}

/// Meters east and north of a point to latitude and longitude.
fn offset([lat, lon]: [f64; 2], [east, north]: [f64; 2]) -> [f64; 2] {
    const EARTH_M: f64 = 6_371_000.0;
    [
        lat + (north / EARTH_M).to_degrees(),
        lon + (east / (EARTH_M * lat.to_radians().cos())).to_degrees(),
    ]
}
