//! The locate step at a checked place.
use super::{Hypothesis, SearchProgress, check};
use crate::{LocateError, LocateRequest, LookAt, Outcome, SceneRequest, TileSource, locate};
use navigate_visual::{AsyncImageMatcher, CameraModel};

/// Photo width for the locate step, pixels; the matcher works at about this size.
const LOCATE_WIDTH: u32 = 640;

/// The photo reduced to the locate step's width, and the focal length at that size.
pub(super) fn working_size(gray: &image::GrayImage, focal_px: f64) -> (image::GrayImage, f64) {
    let (w, h) = gray.dimensions();
    if w <= LOCATE_WIDTH {
        return (gray.clone(), focal_px);
    }
    let scale = f64::from(LOCATE_WIDTH) / f64::from(w);
    let height = (f64::from(h) * scale).round().max(1.0) as u32;
    let small = image::imageops::resize(
        gray,
        LOCATE_WIDTH,
        height,
        image::imageops::FilterType::Triangle,
    );
    (small, focal_px * scale)
}

/// The locate pipeline aimed where the checked view's optical axis meets the ground.
pub(super) async fn locate_at<T: TileSource, M: AsyncImageMatcher>(
    tiles: &T,
    matcher: M,
    gray: &image::GrayImage,
    hypothesis: &Hypothesis,
    found: &check::Check,
    focal_px: f64,
    progress: &mut impl FnMut(SearchProgress),
) -> Result<Outcome, LocateError> {
    let view = &found.view;
    let axis_m = view.height_m / view.pitch_deg.clamp(5.0, 90.0).to_radians().tan();
    let along = |heading: f64, d: f64| {
        [
            d * heading.to_radians().sin(),
            d * heading.to_radians().cos(),
        ]
    };
    // From the footprint centre to the axis point, turned by the fitted heading change.
    let centre_back = hypothesis.footprint[0].hypot(hypothesis.footprint[1]);
    let shift = along(found.heading_deg, axis_m - centre_back);
    let target = offset(found.ground, shift);
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
    let request = LocateRequest {
        image: gray.clone(),
        camera,
        target,
        look_at: LookAt {
            heading_deg: found.heading_deg,
            heading_offsets_deg: vec![-8.0, 0.0, 8.0],
            pitches_deg: vec![view.pitch_deg - 6.0, view.pitch_deg, view.pitch_deg + 6.0],
            heights_m: vec![view.height_m * 0.8, view.height_m, view.height_m * 1.25],
        },
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
        position_radius_m: (0.35 * reach).max(400.0),
        attitude_radius_rad: 0.9,
    };
    tracing::debug!(
        ?target,
        heading = found.heading_deg,
        radius_m,
        inliers = found.inliers,
        "locating at the best place"
    );
    let outcome = locate(tiles, matcher, request, |p| {
        progress(SearchProgress::Locating(p))
    })
    .await?;
    if let Outcome::NotFound {
        candidates,
        best_matches,
    } = &outcome
    {
        tracing::debug!(candidates, best_matches, "the locate step found no pose");
    }
    Ok(outcome)
}

/// Metres east and north of a point to latitude and longitude.
fn offset([lat, lon]: [f64; 2], [east, north]: [f64; 2]) -> [f64; 2] {
    const EARTH_M: f64 = 6_371_000.0;
    [
        lat + (north / EARTH_M).to_degrees(),
        lon + (east / (EARTH_M * lat.to_radians().cos())).to_degrees(),
    ]
}
