//! A quick check of one ranked place: the photo on flat ground against a map window.
use super::project::{View, grey};
use super::similarity::fit;
use crate::{
    LocateError,
    area::tile_xy,
    scene::{self, SceneRequest, TileSource},
};
use image::GrayImage;
use nalgebra::Vector2;
use navigate_visual::AsyncImageMatcher;
use navigate_visual_render::ImageAtlas;

/// Matcher input size.
const WINDOW: [u32; 2] = [640, 480];
const EQUATOR_M: f64 = 40_075_016.686;

/// The result of one check.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Check {
    /// Matches that one similarity explains.
    pub inliers: usize,
    /// Latitude and longitude of the ground at the footprint centre, after the fit.
    pub ground: [f64; 2],
    /// Heading after the fit, degrees.
    pub heading_deg: f64,
}

/// Imagery zoom whose pixels make a window about 1.25 crop sides wide.
pub fn zoom_for(side_m: f64, latitude: f64) -> u32 {
    let gsd = side_m * 1.25 / f64::from(WINDOW[0]);
    let zoom = (EQUATOR_M * latitude.to_radians().cos() / (256.0 * gsd))
        .log2()
        .round();
    zoom.clamp(12.0, 18.0) as u32
}

fn ground_sample_m(atlas: &ImageAtlas, latitude: f64) -> f64 {
    let n = f64::from(1_u32 << atlas.layout.zoom) * f64::from(atlas.layout.tile_pixels);
    EQUATOR_M / n * latitude.to_radians().cos()
}

fn atlas_pixel(atlas: &ImageAtlas, point: [f64; 2]) -> Vector2<f64> {
    let [tx, ty] = tile_xy(point, atlas.layout.zoom);
    let scale = f64::from(atlas.layout.tile_pixels);
    Vector2::new(
        (tx - f64::from(atlas.layout.origin_tile[0])) * scale,
        (ty - f64::from(atlas.layout.origin_tile[1])) * scale,
    )
}

fn lat_lon(atlas: &ImageAtlas, pixel: Vector2<f64>) -> [f64; 2] {
    let layout = &atlas.layout;
    let scale = f64::from(layout.tile_pixels);
    let n = f64::from(1_u32 << layout.zoom);
    let tx = f64::from(layout.origin_tile[0]) + pixel.x / scale;
    let ty = f64::from(layout.origin_tile[1]) + pixel.y / scale;
    let lat = (std::f64::consts::PI * (1.0 - 2.0 * ty / n))
        .sinh()
        .atan()
        .to_degrees();
    [lat, tx / n * 360.0 - 180.0]
}

/// Histogram-equalized grey values; zero stays zero, so empty ground stays empty.
fn equalize(mut image: GrayImage) -> GrayImage {
    let mut histogram = [0_usize; 256];
    for p in image.pixels() {
        histogram[usize::from(p.0[0])] += 1;
    }
    let total: usize = histogram[1..].iter().sum();
    if total == 0 {
        return image;
    }
    let mut map = [0_u8; 256];
    let mut running = 0;
    for level in 1..256 {
        running += histogram[level];
        map[level] = (1 + running * 254 / total) as u8;
    }
    image
        .pixels_mut()
        .for_each(|p| p.0[0] = map[usize::from(p.0[0])]);
    image
}

/// Match the photo, laid on flat ground with `view`, against the map window at
/// `place`, the ground at the footprint centre that the ranking proposed.
///
/// # Errors
///
/// Returns an error for a failed tile fetch or a matcher failure.
pub async fn check<S: TileSource, M: AsyncImageMatcher>(
    source: &S,
    matcher: &mut M,
    photo: &GrayImage,
    view: &View,
    footprint: [f64; 2],
    place: [f64; 2],
    side_m: f64,
) -> Result<Option<Check>, LocateError> {
    let request = SceneRequest {
        anchor: place,
        radius_m: side_m,
        imagery_zoom: zoom_for(side_m, place[0]),
        terrain_zoom: 10,
        imagery_tile_pixels: 256,
        max_bytes: 256 << 20,
        concurrency: 16,
    };
    let scene = scene::build(source, &request, |_, _| {}).await?;
    let atlas = &scene.imagery;
    let gsd = ground_sample_m(atlas, place[0]);
    let [w, h] = WINDOW;
    let corner = atlas_pixel(atlas, place) - Vector2::new(f64::from(w) / 2.0, f64::from(h) / 2.0);
    let [aw, ah] = atlas.layout.size;
    let window = equalize(GrayImage::from_fn(w, h, |u, v| {
        let (x, y) = (corner.x + f64::from(u), corner.y + f64::from(v));
        if x < 0.0 || y < 0.0 || x >= f64::from(aw) || y >= f64::from(ah) {
            return image::Luma([0]);
        }
        let i = y as usize * aw as usize + x as usize;
        image::Luma([atlas.pixels.get(i).copied().unwrap_or(0)])
    }));
    let projection = equalize(grey(photo, view, footprint, gsd, WINDOW));
    let matches = matcher.match_images(&window, &projection).await?;
    let Some((inliers, model)) = fit(&matches) else {
        return Ok(None);
    };
    let centre = Vector2::new(f64::from(w) / 2.0, f64::from(h) / 2.0);
    Ok(Some(Check {
        inliers,
        ground: lat_lon(atlas, model.apply(centre) + corner),
        heading_deg: (view.heading_deg + model.angle_deg()).rem_euclid(360.0),
    }))
}
