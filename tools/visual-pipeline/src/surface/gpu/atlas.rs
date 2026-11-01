//! Resident DEM and image atlas builders and `<= 60 x 60` sub-atlas windowing for GPU rendering.

use super::Surface;
use image::imageops::{FilterType, colorops::grayscale, resize};
use navigate_visual_render::{AtlasLayout, ImageAtlas, TerrainAtlas};

pub(super) const GPU_TILE_PIXELS: u32 = 256;
pub(super) const MAX_ATLAS_DIM_PX: u32 = 16_384;
pub(super) const MAX_SUBATLAS_TILES: u32 = 60;
pub(super) const SUBATLAS_CORE_MARGIN_TILES: u32 = 10;
pub(super) const MAX_SCENE_BYTES: usize = 1536 * 1024 * 1024;

/// Return `[min_x, max_x, min_y, max_y]` Web Mercator imagery tile bounds of `surface`.
pub(super) fn surface_imagery_bounds(surface: &Surface) -> Option<[u32; 4]> {
    if surface.imagery_tiles.is_empty() {
        return None;
    }
    let (mut min_x, mut max_x) = (u32::MAX, 0_u32);
    let (mut min_y, mut max_y) = (u32::MAX, 0_u32);
    for &(tx, ty) in surface.imagery_tiles.keys() {
        min_x = min_x.min(tx);
        max_x = max_x.max(tx);
        min_y = min_y.min(ty);
        max_y = max_y.max(ty);
    }
    Some([min_x, max_x, min_y, max_y])
}

fn clamp_1d_window(min_b: u32, max_b: u32, center: u32, max_span: u32) -> (u32, u32) {
    let span = max_b.saturating_sub(min_b).saturating_add(1);
    if span <= max_span {
        return (min_b, max_b);
    }
    let half = max_span / 2;
    let mut start = center.saturating_sub(half).max(min_b);
    let mut end = start.saturating_add(max_span - 1);
    if end > max_b {
        end = max_b;
        start = end.saturating_sub(max_span - 1).max(min_b);
    }
    (start, end)
}

/// Compute a `<= max_span x max_span` imagery tile window inside `bounds` centered on `(cx, cy)`.
pub(super) fn subatlas_window_around(
    bounds: [u32; 4],
    cx: u32,
    cy: u32,
    max_span: u32,
) -> [u32; 4] {
    let span = max_span.max(1);
    let (wx0, wx1) = clamp_1d_window(bounds[0], bounds[1], cx, span);
    let (wy0, wy1) = clamp_1d_window(bounds[2], bounds[3], cy, span);
    [wx0, wx1, wy0, wy1]
}

/// Return `true` when `(cx, cy)` stays inside `window` without needing a sub-atlas shift.
pub(super) fn subatlas_contains_comfortably(
    bounds: [u32; 4],
    window: [u32; 4],
    cx: u32,
    cy: u32,
) -> bool {
    let mx = SUBATLAS_CORE_MARGIN_TILES;
    let in_x = (cx >= window[0].saturating_add(mx) || window[0] == bounds[0])
        && (cx <= window[1].saturating_sub(mx) || window[1] == bounds[1]);
    let in_y = (cy >= window[2].saturating_add(mx) || window[2] == bounds[2])
        && (cy <= window[3].saturating_sub(mx) || window[3] == bounds[3]);
    in_x && in_y
}

/// Construct a full-surface image atlas when all imagery tiles fit within `MAX_ATLAS_DIM_PX`.
pub(super) fn build_image_atlas(surface: &Surface) -> Option<ImageAtlas> {
    let bounds = surface_imagery_bounds(surface)?;
    let cols = bounds[1].checked_sub(bounds[0])?.checked_add(1)?;
    let rows = bounds[3].checked_sub(bounds[2])?.checked_add(1)?;
    if cols > MAX_SUBATLAS_TILES || rows > MAX_SUBATLAS_TILES {
        return None;
    }
    build_image_atlas_window(surface, bounds, MAX_ATLAS_DIM_PX)
}

/// Construct an image atlas restricted to `window = [wx0, wx1, wy0, wy1]` and `max_tex_dim`.
pub(super) fn build_image_atlas_window(
    surface: &Surface,
    window: [u32; 4],
    max_tex_dim: u32,
) -> Option<ImageAtlas> {
    let (mut min_x, mut max_x) = (u32::MAX, 0_u32);
    let (mut min_y, mut max_y) = (u32::MAX, 0_u32);
    let mut matched = false;
    for &(tx, ty) in surface.imagery_tiles.keys() {
        if tx >= window[0] && tx <= window[1] && ty >= window[2] && ty <= window[3] {
            min_x = min_x.min(tx);
            max_x = max_x.max(tx);
            min_y = min_y.min(ty);
            max_y = max_y.max(ty);
            matched = true;
        }
    }
    if !matched {
        return None;
    }
    let cols = max_x.checked_sub(min_x)?.checked_add(1)?;
    let rows = max_y.checked_sub(min_y)?.checked_add(1)?;
    let width = cols.checked_mul(GPU_TILE_PIXELS)?;
    let height = rows.checked_mul(GPU_TILE_PIXELS)?;
    if width > max_tex_dim || height > max_tex_dim {
        return None;
    }
    let total = (width as usize).checked_mul(height as usize)?;
    if total.saturating_mul(2) > MAX_SCENE_BYTES {
        return None;
    }
    let mut pixels = vec![0_u8; total];
    let mut coverage = vec![0_u8; total];
    let tp = GPU_TILE_PIXELS as usize;
    let stride = width as usize;

    for (&(tx, ty), rgb) in &surface.imagery_tiles {
        if tx < min_x || tx > max_x || ty < min_y || ty > max_y {
            continue;
        }
        let gray = grayscale(rgb);
        let tile = if gray.dimensions() == (GPU_TILE_PIXELS, GPU_TILE_PIXELS) {
            gray
        } else {
            resize(
                &gray,
                GPU_TILE_PIXELS,
                GPU_TILE_PIXELS,
                FilterType::Triangle,
            )
        };
        let ox = ((tx - min_x) as usize) * tp;
        let oy = ((ty - min_y) as usize) * tp;
        let raw = tile.as_raw();
        for r in 0..tp {
            let dst = (oy + r) * stride + ox;
            pixels[dst..dst + tp].copy_from_slice(&raw[r * tp..(r + 1) * tp]);
            coverage[dst..dst + tp].fill(255);
        }
    }
    Some(ImageAtlas {
        layout: AtlasLayout {
            zoom: surface.imagery_zoom,
            origin_tile: [min_x, min_y],
            tile_pixels: GPU_TILE_PIXELS,
            size: [width, height],
        },
        pixels,
        coverage,
    })
}

/// Construct a full-surface terrain elevation atlas when DEM tiles fit within `MAX_ATLAS_DIM_PX`.
pub(super) fn build_terrain_atlas(surface: &Surface) -> Option<TerrainAtlas> {
    if surface.dem_tiles.is_empty() {
        return None;
    }
    let (mut min_x, mut max_x) = (u32::MAX, 0_u32);
    let (mut min_y, mut max_y) = (u32::MAX, 0_u32);
    for &(tx, ty) in surface.dem_tiles.keys() {
        min_x = min_x.min(tx);
        max_x = max_x.max(tx);
        min_y = min_y.min(ty);
        max_y = max_y.max(ty);
    }
    let cols = max_x.checked_sub(min_x)?.checked_add(1)?;
    let rows = max_y.checked_sub(min_y)?.checked_add(1)?;
    if cols > MAX_SUBATLAS_TILES || rows > MAX_SUBATLAS_TILES {
        return None;
    }
    build_terrain_atlas_window(surface, [min_x, max_x, min_y, max_y], MAX_ATLAS_DIM_PX)
}

/// Construct a terrain elevation atlas covering `img_window` (in `surface.imagery_zoom` tiles).
pub(super) fn build_terrain_atlas_for_imagery_window(
    surface: &Surface,
    img_window: [u32; 4],
    max_tex_dim: u32,
) -> Option<TerrainAtlas> {
    if surface.dem_tiles.is_empty() {
        return None;
    }
    let (mut dem_min_x, mut dem_max_x) = (u32::MAX, 0_u32);
    let (mut dem_min_y, mut dem_max_y) = (u32::MAX, 0_u32);
    for &(tx, ty) in surface.dem_tiles.keys() {
        dem_min_x = dem_min_x.min(tx);
        dem_max_x = dem_max_x.max(tx);
        dem_min_y = dem_min_y.min(ty);
        dem_max_y = dem_max_y.max(ty);
    }
    let (wx0, wx1, wy0, wy1) = if surface.imagery_zoom >= surface.dem_zoom {
        let shift = (surface.imagery_zoom - surface.dem_zoom).min(24);
        (
            (img_window[0] >> shift).saturating_sub(2),
            (img_window[1] >> shift).saturating_add(2),
            (img_window[2] >> shift).saturating_sub(2),
            (img_window[3] >> shift).saturating_add(2),
        )
    } else {
        let shift = (surface.dem_zoom - surface.imagery_zoom).min(24);
        let scale = 1_u32 << shift;
        (
            img_window[0].saturating_mul(scale).saturating_sub(2),
            img_window[1]
                .saturating_add(1)
                .saturating_mul(scale)
                .saturating_add(1),
            img_window[2].saturating_mul(scale).saturating_sub(2),
            img_window[3]
                .saturating_add(1)
                .saturating_mul(scale)
                .saturating_add(1),
        )
    };
    let cx = wx0.midpoint(wx1);
    let cy = wy0.midpoint(wy1);
    let bounded = subatlas_window_around(
        [
            dem_min_x.max(wx0),
            dem_max_x.min(wx1).max(dem_min_x.max(wx0)),
            dem_min_y.max(wy0),
            dem_max_y.min(wy1).max(dem_min_y.max(wy0)),
        ],
        cx,
        cy,
        MAX_SUBATLAS_TILES,
    );
    build_terrain_atlas_window(surface, bounded, max_tex_dim).or_else(|| {
        let fallback = subatlas_window_around(
            [dem_min_x, dem_max_x, dem_min_y, dem_max_y],
            cx,
            cy,
            MAX_SUBATLAS_TILES,
        );
        build_terrain_atlas_window(surface, fallback, max_tex_dim)
    })
}

fn build_terrain_atlas_window(
    surface: &Surface,
    window: [u32; 4],
    max_tex_dim: u32,
) -> Option<TerrainAtlas> {
    let (mut min_x, mut max_x) = (u32::MAX, 0_u32);
    let (mut min_y, mut max_y) = (u32::MAX, 0_u32);
    let mut matched = false;
    for &(tx, ty) in surface.dem_tiles.keys() {
        if tx >= window[0] && tx <= window[1] && ty >= window[2] && ty <= window[3] {
            min_x = min_x.min(tx);
            max_x = max_x.max(tx);
            min_y = min_y.min(ty);
            max_y = max_y.max(ty);
            matched = true;
        }
    }
    if !matched {
        return None;
    }
    let cols = max_x.checked_sub(min_x)?.checked_add(1)?;
    let rows = max_y.checked_sub(min_y)?.checked_add(1)?;
    let width = cols.checked_mul(GPU_TILE_PIXELS)?;
    let height = rows.checked_mul(GPU_TILE_PIXELS)?;
    if width > max_tex_dim || height > max_tex_dim {
        return None;
    }
    let total = (width as usize).checked_mul(height as usize)?;
    if total.saturating_mul(4) > MAX_SCENE_BYTES {
        return None;
    }
    let mut heights_m = vec![0.0_f32; total];
    let mut coverage = vec![0_u8; total];
    let tp = GPU_TILE_PIXELS as usize;
    let stride = width as usize;
    let src_s = surface.dem_tile_size.max(1) as usize;

    for (&(tx, ty), elev) in &surface.dem_tiles {
        if tx < min_x || tx > max_x || ty < min_y || ty > max_y || elev.len() != src_s * src_s {
            continue;
        }
        let resampled = resample_dem_tile(elev, src_s, tp);
        let ox = ((tx - min_x) as usize) * tp;
        let oy = ((ty - min_y) as usize) * tp;
        for r in 0..tp {
            let dst = (oy + r) * stride + ox;
            heights_m[dst..dst + tp].copy_from_slice(&resampled[r * tp..(r + 1) * tp]);
            coverage[dst..dst + tp].fill(255);
        }
    }
    Some(TerrainAtlas {
        layout: AtlasLayout {
            zoom: surface.dem_zoom,
            origin_tile: [min_x, min_y],
            tile_pixels: GPU_TILE_PIXELS,
            size: [width, height],
        },
        heights_m,
        coverage,
    })
}

/// Resample a DEM tile to the destination tile dimension.
pub(super) fn resample_dem_tile(src: &[f32], src_s: usize, dst_s: usize) -> Vec<f32> {
    let clean = |v: f32| {
        if v.is_finite() {
            v.clamp(-99_000.0, 99_000.0)
        } else {
            0.0
        }
    };
    if src_s == dst_s {
        return src.iter().copied().map(clean).collect();
    }
    let mut out = vec![0.0_f32; dst_s * dst_s];
    for y in 0..dst_s {
        let sy = ((y as f64 + 0.5) * (src_s as f64) / (dst_s as f64) - 0.5)
            .clamp(0.0, (src_s - 1) as f64);
        let y0 = sy.floor() as usize;
        let y1 = (y0 + 1).min(src_s - 1);
        let fy = (sy - (y0 as f64)) as f32;
        for x in 0..dst_s {
            let sx = ((x as f64 + 0.5) * (src_s as f64) / (dst_s as f64) - 0.5)
                .clamp(0.0, (src_s - 1) as f64);
            let x0 = sx.floor() as usize;
            let x1 = (x0 + 1).min(src_s - 1);
            let fx = (sx - (x0 as f64)) as f32;
            let h00 = clean(src[y0 * src_s + x0]);
            let h10 = clean(src[y0 * src_s + x1]);
            let h01 = clean(src[y1 * src_s + x0]);
            let h11 = clean(src[y1 * src_s + x1]);
            out[y * dst_s + x] = h00 * (1.0 - fx) * (1.0 - fy)
                + h10 * fx * (1.0 - fy)
                + h01 * (1.0 - fx) * fy
                + h11 * fx * fy;
        }
    }
    out
}

/// Sample optical-axis depth at subpixel coordinates.
pub(super) fn sample_depth(
    depth_m: &[f32],
    width: u32,
    height: u32,
    u: f64,
    v: f64,
) -> Option<f32> {
    if !u.is_finite() || !v.is_finite() || width == 0 || height == 0 {
        return None;
    }
    let (w, h) = (width as usize, height as usize);
    if u < 0.0 || v < 0.0 || u > f64::from(width - 1) || v > f64::from(height - 1) {
        return None;
    }
    let x0 = (u.floor() as usize).min(w - 1);
    let y0 = (v.floor() as usize).min(h - 1);
    let x1 = (x0 + 1).min(w - 1);
    let y1 = (y0 + 1).min(h - 1);
    let d00 = depth_m[y0 * w + x0];
    let d10 = depth_m[y0 * w + x1];
    let d01 = depth_m[y1 * w + x0];
    let d11 = depth_m[y1 * w + x1];
    if d00 > 0.5 && d10 > 0.5 && d01 > 0.5 && d11 > 0.5 {
        let d_min = d00.min(d10).min(d01).min(d11);
        let d_max = d00.max(d10).max(d01).max(d11);
        if (d_max - d_min) <= 0.025 * d_max {
            let fx = (u - (x0 as f64)) as f32;
            let fy = (v - (y0 as f64)) as f32;
            return Some(
                d00 * (1.0 - fx) * (1.0 - fy)
                    + d10 * fx * (1.0 - fy)
                    + d01 * (1.0 - fx) * fy
                    + d11 * fx * fy,
            );
        }
    }
    let xn = (u.round() as usize).min(w - 1);
    let yn = (v.round() as usize).min(h - 1);
    let dn = depth_m[yn * w + xn];
    (dn > 0.5).then_some(dn)
}
