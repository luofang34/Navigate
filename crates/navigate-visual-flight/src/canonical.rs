//! Scale-adaptive canonical tile poses and frustum tile selection for visual corridors.

use crate::surface::Surface;
use nalgebra::Matrix3;
use std::collections::BTreeSet;

/// Coarse geographic waypoint prior used to select corridor tiles.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PriorWaypoint {
    /// Latitude in degrees.
    pub lat_deg: f64,
    /// Longitude in degrees.
    pub lon_deg: f64,
    /// Optional altitude above mean sea level in meters.
    pub alt_msl_m: Option<f64>,
    /// Optional compass heading in degrees clockwise from North.
    pub heading_deg: Option<f64>,
}

/// Compute canonical camera poses across shallow, mid, steep oblique, and nadir view bands.
///
/// When `include_obliques` is true and `tile_m <= agl_m * 2.5`, this function emits `31` poses
/// centered on the tile (`12` at `12 deg`, `12` at `28 deg`, `6` at `48 deg`, plus `1` nadir).
/// When `tile_m > agl_m * 2.5`, this function targets a `2 x 2` sub-tile grid at
/// `[(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)] * tile_m` so wide corridor tiles
/// preserve ground-sample density across the entire tile footprint.
#[must_use]
pub fn canonical_tile_poses(
    surface: &Surface,
    (ce, cn, ground_msl): (f64, f64, f64),
    tile_m: f64,
    agl_m: f64,
    include_obliques: bool,
) -> Vec<(u8, [f64; 3], Matrix3<f64>)> {
    let [clat, clon, _] = surface.geodetic([ce, cn, 0.0]);
    let nadir_eye = surface.local(clat, clon, ground_msl + (tile_m * 1.45).max(agl_m));
    let nadir_rot = surface.camera_rotation(clat, clon, 0.0, 89.0, 0.0);
    if !include_obliques {
        if !cfg!(debug_assertions) && agl_m <= 250.0 && tile_m * 1.45 > agl_m * 1.6 {
            let sub_m = tile_m * 0.5;
            let sub_agl = (sub_m * 1.45).max(agl_m);
            let mut sub_views = Vec::with_capacity(4);
            for sy in [-0.25_f64, 0.25_f64] {
                for sx in [-0.25_f64, 0.25_f64] {
                    let [slat, slon, _] =
                        surface.geodetic([ce + sx * tile_m, cn + sy * tile_m, 0.0]);
                    let sdem = surface.sample_dem(slat, slon).unwrap_or(ground_msl);
                    let seye = surface.local(slat, slon, sdem + sub_agl);
                    let srot = surface.camera_rotation(slat, slon, 0.0, 89.0, 0.0);
                    sub_views.push((0, seye, srot));
                }
            }
            return sub_views;
        }
        return vec![(0, nadir_eye, nadir_rot)];
    }

    let min_clear_m = agl_m.clamp(25.0, 140.0);
    let wide_tile = tile_m > agl_m * 2.5;
    let sub_offsets: &[(f64, f64)] = if wide_tile {
        &[(-0.25, -0.25), (0.25, -0.25), (-0.25, 0.25), (0.25, 0.25)]
    } else {
        &[(0.0, 0.0)]
    };
    let sub_m = if wide_tile { tile_m * 0.5 } else { tile_m };
    let mut out = Vec::with_capacity(if wide_tile { 68 } else { 31 });

    for &(sx, sy) in sub_offsets {
        let te = ce + sx * tile_m;
        let tn = cn + sy * tile_m;
        let [tlat, tlon, _] = surface.geodetic([te, tn, 0.0]);
        let t_ground = surface.sample_dem(tlat, tlon).unwrap_or(ground_msl);
        let mut band = 0_u8;
        let oblique_rings: &[(f64, u32, f64)] = if wide_tile {
            // 6 headings at 12 deg, 6 at 28 deg, 4 at 48 deg + 1 sub-nadir = 17 views x 4 = 68 views.
            &[
                (12.0_f64, 6_u32, 60.0_f64),
                (28.0_f64, 6_u32, 60.0_f64),
                (48.0_f64, 4_u32, 90.0_f64),
            ]
        } else {
            &[
                (12.0_f64, 12_u32, 30.0_f64),
                (28.0_f64, 12_u32, 30.0_f64),
                (48.0_f64, 6_u32, 60.0_f64),
            ]
        };
        for &(down_deg, heading_count, heading_step_deg) in oblique_rings {
            let horiz_back = agl_m / down_deg.to_radians().tan();
            for step in 0..heading_count {
                let heading_deg = f64::from(step) * heading_step_deg;
                let h = heading_deg.to_radians();
                let eye_e = te - h.sin() * horiz_back;
                let eye_n = tn - h.cos() * horiz_back;
                let [elat, elon, _] = surface.geodetic([eye_e, eye_n, 0.0]);
                let base_dem = surface.sample_dem(elat, elon).unwrap_or(t_ground);
                let eye_alt = (t_ground + agl_m).max(base_dem + min_clear_m);
                let eye = surface.local(elat, elon, eye_alt);
                let rot = surface.camera_rotation(elat, elon, heading_deg, down_deg, 0.0);
                out.push((band, eye, rot));
                band = band.saturating_add(1);
            }
        }
        let sub_nadir_eye = surface.local(tlat, tlon, t_ground + (sub_m * 1.45).max(agl_m));
        let sub_nadir_rot = surface.camera_rotation(tlat, tlon, 0.0, 89.0, 0.0);
        out.push((band, sub_nadir_eye, sub_nadir_rot));
    }
    out
}

/// Select corridor tile `(tile_x, tile_y)` keys around `priors` with forward frustum rays.
#[must_use]
pub fn select_corridor_tiles_from_priors(
    surface: &Surface,
    priors: &[PriorWaypoint],
    tile_m: f64,
    radius_m: f64,
    nominal_agl_m: f64,
) -> Vec<(i32, i32)> {
    let sw = surface.local(surface.bbox[1], surface.bbox[0], 0.0);
    let ne = surface.local(surface.bbox[3], surface.bbox[2], 0.0);
    let cat_min_x = (sw[0].min(ne[0]) / tile_m).floor() as i32;
    let cat_max_x = (sw[0].max(ne[0]) / tile_m).floor() as i32;
    let cat_min_y = (sw[1].min(ne[1]) / tile_m).floor() as i32;
    let cat_max_y = (sw[1].max(ne[1]) / tile_m).floor() as i32;

    let mut selected = BTreeSet::new();
    let r = radius_m.max(0.0);
    let sector_dist_m = (nominal_agl_m * 6.0).clamp(2_500.0, 6_000.0);
    for wp in priors {
        let p = surface.local(wp.lat_deg, wp.lon_deg, 0.0);
        let x0 = (((p[0] - r) / tile_m).floor() as i32).max(cat_min_x);
        let x1 = (((p[0] + r) / tile_m).floor() as i32).min(cat_max_x);
        let y0 = (((p[1] - r) / tile_m).floor() as i32).max(cat_min_y);
        let y1 = (((p[1] + r) / tile_m).floor() as i32).min(cat_max_y);
        for ty in y0..=y1 {
            for tx in x0..=x1 {
                selected.insert((tx, ty));
            }
        }

        if let Some(hdg_deg) = wp.heading_deg.filter(|d| d.is_finite()) {
            let hdg_rad = hdg_deg.to_radians();
            let fwd_e = hdg_rad.sin();
            let fwd_n = hdg_rad.cos();
            let sx0 = (((p[0] - sector_dist_m) / tile_m).floor() as i32).max(cat_min_x);
            let sx1 = (((p[0] + sector_dist_m) / tile_m).floor() as i32).min(cat_max_x);
            let sy0 = (((p[1] - sector_dist_m) / tile_m).floor() as i32).max(cat_min_y);
            let sy1 = (((p[1] + sector_dist_m) / tile_m).floor() as i32).min(cat_max_y);
            let cos_half_sector = std::f64::consts::FRAC_1_SQRT_2;
            for ty in sy0..=sy1 {
                let dy = (f64::from(ty) + 0.5) * tile_m - p[1];
                for tx in sx0..=sx1 {
                    let dx = (f64::from(tx) + 0.5) * tile_m - p[0];
                    let dist = dx.hypot(dy);
                    if dist <= sector_dist_m
                        && (dist <= 1.0e-6 || (dx * fwd_e + dy * fwd_n) >= dist * cos_half_sector)
                    {
                        selected.insert((tx, ty));
                    }
                }
            }

            let dem = surface.sample_dem(wp.lat_deg, wp.lon_deg).unwrap_or(0.0);
            let alt = wp.alt_msl_m.unwrap_or(dem + nominal_agl_m.max(500.0));
            let agl_m = (alt - dem).max(10.0);
            let horizon_dist_m = (2.0 * 6_371_000.0 * agl_m).sqrt();
            let max_ray_dist_m = 12_000.0_f64.min(horizon_dist_m).max(4_000.0);

            for fov_offset in [
                -std::f64::consts::FRAC_PI_6,
                0.0,
                std::f64::consts::FRAC_PI_6,
            ] {
                let ray_hdg = hdg_rad + fov_offset;
                for step_dist in (500..=(max_ray_dist_m as i32)).step_by(500) {
                    let d = f64::from(step_dist);
                    let rx = p[0] + d * ray_hdg.sin();
                    let ry = p[1] + d * ray_hdg.cos();
                    let tx = (rx / tile_m).floor() as i32;
                    let ty = (ry / tile_m).floor() as i32;
                    if tx >= cat_min_x && tx <= cat_max_x && ty >= cat_min_y && ty <= cat_max_y {
                        selected.insert((tx, ty));
                    }
                }
            }
        }
    }
    if selected.is_empty() {
        for ty in cat_min_y..=cat_max_y {
            for tx in cat_min_x..=cat_max_x {
                selected.insert((tx, ty));
            }
        }
    }
    selected.into_iter().collect()
}
