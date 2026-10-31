//! Canonical tile poses and frustum DEM raycasting for visual corridor navigation.

use crate::{surface::Surface, testset::PhotoCase};
use nalgebra::Matrix3;
use std::collections::BTreeSet;

/// Computes canonical poses across shallow oblique, oblique, and nadir viewing bands.
pub(crate) fn canonical_tile_poses(
    surface: &Surface,
    (ce, cn, ground_msl): (f64, f64, f64),
    tile_m: f64,
    agl_m: f64,
    include_obliques: bool,
) -> Vec<(u8, [f64; 3], Matrix3<f64>)> {
    let [clat, clon, _] = surface.geodetic([ce, cn, 0.0]);
    let nadir_eye = surface.local(clat, clon, ground_msl + (tile_m * 0.9).max(agl_m));
    let nadir_rot = surface.camera_rotation(clat, clon, 0.0, 89.0, 0.0);
    if !include_obliques {
        return vec![(0, nadir_eye, nadir_rot)];
    }

    let mut out = Vec::with_capacity(37);
    let mut band = 0_u8;
    let min_clear_m = agl_m.clamp(25.0, 140.0);
    // Bands: 12.0 deg (shallow grazing), 26.0 deg (mid-oblique), 50.0 deg (steep oblique)
    for down_deg in [12.0_f64, 26.0_f64, 50.0_f64] {
        let horiz_back = agl_m / down_deg.to_radians().tan();
        for step in 0..12_u32 {
            let heading_deg = f64::from(step) * 30.0;
            let h = heading_deg.to_radians();
            let eye_e = ce - h.sin() * horiz_back;
            let eye_n = cn - h.cos() * horiz_back;
            let [elat, elon, _] = surface.geodetic([eye_e, eye_n, 0.0]);
            let base_dem = surface.sample_dem(elat, elon).unwrap_or(ground_msl);
            let eye_alt = (ground_msl + agl_m).max(base_dem + min_clear_m);
            let eye = surface.local(elat, elon, eye_alt);
            let rot = surface.camera_rotation(elat, elon, heading_deg, down_deg, 0.0);
            out.push((band, eye, rot));
            band += 1;
        }
    }
    out.push((band, nadir_eye, nadir_rot));
    out
}

/// Selects corridor tiles by intersecting camera frustum rays with the DEM surface out to 12 km.
pub(crate) fn select_corridor_tiles(
    surface: &Surface,
    cases: &[PhotoCase],
    tile_m: f64,
    radius_m: f64,
) -> Vec<(i32, i32)> {
    let sw = surface.local(surface.bbox[1], surface.bbox[0], 0.0);
    let ne = surface.local(surface.bbox[3], surface.bbox[2], 0.0);
    let cat_min_x = (sw[0].min(ne[0]) / tile_m).floor() as i32;
    let cat_max_x = (sw[0].max(ne[0]) / tile_m).floor() as i32;
    let cat_min_y = (sw[1].min(ne[1]) / tile_m).floor() as i32;
    let cat_max_y = (sw[1].max(ne[1]) / tile_m).floor() as i32;

    let mut selected = BTreeSet::new();
    let r = radius_m.max(0.0);
    for case in cases {
        if let Some(pos) = case.prior.as_ref().and_then(|p| p.position.as_ref()) {
            let p = surface.local(pos.latitude, pos.longitude, 0.0);
            let x0 = (((p[0] - r) / tile_m).floor() as i32).max(cat_min_x);
            let x1 = (((p[0] + r) / tile_m).floor() as i32).min(cat_max_x);
            let y0 = (((p[1] - r) / tile_m).floor() as i32).max(cat_min_y);
            let y1 = (((p[1] + r) / tile_m).floor() as i32).min(cat_max_y);
            for ty in y0..=y1 {
                for tx in x0..=x1 {
                    selected.insert((tx, ty));
                }
            }

            // Raycast frustum ground rays out to 12 km (or horizon distance at high AGL)
            if let Some(hdg_deg) = case.prior.as_ref().and_then(|p| p.direction_deg) {
                let dem = surface
                    .sample_dem(pos.latitude, pos.longitude)
                    .unwrap_or(0.0);
                let alt = pos.altitude_m.unwrap_or(dem + 500.0);
                let agl_m = (alt - dem).max(10.0);
                let horizon_dist_m = (2.0 * 6_371_000.0 * agl_m).sqrt();
                let max_ray_dist_m = 12_000.0_f64.min(horizon_dist_m).max(4_000.0);

                let hdg_rad = hdg_deg.to_radians();
                // Raycast in 3 directions across horizontal field-of-view (+/- 30 deg)
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
                        if tx >= cat_min_x && tx <= cat_max_x && ty >= cat_min_y && ty <= cat_max_y
                        {
                            selected.insert((tx, ty));
                        }
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
