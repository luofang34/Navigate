//! Canonical tile poses and frustum DEM raycasting for visual corridor navigation.

use crate::{surface::Surface, testset::PhotoCase};
use nalgebra::Matrix3;
use std::collections::BTreeSet;

/// Computes canonical poses across shallow oblique, mid-oblique, and nadir viewing bands.
pub(crate) fn canonical_tile_poses(
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
    let mut band = 0_u8;
    let mut out = Vec::with_capacity(25);
    for down_deg in [12.0_f64, 28.0_f64] {
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

/// Selects corridor tiles by isotropic radius plus a forward frustum sector and horizon rays.
pub(crate) fn select_corridor_tiles(
    surface: &Surface,
    cases: &[PhotoCase],
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

            if let Some(hdg_deg) = case.prior.as_ref().and_then(|p| p.direction_deg) {
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
                            && (dist <= 1.0e-6
                                || (dx * fwd_e + dy * fwd_n) >= dist * cos_half_sector)
                        {
                            selected.insert((tx, ty));
                        }
                    }
                }

                let dem = surface
                    .sample_dem(pos.latitude, pos.longitude)
                    .unwrap_or(0.0);
                let alt = pos.altitude_m.unwrap_or(dem + nominal_agl_m.max(500.0));
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

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};
    use serde_json::json;

    #[test]
    fn select_corridor_tiles_fills_forward_frustum_sector_without_backward_bloat() {
        let img = RgbImage::from_pixel(256, 256, Rgb([100, 110, 95]));
        let mut surface = Surface::synthetic(41.0, -75.0, 100.0, img);
        surface.bbox = [-75.15, 40.90, -74.85, 41.10];
        let case: PhotoCase = serde_json::from_value(json!({
            "id": "oblique-east",
            "image": "img.png",
            "image_sha256": "0".repeat(64),
            "size": [800, 600],
            "valid_mask": "",
            "mask_sha256": "",
            "mask_status": "full_image_unmasked",
            "prior": {
                "position": {
                    "latitude": 41.0,
                    "longitude": -75.0,
                    "altitude_m": 700.0,
                    "unsigned_altitude_m": 700.0,
                    "altitude_datum": "msl"
                },
                "direction_deg": 90.0
            }
        }))
        .expect("valid PhotoCase JSON");
        let tiles = select_corridor_tiles(&surface, &[case], 500.0, 500.0, 600.0);
        assert!(
            tiles.contains(&(4, 3)),
            "forward frustum sector must cover +37.9 deg off-ray oblique tiles within +/- 45 deg"
        );
        assert!(
            tiles.contains(&(5, -4)),
            "southern half of forward sector (-32.5 deg off East) must be covered"
        );
        assert!(
            !tiles.contains(&(-5, 0)),
            "tiles 2.5 km behind camera outside corridor_radius_m must not be selected"
        );
    }

    #[test]
    fn canonical_tile_poses_uses_25_full_azimuth_oblique_and_nadir_views() {
        let img = RgbImage::from_pixel(256, 256, Rgb([100, 110, 95]));
        let surface = Surface::synthetic(41.0, -75.0, 100.0, img);
        let views = canonical_tile_poses(&surface, (500.0, 200.0, 100.0), 500.0, 600.0, true);
        assert_eq!(views.len(), 25);
    }
}
