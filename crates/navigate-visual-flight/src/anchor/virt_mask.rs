//! Valid-region mask and oblique fallback ranking helpers for anchor hypothesis selection.

use super::{nadir_screen::rot_to_row_major, seeds::SeedPose};
use crate::error::FlightVisualError;
use image::GrayImage;
use navigate_match::{
    CorridorTileStore, KashinScratch, PreparedKashinQuery, VirtualViewCamera, kashin_screen_stats,
};
use std::cmp::Reverse;

/// Rank candidate oblique fallback poses by `Kashin` descriptor agreement against `prepared`.
pub(crate) fn rank_oblique_fallbacks(
    store: &CorridorTileStore,
    scratch: &mut KashinScratch,
    prepared: &PreparedKashinQuery,
    intrinsics: [f64; 4],
    render_size: [u32; 2],
    mask_small: Option<&GrayImage>,
    mut fallbacks: Vec<SeedPose>,
) -> Result<Vec<SeedPose>, FlightVisualError> {
    if fallbacks.len() <= 1 {
        return Ok(fallbacks);
    }
    let sub_mask = mask_small.filter(|m| mask_valid_fraction(m) < 0.70);
    let proj_cap = if sub_mask.is_some() { 1024 } else { 384 };
    let mut ranked: Vec<((usize, i64), SeedPose)> = Vec::with_capacity(fallbacks.len());
    for (eye, rot) in fallbacks.drain(..) {
        let mut virt = store.project_virtual_kashin(&VirtualViewCamera {
            eye_enu: eye,
            world_to_cam: rot_to_row_major(&rot),
            intrinsics,
            image_size: render_size,
            max_landmarks: proj_cap,
            min_cos_incidence: 0.05,
            min_cos_orientation: 0.35,
        })?;
        if let Some(mask) = sub_mask {
            retain_unmasked_virtual_landmarks(&mut virt, mask, render_size, 384);
        }
        let (strict_cnt, smooth) = kashin_screen_stats(scratch, &virt, prepared, 384, 32);
        ranked.push(((strict_cnt, (smooth * 1_000.0).round() as i64), (eye, rot)));
    }
    ranked.sort_by_key(|a| Reverse(a.0));
    Ok(ranked.into_iter().map(|(_, p)| p).collect())
}

/// Compute the fraction of pixels in `mask` with intensity `>= 128`.
#[must_use]
pub(crate) fn mask_valid_fraction(mask: &GrayImage) -> f64 {
    let raw = mask.as_raw();
    if raw.is_empty() {
        return 1.0;
    }
    let valid = raw.iter().filter(|&&v| v >= 128).count();
    (valid as f64) / (raw.len() as f64)
}

/// Select the oblique fallback look-down angle in degrees for an optional valid-region `mask`.
///
/// When a sub-window mask covers less than `70%` of the frame and has its valid vertical
/// centroid below `67%` of image height, the valid pixels already look downward relative to the
/// principal point. Aiming the optical axis at `8.0` degrees keeps distant horizon terrain inside
/// the lower-frame crop instead of pitching those rays into foreground ground.
#[must_use]
pub(crate) fn subwindow_oblique_down_deg(
    mask_small: Option<&GrayImage>,
    default_down_deg: f64,
) -> f64 {
    let Some(mask) = mask_small else {
        return default_down_deg;
    };
    let (w, h) = mask.dimensions();
    if w == 0 || h == 0 {
        return default_down_deg;
    }
    let step = if w >= 16 && h >= 16 { 4_usize } else { 1_usize };
    let raw = mask.as_raw();
    let w_usize = w as usize;
    let h_f64 = f64::from(h);
    let mut total = 0_usize;
    let mut valid = 0_usize;
    let mut sum_v = 0.0_f64;
    for y in (0..(h as usize)).step_by(step) {
        let v_norm = ((y as f64) + 0.5) / h_f64;
        let row = &raw[(y * w_usize)..((y + 1) * w_usize)];
        for &px in row.iter().step_by(step) {
            total += 1;
            if px >= 128 {
                valid += 1;
                sum_v += v_norm;
            }
        }
    }
    if total == 0 || valid == 0 {
        return default_down_deg;
    }
    let valid_frac = (valid as f64) / (total as f64);
    let v_mid = sum_v / (valid as f64);
    if valid_frac < 0.70 && v_mid >= 0.67 {
        8.0
    } else {
        default_down_deg
    }
}

/// Retain only virtual landmarks whose pixel coordinates lie inside `mask` (`>= 128`).
///
/// Keeps all parallel vectors in [`navigate_match::VirtualCorridorView`] synchronized and caps
/// output length at `max_landmarks`.
pub(crate) fn retain_unmasked_virtual_landmarks(
    virt: &mut navigate_match::VirtualCorridorView,
    mask: &GrayImage,
    render_size: [u32; 2],
    max_landmarks: usize,
) {
    let (mw, mh) = mask.dimensions();
    let [rw, rh] = render_size;
    if mw == 0 || mh == 0 || rw == 0 || rh == 0 || virt.world_points_enu.is_empty() {
        return;
    }
    let sx = f64::from(mw) / f64::from(rw);
    let sy = f64::from(mh) / f64::from(rh);
    let raw = mask.as_raw();
    let mw_usize = mw as usize;
    let has_desc = virt.features.descriptors.len() == virt.world_points_enu.len() * 64;

    let mut write = 0_usize;
    let total = virt.world_points_enu.len();
    for read in 0..total {
        if write >= max_landmarks {
            break;
        }
        let px = virt.features.pixels[read];
        let mx = ((px[0] * sx).floor() as isize).clamp(0, (mw as isize) - 1) as usize;
        let my = ((px[1] * sy).floor() as isize).clamp(0, (mh as isize) - 1) as usize;
        if raw[my * mw_usize + mx] < 128 {
            continue;
        }
        if write != read {
            virt.features.pixels[write] = px;
            virt.features.model_pixels[write] = virt.features.model_pixels[read];
            virt.features.scores[write] = virt.features.scores[read];
            if has_desc {
                virt.features
                    .descriptors
                    .copy_within((read * 64)..((read + 1) * 64), write * 64);
            }
            virt.world_points_enu[write] = virt.world_points_enu[read];
            virt.sign_bits[write] = virt.sign_bits[read];
            virt.kashin_i8[write] = virt.kashin_i8[read];
        }
        write += 1;
    }

    virt.features.pixels.truncate(write);
    virt.features.model_pixels.truncate(write);
    virt.features.scores.truncate(write);
    if has_desc {
        virt.features.descriptors.truncate(write * 64);
    }
    virt.world_points_enu.truncate(write);
    virt.sign_bits.truncate(write);
    virt.kashin_i8.truncate(write);
}

#[cfg(test)]
mod tests {
    use super::*;
    use navigate_match::{SparseFeatures, VirtualCorridorView};

    #[test]
    fn retain_unmasked_virtual_landmarks_drops_masked_pixels_and_syncs_fields() {
        let mut mask = GrayImage::new(100, 100);
        for y in 50..100 {
            for x in 50..100 {
                mask.put_pixel(x, y, image::Luma([255]));
            }
        }
        assert!((mask_valid_fraction(&mask) - 0.25).abs() < 1e-6);

        let mut virt = VirtualCorridorView {
            features: SparseFeatures {
                pixels: vec![[10.0, 10.0], [160.0, 160.0], [180.0, 180.0]],
                model_pixels: vec![[10.0, 10.0], [160.0, 160.0], [180.0, 180.0]],
                scores: vec![0.9, 0.8, 0.7],
                descriptors: (0..(3 * 64)).map(|i| i as f32).collect(),
            },
            world_points_enu: vec![[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]],
            sign_bits: vec![11, 22, 33],
            kashin_i8: vec![[1; 64], [2; 64], [3; 64]],
            visible_tiles: 2,
        };
        retain_unmasked_virtual_landmarks(&mut virt, &mask, [200, 200], 2);
        assert_eq!(
            virt.world_points_enu,
            vec![[2.0, 0.0, 0.0], [3.0, 0.0, 0.0]]
        );
        assert_eq!(virt.sign_bits, vec![22, 33]);
        assert_eq!(virt.features.pixels.len(), 2);
        assert_eq!(virt.features.descriptors.len(), 128);
        assert!((virt.features.descriptors[0] - 64.0).abs() < 1e-6);
    }

    #[test]
    fn subwindow_oblique_down_deg_adapts_only_for_lower_frame_crop() {
        assert!((subwindow_oblique_down_deg(None, 22.0) - 22.0).abs() < 1e-9);

        let mut mid_crop = GrayImage::new(80, 60);
        for y in 14..58 {
            for x in 30..75 {
                mid_crop.put_pixel(x, y, image::Luma([255]));
            }
        }
        assert!((subwindow_oblique_down_deg(Some(&mid_crop), 22.0) - 22.0).abs() < 1e-9);

        let mut bottom_crop = GrayImage::new(80, 60);
        for y in 28..58 {
            for x in 42..76 {
                bottom_crop.put_pixel(x, y, image::Luma([255]));
            }
        }
        assert!((subwindow_oblique_down_deg(Some(&bottom_crop), 22.0) - 8.0).abs() < 1e-9);
    }
}
