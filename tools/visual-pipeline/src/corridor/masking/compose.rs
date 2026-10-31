//! Per-frame composition of cached `Segformer` layers, attitude sky warping, and propeller gating.

use nalgebra::Matrix3;

pub(super) fn compose_fallback_u8(
    total: usize,
    fb_raw: &[u8],
    prop_occluded: &[bool],
) -> (Vec<u8>, usize) {
    let mut out = vec![0u8; total];
    let mut valid = 0usize;
    for i in 0..total.min(fb_raw.len()).min(prop_occluded.len()) {
        if fb_raw[i] >= 128 && !prop_occluded[i] {
            out[i] = 255;
            valid += 1;
        }
    }
    (out, valid)
}

pub(super) fn compose_unmasked_u8(total: usize, prop_occluded: &[bool]) -> (Vec<u8>, usize) {
    let mut out = vec![0u8; total];
    let mut valid = 0usize;
    for (dst, &blade) in out.iter_mut().zip(prop_occluded.iter().take(total)) {
        if !blade {
            *dst = 255;
            valid += 1;
        }
    }
    (out, valid)
}

fn compose_static_u8(
    airframe_keep: &[u8],
    sky_keep: &[u8],
    prop_occluded: &[bool],
) -> (Vec<u8>, usize) {
    let total = airframe_keep
        .len()
        .min(sky_keep.len())
        .min(prop_occluded.len());
    let mut out = vec![0u8; total];
    let mut valid = 0usize;
    for i in 0..total {
        let keep = if prop_occluded[i] {
            0
        } else {
            airframe_keep[i] & sky_keep[i]
        };
        out[i] = keep;
        valid += usize::from(keep != 0);
    }
    (out, valid)
}

pub(super) fn compose_amortized_u8(
    [w, h]: [u32; 2],
    airframe_keep: &[u8],
    sky_keep: &[u8],
    prop_occluded: &[bool],
    key_rot: Option<&Matrix3<f64>>,
    cur_rot: Option<&Matrix3<f64>>,
    k: &Matrix3<f64>,
) -> (Vec<u8>, usize) {
    let width = w as usize;
    let height = h as usize;
    let total = width * height;
    let Some((r_key, r_cur)) = key_rot.zip(cur_rot) else {
        return compose_static_u8(airframe_keep, sky_keep, prop_occluded);
    };
    let delta_key_from_cur = r_key * r_cur.transpose();
    if (delta_key_from_cur - Matrix3::identity()).norm() < 1e-4 {
        return compose_static_u8(airframe_keep, sky_keep, prop_occluded);
    }

    let fx = k[(0, 0)].max(1e-6);
    let fy = k[(1, 1)].max(1e-6);
    let cx = k[(0, 2)];
    let cy = k[(1, 2)];
    let k_pix = Matrix3::new(fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0);
    let k_inv = Matrix3::new(
        1.0 / fx,
        0.0,
        -cx / fx,
        0.0,
        1.0 / fy,
        -cy / fy,
        0.0,
        0.0,
        1.0,
    );
    let hom = k_pix * delta_key_from_cur * k_inv;
    let mut out = vec![0u8; total];
    let workers = if total >= 120_000 {
        std::thread::available_parallelism()
            .map(|n| n.get().min(8))
            .unwrap_or(1)
    } else {
        1
    };
    if workers <= 1 {
        let valid = warp_sky_rows(
            0,
            [width, height],
            &hom,
            airframe_keep,
            sky_keep,
            prop_occluded,
            &mut out,
        );
        return (out, valid);
    }

    let rows_per_band = height.div_ceil(workers).max(1);
    let valid_total = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        let mut rem = out.as_mut_slice();
        let mut y0 = 0usize;
        while !rem.is_empty() && y0 < height {
            let band_rows = rows_per_band.min(height - y0);
            let (band_out, rest) = rem.split_at_mut(band_rows * width);
            rem = rest;
            let start_y = y0;
            y0 += band_rows;
            handles.push(scope.spawn(move || {
                warp_sky_rows(
                    start_y,
                    [width, height],
                    &hom,
                    airframe_keep,
                    sky_keep,
                    prop_occluded,
                    band_out,
                )
            }));
        }
        handles
            .into_iter()
            .map(|h_join| h_join.join().unwrap_or(0))
            .sum()
    });
    (out, valid_total)
}

fn warp_sky_rows(
    start_y: usize,
    [width, height]: [usize; 2],
    hom: &Matrix3<f64>,
    airframe_keep: &[u8],
    sky_keep: &[u8],
    prop_occluded: &[bool],
    band_out: &mut [u8],
) -> usize {
    let h00 = hom[(0, 0)];
    let h10 = hom[(1, 0)];
    let h20 = hom[(2, 0)];
    let max_u = (width.saturating_sub(1)) as f64;
    let max_v = (height.saturating_sub(1)) as f64;
    let band_rows = band_out.len() / width;
    let mut valid = 0usize;

    for local_y in 0..band_rows {
        let y = start_y + local_y;
        let yf = y as f64;
        let row_px = hom[(0, 1)] * yf + hom[(0, 2)];
        let row_py = hom[(1, 1)] * yf + hom[(1, 2)];
        let row_pz = hom[(2, 1)] * yf + hom[(2, 2)];
        let row_off = y * width;
        let air_row = &airframe_keep[row_off..row_off + width];
        let prop_row = &prop_occluded[row_off..row_off + width];
        let out_row = &mut band_out[local_y * width..(local_y + 1) * width];

        for x in 0..width {
            if air_row[x] == 0 || prop_row[x] {
                continue;
            }
            let xf = x as f64;
            let pz = row_pz + h20 * xf;
            if pz <= 1e-3 {
                continue;
            }
            let inv_z = 1.0 / pz;
            let uk = ((row_px + h00 * xf) * inv_z).round().clamp(0.0, max_u) as usize;
            let vk = ((row_py + h10 * xf) * inv_z).round().clamp(0.0, max_v) as usize;
            let keep = sky_keep[vk * width + uk];
            out_row[x] = keep;
            valid += usize::from(keep != 0);
        }
    }
    valid
}

#[cfg(test)]
mod tests;
