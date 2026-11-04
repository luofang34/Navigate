//! Zero-copy strided anti-aliased downsampling and luminance statistics for [`crate::FramePixels`].

use crate::{error::FlightVisualError, types::FramePixels};
use image::{GrayImage, RgbImage};

const FP_ONE: u32 = 4096;
const FP_HALF: u64 = (FP_ONE as u64 * FP_ONE as u64) / 2;
const FP_SHIFT: u32 = 24;

/// Compute the live tracking render resolution with longest edge capped at `800 px`.
#[must_use]
pub fn tracking_render_size(src_size: [u32; 2]) -> [u32; 2] {
    let w = src_size[0].max(1);
    let h = src_size[1].max(1);
    let longest = w.max(h);
    if longest <= 800 {
        return [w.max(16), h.max(16)];
    }
    let scale = 800.0 / f64::from(longest);
    [
        ((f64::from(w) * scale).round() as u32).max(16),
        ((f64::from(h) * scale).round() as u32).max(16),
    ]
}

/// Validate buffer stride and downsample `pixels` directly into `target` luma (and optional `RGB`).
///
/// For `Nv12YPlane` and `Gray8`, this function samples the strided luma plane directly into
/// `target` without allocating a full-resolution copy. When `need_rgb` is true on a luma-only
/// input, it replicates the small `target` luma into a small `RgbImage`.
///
/// # Errors
///
/// Returns [`FlightVisualError::Invalid`] when `pixels` dimensions or byte strides are invalid.
pub fn prepare_tracking_buffers(
    pixels: FramePixels<'_>,
    target: [u32; 2],
    need_rgb: bool,
) -> Result<(GrayImage, Option<RgbImage>), FlightVisualError> {
    let [tw, th] = [target[0].max(1), target[1].max(1)];
    match pixels {
        FramePixels::Gray8 {
            data,
            width,
            height,
            stride,
        }
        | FramePixels::Nv12YPlane {
            y_plane: data,
            width,
            height,
            y_stride: stride,
        } => {
            validate_plane(data.len(), width, height, stride, 1)?;
            let gray = downsample_luma_strided(data, width, height, stride, tw, th);
            let rgb = need_rgb.then(|| gray_to_rgb(&gray));
            Ok((gray, rgb))
        }
        FramePixels::Rgb8 {
            data,
            width,
            height,
            stride,
        } => {
            validate_plane(data.len(), width, height, stride, 3)?;
            if need_rgb {
                let (gray, rgb) = downsample_rgb_and_luma(data, width, height, stride, tw, th);
                Ok((gray, Some(rgb)))
            } else {
                let gray = downsample_rgb_to_luma_only(data, width, height, stride, tw, th);
                Ok((gray, None))
            }
        }
    }
}

fn validate_plane(
    len: usize,
    width: u32,
    height: u32,
    stride: usize,
    bytes_per_px: usize,
) -> Result<(), FlightVisualError> {
    if width == 0 || height == 0 {
        return Err(FlightVisualError::Invalid(
            "frame pixel dimensions must be positive".into(),
        ));
    }
    let min_row = (width as usize).saturating_mul(bytes_per_px);
    if stride < min_row {
        return Err(FlightVisualError::Invalid(format!(
            "frame stride {stride} is smaller than row byte width {min_row}"
        )));
    }
    let needed = (height as usize - 1)
        .checked_mul(stride)
        .and_then(|base| base.checked_add(min_row))
        .ok_or_else(|| FlightVisualError::Invalid("frame buffer size overflow".into()))?;
    if len < needed {
        return Err(FlightVisualError::Invalid(format!(
            "frame buffer length {len} is smaller than required {needed} bytes"
        )));
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct TentSpan {
    start: usize,
    len: usize,
    off: usize,
}

struct TentKernel {
    spans: Vec<TentSpan>,
    weights: Vec<u32>,
    ring_rows: usize,
}

fn build_tent_kernel(src_len: usize, dst_len: usize) -> TentKernel {
    let src_n = src_len.max(1);
    let dst_n = dst_len.max(1);
    let ratio = (src_n as f64) / (dst_n as f64);
    let scale = ratio.max(1.0);
    let max_idx = src_n.saturating_sub(1) as isize;
    let est_taps = ((2.0 * scale).ceil() as usize + 2) * dst_len;
    let mut spans = Vec::with_capacity(dst_len);
    let mut weights = Vec::with_capacity(est_taps);
    let mut raw_w = Vec::with_capacity((2.0 * scale).ceil() as usize + 4);
    let mut max_span = 1_usize;

    for d in 0..dst_len {
        let center = (d as f64 + 0.5) * ratio - 0.5;
        let s0 = ((center - scale).ceil() as isize).clamp(0, max_idx) as usize;
        let s1 = ((center + scale).floor() as isize).clamp(0, max_idx) as usize;
        let (start, len) = if s1 >= s0 {
            (s0, s1 - s0 + 1)
        } else {
            (center.round().clamp(0.0, max_idx as f64) as usize, 1)
        };
        raw_w.clear();
        let mut sum_f = 0.0_f64;
        for s in start..start + len {
            let wf = (1.0 - ((s as f64) - center).abs() / scale).max(0.0);
            raw_w.push(wf);
            sum_f += wf;
        }
        let off = weights.len();
        if sum_f <= 0.0 || len == 1 {
            let nearest = center.round().clamp(0.0, max_idx as f64) as usize;
            weights.push(FP_ONE);
            spans.push(TentSpan {
                start: nearest,
                len: 1,
                off,
            });
            continue;
        }
        max_span = max_span.max(len);
        let mut acc = 0_u32;
        for &wf in &raw_w[..len - 1] {
            let q =
                (((wf / sum_f) * f64::from(FP_ONE)).round() as u32).min(FP_ONE.saturating_sub(acc));
            weights.push(q);
            acc += q;
        }
        weights.push(FP_ONE.saturating_sub(acc));
        spans.push(TentSpan { start, len, off });
    }
    let ring_rows = (max_span + 4).next_power_of_two().max(8);
    TentKernel {
        spans,
        weights,
        ring_rows,
    }
}

#[inline]
fn rgb_to_luma_bt709(r: u32, g: u32, b: u32) -> u8 {
    ((13_933 * r + 46_871 * g + 4_732 * b + 32_768) >> 16).min(255) as u8
}

fn downsample_luma_strided(
    src: &[u8],
    sw: u32,
    sh: u32,
    stride: usize,
    dw: u32,
    dh: u32,
) -> GrayImage {
    let (sw_u, sh_u, dw_u, dh_u) = (sw as usize, sh as usize, dw as usize, dh as usize);
    if sw == dw && sh == dh {
        let mut out = vec![0_u8; dw_u * dh_u];
        for y in 0..dh_u {
            out[y * dw_u..(y + 1) * dw_u].copy_from_slice(&src[y * stride..y * stride + dw_u]);
        }
        return GrayImage::from_raw(dw, dh, out).unwrap_or_else(|| GrayImage::new(dw, dh));
    }

    let xk = build_tent_kernel(sw_u, dw_u);
    let yk = build_tent_kernel(sh_u, dh_u);
    let ring_mask = yk.ring_rows - 1;
    let mut ring_y = vec![usize::MAX; yk.ring_rows];
    let mut ring_buf = vec![0_u32; yk.ring_rows * dw_u];
    let mut out = vec![0_u8; dw_u * dh_u];

    for (dy, yt) in yk.spans.iter().enumerate() {
        for sy in yt.start..yt.start + yt.len {
            let slot = sy & ring_mask;
            if ring_y[slot] != sy {
                ring_y[slot] = sy;
                let srow = &src[sy * stride..sy * stride + sw_u];
                let hrow = &mut ring_buf[slot * dw_u..(slot + 1) * dw_u];
                for (xt, h_dst) in xk.spans.iter().zip(hrow.iter_mut()) {
                    let s_slice = &srow[xt.start..xt.start + xt.len];
                    let w_slice = &xk.weights[xt.off..xt.off + xt.len];
                    let mut acc = 0_u32;
                    for (&px, &wx) in s_slice.iter().zip(w_slice) {
                        acc += u32::from(px) * wx;
                    }
                    *h_dst = acc;
                }
            }
        }
        let drow = &mut out[dy * dw_u..(dy + 1) * dw_u];
        let wy_slice = &yk.weights[yt.off..yt.off + yt.len];
        for (dx, dst) in drow.iter_mut().enumerate() {
            let mut acc = 0_u64;
            for (k, &wy) in wy_slice.iter().enumerate() {
                let s_off = ((yt.start + k) & ring_mask) * dw_u + dx;
                acc += u64::from(ring_buf[s_off]) * u64::from(wy);
            }
            *dst = ((acc + FP_HALF) >> FP_SHIFT).min(255) as u8;
        }
    }

    GrayImage::from_raw(dw, dh, out).unwrap_or_else(|| GrayImage::new(dw, dh))
}

fn downsample_rgb_and_luma(
    src: &[u8],
    sw: u32,
    sh: u32,
    stride: usize,
    dw: u32,
    dh: u32,
) -> (GrayImage, RgbImage) {
    let (sw_u, sh_u, dw_u, dh_u) = (sw as usize, sh as usize, dw as usize, dh as usize);
    if sw == dw && sh == dh {
        let mut gray_raw = vec![0_u8; dw_u * dh_u];
        let mut rgb_raw = vec![0_u8; dw_u * dh_u * 3];
        for y in 0..dh_u {
            let srow = &src[y * stride..y * stride + dw_u * 3];
            rgb_raw[y * dw_u * 3..(y + 1) * dw_u * 3].copy_from_slice(srow);
            for x in 0..dw_u {
                let p = &srow[x * 3..x * 3 + 3];
                gray_raw[y * dw_u + x] =
                    rgb_to_luma_bt709(u32::from(p[0]), u32::from(p[1]), u32::from(p[2]));
            }
        }
        let gray = GrayImage::from_raw(dw, dh, gray_raw).unwrap_or_else(|| GrayImage::new(dw, dh));
        let rgb = RgbImage::from_raw(dw, dh, rgb_raw).unwrap_or_else(|| RgbImage::new(dw, dh));
        return (gray, rgb);
    }

    let xk = build_tent_kernel(sw_u, dw_u);
    let yk = build_tent_kernel(sh_u, dh_u);
    let ring_mask = yk.ring_rows - 1;
    let row_elems = dw_u * 3;
    let mut ring_y = vec![usize::MAX; yk.ring_rows];
    let mut ring_buf = vec![0_u32; yk.ring_rows * row_elems];
    let mut gray_raw = vec![0_u8; dw_u * dh_u];
    let mut rgb_raw = vec![0_u8; dh_u * row_elems];

    for (dy, yt) in yk.spans.iter().enumerate() {
        for sy in yt.start..yt.start + yt.len {
            let slot = sy & ring_mask;
            if ring_y[slot] != sy {
                ring_y[slot] = sy;
                let srow = &src[sy * stride..sy * stride + sw_u * 3];
                let hrow = &mut ring_buf[slot * row_elems..(slot + 1) * row_elems];
                for (dx, xt) in xk.spans.iter().enumerate() {
                    let s_slice = &srow[xt.start * 3..(xt.start + xt.len) * 3];
                    let w_slice = &xk.weights[xt.off..xt.off + xt.len];
                    let (mut ar, mut ag, mut ab) = (0_u32, 0_u32, 0_u32);
                    for (rgb_px, &wx) in s_slice.as_chunks::<3>().0.iter().zip(w_slice) {
                        ar += u32::from(rgb_px[0]) * wx;
                        ag += u32::from(rgb_px[1]) * wx;
                        ab += u32::from(rgb_px[2]) * wx;
                    }
                    hrow[dx * 3] = ar;
                    hrow[dx * 3 + 1] = ag;
                    hrow[dx * 3 + 2] = ab;
                }
            }
        }
        let wy_slice = &yk.weights[yt.off..yt.off + yt.len];
        for dx in 0..dw_u {
            let c = dx * 3;
            let (mut ar, mut ag, mut ab) = (0_u64, 0_u64, 0_u64);
            for (k, &wy) in wy_slice.iter().enumerate() {
                let s_off = ((yt.start + k) & ring_mask) * row_elems + c;
                let wy64 = u64::from(wy);
                ar += u64::from(ring_buf[s_off]) * wy64;
                ag += u64::from(ring_buf[s_off + 1]) * wy64;
                ab += u64::from(ring_buf[s_off + 2]) * wy64;
            }
            let r = ((ar + FP_HALF) >> FP_SHIFT).min(255) as u32;
            let g = ((ag + FP_HALF) >> FP_SHIFT).min(255) as u32;
            let b = ((ab + FP_HALF) >> FP_SHIFT).min(255) as u32;
            let idx = dy * dw_u + dx;
            rgb_raw[idx * 3] = r as u8;
            rgb_raw[idx * 3 + 1] = g as u8;
            rgb_raw[idx * 3 + 2] = b as u8;
            gray_raw[idx] = rgb_to_luma_bt709(r, g, b);
        }
    }

    let gray = GrayImage::from_raw(dw, dh, gray_raw).unwrap_or_else(|| GrayImage::new(dw, dh));
    let rgb = RgbImage::from_raw(dw, dh, rgb_raw).unwrap_or_else(|| RgbImage::new(dw, dh));
    (gray, rgb)
}

fn downsample_rgb_to_luma_only(
    src: &[u8],
    sw: u32,
    sh: u32,
    stride: usize,
    dw: u32,
    dh: u32,
) -> GrayImage {
    let (sw_u, sh_u, dw_u, dh_u) = (sw as usize, sh as usize, dw as usize, dh as usize);
    if sw == dw && sh == dh {
        let mut out = vec![0_u8; dw_u * dh_u];
        for y in 0..dh_u {
            let srow = &src[y * stride..y * stride + dw_u * 3];
            for x in 0..dw_u {
                let p = &srow[x * 3..x * 3 + 3];
                out[y * dw_u + x] =
                    rgb_to_luma_bt709(u32::from(p[0]), u32::from(p[1]), u32::from(p[2]));
            }
        }
        return GrayImage::from_raw(dw, dh, out).unwrap_or_else(|| GrayImage::new(dw, dh));
    }

    let xk = build_tent_kernel(sw_u, dw_u);
    let yk = build_tent_kernel(sh_u, dh_u);
    let ring_mask = yk.ring_rows - 1;
    let mut ring_y = vec![usize::MAX; yk.ring_rows];
    let mut ring_buf = vec![0_u32; yk.ring_rows * dw_u];
    let mut luma_row = vec![0_u8; sw_u];
    let mut out = vec![0_u8; dw_u * dh_u];

    for (dy, yt) in yk.spans.iter().enumerate() {
        for sy in yt.start..yt.start + yt.len {
            let slot = sy & ring_mask;
            if ring_y[slot] != sy {
                ring_y[slot] = sy;
                let srow = &src[sy * stride..sy * stride + sw_u * 3];
                for (rgb_px, y_dst) in srow.as_chunks::<3>().0.iter().zip(luma_row.iter_mut()) {
                    *y_dst = rgb_to_luma_bt709(
                        u32::from(rgb_px[0]),
                        u32::from(rgb_px[1]),
                        u32::from(rgb_px[2]),
                    );
                }
                let hrow = &mut ring_buf[slot * dw_u..(slot + 1) * dw_u];
                for (xt, h_dst) in xk.spans.iter().zip(hrow.iter_mut()) {
                    let s_slice = &luma_row[xt.start..xt.start + xt.len];
                    let w_slice = &xk.weights[xt.off..xt.off + xt.len];
                    let mut acc = 0_u32;
                    for (&y8, &wx) in s_slice.iter().zip(w_slice) {
                        acc += u32::from(y8) * wx;
                    }
                    *h_dst = acc;
                }
            }
        }
        let drow = &mut out[dy * dw_u..(dy + 1) * dw_u];
        let wy_slice = &yk.weights[yt.off..yt.off + yt.len];
        for (dx, dst) in drow.iter_mut().enumerate() {
            let mut acc = 0_u64;
            for (k, &wy) in wy_slice.iter().enumerate() {
                let s_off = ((yt.start + k) & ring_mask) * dw_u + dx;
                acc += u64::from(ring_buf[s_off]) * u64::from(wy);
            }
            *dst = ((acc + FP_HALF) >> FP_SHIFT).min(255) as u8;
        }
    }

    GrayImage::from_raw(dw, dh, out).unwrap_or_else(|| GrayImage::new(dw, dh))
}

fn gray_to_rgb(gray: &GrayImage) -> RgbImage {
    let (w, h) = gray.dimensions();
    let raw = gray.as_raw();
    let mut rgb = Vec::with_capacity(raw.len() * 3);
    for &y in raw {
        rgb.extend_from_slice(&[y, y, y]);
    }
    RgbImage::from_raw(w, h, rgb).unwrap_or_else(|| RgbImage::new(w, h))
}

/// Compute the exact median pixel luminance (`0..=255`) using a 256-bin histogram.
#[must_use]
pub fn median_luma_u8(gray: &GrayImage) -> u8 {
    let raw = gray.as_raw();
    if raw.is_empty() {
        return 0;
    }
    let mut hist = [0_u32; 256];
    for &b in raw {
        hist[b as usize] += 1;
    }
    let target = (raw.len() / 2) as u32;
    let mut acc = 0_u32;
    for (val, &cnt) in hist.iter().enumerate() {
        acc += cnt;
        if acc > target {
            return val as u8;
        }
    }
    255
}
