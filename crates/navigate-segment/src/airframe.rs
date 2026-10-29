//! Airframe strut and wing-border occlusion detectors for cockpit/cabin photography.

use crate::morphology::{
    LineSegment, connected_components_8, convex_hull, detect_line_segments, dilate_rect,
    fill_convex_poly, morph_close,
};
use image::{DynamicImage, GrayImage, RgbImage};

/// Detect elongated structural struts (`fence = 32`, `railing = 38`, `pole = 93`) anchored to
/// image borders or detected airframe pixels.
pub fn support_mask(
    rgb: &RgbImage,
    labels: &[u8],
    confidence: &[f32],
    airframe: &[bool],
) -> Vec<bool> {
    let w = rgb.width() as usize;
    let h = rgb.height() as usize;
    let n = w * h;
    if n == 0 {
        return Vec::new();
    }
    let gray = DynamicImage::ImageRgb8(rgb.clone()).to_luma8();
    let segments = detect_line_segments(&gray, 40.0, 120.0, (w.min(h) as f64) * 0.08);
    let mut line_mask = vec![false; n];
    for seg in &segments {
        rasterize_thick_segment(&mut line_mask, w, h, *seg, 4);
    }
    let anchor = dilate_rect(airframe, w, h, 10);
    let candidate: Vec<bool> = (0..n)
        .map(|i| matches!(labels[i], 32 | 38 | 93) && confidence[i] >= 0.2 && line_mask[i])
        .collect();
    let closed = morph_close(&candidate, w, h, 3);
    let (comp_labels, stats) = connected_components_8(&closed, w, h);
    let mut keep_label = vec![false; stats.len() + 1];
    for stat in &stats {
        let major = stat.width.max(stat.height);
        let minor = stat.width.min(stat.height).max(1);
        let touches_border = stat.left <= 4
            || stat.top <= 4
            || stat.left + stat.width >= w.saturating_sub(4)
            || stat.top + stat.height >= h.saturating_sub(4);
        let elongated = stat.area >= 400 && (major as f64 / minor as f64 >= 2.5 || major >= 120);
        if elongated && touches_border {
            keep_label[stat.label as usize] = true;
        }
    }
    for i in 0..n {
        let lbl = comp_labels[i] as usize;
        if lbl != 0 && !keep_label[lbl] && anchor[i] {
            let stat = stats[lbl - 1];
            let major = stat.width.max(stat.height);
            let minor = stat.width.min(stat.height).max(1);
            if stat.area >= 400 && (major as f64 / minor as f64 >= 2.5 || major >= 120) {
                keep_label[lbl] = true;
            }
        }
    }
    comp_labels
        .into_iter()
        .map(|l| l != 0 && keep_label[l as usize])
        .collect()
}

/// Detect straight wing/strut line pairs that bracket semantic airplane pixels (`label == 90`)
/// and touch the image border.
pub fn border_support_mask(
    rgb: &RgbImage,
    labels: &[u8],
    airplane_probs: Option<&[f32]>,
) -> Vec<bool> {
    let w = rgb.width() as usize;
    let h = rgb.height() as usize;
    let n = w * h;
    if n == 0 {
        return Vec::new();
    }
    let seed: Vec<bool> = (0..n)
        .map(|i| {
            labels[i] == 90
                || airplane_probs.is_some_and(|p| p.get(i).copied().unwrap_or(0.0) >= 0.05)
        })
        .collect();
    if !seed.iter().any(|&b| b) {
        return vec![false; n];
    }

    let gray: GrayImage = DynamicImage::ImageRgb8(rgb.clone()).to_luma8();
    let short_side = w.min(h) as f64;
    let segments = detect_line_segments(&gray, 35.0, 110.0, short_side * 0.12);
    let border_tol = 24.0;
    let border_segs: Vec<LineSegment> = segments
        .into_iter()
        .filter(|s| {
            touches_border(s.x1, s.y1, w as f64, h as f64, border_tol)
                || touches_border(s.x2, s.y2, w as f64, h as f64, border_tol)
        })
        .collect();

    let mut out = vec![false; n];
    for i in 0..border_segs.len() {
        for j in (i + 1)..border_segs.len() {
            let a = border_segs[i];
            let b = border_segs[j];
            let angle_a = (a.y2 - a.y1).atan2(a.x2 - a.x1);
            let angle_b = (b.y2 - b.y1).atan2(b.x2 - b.x1);
            let diff = (angle_a - angle_b).abs().rem_euclid(std::f64::consts::PI);
            let acute = diff.min(std::f64::consts::PI - diff).to_degrees();
            if acute > 28.0 {
                continue;
            }
            let sep = point_segment_distance((a.x1 + a.x2) * 0.5, (a.y1 + a.y2) * 0.5, b);
            if !(8.0..=short_side * 0.28).contains(&sep) {
                continue;
            }
            let hull = convex_hull(&[[a.x1, a.y1], [a.x2, a.y2], [b.x1, b.y1], [b.x2, b.y2]]);
            let mut candidate = vec![false; n];
            fill_convex_poly(&mut candidate, w, h, &hull);
            let area: usize = candidate.iter().map(|&v| usize::from(v)).sum();
            if area == 0 || area > n / 3 {
                continue;
            }
            let seed_hits: usize = candidate
                .iter()
                .zip(&seed)
                .map(|(&c, &s)| usize::from(c && s))
                .sum();
            if seed_hits >= 25 && (seed_hits as f64 / area as f64) >= 0.02 {
                for (dst, src) in out.iter_mut().zip(&candidate) {
                    *dst |= *src;
                }
            }
        }
    }
    out
}

fn touches_border(x: f64, y: f64, w: f64, h: f64, tol: f64) -> bool {
    x <= tol || y <= tol || x >= w - 1.0 - tol || y >= h - 1.0 - tol
}

fn point_segment_distance(px: f64, py: f64, s: LineSegment) -> f64 {
    let dx = s.x2 - s.x1;
    let dy = s.y2 - s.y1;
    let l2 = dx * dx + dy * dy;
    if l2 <= 1e-9 {
        return (px - s.x1).hypot(py - s.y1);
    }
    let t = (((px - s.x1) * dx + (py - s.y1) * dy) / l2).clamp(0.0, 1.0);
    (px - (s.x1 + t * dx)).hypot(py - (s.y1 + t * dy))
}

fn rasterize_thick_segment(
    mask: &mut [bool],
    width: usize,
    height: usize,
    seg: LineSegment,
    radius: isize,
) {
    let steps = ((seg.x2 - seg.x1).hypot(seg.y2 - seg.y1).ceil() as usize).max(1);
    for step in 0..=steps {
        let t = step as f64 / steps as f64;
        let cx = (seg.x1 + t * (seg.x2 - seg.x1)).round() as isize;
        let cy = (seg.y1 + t * (seg.y2 - seg.y1)).round() as isize;
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                let x = cx + dx;
                let y = cy + dy;
                if x >= 0 && y >= 0 && (x as usize) < width && (y as usize) < height {
                    mask[y as usize * width + x as usize] = true;
                }
            }
        }
    }
}
