//! Binary morphology, 8-connected component analysis, convex hulls, and edge line detection.

use image::GrayImage;

/// Connected-component summary for 8-connected foreground pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComponentStats {
    /// Positive component label (`1..=count`).
    pub label: u32,
    /// Minimum column index.
    pub left: usize,
    /// Minimum row index.
    pub top: usize,
    /// Bounding-box width in pixels.
    pub width: usize,
    /// Bounding-box height in pixels.
    pub height: usize,
    /// Foreground pixel count.
    pub area: usize,
}

/// Detected straight line segment `(x1, y1, x2, y2)` in image coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineSegment {
    /// Start column.
    pub x1: f64,
    /// Start row.
    pub y1: f64,
    /// End column.
    pub x2: f64,
    /// End row.
    pub y2: f64,
}

/// Erode a binary mask (`width × height`) with a `(2 * radius + 1)` square structuring element.
///
/// Out-of-bounds pixels equal `border_value`.
pub fn erode_rect(
    mask: &[bool],
    width: usize,
    height: usize,
    radius: usize,
    border_value: bool,
) -> Vec<bool> {
    rect_filter(mask, width, height, radius, border_value, false)
}

/// Dilate a binary mask (`width × height`) with a `(2 * radius + 1)` square structuring element.
pub fn dilate_rect(mask: &[bool], width: usize, height: usize, radius: usize) -> Vec<bool> {
    rect_filter(mask, width, height, radius, true, true)
}

fn rect_filter(
    mask: &[bool],
    width: usize,
    height: usize,
    radius: usize,
    border_value: bool,
    dilate: bool,
) -> Vec<bool> {
    if radius == 0 || width == 0 || height == 0 {
        return mask.to_vec();
    }
    let mut horiz = vec![false; width * height];
    let mut prefix = vec![0u32; width + 1];
    for y in 0..height {
        let row = &mask[y * width..(y + 1) * width];
        let h_row = &mut horiz[y * width..(y + 1) * width];
        prefix[0] = 0;
        for x in 0..width {
            let hit = if dilate { row[x] } else { !row[x] };
            prefix[x + 1] = prefix[x] + u32::from(hit);
        }
        for (x, h_slot) in h_row.iter_mut().enumerate() {
            let x0 = x.saturating_sub(radius);
            let x1 = (x + radius).min(width - 1);
            let hits = prefix[x1 + 1] - prefix[x0];
            *h_slot = if dilate {
                hits > 0
            } else {
                let outside_missing = !border_value && (x < radius || x + radius >= width);
                !outside_missing && hits == 0
            };
        }
    }

    let mut out = vec![false; width * height];
    let mut col_hits = vec![0u32; width];
    let init_y1 = radius.min(height - 1);
    for ry in 0..=init_y1 {
        let h_row = &horiz[ry * width..(ry + 1) * width];
        for x in 0..width {
            let hit = if dilate { h_row[x] } else { !h_row[x] };
            col_hits[x] += u32::from(hit);
        }
    }
    for y in 0..height {
        let y0 = y.saturating_sub(radius);
        let y1 = (y + radius).min(height - 1);
        let outside_missing = !dilate && !border_value && (y < radius || y + radius >= height);
        if !outside_missing {
            let out_row = &mut out[y * width..(y + 1) * width];
            for x in 0..width {
                out_row[x] = if dilate {
                    col_hits[x] > 0
                } else {
                    col_hits[x] == 0
                };
            }
        }
        if y + 1 < height {
            let next_y0 = (y + 1).saturating_sub(radius);
            let next_y1 = (y + 1 + radius).min(height - 1);
            if next_y1 > y1 {
                let add_row = &horiz[next_y1 * width..(next_y1 + 1) * width];
                for x in 0..width {
                    let hit = if dilate { add_row[x] } else { !add_row[x] };
                    col_hits[x] += u32::from(hit);
                }
            }
            if next_y0 > y0 {
                let sub_row = &horiz[y0 * width..(y0 + 1) * width];
                for x in 0..width {
                    let hit = if dilate { sub_row[x] } else { !sub_row[x] };
                    col_hits[x] -= u32::from(hit);
                }
            }
        }
    }
    out
}

/// Morphological opening (`erode` then `dilate`) with a `(2 * radius + 1)` square kernel.
pub fn morph_open(mask: &[bool], width: usize, height: usize, radius: usize) -> Vec<bool> {
    let eroded = erode_rect(mask, width, height, radius, true);
    if !eroded.iter().any(|&b| b) {
        return eroded;
    }
    dilate_rect(&eroded, width, height, radius)
}

/// Morphological closing (`dilate` then `erode`) with a `(2 * radius + 1)` square kernel.
pub fn morph_close(mask: &[bool], width: usize, height: usize, radius: usize) -> Vec<bool> {
    let dilated = dilate_rect(mask, width, height, radius);
    erode_rect(&dilated, width, height, radius, false)
}

/// Label 8-connected components in `mask` (`width × height`).
///
/// Returns `(labels, stats)` where `labels[idx] == 0` is background and `stats[k - 1]` describes
/// component `k`.
pub fn connected_components_8(
    mask: &[bool],
    width: usize,
    height: usize,
) -> (Vec<u32>, Vec<ComponentStats>) {
    let mut labels = vec![0u32; width * height];
    let mut stats = Vec::new();
    let mut queue: Vec<(u32, u32)> = Vec::new();
    let mut next_label = 1u32;

    for sy in 0..height {
        for sx in 0..width {
            let start = sy * width + sx;
            if !mask[start] || labels[start] != 0 {
                continue;
            }
            let label = next_label;
            next_label += 1;
            labels[start] = label;
            queue.clear();
            queue.push((sx as u32, sy as u32));

            let mut min_x = sx;
            let mut max_x = sx;
            let mut min_y = sy;
            let mut max_y = sy;
            let mut area = 0usize;
            let mut head = 0usize;

            while head < queue.len() {
                let (xu, yu) = queue[head];
                head += 1;
                let x = xu as usize;
                let y = yu as usize;
                area += 1;
                min_x = min_x.min(x);
                max_x = max_x.max(x);
                min_y = min_y.min(y);
                max_y = max_y.max(y);

                let y0 = y.saturating_sub(1);
                let y1 = (y + 1).min(height - 1);
                let x0 = x.saturating_sub(1);
                let x1 = (x + 1).min(width - 1);
                for ny in y0..=y1 {
                    let row_off = ny * width;
                    for nx in x0..=x1 {
                        let nidx = row_off + nx;
                        if mask[nidx] && labels[nidx] == 0 {
                            labels[nidx] = label;
                            queue.push((nx as u32, ny as u32));
                        }
                    }
                }
            }

            stats.push(ComponentStats {
                label,
                left: min_x,
                top: min_y,
                width: max_x - min_x + 1,
                height: max_y - min_y + 1,
                area,
            });
        }
    }
    (labels, stats)
}

/// Compute the 2D convex hull of `points` in counter-clockwise order (Andrew's monotone chain).
pub fn convex_hull(points: &[[f64; 2]]) -> Vec<[f64; 2]> {
    if points.len() <= 2 {
        return points.to_vec();
    }
    let mut sorted = points.to_vec();
    sorted.sort_by(|a, b| a[0].total_cmp(&b[0]).then_with(|| a[1].total_cmp(&b[1])));
    sorted.dedup_by(|a, b| (a[0] - b[0]).abs() < 1e-9 && (a[1] - b[1]).abs() < 1e-9);
    if sorted.len() <= 2 {
        return sorted;
    }
    let cross = |o: [f64; 2], a: [f64; 2], b: [f64; 2]| -> f64 {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };
    let mut lower: Vec<[f64; 2]> = Vec::new();
    for &p in &sorted {
        while lower.len() >= 2 && cross(lower[lower.len() - 2], lower[lower.len() - 1], p) <= 0.0 {
            lower.pop();
        }
        lower.push(p);
    }
    let mut upper: Vec<[f64; 2]> = Vec::new();
    for &p in sorted.iter().rev() {
        while upper.len() >= 2 && cross(upper[upper.len() - 2], upper[upper.len() - 1], p) <= 0.0 {
            upper.pop();
        }
        upper.push(p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    lower
}

/// Rasterize a convex polygon `hull` into `mask` (`width × height`).
pub fn fill_convex_poly(mask: &mut [bool], width: usize, height: usize, hull: &[[f64; 2]]) {
    if hull.len() < 3 || width == 0 || height == 0 {
        return;
    }
    let min_x = hull
        .iter()
        .map(|p| p[0].floor().max(0.0) as usize)
        .min()
        .unwrap_or(0)
        .min(width - 1);
    let max_x = hull
        .iter()
        .map(|p| p[0].ceil().max(0.0) as usize)
        .max()
        .unwrap_or(0)
        .min(width - 1);
    let min_y = hull
        .iter()
        .map(|p| p[1].floor().max(0.0) as usize)
        .min()
        .unwrap_or(0)
        .min(height - 1);
    let max_y = hull
        .iter()
        .map(|p| p[1].ceil().max(0.0) as usize)
        .max()
        .unwrap_or(0)
        .min(height - 1);

    for y in min_y..=max_y {
        let py = y as f64;
        for x in min_x..=max_x {
            let px = x as f64;
            let mut pos = false;
            let mut neg = false;
            for i in 0..hull.len() {
                let a = hull[i];
                let b = hull[(i + 1) % hull.len()];
                let c = (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0]);
                if c > 1e-6 {
                    pos = true;
                } else if c < -1e-6 {
                    neg = true;
                }
                if pos && neg {
                    break;
                }
            }
            if !(pos && neg) {
                mask[y * width + x] = true;
            }
        }
    }
}

/// Detect straight edge segments in `gray` via Sobel non-maximum suppression and line fitting.
pub fn detect_line_segments(
    gray: &GrayImage,
    low_threshold: f32,
    high_threshold: f32,
    min_length: f64,
) -> Vec<LineSegment> {
    let w = gray.width() as usize;
    let h = gray.height() as usize;
    if w < 5 || h < 5 {
        return Vec::new();
    }
    let raw = gray.as_raw();
    let mut blur = vec![0.0_f32; w * h];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let mut sum = 0.0_f32;
            for dy in 0..3 {
                for dx in 0..3 {
                    let k = match (dx, dy) {
                        (1, 1) => 4.0,
                        (1, _) | (_, 1) => 2.0,
                        _ => 1.0,
                    };
                    sum += k * f32::from(raw[(y + dy - 1) * w + (x + dx - 1)]);
                }
            }
            blur[y * w + x] = sum / 16.0;
        }
    }

    let mut mag = vec![0.0_f32; w * h];
    let mut dir = vec![0u8; w * h];
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let gx = -blur[(y - 1) * w + (x - 1)] + blur[(y - 1) * w + (x + 1)]
                - 2.0 * blur[y * w + (x - 1)]
                + 2.0 * blur[y * w + (x + 1)]
                - blur[(y + 1) * w + (x - 1)]
                + blur[(y + 1) * w + (x + 1)];
            let gy = -blur[(y - 1) * w + (x - 1)]
                - 2.0 * blur[(y - 1) * w + x]
                - blur[(y - 1) * w + (x + 1)]
                + blur[(y + 1) * w + (x - 1)]
                + 2.0 * blur[(y + 1) * w + x]
                + blur[(y + 1) * w + (x + 1)];
            mag[y * w + x] = gx.hypot(gy);
            let angle = gy.atan2(gx).to_degrees().rem_euclid(180.0);
            dir[y * w + x] = if !(22.5..157.5).contains(&angle) {
                0
            } else if angle < 67.5 {
                1
            } else if angle < 112.5 {
                2
            } else {
                3
            };
        }
    }

    let mut strong = vec![false; w * h];
    let mut weak = vec![false; w * h];
    for y in 2..h - 2 {
        for x in 2..w - 2 {
            let idx = y * w + x;
            let m = mag[idx];
            if m < low_threshold {
                continue;
            }
            let (m1, m2) = match dir[idx] {
                0 => (mag[y * w + x - 1], mag[y * w + x + 1]),
                1 => (mag[(y - 1) * w + x - 1], mag[(y + 1) * w + x + 1]),
                2 => (mag[(y - 1) * w + x], mag[(y + 1) * w + x]),
                _ => (mag[(y - 1) * w + x + 1], mag[(y + 1) * w + x - 1]),
            };
            if m >= m1 && m >= m2 {
                if m >= high_threshold {
                    strong[idx] = true;
                } else {
                    weak[idx] = true;
                }
            }
        }
    }

    let mut edges = strong.clone();
    let mut stack: Vec<usize> = strong
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| b.then_some(i))
        .collect();
    while let Some(idx) = stack.pop() {
        let x = idx % w;
        let y = idx / w;
        for ny in y.saturating_sub(1)..=(y + 1).min(h - 1) {
            for nx in x.saturating_sub(1)..=(x + 1).min(w - 1) {
                let n = ny * w + nx;
                if weak[n] && !edges[n] {
                    edges[n] = true;
                    stack.push(n);
                }
            }
        }
    }

    let (labels, stats) = connected_components_8(&edges, w, h);
    let mut points_by_label: Vec<Vec<[f64; 2]>> = vec![Vec::new(); stats.len()];
    for y in 0..h {
        for x in 0..w {
            let lbl = labels[y * w + x];
            if lbl != 0 {
                points_by_label[lbl as usize - 1].push([x as f64, y as f64]);
            }
        }
    }

    let mut out = Vec::new();
    for pts in points_by_label {
        if pts.len() < 12 {
            continue;
        }
        let n = pts.len() as f64;
        let mx = pts.iter().map(|p| p[0]).sum::<f64>() / n;
        let my = pts.iter().map(|p| p[1]).sum::<f64>() / n;
        let mut sxx = 0.0;
        let mut syy = 0.0;
        let mut sxy = 0.0;
        for p in &pts {
            let dx = p[0] - mx;
            let dy = p[1] - my;
            sxx += dx * dx;
            syy += dy * dy;
            sxy += dx * dy;
        }
        let theta = 0.5 * (2.0 * sxy).atan2(sxx - syy);
        let ux = theta.cos();
        let uy = theta.sin();
        let nx = -uy;
        let ny = ux;
        let max_perp = pts
            .iter()
            .map(|p| ((p[0] - mx) * nx + (p[1] - my) * ny).abs())
            .fold(0.0_f64, f64::max);
        if max_perp > 6.0 {
            continue;
        }
        let mut min_t = f64::INFINITY;
        let mut max_t = f64::NEG_INFINITY;
        for p in &pts {
            let t = (p[0] - mx) * ux + (p[1] - my) * uy;
            min_t = min_t.min(t);
            max_t = max_t.max(t);
        }
        if max_t - min_t >= min_length {
            out.push(LineSegment {
                x1: mx + min_t * ux,
                y1: my + min_t * uy,
                x2: mx + max_t * ux,
                y2: my + max_t * uy,
            });
        }
    }
    out
}
