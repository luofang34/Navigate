//! The camera angle below the horizon, from the sky boundary of a photo.
//!
//! A photo from an aircraft often shows the horizon. Each sampled column gives the
//! row where the bright, smooth sky that starts at the top of the photo ends. A line
//! through those rows, fitted with a fixed-seed sample, gives the horizon row at the
//! photo center and the roll. A photo without sky above most columns gives no angle.
use image::RgbImage;

/// Sampled columns.
const COLUMNS: u32 = 96;
/// Least share of columns that must agree with the fitted line.
const SUPPORT: f64 = 0.35;
/// Largest row distance of an agreeing column from the line, in photo heights.
const AGREE: f64 = 0.02;

/// The horizon that a photo shows.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Horizon {
    /// Angle of the optical axis below the horizon, degrees.
    pub pitch_deg: f64,
    /// Roll of the horizon, degrees, positive when the right side is lower.
    pub roll_deg: f64,
    /// Share of the sampled columns that agree.
    pub support: f64,
}

fn luma(p: [u8; 3]) -> f64 {
    0.299 * f64::from(p[0]) + 0.587 * f64::from(p[1]) + 0.114 * f64::from(p[2])
}

/// The row where the sky at the top of column `x` ends, if the column starts with sky.
fn sky_end(photo: &RgbImage, x: u32) -> Option<f64> {
    let h = photo.height();
    let top = photo.get_pixel(x, 0).0;
    // Sky is bright and not much redder than it is blue.
    if luma(top) < 110.0 || f64::from(top[0]) > f64::from(top[2]) + 25.0 {
        return None;
    }
    let step = (h / 200).max(1);
    let mut previous = luma(top);
    let mut sky = top.map(f64::from);
    let mut y = step;
    while y < h {
        let p = photo.get_pixel(x, y).0;
        let value = luma(p);
        let colour = (f64::from(p[0]) - sky[0]).abs()
            + (f64::from(p[1]) - sky[1]).abs()
            + (f64::from(p[2]) - sky[2]).abs();
        if (value - previous).abs() > 18.0 || colour > 75.0 {
            return Some(f64::from(y));
        }
        // Sky color changes slowly towards the horizon; follow it.
        sky = std::array::from_fn(|i| 0.8 * sky[i] + 0.2 * f64::from(p[i]));
        previous = value;
        y += step;
    }
    None
}

/// The horizon of a photo with focal length `focal_px`, if enough columns show it.
pub fn horizon(photo: &RgbImage, focal_px: f64) -> Option<Horizon> {
    let (w, h) = photo.dimensions();
    let points: Vec<(f64, f64)> = (0..COLUMNS)
        .filter_map(|i| {
            let x = (f64::from(i) + 0.5) * f64::from(w) / f64::from(COLUMNS);
            sky_end(photo, x as u32).map(|y| (x, y))
        })
        .collect();
    if (points.len() as f64) < SUPPORT * f64::from(COLUMNS) {
        return None;
    }
    let tolerance = AGREE * f64::from(h);
    let mut best: Option<(usize, f64, f64)> = None;
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    for _ in 0..300 {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let i = ((state >> 33) as usize) % points.len();
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1);
        let j = ((state >> 33) as usize) % points.len();
        let ((x0, y0), (x1, y1)) = (points[i], points[j]);
        if (x1 - x0).abs() < f64::from(w) * 0.2 {
            continue;
        }
        let slope = (y1 - y0) / (x1 - x0);
        let count = points
            .iter()
            .filter(|(x, y)| (y0 + slope * (x - x0) - y).abs() < tolerance)
            .count();
        if best.is_none_or(|(c, _, _)| count > c) {
            best = Some((count, slope, y0 + slope * (f64::from(w) / 2.0 - x0)));
        }
    }
    let (count, slope, centre_row) = best?;
    let support = count as f64 / f64::from(COLUMNS);
    (support >= SUPPORT).then(|| Horizon {
        // A horizon above the photo center means the camera looks down.
        pitch_deg: ((f64::from(h) / 2.0 - centre_row) / focal_px)
            .atan()
            .to_degrees(),
        roll_deg: slope.atan().to_degrees(),
        support,
    })
}
