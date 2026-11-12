//! Local contrast normalisation: each pixel as its difference from the mean
//! of a window about it, over the standard deviation of that window.
//!
//! A camera frame and a map render differ in exposure, response, contrast
//! and the light across the ground; the tracker compares grey levels and
//! loses most points to that difference, and the corner score of faint
//! ground stays under its threshold. After normalisation both images show
//! the same structure at the same contrast, so the tracker compares like
//! with like and the corner threshold is relative to the local contrast.

use image::{GrayImage, Luma};

/// Half side of the window, pixels: a 15 by 15 window spans the patches of
/// the tracker with room for their shift.
const RADIUS: i64 = 7;

/// Grey levels for one standard deviation of the window.
const GAIN: f64 = 40.0;

/// Added to the standard deviation, grey levels, so that flat ground and
/// sensor noise are not raised to full contrast.
const FLOOR: f64 = 4.0;

/// Sums of `image` and of its squares over every rectangle from the
/// origin, one row and one column larger than the image.
struct Integral {
    width: usize,
    sum: Vec<f64>,
    squares: Vec<f64>,
}

impl Integral {
    fn of(image: &GrayImage) -> Self {
        let (w, h) = (image.width() as usize, image.height() as usize);
        let width = w + 1;
        let mut sum = vec![0.0; width * (h + 1)];
        let mut squares = vec![0.0; width * (h + 1)];
        for (y, row) in image.as_raw().chunks(w.max(1)).enumerate() {
            let (mut run, mut run2) = (0.0, 0.0);
            for (x, value) in row.iter().enumerate() {
                let v = f64::from(*value);
                run += v;
                run2 += v * v;
                let above = y * width + x + 1;
                let here = above + width;
                let (s, q) = (
                    sum.get(above).copied().unwrap_or(0.0),
                    squares.get(above).copied().unwrap_or(0.0),
                );
                if let (Some(a), Some(b)) = (sum.get_mut(here), squares.get_mut(here)) {
                    *a = s + run;
                    *b = q + run2;
                }
            }
        }
        Self {
            width,
            sum,
            squares,
        }
    }

    /// The sum and the sum of squares over columns `x0..x1` and rows
    /// `y0..y1`.
    fn window(&self, (x0, x1): (usize, usize), (y0, y1): (usize, usize)) -> (f64, f64) {
        let at = |table: &[f64], x: usize, y: usize| {
            table.get(y * self.width + x).copied().unwrap_or(0.0)
        };
        let rect = |t: &[f64]| at(t, x1, y1) - at(t, x0, y1) - at(t, x1, y0) + at(t, x0, y0);
        (rect(&self.sum), rect(&self.squares))
    }
}

/// `image` with its local contrast normalised.
pub(super) fn normalised(image: &GrayImage) -> GrayImage {
    let integral = Integral::of(image);
    let (w, h) = (i64::from(image.width()), i64::from(image.height()));
    let span = |c: i64, len: i64| {
        let low = (c - RADIUS).clamp(0, len) as usize;
        let high = (c + RADIUS + 1).clamp(0, len) as usize;
        (low, high)
    };
    GrayImage::from_fn(image.width(), image.height(), |x, y| {
        let (xs, ys) = (span(i64::from(x), w), span(i64::from(y), h));
        let n = ((xs.1 - xs.0) * (ys.1 - ys.0)).max(1) as f64;
        let (s, q) = integral.window(xs, ys);
        let mean = s / n;
        let sd = (q / n - mean * mean).max(0.0).sqrt();
        let v = image
            .get_pixel_checked(x, y)
            .map_or(mean, |p| f64::from(p.0[0]));
        let out = 128.0 + GAIN * (v - mean) / (sd + FLOOR);
        Luma([out.round().clamp(0.0, 255.0) as u8])
    })
}
