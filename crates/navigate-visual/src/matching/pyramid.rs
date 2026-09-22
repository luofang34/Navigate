//! Image pyramids and spatially distributed corner selection.

use image::{
    GrayImage,
    imageops::{FilterType, resize},
};
use nalgebra::Vector2;

pub(crate) fn build(image: &GrayImage) -> Vec<GrayImage> {
    let mut levels = vec![image.clone()];
    for _ in 0..3 {
        let Some(last) = levels.last() else {
            break;
        };
        if last.width() < 64 || last.height() < 64 {
            break;
        }
        levels.push(resize(
            last,
            last.width() / 2,
            last.height() / 2,
            FilterType::Triangle,
        ));
    }
    levels
}

pub(super) fn sample(image: &GrayImage, x: f64, y: f64) -> f64 {
    let ix = x.floor() as u32;
    let iy = y.floor() as u32;
    let dx = x - f64::from(ix);
    let dy = y - f64::from(iy);
    let at = |x, y| f64::from(image.get_pixel(x, y)[0]);
    (1.0 - dy) * ((1.0 - dx) * at(ix, iy) + dx * at(ix + 1, iy))
        + dy * ((1.0 - dx) * at(ix, iy + 1) + dx * at(ix + 1, iy + 1))
}

pub(crate) fn corners(image: &GrayImage) -> Vec<Vector2<f64>> {
    let mut points = Vec::new();
    if image.width() < 32 || image.height() < 32 {
        return points;
    }
    for by in (12..image.height() - 12).step_by(18) {
        for bx in (12..image.width() - 12).step_by(18) {
            let mut best = (80.0, bx, by);
            for y in (by..(by + 18).min(image.height() - 12)).step_by(2) {
                for x in (bx..(bx + 18).min(image.width() - 12)).step_by(2) {
                    let score = corner_score(image, x, y);
                    if score > best.0 {
                        best = (score, x, y);
                    }
                }
            }
            if best.0 > 80.0 {
                points.push(Vector2::new(f64::from(best.1), f64::from(best.2)));
            }
        }
    }
    points
}

fn corner_score(image: &GrayImage, x: u32, y: u32) -> f64 {
    let (mut xx, mut xy, mut yy) = (0.0, 0.0, 0.0);
    for py in y - 2..=y + 2 {
        for px in x - 2..=x + 2 {
            let gx = f64::from(image.get_pixel(px + 1, py)[0])
                - f64::from(image.get_pixel(px - 1, py)[0]);
            let gy = f64::from(image.get_pixel(px, py + 1)[0])
                - f64::from(image.get_pixel(px, py - 1)[0]);
            xx += gx * gx;
            xy += gx * gy;
            yy += gy * gy;
        }
    }
    0.5 * (xx + yy - ((xx - yy).powi(2) + 4.0 * xy * xy).sqrt())
}
