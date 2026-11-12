//! The tracker against a copy of its plain form: a vector for each patch
//! and pixels read through `get_pixel`. The two must give the same bits.

use super::super::pyramid;
use super::super::tracking;
use image::GrayImage;
use nalgebra::{Matrix2, Vector2};

/// The plain bilinear sample.
fn sample(image: &GrayImage, x: f64, y: f64) -> f64 {
    let ix = x.floor() as u32;
    let iy = y.floor() as u32;
    let dx = x - f64::from(ix);
    let dy = y - f64::from(iy);
    let at = |x, y| f64::from(image.get_pixel(x, y)[0]);
    (1.0 - dy) * ((1.0 - dx) * at(ix, iy) + dx * at(ix + 1, iy))
        + dy * ((1.0 - dx) * at(ix, iy + 1) + dx * at(ix + 1, iy + 1))
}

const RADIUS: i32 = 4;

fn track(
    source: &[GrayImage],
    target: &[GrayImage],
    point: Vector2<f64>,
    initial: Vector2<f64>,
) -> Option<Vector2<f64>> {
    let mut found = initial;
    let mut seeded = false;
    for level in (0..source.len()).rev() {
        let scale = 2_f64.powi(level as i32);
        let p = (point + Vector2::repeat(0.5)) / scale - Vector2::repeat(0.5);
        let mut guess = (found + Vector2::repeat(0.5)) / scale - Vector2::repeat(0.5);
        if !seeded {
            if !inside(&source[level], p) {
                continue;
            }
            guess = coarse_seed(&source[level], &target[level], p, guess)?;
            seeded = true;
        }
        let aligned = align(&source[level], &target[level], p, guess)?;
        found = (aligned + Vector2::repeat(0.5)) * scale - Vector2::repeat(0.5);
    }
    seeded.then_some(found)
}

fn coarse_seed(
    source: &GrayImage,
    target: &GrayImage,
    p: Vector2<f64>,
    initial: Vector2<f64>,
) -> Option<Vector2<f64>> {
    if !inside(source, p) {
        return None;
    }
    let template = patch(source, p);
    let mut best = None;
    let mut best_error = f64::INFINITY;
    for y in -8..=8 {
        for x in -8..=8 {
            let q = initial + Vector2::new(f64::from(x), f64::from(y));
            if !inside(target, q) {
                continue;
            }
            let values = patch(target, q);
            let error = template
                .iter()
                .zip(values)
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>();
            if error < best_error {
                best_error = error;
                best = Some(q);
            }
        }
    }
    best
}

fn inside(image: &GrayImage, p: Vector2<f64>) -> bool {
    let border = f64::from(RADIUS + 2);
    p.x >= border
        && p.y >= border
        && p.x < f64::from(image.width()) - border
        && p.y < f64::from(image.height()) - border
}

fn patch(image: &GrayImage, p: Vector2<f64>) -> Vec<f64> {
    let mut values = Vec::with_capacity(81);
    for y in -RADIUS..=RADIUS {
        for x in -RADIUS..=RADIUS {
            values.push(sample(image, p.x + f64::from(x), p.y + f64::from(y)));
        }
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    values.iter_mut().for_each(|v| *v -= mean);
    values
}

fn align(
    source: &GrayImage,
    target: &GrayImage,
    p: Vector2<f64>,
    mut q: Vector2<f64>,
) -> Option<Vector2<f64>> {
    if !inside(source, p) {
        return None;
    }
    let template = patch(source, p);
    for _ in 0..25 {
        if !inside(target, q) {
            return None;
        }
        let values = patch(target, q);
        let mut h = Matrix2::zeros();
        let mut b = Vector2::zeros();
        for (index, (a, v)) in template.iter().zip(&values).enumerate() {
            let x = q.x + (index % 9) as f64 - f64::from(RADIUS);
            let y = q.y + (index / 9) as f64 - f64::from(RADIUS);
            let gradient = Vector2::new(
                (sample(target, x + 1.0, y) - sample(target, x - 1.0, y)) * 0.5,
                (sample(target, x, y + 1.0) - sample(target, x, y - 1.0)) * 0.5,
            );
            let error = a - v;
            let weight = 20.0 / error.abs().max(20.0);
            h += gradient * gradient.transpose() * weight;
            b += gradient * error * weight;
        }
        if h.determinant() < 1.0 || h.trace() < 40.0 {
            return None;
        }
        let step = h.try_inverse()? * b;
        q += step / (step.norm() / 3.0).max(1.0);
        if step.norm() < 0.02 {
            break;
        }
    }
    if !inside(target, q) {
        return None;
    }
    let values = patch(target, q);
    let error = template
        .iter()
        .zip(values)
        .map(|(a, b)| (a - b).abs())
        .sum::<f64>()
        / 81.0;
    (error < 22.0).then_some(q)
}

/// A textured image of `seed` with a few hard edges.
fn image(seed: u32) -> GrayImage {
    GrayImage::from_fn(320, 240, |x, y| {
        let h = (x.wrapping_mul(73_856_093)
            ^ y.wrapping_mul(19_349_663)
            ^ seed.wrapping_mul(83_492_791))
            % 97;
        let smooth =
            ((f64::from(x) * 0.11).sin() * 60.0 + (f64::from(y) * 0.07).cos() * 50.0) as i32;
        let edge = if (x / 23 + y / 17 + seed).is_multiple_of(3) {
            70
        } else {
            0
        };
        image::Luma([(110 + smooth + edge + h as i32 / 4).clamp(0, 255) as u8])
    })
}

#[test]
fn the_tracker_gives_the_bits_of_its_plain_form() {
    for seed in 0..2 {
        let original = image(seed);
        let source = pyramid::build(&original);
        // The target: the source moved by a few pixels and brightened.
        let moved = GrayImage::from_fn(320, 240, |x, y| {
            let p = original.get_pixel((x + 317) % 320, (y + 238) % 240)[0];
            image::Luma([p.saturating_add(9)])
        });
        let target = pyramid::build(&moved);
        let mut found = 0;
        for point in pyramid::corners(&original) {
            for initial in [point, point + Vector2::new(1.7, -2.3)] {
                let fast = tracking::track(&source, &target, point, initial, f64::INFINITY);
                let plain = track(&source, &target, point, initial);
                assert_eq!(
                    fast.map(|q| (q.x.to_bits(), q.y.to_bits())),
                    plain.map(|q| (q.x.to_bits(), q.y.to_bits()))
                );
                found += usize::from(fast.is_some());
            }
        }
        assert!(found > 100, "{found}");
    }
}
