//! Coarse-to-fine patch alignment with an additive intensity offset.

use super::pyramid::sample;
use image::GrayImage;
use nalgebra::{Matrix2, Vector2};

const RADIUS: i32 = 4;

pub(super) fn track(
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
