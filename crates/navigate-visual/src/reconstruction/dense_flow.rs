//! Interpolate local dense correspondences without assigning a new feature identity.
use crate::PixelMatch;
use nalgebra::Vector2;
use nalgebra::{Matrix3, Vector3};
use std::collections::HashMap;
pub(super) struct Flow<'a> {
    pairs: &'a [PixelMatch],
    cells: HashMap<(i32, i32), Vec<usize>>,
}
impl<'a> Flow<'a> {
    pub fn new(pairs: &'a [PixelMatch]) -> Self {
        let mut cells: HashMap<_, Vec<_>> = HashMap::new();
        for (i, p) in pairs.iter().enumerate() {
            cells.entry(key(p.reference)).or_default().push(i)
        }
        Self { pairs, cells }
    }
    pub fn at(&self, pixel: Vector2<f64>) -> Option<Vector2<f64>> {
        let (x, y) = key(pixel);
        let mut near = Vec::new();
        for cy in y - 1..=y + 1 {
            for cx in x - 1..=x + 1 {
                for &i in self.cells.get(&(cx, cy)).into_iter().flatten() {
                    let d = (self.pairs[i].reference - pixel).norm_squared();
                    if d < 900.0 {
                        near.push((i, d))
                    }
                }
            }
        }
        near.sort_by(|a, b| a.1.total_cmp(&b.1));
        near.truncate(9);
        if near.len() < 6 {
            return None;
        }
        let mut h = Matrix3::zeros();
        let mut bx = Vector3::zeros();
        let mut by = Vector3::zeros();
        for &(i, _) in &near {
            let p = &self.pairs[i];
            let x = Vector3::new(
                (p.reference[0] - pixel.x) / 20.0,
                (p.reference[1] - pixel.y) / 20.0,
                1.0,
            );
            h += x * x.transpose();
            bx += x * p.query[0];
            by += x * p.query[1]
        }
        let inverse = h.try_inverse()?;
        let ax = inverse * bx;
        let ay = inverse * by;
        let mut errors = Vec::new();
        for &(i, _) in &near {
            let p = &self.pairs[i];
            let x = Vector3::new(
                (p.reference[0] - pixel.x) / 20.0,
                (p.reference[1] - pixel.y) / 20.0,
                1.0,
            );
            errors.push((Vector2::new(ax.dot(&x), ay.dot(&x)) - p.query).norm());
        }
        if errors.iter().any(|&x| x > 2.0) {
            return None;
        }
        Some(Vector2::new(ax.z, ay.z))
    }
}
fn key(pixel: Vector2<f64>) -> (i32, i32) {
    (
        (pixel.x / 24.0).floor() as i32,
        (pixel.y / 24.0).floor() as i32,
    )
}

#[cfg(test)]
mod tests;
