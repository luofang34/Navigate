//! Full-resolution local alignment initialized by an external prediction.
use super::{tracking::align, validate_refinement};
use crate::VisualError;
use image::GrayImage;
use nalgebra::Vector2;

pub(super) fn run(
    reference: &GrayImage,
    query: &GrayImage,
    points: &[Vector2<f64>],
    initial: &[Vector2<f64>],
) -> Result<Vec<Option<Vector2<f64>>>, VisualError> {
    validate_refinement(reference, query, points, initial)?;
    Ok(points
        .iter()
        .zip(initial)
        .map(|(&point, &guess)| {
            let found = align(reference, query, point, guess)?;
            let back = align(query, reference, found, point)?;
            ((back - point).norm() <= 1.0).then_some(found)
        })
        .collect())
}

#[cfg(test)]
mod tests;
