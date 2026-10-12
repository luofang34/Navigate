//! Separable gray reduction with one float row of intermediate storage.

use image::{GenericImageView, GrayImage};

struct Kernel {
    first: usize,
    weights: Vec<f32>,
}

fn kernel(source: usize, target: usize, index: usize) -> Kernel {
    let ratio = source as f32 / target as f32;
    let center = (index as f32 + 0.5) * ratio;
    let first = ((center - ratio).floor() as i64).clamp(0, source as i64 - 1) as usize;
    let end = ((center + ratio).ceil() as i64).clamp(first as i64 + 1, source as i64) as usize;
    let center = center - 0.5;
    let mut weights = Vec::with_capacity(end - first);
    let mut total = 0.0;
    for sample in first..end {
        let distance = ((sample as f32 - center) / ratio).abs();
        let weight = if distance < 1.0 { 1.0 - distance } else { 0.0 };
        weights.push(weight);
        total += weight;
    }
    for weight in &mut weights {
        *weight /= total;
    }
    Kernel { first, weights }
}

pub(super) fn half(image: &GrayImage) -> GrayImage {
    let width = image.width() as usize;
    let target_width = width / 2;
    let target_height = image.height() as usize / 2;
    let mut result = image.buffer_with_dimensions(target_width as u32, target_height as u32);
    if target_width == 0 || target_height == 0 {
        return result;
    }
    let columns: Vec<_> = (0..target_width)
        .map(|index| kernel(width, target_width, index))
        .collect();
    let mut row = vec![0.0; width];
    for (y, output) in result.as_mut().chunks_exact_mut(target_width).enumerate() {
        row.fill(0.0);
        let vertical = kernel(image.height() as usize, target_height, y);
        for (offset, weight) in vertical.weights.iter().enumerate() {
            let start = (vertical.first + offset) * width;
            for (value, sample) in row.iter_mut().zip(&image.as_raw()[start..start + width]) {
                *value += f32::from(*sample) * weight;
            }
        }
        for (pixel, column) in output.iter_mut().zip(&columns) {
            // Keep the vertical-then-horizontal sum order and quantize only once.
            let mut value = 0.0;
            for (offset, weight) in column.weights.iter().enumerate() {
                value += row[column.first + offset] * weight;
            }
            *pixel = value.clamp(0.0, 255.0).round() as u8;
        }
    }
    result
}

#[cfg(test)]
mod tests;
