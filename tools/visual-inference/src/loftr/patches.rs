//! Crop coordinates stay in the original observation and reference images.
use crate::InferenceError;
use image::{GrayImage, imageops};
use navigate_visual::PixelMatch;

pub(super) fn match_blocking(
    reference: &GrayImage,
    query: &GrayImage,
    mut match_pair: impl FnMut(&GrayImage, &GrayImage) -> Result<Vec<PixelMatch>, InferenceError>,
) -> Result<Vec<PixelMatch>, InferenceError> {
    let mut groups = vec![match_pair(reference, query)?];
    let (width, height) = reference.dimensions();
    if query.dimensions() != reference.dimensions() || width.min(height) < 256 {
        return Ok(groups.remove(0));
    }
    let crop_width = (u64::from(width) * 5).div_ceil(8) as u32;
    let crop_height = (u64::from(height) * 5).div_ceil(8) as u32;
    for y in [0, height - crop_height] {
        for x in [0, width - crop_width] {
            let a = imageops::crop_imm(reference, x, y, crop_width, crop_height).to_image();
            let b = imageops::crop_imm(query, x, y, crop_width, crop_height).to_image();
            let mut pairs = match_pair(&a, &b)?;
            for pair in &mut pairs {
                pair.reference.x += f64::from(x);
                pair.reference.y += f64::from(y);
                pair.query.x += f64::from(x);
                pair.query.y += f64::from(y);
            }
            groups.push(pairs);
        }
    }
    let mut pairs = Vec::new();
    for index in 0..groups.iter().map(Vec::len).max().unwrap_or(0) {
        for group in &groups {
            if let Some(pair) = group.get(index) {
                pairs.push(*pair);
                if pairs.len() == 4096 {
                    return Ok(pairs);
                }
            }
        }
    }
    Ok(pairs)
}

#[cfg(test)]
mod tests;
