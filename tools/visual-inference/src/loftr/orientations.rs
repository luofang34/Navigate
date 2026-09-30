//! Query rotations return correspondences in the unchanged camera pixel grid.
use crate::InferenceError;
use image::{GrayImage, imageops};
use navigate_visual::PixelMatch;

pub(super) fn match_blocking(
    reference: &GrayImage,
    query: &GrayImage,
    attempt: u32,
    mut match_pair: impl FnMut(&GrayImage, &GrayImage) -> Result<Vec<PixelMatch>, InferenceError>,
) -> Result<Option<Vec<PixelMatch>>, InferenceError> {
    let rotated = match attempt {
        0 => imageops::rotate90(query),
        1 => imageops::rotate180(query),
        2 => imageops::rotate270(query),
        _ => return Ok(None),
    };
    let mut pairs = match_pair(reference, &rotated)?;
    let width = f64::from(query.width());
    let height = f64::from(query.height());
    for pair in &mut pairs {
        let [x, y] = [pair.query.x, pair.query.y];
        pair.query = match attempt {
            0 => [y, height - 1.0 - x],
            1 => [width - 1.0 - x, height - 1.0 - y],
            _ => [width - 1.0 - y, x],
        }
        .into();
    }
    Ok(Some(pairs))
}

#[cfg(test)]
mod tests;
