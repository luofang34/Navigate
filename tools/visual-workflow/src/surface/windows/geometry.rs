//! Window and rotation coordinates are fixed by the source raster.
use crate::Error;
use navigate_visual::PixelMatch;
use std::collections::BTreeSet;

pub(super) fn windows((width, height): (u32, u32)) -> Result<Vec<[u32; 4]>, Error> {
    let crop = width / 4 * 3;
    if crop < 32 || height < 32 {
        return Err(Error::Input("map is too small for three windows".into()));
    }
    Ok(vec![
        [0, 0, crop, height],
        [(width - crop) / 2, 0, crop, height],
        [width - crop, 0, crop, height],
    ])
}
pub(super) fn validate_rotations(rotations: &[u8], limit: usize) -> Result<(), Error> {
    if rotations.is_empty()
        || rotations.len() > 4
        || rotations.len() * 3 > limit
        || rotations.iter().any(|r| *r > 3)
        || rotations.iter().copied().collect::<BTreeSet<_>>().len() != rotations.len()
    {
        return Err(Error::Input(
            "three windows per unique quarter turn exceed candidate budget".into(),
        ));
    }
    Ok(())
}
pub(super) fn restore_crop(pair: &mut PixelMatch, window: [u32; 4], rotation: u8) {
    super::super::matching::restore(pair, rotation, (window[2], window[3]));
    pair.reference.x += f64::from(window[0]);
    pair.reference.y += f64::from(window[1]);
}
pub(super) fn contains(pair: &PixelMatch, area: [u32; 4]) -> bool {
    contains_pixel(pair.reference, area)
}
pub(super) fn contains_pixel(pixel: nalgebra::Vector2<f64>, area: [u32; 4]) -> bool {
    let [x, y, w, h] = valid_area(area);
    pixel.x >= f64::from(x)
        && pixel.y >= f64::from(y)
        && pixel.x < f64::from(x + w)
        && pixel.y < f64::from(y + h)
}

#[cfg(test)]
mod tests;

pub(super) fn valid_area([x, y, w, h]: [u32; 4]) -> [u32; 4] {
    // The asymmetric half-open bounds stay inside the decoder margin under pixel-centre reflections.
    [x + 4, y + 4, w - 9, h - 9]
}
pub(super) fn model_transform((width, height): (u32, u32)) -> serde_json::Value {
    let scale = (800.0 / f64::from(width)).min(600.0 / f64::from(height));
    serde_json::json!({"rotated_input_size":[width,height],"letterbox_size":[800,600],"scale":scale,
        "offset":[(800.0-f64::from(width)*scale+scale-1.0)/2.0,(600.0-f64::from(height)*scale+scale-1.0)/2.0],
        "dense_grid":"Dense-image XFeat normalizes the full letterbox then resizes to 800x576. Grid is 100x72. Context is the declared rotated source image; no halo-equivalence assumption."})
}
