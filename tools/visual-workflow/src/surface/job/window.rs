//! Integer map windows preserve the source grid and nearest-height samples.
use crate::Error;
use navigate_visual::OrthographicReference;

pub(super) fn crop(
    source: OrthographicReference,
    window: [u32; 4],
    byte_limit: usize,
) -> Result<OrthographicReference, Error> {
    if window[2] >= 32
        && window[3] >= 32
        && window == [0, 0, source.image.width(), source.image.height()]
    {
        return Ok(source);
    }
    crop_borrowed(&source, window, byte_limit)
}

pub(in crate::surface) fn crop_borrowed(
    source: &OrthographicReference,
    [x, y, width, height]: [u32; 4],
    byte_limit: usize,
) -> Result<OrthographicReference, Error> {
    let (source_width, source_height) = source.image.dimensions();
    if width < 32
        || height < 32
        || x.checked_add(width).is_none_or(|end| end > source_width)
        || y.checked_add(height).is_none_or(|end| end > source_height)
    {
        return Err(Error::Input(
            "surface window must fit the source and be at least 32 by 32".into(),
        ));
    }
    let crop_pixels = width as usize * height as usize;
    if source
        .image
        .as_raw()
        .len()
        .saturating_add(crop_pixels)
        .saturating_mul(5)
        > byte_limit
    {
        return Err(Error::Input(
            "source and surface window exceed reference memory limit".into(),
        ));
    }
    let mut heights_m = Vec::with_capacity(crop_pixels);
    for row in y..y + height {
        let start = row as usize * source_width as usize + x as usize;
        heights_m.extend_from_slice(&source.heights_m[start..start + width as usize]);
    }
    let image = image::imageops::crop_imm(&source.image, x, y, width, height).to_image();
    let mut geometry = source.geometry;
    geometry.origin += geometry.column * f64::from(x) + geometry.row * f64::from(y);
    let cropped = OrthographicReference {
        map: source.map.clone(),
        image,
        heights_m,
        geometry,
    };
    cropped.validate()?;
    Ok(cropped)
}

#[cfg(test)]
mod tests;
