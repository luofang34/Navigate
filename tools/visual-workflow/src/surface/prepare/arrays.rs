//! Read bounded numeric arrays without executing object serialization.
use crate::{Error, assets};
use image::{DynamicImage, RgbImage};
use npyz::{Deserialize, NpyFile, Order};
use std::path::Path;

pub(super) fn read_blocking<T: Deserialize>(path: &Path) -> Result<(Vec<u64>, Vec<T>), Error> {
    let bytes = assets::read_blocking(path, 64 * 1024 * 1024)?;
    let file = header(&bytes, path)?;
    let shape = file.shape().to_vec();
    let values = file.into_vec::<T>().map_err(|source| Error::Io {
        operation: "decode array values",
        path: path.into(),
        source,
    })?;
    Ok((shape, values))
}
fn header<'a>(bytes: &'a [u8], path: &Path) -> Result<NpyFile<&'a [u8]>, Error> {
    let file = NpyFile::new(bytes).map_err(|source| Error::Io {
        operation: "decode array header",
        path: path.into(),
        source,
    })?;
    if file.order() != Order::C || file.len() > 12_000_000 {
        return Err(Error::Input(format!(
            "{} exceeds array bounds or is not row-major",
            path.display()
        )));
    }
    Ok(file)
}

pub(super) fn float_blocking(path: &Path) -> Result<(Vec<u64>, Vec<f64>), Error> {
    let bytes = assets::read_blocking(path, 64 * 1024 * 1024)?;
    let file = header(&bytes, path)?;
    let shape = file.shape().to_vec();
    let values = match file.dtype() {
        npyz::DType::Plain(t) if t.type_char() == npyz::TypeChar::Float && t.size_field() == 4 => {
            file.into_vec::<f32>()
                .map(|values| values.into_iter().map(f64::from).collect())
        }
        npyz::DType::Plain(t) if t.type_char() == npyz::TypeChar::Float && t.size_field() == 8 => {
            file.into_vec::<f64>()
        }
        dtype => {
            return Err(Error::Input(format!(
                "{} needs f32 or f64, found {}",
                path.display(),
                dtype.descr()
            )));
        }
    }
    .map_err(|source| Error::Io {
        operation: "decode floating array",
        path: path.into(),
        source,
    })?;
    Ok((shape, values))
}
pub(super) fn image_blocking(path: &Path) -> Result<DynamicImage, Error> {
    let (shape, pixels) = read_blocking::<u8>(path)?;
    let [height, width, 3] = shape.as_slice() else {
        return Err(Error::Input(format!(
            "{} requires H by W by 3 RGB bytes",
            path.display()
        )));
    };
    if !(32..=8192).contains(width) || !(32..=8192).contains(height) || width * height > 4_000_000 {
        return Err(Error::Input(format!(
            "{} image dimensions exceed bounds",
            path.display()
        )));
    }
    let image = RgbImage::from_raw(*width as u32, *height as u32, pixels)
        .ok_or_else(|| Error::Input(format!("{} RGB byte count", path.display())))?;
    Ok(DynamicImage::ImageRgb8(image))
}
