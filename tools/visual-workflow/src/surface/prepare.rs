//! Prepare local surface inputs from unpacked numeric benchmark arrays.
mod arrays;
use super::job::{Surface, SurfaceJob};
use crate::{Error, assets, job::Image};
use image::{DynamicImage, GenericImageView, ImageFormat};
use nalgebra::Vector2;
use navigate_recording::{Camera, Pose};
use navigate_visual::RasterGeometry;
use sha2::{Digest, Sha256};
use std::{io::Cursor, path::Path};

pub fn run_blocking(input: &Path, output: &Path) -> Result<(), Error> {
    let query = arrays::image_blocking(&input.join("image_query.npy"))?;
    let reference = arrays::image_blocking(&input.join("image_dop.npy"))?;
    let camera = calibration_blocking(&input.join("intrinsics.npy"), query.dimensions())?;
    let (geometry, heights) = surface_blocking(&input.join("dsm.npy"), reference.dimensions())?;
    let maximum = heights
        .iter()
        .filter(|v| v.is_finite())
        .copied()
        .reduce(f32::max)
        .ok_or_else(|| Error::Input("surface has no finite heights".into()))?;
    let centre = geometry.origin
        + geometry.column * f64::from(reference.width() - 1) * 0.5
        + geometry.row * f64::from(reference.height() - 1) * 0.5;
    std::fs::create_dir(output).map_err(|source| Error::Io {
        operation: "create output directory",
        path: output.into(),
        source,
    })?;
    let query = save_image_blocking(output, "query.png", query)?;
    let image = save_image_blocking(output, "reference.png", reference)?;
    let bytes: Vec<_> = heights.iter().flat_map(|h| h.to_le_bytes()).collect();
    assets::write_blocking(&output.join("heights.bin"), &bytes)?;
    let heights = Image {
        path: "heights.bin".into(),
        sha256: format!("{:x}", Sha256::digest(&bytes)),
    };
    let mut hash = Sha256::new();
    hash.update(b"navigate-orthographic-map-v1");
    hash.update(image.sha256.as_bytes());
    hash.update(heights.sha256.as_bytes());
    for v in geometry
        .origin
        .iter()
        .chain(geometry.column.iter())
        .chain(geometry.row.iter())
    {
        hash.update(v.to_le_bytes());
    }
    let reference = Surface {
        image,
        heights,
        origin: geometry.origin.into(),
        column: geometry.column.into(),
        row: geometry.row.into(),
        map_release: "unpacked-orthographic-arrays".into(),
        map_sha256: format!("{:x}", hash.finalize()),
    };
    let job = SurfaceJob {
        query,
        camera,
        reference,
        prior: Pose {
            position: [centre.x, centre.y, f64::from(maximum) + 100.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
        },
        position_radius_m: 1000.0,
        attitude_radius_rad: std::f64::consts::PI,
        reference_quarter_turns: vec![0, 1, 2, 3],
        record_matches: false,
    };
    assets::write_json_blocking(&output.join("job.json"), &job)?;
    tracing::info!(path=%output.display(),"Prepared local surface; set navigation bounds for the intended deployment");
    Ok(())
}
fn calibration_blocking(path: &Path, size: (u32, u32)) -> Result<Camera, Error> {
    let (shape, values) = arrays::read_blocking::<f32>(path)?;
    if shape != [3, 3]
        || values.len() != 9
        || !values.iter().all(|v| v.is_finite())
        || values[1] != 0.0
        || values[3] != 0.0
        || values[6] != 0.0
        || values[7] != 0.0
        || values[8] != 1.0
    {
        return Err(Error::Input(
            "intrinsics require a finite unskewed 3 by 3 pinhole matrix".into(),
        ));
    }
    let camera = Camera {
        size: [size.0, size.1],
        focal: [f64::from(values[0]), f64::from(values[4])],
        principal: [f64::from(values[2]), f64::from(values[5])],
    };
    camera.model().validate()?;
    Ok(camera)
}
fn surface_blocking(path: &Path, size: (u32, u32)) -> Result<(RasterGeometry, Vec<f32>), Error> {
    let (shape, values) = arrays::read_blocking::<f32>(path)?;
    let (width, height) = size;
    if shape != [u64::from(height), u64::from(width), 3]
        || values.len() != width as usize * height as usize * 3
    {
        return Err(Error::Input(
            "surface requires image-sized H by W by 3 metric coordinates".into(),
        ));
    }
    let xy = |x: usize, y: usize| {
        let i = (y * width as usize + x) * 3;
        Vector2::new(f64::from(values[i]), f64::from(values[i + 1]))
    };
    let origin = xy(0, 0);
    let geometry = RasterGeometry {
        origin,
        column: (xy(width as usize - 1, 0) - origin) / f64::from(width - 1),
        row: (xy(0, height as usize - 1) - origin) / f64::from(height - 1),
    };
    geometry.validate()?;
    let mut heights = Vec::with_capacity(width as usize * height as usize);
    for (i, point) in values.as_chunks::<3>().0.iter().enumerate() {
        let expected = origin
            + geometry.column * (i % width as usize) as f64
            + geometry.row * (i / width as usize) as f64;
        if !point[..2].iter().all(|v| v.is_finite())
            || (Vector2::new(f64::from(point[0]), f64::from(point[1])) - expected).norm() > 0.001
        {
            return Err(Error::Input(format!(
                "surface pixel {i} is not on one affine grid within 1 mm"
            )));
        }
        heights.push(point[2]);
    }
    Ok((geometry, heights))
}
fn save_image_blocking(root: &Path, name: &str, image: DynamicImage) -> Result<Image, Error> {
    let path = root.join(name);
    let mut bytes = Cursor::new(Vec::new());
    image
        .write_to(&mut bytes, ImageFormat::Png)
        .map_err(|source| Error::Image {
            path: path.clone(),
            source,
        })?;
    assets::write_blocking(&path, bytes.get_ref())?;
    Ok(Image {
        path: name.into(),
        sha256: format!("{:x}", Sha256::digest(bytes.get_ref())),
    })
}

#[cfg(test)]
mod tests;
