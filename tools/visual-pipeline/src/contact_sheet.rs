//! Thumbnail contact sheets for a photo test set.

use crate::PipelineError;
use image::{Rgb, RgbImage, imageops::FilterType};
use std::path::Path;

/// Write 4-column thumbnail contact sheets (`contact-sheet-01.jpg`, ...) into `root`.
///
/// # Errors
///
/// Returns [`PipelineError`] when thumbnail reading or sheet writing fails.
pub fn write_contact_sheets(root: &Path, ids: &[String]) -> Result<(), PipelineError> {
    if ids.is_empty() {
        return Ok(());
    }
    let cols = 4u32;
    let cell_w = 320u32;
    let cell_h = 240u32;
    let per_page = 24usize;
    for (page, group) in ids.chunks(per_page).enumerate() {
        let rows = (group.len() as u32).div_ceil(cols);
        let mut sheet = RgbImage::from_pixel(cols * cell_w, rows * cell_h, Rgb([24, 24, 24]));
        for (i, id) in group.iter().enumerate() {
            let thumb_path = root.join("thumbnails").join(format!("{id}.jpg"));
            if !thumb_path.is_file() {
                continue;
            }
            let img = image::open(&thumb_path)
                .map_err(|source| PipelineError::Image {
                    path: thumb_path,
                    source,
                })?
                .to_rgb8();
            let scale = (f64::from(cell_w) / f64::from(img.width().max(1)))
                .min(f64::from(cell_h) / f64::from(img.height().max(1)));
            let tw = ((f64::from(img.width()) * scale).round() as u32).clamp(1, cell_w);
            let th = ((f64::from(img.height()) * scale).round() as u32).clamp(1, cell_h);
            let resized = image::imageops::resize(&img, tw, th, FilterType::Triangle);
            let ox = (i as u32 % cols) * cell_w + (cell_w - tw) / 2;
            let oy = (i as u32 / cols) * cell_h + (cell_h - th) / 2;
            for y in 0..th {
                for x in 0..tw {
                    sheet.put_pixel(ox + x, oy + y, *resized.get_pixel(x, y));
                }
            }
        }
        let out_path = root.join(format!("contact-sheet-{:02}.jpg", page + 1));
        sheet
            .save(&out_path)
            .map_err(|source| PipelineError::Image {
                path: out_path,
                source,
            })?;
    }
    Ok(())
}
