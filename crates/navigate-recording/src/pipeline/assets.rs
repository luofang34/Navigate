//! Asset decoding and calibrated map-reference construction.

use crate::{Configuration, Error, Reference};
use navigate_visual::{LocalFrame, MapRevision, PixelMatch, ReferenceView};
use std::path::{Path, PathBuf};

fn path_blocking(root: &Path, name: &str) -> Result<PathBuf, Error> {
    let path = root.join(name);
    let read = |source| Error::Io {
        path: path.clone(),
        source,
    };
    let root = root.canonicalize().map_err(read)?;
    let canonical = path.canonicalize().map_err(read)?;
    if !canonical.starts_with(root) {
        return Err(Error::Invalid(format!(
            "asset escapes evidence directory: {name}"
        )));
    }
    Ok(canonical)
}

pub(super) fn image_blocking(root: &Path, name: &str) -> Result<image::GrayImage, Error> {
    let path = path_blocking(root, name)?;
    image::open(&path)
        .map(|i| i.to_luma8())
        .map_err(|source| Error::Image { path, source })
}

pub(super) fn reference_blocking(
    root: &Path,
    source: &Reference,
    config: &Configuration,
    map: MapRevision,
) -> Result<ReferenceView, Error> {
    let path = path_blocking(root, &source.depth)?;
    let bytes = std::fs::read(&path).map_err(|source| Error::Io { path, source })?;
    let pixels = config.camera.size[0] as usize * config.camera.size[1] as usize;
    if bytes.len() != pixels * 4 {
        return Err(Error::Invalid(
            "depth does not match camera dimensions".into(),
        ));
    }
    let depth_m = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    Ok(ReferenceView {
        map,
        frame: LocalFrame::anchor_mercator(config.anchor[0], config.anchor[1])?,
        pose: source.pose.camera()?,
        image: image_blocking(root, &source.image)?,
        depth_m,
    })
}

pub(super) fn matches(input: &[[[f64; 2]; 2]]) -> Vec<PixelMatch> {
    input
        .iter()
        .map(|pair| PixelMatch {
            reference: pair[0].into(),
            query: pair[1].into(),
        })
        .collect()
}
