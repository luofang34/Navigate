//! Bounded reference orientations share one query feature extraction.
use crate::{
    Error,
    config::{Backend, Config},
};
use image::{GrayImage, imageops};
use navigate_visual::{ImageMatcher, PixelMatch};
use navigate_visual_onnx::XFeatFeatures;
use navigate_visual_workflow::FeatureBackend;
use std::{collections::BTreeSet, path::Path, time::Instant};

struct Reference {
    rotation: u8,
    image: GrayImage,
    features: Option<XFeatFeatures>,
}
pub(super) struct Matcher {
    backend: Backend,
    references: Vec<Reference>,
    original_size: (u32, u32),
    pub resident_bytes: usize,
}
pub(super) struct Matches {
    pub rotation: u8,
    pub pairs: Vec<PixelMatch>,
    pub matching_ms: f64,
}
impl Matcher {
    pub fn new_blocking(
        config: &Config,
        root: &Path,
        image: &GrayImage,
        rotations: &[u8],
        surface_bytes: usize,
    ) -> Result<Self, Error> {
        validate_rotations(rotations, config.limits.candidates)?;
        let mut backend = config.load_blocking(root)?;
        let mut references = Vec::with_capacity(rotations.len());
        let mut resident_bytes = surface_bytes;
        for &rotation in rotations {
            if resident_bytes.saturating_add(image.as_raw().len()) > config.limits.reference_bytes {
                return Err(Error::Input(
                    "surface orientations exceed reference memory limit".into(),
                ));
            }
            let image = rotate(image, rotation);
            resident_bytes = resident_bytes.saturating_add(image.as_raw().len());
            let features = match &mut backend {
                Backend::Features(b) => {
                    let f = b.extract_features_blocking(&image)?;
                    let metadata = b.validate_features(&f, &image)?;
                    resident_bytes = resident_bytes.saturating_add(metadata.resident_bytes);
                    if metadata.count > config.limits.features_per_image
                        || resident_bytes > config.limits.reference_bytes
                    {
                        return Err(Error::Input(
                            "surface features exceed reference memory limit".into(),
                        ));
                    }
                    Some(f)
                }
                Backend::Pair(_) => None,
            };
            references.push(Reference {
                rotation,
                image,
                features,
            });
        }
        Ok(Self {
            backend,
            references,
            original_size: image.dimensions(),
            resident_bytes,
        })
    }
    pub fn identity(&self) -> &str {
        match &self.backend {
            Backend::Features(b) => b.identity(),
            Backend::Pair(b) => b.identity(),
        }
    }
    pub fn matches_blocking(
        &mut self,
        query: &GrayImage,
        config: &Config,
    ) -> Result<(f64, Vec<Matches>), Error> {
        let start = Instant::now();
        let features = match &mut self.backend {
            Backend::Features(b) => {
                let f = b.extract_features_blocking(query)?;
                let metadata = b.validate_features(&f, query)?;
                if metadata.count > config.limits.features_per_image
                    || metadata.resident_bytes > config.limits.query_feature_bytes
                {
                    return Err(Error::Input(
                        "surface query features exceed memory limit".into(),
                    ));
                }
                Some(f)
            }
            Backend::Pair(_) => None,
        };
        let extraction_ms = if features.is_some() {
            start.elapsed().as_secs_f64() * 1000.0
        } else {
            0.0
        };
        let mut results = Vec::with_capacity(self.references.len());
        for reference in &self.references {
            let start = Instant::now();
            let mut pairs = match (&mut self.backend, &reference.features, &features) {
                (Backend::Features(b), Some(r), Some(q)) => b.match_features_blocking(r, q)?,
                (Backend::Pair(b), None, None) => {
                    b.match_images_blocking(&reference.image, query)?
                }
                _ => return Err(Error::Input("surface feature contract mismatch".into())),
            };
            for pair in &mut pairs {
                restore(pair, reference.rotation, self.original_size);
            }
            results.push(Matches {
                rotation: reference.rotation,
                pairs,
                matching_ms: start.elapsed().as_secs_f64() * 1000.0,
            });
        }
        Ok((extraction_ms, results))
    }
}
fn validate_rotations(rotations: &[u8], limit: usize) -> Result<(), Error> {
    let unique: BTreeSet<_> = rotations.iter().copied().collect();
    if rotations.is_empty()
        || rotations.len() > 4
        || rotations.len() > limit
        || unique.len() != rotations.len()
        || rotations.iter().any(|v| *v > 3)
    {
        return Err(Error::Input(
            "surface orientations must be unique quarter turns within the candidate limit".into(),
        ));
    }
    Ok(())
}
fn rotate(image: &GrayImage, rotation: u8) -> GrayImage {
    match rotation {
        1 => imageops::rotate90(image),
        2 => imageops::rotate180(image),
        3 => imageops::rotate270(image),
        _ => image.clone(),
    }
}
fn restore(pair: &mut PixelMatch, rotation: u8, size: (u32, u32)) {
    let p = pair.reference;
    let w = f64::from(size.0) - 1.0;
    let h = f64::from(size.1) - 1.0;
    pair.reference = match rotation {
        1 => [p.y, h - p.x].into(),
        2 => [w - p.x, h - p.y].into(),
        3 => [w - p.y, p.x].into(),
        _ => p,
    };
}
#[cfg(test)]
mod tests;
