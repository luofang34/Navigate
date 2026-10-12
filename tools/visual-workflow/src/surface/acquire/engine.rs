//! Share one model and one validated query feature packet across map windows.
use super::super::matching::{Matches, restore, rotate};
use crate::{
    Error,
    config::{Backend, Config},
};
use image::GrayImage;
use navigate_visual_onnx::{XFeatFeatures, XFeatStages};
use navigate_visual_workflow::FeatureBackend;
use std::time::Instant;

pub(in crate::surface) struct Engine {
    pub backend: Box<XFeatStages>,
    query: Option<XFeatFeatures>,
    pub query_bytes: usize,
    pub reference_encodings: usize,
    pub peak_reference_bytes: usize,
}
impl Engine {
    pub fn new(backend: Backend) -> Result<Self, Error> {
        let Backend::Features(backend) = backend else {
            return Err(Error::Input(
                "acquisition requires a feature backend".into(),
            ));
        };
        Ok(Self {
            backend,
            query: None,
            query_bytes: 0,
            reference_encodings: 0,
            peak_reference_bytes: 0,
        })
    }
    pub fn query_blocking(&mut self, query: &GrayImage, config: &Config) -> Result<(), Error> {
        self.query = None;
        self.query_bytes = 0;
        self.reference_encodings = 0;
        self.peak_reference_bytes = 0;
        let features = self.backend.extract_features_blocking(query)?;
        let info = self.backend.validate_features(&features, query)?;
        if info.count > config.limits.features_per_image
            || info.resident_bytes > config.limits.query_feature_bytes
        {
            return Err(Error::Input("acquisition query feature budget".into()));
        }
        self.query = Some(features);
        self.query_bytes = info.resident_bytes;
        Ok(())
    }
    pub fn match_blocking(
        &mut self,
        image: &GrayImage,
        rotation: u8,
        resident_raster_bytes: usize,
        config: &Config,
    ) -> Result<(Matches, f64), Error> {
        if self.reference_encodings >= config.limits.candidates {
            return Err(Error::Input(
                "acquisition candidate budget exhausted".into(),
            ));
        }
        let overlap = resident_raster_bytes.saturating_add(image.as_raw().len());
        if overlap > config.limits.reference_bytes {
            return Err(Error::Input("acquisition rotated raster budget".into()));
        }
        let started = Instant::now();
        let rotated = rotate(image, rotation);
        let features = self.backend.extract_features_blocking(&rotated)?;
        let info = self.backend.validate_features(&features, &rotated)?;
        let peak = overlap.saturating_add(info.resident_bytes);
        if peak > config.limits.reference_bytes || info.count > config.limits.features_per_image {
            return Err(Error::Input("acquisition reference feature budget".into()));
        }
        self.peak_reference_bytes = self.peak_reference_bytes.max(peak);
        self.reference_encodings = self.reference_encodings.wrapping_add(1);
        drop(rotated);
        let extraction_ms = started.elapsed().as_secs_f64() * 1000.0;
        let started = Instant::now();
        let mut pairs = self.backend.match_features_blocking(
            &features,
            self.query
                .as_ref()
                .ok_or_else(|| Error::Input("acquisition query is not encoded".into()))?,
        )?;
        for pair in &mut pairs {
            restore(pair, rotation, image.dimensions());
        }
        Ok((
            Matches {
                rotation,
                pairs,
                matching_ms: started.elapsed().as_secs_f64() * 1000.0,
            },
            extraction_ms,
        ))
    }
}
