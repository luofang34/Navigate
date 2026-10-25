//! Retain the reference encoder result across a sequence.
use crate::{Error, config::Backend};
use image::GrayImage;
use navigate_visual::{ImageMatcher, PixelMatch};
use navigate_visual_onnx::XFeatFeatures;
use navigate_visual_workflow::FeatureBackend;

pub(super) struct Matcher {
    backend: Backend,
    reference_features: Option<XFeatFeatures>,
}
impl Matcher {
    pub fn new_blocking(mut backend: Backend, reference: &GrayImage) -> Result<Self, Error> {
        let reference_features = match &mut backend {
            Backend::Features(b) => Some(b.extract_features_blocking(reference)?),
            Backend::Pair(_) => None,
        };
        Ok(Self {
            backend,
            reference_features,
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
        reference: &GrayImage,
        query: &GrayImage,
    ) -> Result<Vec<PixelMatch>, Error> {
        Ok(match (&mut self.backend, &self.reference_features) {
            (Backend::Features(b), Some(features)) => {
                let query = b.extract_features_blocking(query)?;
                b.match_features_blocking(features, &query)?
            }
            (Backend::Pair(b), _) => b.match_images_blocking(reference, query)?,
            _ => return Err(Error::Input("missing reference features".into())),
        })
    }
}
