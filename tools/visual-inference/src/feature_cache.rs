//! Bounded feature reuse within one immutable model and execution configuration.
use crate::{InferenceError, features::Features};
use image::GrayImage;
use sha2::{Digest, Sha256};
use std::{collections::VecDeque, sync::Arc};

#[derive(Default)]
pub(crate) struct FeatureCache {
    entries: VecDeque<([u8; 32], Arc<Features>)>,
}
impl FeatureCache {
    pub fn get_or_extract(
        &mut self,
        image: &GrayImage,
        extract: &mut impl FnMut(&GrayImage) -> Result<Features, InferenceError>,
    ) -> Result<Arc<Features>, InferenceError> {
        let mut hash = Sha256::new();
        hash.update(image.width().to_le_bytes());
        hash.update(image.height().to_le_bytes());
        hash.update(image.as_raw());
        let key: [u8; 32] = hash.finalize().into();
        if let Some(index) = self.entries.iter().position(|(digest, _)| *digest == key)
            && let Some(entry) = self.entries.remove(index)
        {
            let features = Arc::clone(&entry.1);
            self.entries.push_back(entry);
            return Ok(features);
        }
        let features = Arc::new(extract(image)?);
        // Two entries retain a query across candidate renders and an adjacent frame pair.
        if self.entries.len() == 2 {
            self.entries.pop_front();
        }
        self.entries.push_back((key, Arc::clone(&features)));
        Ok(features)
    }
}

#[cfg(test)]
mod tests;
