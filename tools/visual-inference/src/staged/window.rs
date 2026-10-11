//! A spatial subset keeps the encoder context and coordinates of its source image.
use super::{Features, InferenceError, XFeatFeatures};

impl XFeatFeatures {
    /// Select features inside a half-open source-pixel rectangle `[x, y, width, height]`.
    ///
    /// Coordinates, normalization, descriptors and source identity stay in the full
    /// source image. This is not equivalent to encoding a cropped image. Validate
    /// the result against the full source image, never against the crop pixels.
    /// The result owns copies of the selected records and can use the packet format.
    ///
    /// # Errors
    /// Rejects empty windows, overflow and windows outside the source image.
    pub fn select_window(&self, window: [u32; 4]) -> Result<Self, InferenceError> {
        let [x, y, width, height] = window;
        let right = x.checked_add(width);
        let bottom = y.checked_add(height);
        if width == 0
            || height == 0
            || right.is_none_or(|v| v > self.image_size[0])
            || bottom.is_none_or(|v| v > self.image_size[1])
        {
            return Err(InferenceError::Invalid(
                "feature window exceeds source image".into(),
            ));
        }
        Ok(self.select_pixels(|pixel| {
            pixel[0] >= f64::from(x)
                && pixel[1] >= f64::from(y)
                && pixel[0] < f64::from(x) + f64::from(width)
                && pixel[1] < f64::from(y) + f64::from(height)
        }))
    }

    /// Select records with a predicate in the unchanged source-image coordinates.
    ///
    /// This keeps source identity, descriptor bits and matcher normalization.
    /// It copies the selected records. Selecting before matching can change the
    /// matcher's context and is not equivalent to filtering its output matches.
    pub fn select_pixels(&self, include: impl Fn([f64; 2]) -> bool) -> Self {
        let mut selected = Features::empty(64);
        for (i, &pixel) in self.features.pixels.iter().enumerate() {
            if !include(pixel) {
                continue;
            }
            selected.pixels.push(pixel);
            selected
                .model_pixels
                .extend_from_slice(&self.features.model_pixels[2 * i..2 * i + 2]);
            selected.scores.push(self.features.scores[i]);
            selected
                .descriptors
                .extend_from_slice(&self.features.descriptors[64 * i..64 * i + 64]);
        }
        Self::new(self.contract, self.image_digest, self.image_size, selected)
    }
}
