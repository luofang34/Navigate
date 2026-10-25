//! LoFTR-DS behind the Navigate matcher port.
use crate::{Correspondence, Loftr, MatchError, input};
use image::GrayImage;
use navigate_visual::PixelMatch;

/// Model input size of the pinned export and of the parity reference.
const INPUT: [usize; 2] = [640, 480];
/// Most correspondences returned for one pair.
const LIMIT: usize = 4096;

/// LoFTR-DS on a burn device, with the input transform of the ONNX adapter.
///
/// Each image has its own letterbox transform. Output pixels refer to the source
/// images. The shared geometry verifier must use each camera's calibration.
pub struct LoftrMatcher {
    model: Loftr,
    identity: String,
}

impl LoftrMatcher {
    /// Wrap a loaded model. `weights_sha256` names the weight file in the identity.
    pub fn new(model: Loftr, weights_sha256: &str) -> Self {
        Self {
            model,
            identity: format!("loftr-dual-softmax-640x480/burn/{weights_sha256}"),
        }
    }

    /// Stable backend and model identity for reports and derived caches.
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Match a reference image and a query image.
    ///
    /// An image with less than two grey levels of contrast gives no matches.
    ///
    /// # Errors
    ///
    /// Returns an error for an unsupported image size or a device failure.
    pub async fn match_images(
        &self,
        reference: &GrayImage,
        query: &GrayImage,
    ) -> Result<Vec<PixelMatch>, MatchError> {
        if [reference, query].iter().any(|image| flat(image)) {
            return Ok(Vec::new());
        }
        let a = input::prepare(reference, INPUT)?;
        let b = input::prepare(query, INPUT)?;
        let correspondences = self
            .model
            .correspondences(a.values.clone(), b.values.clone(), INPUT)
            .await?;
        Ok(correspondences
            .iter()
            .filter_map(|c| pixel_match(c, [&a, &b], [reference, query]))
            .take(LIMIT)
            .collect())
    }
}

fn flat(image: &GrayImage) -> bool {
    let min = image.as_raw().iter().copied().min().unwrap_or(0);
    let max = image.as_raw().iter().copied().max().unwrap_or(0);
    max.saturating_sub(min) < 2
}

fn pixel_match(
    c: &Correspondence,
    inputs: [&input::InputImage; 2],
    images: [&GrayImage; 2],
) -> Option<PixelMatch> {
    let reference = inputs[0].source([f64::from(c.first[0]), f64::from(c.first[1])]);
    let query = inputs[1].source([f64::from(c.second[0]), f64::from(c.second[1])]);
    (input::inside(reference, images[0]) && input::inside(query, images[1])).then(|| PixelMatch {
        reference: reference.into(),
        query: query.into(),
    })
}

#[cfg(not(target_arch = "wasm32"))]
impl navigate_visual::ImageMatcher for LoftrMatcher {
    fn identity(&self) -> &str {
        &self.identity
    }

    fn match_images_blocking(
        &mut self,
        reference: &GrayImage,
        query: &GrayImage,
    ) -> Result<Vec<PixelMatch>, navigate_visual::VisualError> {
        pollster::block_on(self.match_images(reference, query)).map_err(|source| {
            navigate_visual::VisualError::Backend {
                backend: self.identity.clone(),
                source: Box::new(source),
            }
        })
    }
}
