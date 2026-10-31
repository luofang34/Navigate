//! Pure-Rust XFeat dense backbone and sparse keypoint decoder on burn.
//!
//! [`XFeat`] executes the 27 folded convolution layers of `verlab/accelerated_features`
//! on any compiled burn [`Device`], scores keypoint candidates on the device, and
//! decodes unit-length 64-D descriptors and source image keypoints on the host.
mod decode;
mod score;
mod upsample;

use crate::{
    MatchError, Weights, input,
    loftr::layers::{Conv, upload},
};
use burn::tensor::{Device, Tensor, TensorData, Transaction, activation::sigmoid};
use image::GrayImage;
use navigate_visual::PixelMatch;
use score::ScoredMaps;
use upsample::Upsample;

/// Letterbox width before vertical resize to 576 pixels.
const PREPARE_WIDTH: usize = 800;
/// Letterbox height before vertical resize to 576 pixels.
const PREPARE_HEIGHT: usize = 600;
/// Dense network input width in pixels.
pub const XFEAT_INPUT_WIDTH: usize = 800;
/// Dense network input height in pixels.
pub const XFEAT_INPUT_HEIGHT: usize = 576;
/// Coarse grid height (`576 / 8`).
const GRID_HEIGHT: usize = 72;
/// Coarse grid width (`800 / 8`).
const GRID_WIDTH: usize = 100;

/// Sparse keypoints and L2-normalized 64-D descriptors extracted from one image.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseFeatures {
    /// Keypoint positions `[x, y]` in source-image pixel coordinates.
    pub pixels: Vec<[f64; 2]>,
    /// Keypoint positions `[x, y]` in `800 × 600` model-frame coordinates.
    pub model_pixels: Vec<[f32; 2]>,
    /// Detection confidence score of each keypoint.
    pub scores: Vec<f32>,
    /// Row-major L2-normalized descriptors (`64` values per keypoint).
    pub descriptors: Vec<f32>,
}

impl SparseFeatures {
    /// Construct an empty feature set.
    pub fn empty() -> Self {
        Self {
            pixels: Vec::new(),
            model_pixels: Vec::new(),
            scores: Vec::new(),
            descriptors: Vec::new(),
        }
    }

    /// Return the number of keypoints.
    pub fn len(&self) -> usize {
        self.pixels.len()
    }

    /// Return `true` when there are no keypoints.
    pub fn is_empty(&self) -> bool {
        self.pixels.is_empty()
    }

    /// Return normalized LighterGlue coordinates `[(x - 400) / 400, (y - 300) / 400]`.
    pub fn normalized_keypoints(&self) -> Vec<f32> {
        self.model_pixels
            .iter()
            .flat_map(|p| [(p[0] - 400.0) / 400.0, (p[1] - 300.0) / 400.0])
            .collect()
    }

    /// Return up to `max_keypoints` keypoints with the highest confidence scores.
    pub fn truncate_top(&self, max_keypoints: usize) -> Self {
        let count = self
            .pixels
            .len()
            .min(self.model_pixels.len())
            .min(self.scores.len())
            .min(self.descriptors.len() / 64);
        if max_keypoints == 0 || count == 0 {
            return Self::empty();
        }
        let keep = max_keypoints.min(count);
        if self.scores[..count].windows(2).all(|w| w[0] >= w[1]) {
            return Self {
                pixels: self.pixels[..keep].to_vec(),
                model_pixels: self.model_pixels[..keep].to_vec(),
                scores: self.scores[..keep].to_vec(),
                descriptors: self.descriptors[..keep * 64].to_vec(),
            };
        }
        let mut indices: Vec<usize> = (0..count).collect();
        indices.sort_by(|&a, &b| {
            self.scores[b]
                .total_cmp(&self.scores[a])
                .then_with(|| a.cmp(&b))
        });
        indices.truncate(keep);
        let mut out = Self {
            pixels: Vec::with_capacity(keep),
            model_pixels: Vec::with_capacity(keep),
            scores: Vec::with_capacity(keep),
            descriptors: Vec::with_capacity(keep * 64),
        };
        for idx in indices {
            out.pixels.push(self.pixels[idx]);
            out.model_pixels.push(self.model_pixels[idx]);
            out.scores.push(self.scores[idx]);
            out.descriptors
                .extend_from_slice(&self.descriptors[idx * 64..(idx + 1) * 64]);
        }
        out
    }

    /// Match two feature sets with distinct mutual cosine nearest neighbours.
    pub fn mutual_matches(&self, other: &Self) -> Vec<PixelMatch> {
        #[derive(Clone, Copy)]
        struct Nearest {
            index: usize,
            first: f32,
            second: f32,
        }
        impl Nearest {
            fn update(&mut self, index: usize, score: f32) {
                if score > self.first {
                    self.second = self.first;
                    self.first = score;
                    self.index = index;
                } else if score > self.second {
                    self.second = score;
                }
            }
            fn distinct(self) -> bool {
                self.first > 0.82 && 1.0 - self.first < 0.9 * (1.0 - self.second)
            }
        }
        if self.is_empty() || other.is_empty() {
            return Vec::new();
        }
        let empty = Nearest {
            index: 0,
            first: -2.0,
            second: -2.0,
        };
        let mut a = vec![empty; self.len()];
        let mut b = vec![empty; other.len()];
        for (i, da) in self.descriptors.as_chunks::<64>().0.iter().enumerate() {
            for (j, db) in other.descriptors.as_chunks::<64>().0.iter().enumerate() {
                let score: f32 = da.iter().zip(db).map(|(x, y)| x * y).sum();
                a[i].update(j, score);
                b[j].update(i, score);
            }
        }
        a.iter()
            .enumerate()
            .filter_map(|(i, n)| {
                let reverse = b.get(n.index)?;
                (n.distinct() && reverse.distinct() && reverse.index == i).then(|| PixelMatch {
                    reference: self.pixels[i].into(),
                    query: other.pixels[n.index].into(),
                })
            })
            .collect()
    }
}

/// Dense feature maps produced by one XFeat forward pass.
#[derive(Clone, Debug)]
pub struct DenseMaps {
    /// Coarse descriptor map `[1, 64, 72, 100]`.
    pub descriptors: Vec<f32>,
    /// Keypoint cell logits `[1, 65, 72, 100]`.
    pub keypoint_logits: Vec<f32>,
    /// Reliability map `[1, 1, 72, 100]`.
    pub reliability: Vec<f32>,
}

/// Pure-Rust XFeat feature extractor on one burn [`Device`].
#[derive(Clone, Debug)]
pub struct XFeat {
    block1: [Conv; 4],
    skip1: Conv,
    block2: [Conv; 2],
    block3: [Conv; 3],
    block4: [Conv; 3],
    block5: [Conv; 4],
    block_fusion: [Conv; 3],
    heatmap_head: [Conv; 3],
    keypoint_head: [Conv; 4],
    upsample: Upsample,
    device: Device,
}

/// Device tensors of one dense forward pass before read-back.
struct DenseTensors {
    descriptors: Tensor<4>,
    logits: Tensor<4>,
    reliability: Tensor<4>,
}

impl XFeat {
    /// Load named XFeat weights onto `device`.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when any tensor is missing, has an unexpected shape,
    /// or remains unused.
    pub fn load(mut weights: Weights, device: &Device) -> Result<Self, MatchError> {
        let c = |w: &mut Weights, name: &str, shape: [usize; 3], stride: usize| {
            Conv::load(w, name, shape, stride, true, device)
        };
        let model = Self {
            block1: [
                c(&mut weights, "block1.0", [4, 1, 3], 1)?,
                c(&mut weights, "block1.1", [8, 4, 3], 2)?,
                c(&mut weights, "block1.2", [8, 8, 3], 1)?,
                c(&mut weights, "block1.3", [24, 8, 3], 2)?,
            ],
            skip1: c(&mut weights, "skip1.1", [24, 1, 1], 1)?,
            block2: [
                c(&mut weights, "block2.0", [24, 24, 3], 1)?,
                c(&mut weights, "block2.1", [24, 24, 3], 1)?,
            ],
            block3: [
                c(&mut weights, "block3.0", [64, 24, 3], 2)?,
                c(&mut weights, "block3.1", [64, 64, 3], 1)?,
                c(&mut weights, "block3.2", [64, 64, 1], 1)?,
            ],
            block4: [
                c(&mut weights, "block4.0", [64, 64, 3], 2)?,
                c(&mut weights, "block4.1", [64, 64, 3], 1)?,
                c(&mut weights, "block4.2", [64, 64, 3], 1)?,
            ],
            block5: [
                c(&mut weights, "block5.0", [128, 64, 3], 2)?,
                c(&mut weights, "block5.1", [128, 128, 3], 1)?,
                c(&mut weights, "block5.2", [128, 128, 3], 1)?,
                c(&mut weights, "block5.3", [64, 128, 1], 1)?,
            ],
            block_fusion: [
                c(&mut weights, "block_fusion.0", [64, 64, 3], 1)?,
                c(&mut weights, "block_fusion.1", [64, 64, 3], 1)?,
                c(&mut weights, "block_fusion.2", [64, 64, 1], 1)?,
            ],
            heatmap_head: [
                c(&mut weights, "heatmap_head.0", [64, 64, 1], 1)?,
                c(&mut weights, "heatmap_head.1", [64, 64, 1], 1)?,
                c(&mut weights, "heatmap_head.2", [1, 64, 1], 1)?,
            ],
            keypoint_head: [
                c(&mut weights, "keypoint_head.0", [64, 64, 1], 1)?,
                c(&mut weights, "keypoint_head.1", [64, 64, 1], 1)?,
                c(&mut weights, "keypoint_head.2", [64, 64, 1], 1)?,
                c(&mut weights, "keypoint_head.3", [65, 64, 1], 1)?,
            ],
            upsample: Upsample::new(device),
            device: device.clone(),
        };
        weights.finish()?;
        Ok(model)
    }

    /// Borrow the burn device of this model.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Extract up to `limit` sparse keypoints and 64-D descriptors from `image`.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when the image dimensions are outside `16..=8192`,
    /// when `limit` is zero, or when device read-back fails.
    pub async fn extract(
        &self,
        image: &GrayImage,
        limit: usize,
    ) -> Result<SparseFeatures, MatchError> {
        self.extract_masked(image, None, limit).await
    }

    /// Extract up to `limit` sparse keypoints inside `valid_mask` from `image`.
    ///
    /// The decoder discards candidate keypoints where `valid_mask` has a pixel value below `128`.
    /// The decoder ranks the remaining candidates and keeps up to `limit` keypoints.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when `valid_mask` dimensions do not match `image`.
    /// Returns [`MatchError`] when `image` dimensions are outside `16..=8192` or `limit` is outside `1..=4096`.
    /// Returns [`MatchError`] when device read-back fails.
    pub async fn extract_masked(
        &self,
        image: &GrayImage,
        valid_mask: Option<&GrayImage>,
        limit: usize,
    ) -> Result<SparseFeatures, MatchError> {
        if limit == 0 || limit > 4096 {
            return Err(MatchError::Output("feature limit must be 1 through 4096"));
        }
        if let Some(mask) = valid_mask
            && mask.dimensions() != image.dimensions()
        {
            return Err(MatchError::Output(
                "valid mask dimensions do not match image",
            ));
        }
        let min = image.as_raw().iter().copied().min().unwrap_or(0);
        let max = image.as_raw().iter().copied().max().unwrap_or(0);
        if max.saturating_sub(min) < 2 {
            return Ok(SparseFeatures::empty());
        }
        let prepared = input::prepare(image, [PREPARE_WIDTH, PREPARE_HEIGHT])?;
        let normalized = decode::normalize_resized(&prepared.values);
        let maps = self.forward_scored(normalized).await?;
        decode::decode(&maps, &prepared, image, valid_mask, limit)
    }

    /// Run the dense convolutional network on a normalized `576 × 800` buffer.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when `normalized` does not have `576 * 800` values or
    /// when device execution fails.
    pub async fn forward_dense(&self, normalized: Vec<f32>) -> Result<DenseMaps, MatchError> {
        let maps = self.dense_tensors(normalized)?;
        let transaction = Transaction::default()
            .register(maps.descriptors)
            .register(maps.logits)
            .register(maps.reliability);
        let [descriptors, keypoint_logits, reliability] = read_back(transaction).await?;
        Ok(DenseMaps {
            descriptors,
            keypoint_logits,
            reliability,
        })
    }

    /// Run the dense network and score keypoint candidates on the device.
    async fn forward_scored(&self, normalized: Vec<f32>) -> Result<ScoredMaps, MatchError> {
        let maps = self.dense_tensors(normalized)?;
        let scores = score::score_map(maps.logits, maps.reliability, &self.upsample.reliability)?;
        let transaction = Transaction::default()
            .register(maps.descriptors)
            .register(scores);
        let [descriptors, scores] = read_back(transaction).await?;
        Ok(ScoredMaps {
            descriptors,
            scores,
        })
    }

    fn dense_tensors(&self, normalized: Vec<f32>) -> Result<DenseTensors, MatchError> {
        if normalized.len() != XFEAT_INPUT_HEIGHT * XFEAT_INPUT_WIDTH {
            return Err(MatchError::Output("dense XFeat input buffer size mismatch"));
        }
        let x = upload(
            normalized,
            [1, 1, XFEAT_INPUT_HEIGHT, XFEAT_INPUT_WIDTH],
            &self.device,
        );
        let mut x1 = self.block1[0].forward_relu(x.clone());
        for layer in &self.block1[1..] {
            x1 = layer.forward_relu(x1);
        }
        let pooled = x
            .clone()
            .reshape([1, 1, GRID_HEIGHT * 2, 4, GRID_WIDTH * 2, 4])
            .mean_dim(5)
            .squeeze_dim::<5>(5)
            .mean_dim(3)
            .squeeze_dim::<4>(3);
        let skip = self.skip1.forward(pooled);
        let x2 = self.block2[1].forward_relu(self.block2[0].forward_relu(x1 + skip));
        let x3 = self.block3[2]
            .forward_relu(self.block3[1].forward_relu(self.block3[0].forward_relu(x2)));
        let x4 = self.block4[2]
            .forward_relu(self.block4[1].forward_relu(self.block4[0].forward_relu(x3.clone())));
        let x5 = self.block5[3]
            .forward_relu(self.block5[2].forward_relu(
                self.block5[1].forward_relu(self.block5[0].forward_relu(x4.clone())),
            ));
        let fused = x3 + self.upsample.x4.apply(x4)? + self.upsample.x5.apply(x5)?;
        let descriptors = self.block_fusion[2]
            .forward(self.block_fusion[1].forward_relu(self.block_fusion[0].forward_relu(fused)));
        let reliability = sigmoid(
            self.heatmap_head[2].forward(
                self.heatmap_head[1]
                    .forward_relu(self.heatmap_head[0].forward_relu(descriptors.clone())),
            ),
        );
        let patches = x
            .reshape([1, GRID_HEIGHT, 8, GRID_WIDTH, 8])
            .permute([0, 2, 4, 1, 3])
            .reshape([1, 64, GRID_HEIGHT, GRID_WIDTH]);
        let logits = self.keypoint_head[3].forward(self.keypoint_head[2].forward_relu(
            self.keypoint_head[1].forward_relu(self.keypoint_head[0].forward_relu(patches)),
        ));
        Ok(DenseTensors {
            descriptors,
            logits,
            reliability,
        })
    }
}

/// Read back `N` registered tensors as finite host values.
async fn read_back<const N: usize>(transaction: Transaction) -> Result<[Vec<f32>; N], MatchError> {
    let data: Vec<TensorData> =
        transaction
            .execute_async()
            .await
            .map_err(|error| MatchError::Device {
                stage: "xfeat read-back",
                message: format!("{error:?}"),
            })?;
    let values: Vec<Vec<f32>> = data.iter().map(|d| d.iter::<f32>().collect()).collect();
    if values.iter().flatten().any(|v| !v.is_finite()) {
        return Err(MatchError::Output("non-finite dense XFeat output"));
    }
    <[_; N]>::try_from(values).map_err(|_| MatchError::Output("xfeat read-back count"))
}

#[cfg(test)]
pub(crate) mod tests;
