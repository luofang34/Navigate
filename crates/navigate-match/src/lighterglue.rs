//! Pure-Rust LighterGlue feature matcher on burn.
//!
//! [`LighterGlue`] runs the six-layer rotary self-attention and bidirectional
//! cross-attention transformer of `verlab/accelerated_features` and `cvg/LightGlue`
//! (`input_dim = 64`, `descriptor_dim = 96`, `num_heads = 1`).
mod batch;
mod fast;

pub use fast::{FastMatchConfig, FastMatchStats, IndexedMatch};

use crate::{
    MatchError, SparseFeatures, Weights,
    kernels::{Layout, product},
    loftr::layers::{LayerNorm, Linear, upload},
};
use burn::tensor::{
    Device, Tensor, Transaction,
    activation::{gelu, log_softmax, relu, softmax},
};
use navigate_visual::PixelMatch;

const INPUT_DIM: usize = 64;
const MODEL_DIM: usize = 96;
const HALF_DIM: usize = 48;
const FFN_DIM: usize = 192;
const LAYER_COUNT: usize = 6;

#[derive(Clone, Debug)]
struct FeedForward {
    fc0: Linear,
    norm: LayerNorm,
    fc3: Linear,
}

impl FeedForward {
    fn load(weights: &mut Weights, prefix: &str, device: &Device) -> Result<Self, MatchError> {
        Ok(Self {
            fc0: Linear::load(
                weights,
                &format!("{prefix}.0"),
                [FFN_DIM, FFN_DIM],
                true,
                device,
            )?,
            norm: LayerNorm::load(weights, &format!("{prefix}.1"), FFN_DIM, device)?,
            fc3: Linear::load(
                weights,
                &format!("{prefix}.3"),
                [MODEL_DIM, FFN_DIM],
                true,
                device,
            )?,
        })
    }

    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let hidden = gelu(self.norm.forward(self.fc0.forward(x)));
        self.fc3.forward(hidden)
    }
}

#[derive(Clone, Debug)]
struct SelfBlock {
    wqkv: Linear,
    out_proj: Linear,
    ffn: FeedForward,
}

impl SelfBlock {
    fn load(weights: &mut Weights, prefix: &str, device: &Device) -> Result<Self, MatchError> {
        Ok(Self {
            wqkv: Linear::load(
                weights,
                &format!("{prefix}.Wqkv"),
                [MODEL_DIM * 3, MODEL_DIM],
                true,
                device,
            )?,
            out_proj: Linear::load(
                weights,
                &format!("{prefix}.out_proj"),
                [MODEL_DIM, MODEL_DIM],
                true,
                device,
            )?,
            ffn: FeedForward::load(weights, &format!("{prefix}.ffn"), device)?,
        })
    }

    pub(super) fn forward(
        &self,
        x: Tensor<3>,
        cos: &Tensor<3>,
        sin: &Tensor<3>,
        key_mask: Option<&Tensor<3>>,
    ) -> Tensor<3> {
        let [batch, n, _] = x.dims();
        let device = x.device();
        let qkv = self
            .wqkv
            .forward(x.clone())
            .reshape([batch, n, MODEL_DIM, 3]);
        let q = rotary(qkv.clone().narrow(3, 0, 1).squeeze_dim::<3>(3), cos, sin);
        let k = rotary(qkv.clone().narrow(3, 1, 1).squeeze_dim::<3>(3), cos, sin);
        let v = qkv.narrow(3, 2, 1).squeeze_dim::<3>(3);
        let trans_b = Layout {
            trans_b: true,
            ..Layout::default()
        };
        let mut scores =
            product(q, k, Tensor::<1>::zeros([n], &device), trans_b) / (MODEL_DIM as f64).sqrt();
        if let Some(mask) = key_mask {
            scores = scores + mask.clone();
        }
        let context = product(
            softmax(scores, 2),
            v,
            Tensor::<1>::zeros([MODEL_DIM], &device),
            Layout::default(),
        );
        let message = self.out_proj.forward(context);
        x.clone() + self.ffn.forward(Tensor::cat(vec![x, message], 2))
    }
}

#[derive(Clone, Debug)]
struct CrossBlock {
    to_qk: Linear,
    to_v: Linear,
    to_out: Linear,
    ffn: FeedForward,
}

impl CrossBlock {
    fn load(weights: &mut Weights, prefix: &str, device: &Device) -> Result<Self, MatchError> {
        Ok(Self {
            to_qk: Linear::load(
                weights,
                &format!("{prefix}.to_qk"),
                [MODEL_DIM, MODEL_DIM],
                true,
                device,
            )?,
            to_v: Linear::load(
                weights,
                &format!("{prefix}.to_v"),
                [MODEL_DIM, MODEL_DIM],
                true,
                device,
            )?,
            to_out: Linear::load(
                weights,
                &format!("{prefix}.to_out"),
                [MODEL_DIM, MODEL_DIM],
                true,
                device,
            )?,
            ffn: FeedForward::load(weights, &format!("{prefix}.ffn"), device)?,
        })
    }

    pub(super) fn forward(
        &self,
        a: Tensor<3>,
        b: Tensor<3>,
        a_key_mask: Option<&Tensor<3>>,
    ) -> (Tensor<3>, Tensor<3>) {
        let [_, m, _] = b.dims();
        let device = a.device();
        let scale = (MODEL_DIM as f64).powf(0.25);
        let q0 = self.to_qk.forward(a.clone()) / scale;
        let q1 = self.to_qk.forward(b.clone()) / scale;
        let trans_b = Layout {
            trans_b: true,
            ..Layout::default()
        };
        let sim = product(q0, q1, Tensor::<1>::zeros([m], &device), trans_b);
        let vb = self.to_v.forward(b.clone());
        let va = self.to_v.forward(a.clone());
        let zeros = Tensor::<1>::zeros([MODEL_DIM], &device);
        let m0 = self.to_out.forward(product(
            softmax(sim.clone(), 2),
            vb,
            zeros.clone(),
            Layout::default(),
        ));
        let mut sim_ba = sim.swap_dims(1, 2);
        if let Some(mask) = a_key_mask {
            sim_ba = sim_ba + mask.clone();
        }
        let m1 = self
            .to_out
            .forward(product(softmax(sim_ba, 2), va, zeros, Layout::default()));
        (
            a.clone() + self.ffn.forward(Tensor::cat(vec![a, m0], 2)),
            b.clone() + self.ffn.forward(Tensor::cat(vec![b, m1], 2)),
        )
    }
}

#[derive(Clone, Debug)]
struct Layer {
    self_attn: SelfBlock,
    cross_attn: CrossBlock,
}

/// Pure-Rust LighterGlue matcher on one burn [`Device`].
#[derive(Clone, Debug)]
pub struct LighterGlue {
    input_proj: Linear,
    posenc_wr: Linear,
    layers: Vec<Layer>,
    final_proj: Linear,
    matchability: Linear,
    device: Device,
}

impl LighterGlue {
    /// Load named LighterGlue weights onto `device`.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when any tensor is missing, has an unexpected shape,
    /// or remains unused.
    pub fn load(mut weights: Weights, device: &Device) -> Result<Self, MatchError> {
        let input_proj = Linear::load(
            &mut weights,
            "net.input_proj",
            [MODEL_DIM, INPUT_DIM],
            true,
            device,
        )?;
        let posenc_wr = Linear::load(&mut weights, "net.posenc.Wr", [HALF_DIM, 2], false, device)?;
        let mut layers = Vec::with_capacity(LAYER_COUNT);
        for index in 0..LAYER_COUNT {
            let prefix = format!("net.transformers.{index}");
            layers.push(Layer {
                self_attn: SelfBlock::load(&mut weights, &format!("{prefix}.self_attn"), device)?,
                cross_attn: CrossBlock::load(
                    &mut weights,
                    &format!("{prefix}.cross_attn"),
                    device,
                )?,
            });
        }
        let final_proj = Linear::load(
            &mut weights,
            "net.log_assignment.5.final_proj",
            [MODEL_DIM, MODEL_DIM],
            true,
            device,
        )?;
        let matchability = Linear::load(
            &mut weights,
            "net.log_assignment.5.matchability",
            [1, MODEL_DIM],
            true,
            device,
        )?;
        weights.finish()?;
        Ok(Self {
            input_proj,
            posenc_wr,
            layers,
            final_proj,
            matchability,
            device: device.clone(),
        })
    }

    /// Compute the `[N, M]` log-assignment matrix for normalized keypoints and descriptors.
    ///
    /// `keypoints0` has `2 * n` values, `keypoints1` has `2 * m` values, `descriptors0`
    /// has `64 * n` values, and `descriptors1` has `64 * m` values.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when input lengths are inconsistent or read-back fails.
    pub async fn log_assignment(
        &self,
        keypoints0: &[f32],
        keypoints1: &[f32],
        descriptors0: &[f32],
        descriptors1: &[f32],
    ) -> Result<Vec<f32>, MatchError> {
        let scores = self.forward_tensor(keypoints0, keypoints1, descriptors0, descriptors1)?;
        let data = scores
            .into_data_async()
            .await
            .map_err(|error| MatchError::Device {
                stage: "lighterglue read-back",
                message: format!("{error:?}"),
            })?;
        let values: Vec<f32> = data.iter::<f32>().collect();
        if values.iter().any(|v| !v.is_finite()) {
            return Err(MatchError::Output("non-finite LighterGlue log assignment"));
        }
        Ok(values)
    }

    /// Match two [`SparseFeatures`] sets and return accepted [`PixelMatch`] pairs.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when tensor dimensions are invalid or read-back fails.
    pub async fn match_features(
        &self,
        reference: &SparseFeatures,
        query: &SparseFeatures,
    ) -> Result<Vec<PixelMatch>, MatchError> {
        if reference.is_empty() || query.is_empty() {
            return Ok(Vec::new());
        }
        let kp0 = reference.normalized_keypoints();
        let kp1 = query.normalized_keypoints();
        let scores = self.forward_tensor(&kp0, &kp1, &reference.descriptors, &query.descriptors)?;
        let n = reference.len();
        let m = query.len();
        let (row_max, row_argmax) = scores.clone().max_dim_with_indices(2);
        let (_, col_argmax) = scores.max_dim_with_indices(1);
        let data = Transaction::default()
            .register(row_max.reshape([n]))
            .register(row_argmax.reshape([n]))
            .register(col_argmax.reshape([m]))
            .execute_async()
            .await
            .map_err(|error| MatchError::Device {
                stage: "lighterglue assignment read-back",
                message: format!("{error:?}"),
            })?;
        let [rmax, ridx, cidx] = <[_; 3]>::try_from(data)
            .map_err(|_| MatchError::Output("lighterglue read-back count"))?;
        let row_scores: Vec<f32> = rmax.iter::<f32>().collect();
        let row_indices: Vec<i64> = ridx.iter::<i64>().collect();
        let col_indices: Vec<i64> = cidx.iter::<i64>().collect();
        if row_scores.iter().any(|v| !v.is_finite()) {
            return Err(MatchError::Output("non-finite LighterGlue row score"));
        }
        let threshold = 0.1_f32.ln();
        Ok(row_scores
            .into_iter()
            .zip(row_indices)
            .enumerate()
            .filter_map(|(i, (score, j))| {
                let j = usize::try_from(j).ok()?;
                let mutual = col_indices.get(j).copied() == Some(i as i64);
                (score > threshold && mutual).then(|| PixelMatch {
                    reference: reference.pixels[i].into(),
                    query: query.pixels[j].into(),
                })
            })
            .collect())
    }

    fn forward_tensor(
        &self,
        keypoints0: &[f32],
        keypoints1: &[f32],
        descriptors0: &[f32],
        descriptors1: &[f32],
    ) -> Result<Tensor<3>, MatchError> {
        let n = keypoints0.len() / 2;
        let m = keypoints1.len() / 2;
        if n == 0
            || m == 0
            || keypoints0.len() != 2 * n
            || keypoints1.len() != 2 * m
            || descriptors0.len() != INPUT_DIM * n
            || descriptors1.len() != INPUT_DIM * m
        {
            return Err(MatchError::Output("LighterGlue input shape mismatch"));
        }
        let k0 = upload(keypoints0.to_vec(), [1, n, 2], &self.device);
        let k1 = upload(keypoints1.to_vec(), [1, m, 2], &self.device);
        let mut a = self.input_proj.forward(upload(
            descriptors0.to_vec(),
            [1, n, INPUT_DIM],
            &self.device,
        ));
        let mut b = self.input_proj.forward(upload(
            descriptors1.to_vec(),
            [1, m, INPUT_DIM],
            &self.device,
        ));
        let (cos0, sin0) = self.encoding(k0, n);
        let (cos1, sin1) = self.encoding(k1, m);
        for layer in &self.layers {
            a = layer.self_attn.forward(a, &cos0, &sin0, None);
            b = layer.self_attn.forward(b, &cos1, &sin1, None);
            (a, b) = layer.cross_attn.forward(a, b, None);
        }
        let (scores, _, _) = self.assignment_with_logits(a, b, None);
        Ok(scores)
    }

    pub(super) fn assignment_with_logits(
        &self,
        a: Tensor<3>,
        b: Tensor<3>,
        a_key_mask: Option<&Tensor<3>>,
    ) -> (Tensor<3>, Tensor<3>, Tensor<3>) {
        let [_, m, _] = b.dims();
        let scale = (MODEL_DIM as f64).powf(0.25);
        let d0 = self.final_proj.forward(a.clone()) / scale;
        let d1 = self.final_proj.forward(b.clone()) / scale;
        let trans_b = Layout {
            trans_b: true,
            ..Layout::default()
        };
        let sim = product(d0, d1, Tensor::<1>::zeros([m], &self.device), trans_b);
        let sim_col = match a_key_mask {
            Some(mask) => sim.clone() + mask.clone().swap_dims(1, 2),
            None => sim.clone(),
        };
        let ma = self.matchability.forward(a);
        let mb = self.matchability.forward(b);
        let mut scores = log_softmax(sim, 2)
            + log_softmax(sim_col, 1)
            + log_sigmoid(ma.clone())
            + log_sigmoid(mb.clone()).swap_dims(1, 2);
        if let Some(mask) = a_key_mask {
            scores = scores + mask.clone().swap_dims(1, 2);
        }
        (scores, ma, mb)
    }

    pub(super) fn encoding(&self, keypoints: Tensor<3>, count: usize) -> (Tensor<3>, Tensor<3>) {
        let [batch, _, _] = keypoints.dims();
        let projected = self.posenc_wr.forward(keypoints);
        let cos = projected
            .clone()
            .cos()
            .unsqueeze_dim::<4>(3)
            .repeat_dim(3, 2)
            .reshape([batch, count, MODEL_DIM]);
        let sin = projected
            .sin()
            .unsqueeze_dim::<4>(3)
            .repeat_dim(3, 2)
            .reshape([batch, count, MODEL_DIM]);
        (cos, sin)
    }
}

fn rotary(t: Tensor<3>, cos: &Tensor<3>, sin: &Tensor<3>) -> Tensor<3> {
    let [batch, n, _] = t.dims();
    let pairs = t.clone().reshape([batch, n, HALF_DIM, 2]);
    let p0 = pairs.clone().narrow(3, 0, 1);
    let p1 = pairs.narrow(3, 1, 1);
    let rotated = Tensor::cat(vec![p1.neg(), p0], 3).reshape([batch, n, MODEL_DIM]);
    t * cos.clone() + rotated * sin.clone()
}

fn log_sigmoid<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    relu(x.clone().neg()).neg() - (x.abs().neg().exp() + 1.0).log()
}

#[cfg(test)]
mod tests;
