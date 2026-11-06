//! Hardware-portable [`SparseMatcher`] trait and [`ModelSessionMatcher`] adapter.
//!
//! [`SparseMatcher`] mirrors [`crate::SparseFeatureExtractor`] so corridor tracking and
//! anchor pipelines execute identically on Burn (`burn-f32`, `burn-f16`) and on fixed-shape
//! ONNX Runtime sessions (`CoreML`, `TensorRT`, `CUDA`, or vendor neural processors).

use super::{
    FastMatchConfig, FastMatchStats, INPUT_DIM, IndexedMatch,
    batch::quantize_static_token_bucket,
    fast::{decode_mutual, select_input},
};
use crate::{
    MatchError, SparseFeatures,
    traits::{HostTensor, ModelInference, NamedTensor},
};
use navigate_visual::PixelMatch;
use std::sync::Mutex;

const PAD_LOGIT_MASK: f32 = -1.0e9;

/// Thread-safe sparse `LighterGlue` feature matcher across Burn and ONNX Runtime backends.
pub trait SparseMatcher: Send + Sync {
    /// Return the active runtime identifier (for example `"burn"`, `"coreml"`, or `"tensorrt"`).
    fn runtime_name(&self) -> &str;

    /// Match `references` against `query` in static-bucket batches and return [`IndexedMatch`]es.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when `config` or feature shapes fail validation or device execution fails.
    fn match_batch_indexed_blocking(
        &self,
        references: &[&SparseFeatures],
        query: &SparseFeatures,
        config: &FastMatchConfig,
    ) -> Result<Vec<(Vec<IndexedMatch>, FastMatchStats)>, MatchError>;

    /// Match one `reference` against `query` and return [`IndexedMatch`] records.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when `config` or feature shapes fail validation or device execution fails.
    fn match_indexed_blocking(
        &self,
        reference: &SparseFeatures,
        query: &SparseFeatures,
        config: &FastMatchConfig,
    ) -> Result<(Vec<IndexedMatch>, FastMatchStats), MatchError> {
        self.match_batch_indexed_blocking(&[reference], query, config)?
            .pop()
            .ok_or(MatchError::Output("empty single-pair matcher result"))
    }

    /// Match one `reference` against `query` and return accepted [`PixelMatch`] pairs.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when `config` or feature shapes fail validation or device execution fails.
    fn match_blocking(
        &self,
        reference: &SparseFeatures,
        query: &SparseFeatures,
        config: &FastMatchConfig,
    ) -> Result<(Vec<PixelMatch>, FastMatchStats), MatchError> {
        let (indexed, stats) = self.match_indexed_blocking(reference, query, config)?;
        Ok((indexed.into_iter().map(|m| m.pixel_match).collect(), stats))
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl SparseMatcher for super::LighterGlue {
    fn runtime_name(&self) -> &str {
        "burn"
    }

    fn match_batch_indexed_blocking(
        &self,
        references: &[&SparseFeatures],
        query: &SparseFeatures,
        config: &FastMatchConfig,
    ) -> Result<Vec<(Vec<IndexedMatch>, FastMatchStats)>, MatchError> {
        pollster::block_on(self.match_features_batch_indexed_fast(references, query, config))
    }

    fn match_indexed_blocking(
        &self,
        reference: &SparseFeatures,
        query: &SparseFeatures,
        config: &FastMatchConfig,
    ) -> Result<(Vec<IndexedMatch>, FastMatchStats), MatchError> {
        pollster::block_on(self.match_features_indexed_fast(reference, query, config))
    }
}

/// Static shape contract of a [`ModelSessionMatcher`] graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct StaticMatcherShape {
    /// Fixed batch dimension `B` (`1` or `4`, or `None` when dynamic).
    pub static_batch: Option<usize>,
    /// Fixed reference token count `N` (`None` when quantized to `512`/`1024` per call).
    pub static_ref_tokens: Option<usize>,
    /// Fixed query token count `M` (`None` when quantized to `512`/`1024` per call).
    pub static_query_tokens: Option<usize>,
    /// `true` when the graph expects `key_bias0` and `key_bias1` additive logit masks.
    pub has_key_bias: bool,
}

/// [`SparseMatcher`] adapter around a [`ModelInference`] session (`CoreML`, `TensorRT`, `ORT`, or NPU).
pub struct ModelSessionMatcher<S> {
    session: Mutex<S>,
    shape: StaticMatcherShape,
    runtime_label: String,
}

impl<S> ModelSessionMatcher<S>
where
    S: ModelInference<Tensor = HostTensor> + Send,
{
    /// Wrap `session` with its `shape` contract and `runtime_label`.
    pub fn new(session: S, shape: StaticMatcherShape, runtime_label: impl Into<String>) -> Self {
        Self {
            session: Mutex::new(session),
            shape,
            runtime_label: runtime_label.into(),
        }
    }
}

#[cfg(all(not(target_arch = "wasm32"), feature = "ort"))]
impl ModelSessionMatcher<crate::OrtSession> {
    /// Inspect `session` input declarations and build a [`ModelSessionMatcher`].
    #[must_use]
    pub fn from_ort(session: crate::OrtSession, runtime_label: impl Into<String>) -> Self {
        let mut static_batch = None;
        let mut static_ref_tokens = None;
        let mut static_query_tokens = None;
        let mut has_key_bias = false;
        for (name, dims) in session.input_specs() {
            match name.as_str() {
                "keypoints0" | "descriptors0" => {
                    static_batch = static_batch.or_else(|| dims.first().copied().flatten());
                    static_ref_tokens =
                        static_ref_tokens.or_else(|| dims.get(1).copied().flatten());
                }
                "keypoints1" | "descriptors1" => {
                    static_query_tokens =
                        static_query_tokens.or_else(|| dims.get(1).copied().flatten());
                }
                "key_bias0" | "key_bias1" => {
                    has_key_bias = true;
                }
                _ => {}
            }
        }
        Self::new(
            session,
            StaticMatcherShape {
                static_batch,
                static_ref_tokens,
                static_query_tokens,
                has_key_bias,
            },
            runtime_label,
        )
    }
}

impl<S> SparseMatcher for ModelSessionMatcher<S>
where
    S: ModelInference<Tensor = HostTensor> + Send,
    S::Error: std::fmt::Display,
{
    fn runtime_name(&self) -> &str {
        &self.runtime_label
    }

    fn match_batch_indexed_blocking(
        &self,
        references: &[&SparseFeatures],
        query: &SparseFeatures,
        config: &FastMatchConfig,
    ) -> Result<Vec<(Vec<IndexedMatch>, FastMatchStats)>, MatchError> {
        config.validate()?;
        if references.is_empty() {
            return Ok(Vec::new());
        }
        let max_q = self
            .shape
            .static_query_tokens
            .map_or(config.max_keypoints, |q| config.max_keypoints.min(q));
        let max_r = self
            .shape
            .static_ref_tokens
            .map_or(config.max_keypoints, |r| config.max_keypoints.min(r));
        let (kp_q, desc_q, idx_q) = select_input(query, max_q)?;
        let m = idx_q.len();
        let mut out: Vec<Option<(Vec<IndexedMatch>, FastMatchStats)>> =
            vec![None; references.len()];
        let mut active = Vec::with_capacity(references.len());

        for (out_idx, &reference) in references.iter().enumerate() {
            let (kp_r, desc_r, idx_r) = select_input(reference, max_r)?;
            let n = idx_r.len();
            if n == 0 || m == 0 {
                out[out_idx] = Some((
                    Vec::new(),
                    FastMatchStats {
                        initial_ref_tokens: n,
                        initial_query_tokens: m,
                        ..FastMatchStats::default()
                    },
                ));
            } else {
                active.push((out_idx, reference, kp_r, desc_r, idx_r));
            }
        }
        if active.is_empty() {
            return Ok(out.into_iter().flatten().collect());
        }

        let pad_m = self
            .shape
            .static_query_tokens
            .unwrap_or_else(|| quantize_static_token_bucket(m));
        let chunk_cap = self.shape.static_batch.unwrap_or(4).max(1);
        for chunk in active.chunks(chunk_cap) {
            let batch = self
                .shape
                .static_batch
                .unwrap_or(if chunk.len() == 1 { 1 } else { 4 });
            let max_n = chunk.iter().map(|c| c.4.len()).max().unwrap_or(1);
            let pad_n = self
                .shape
                .static_ref_tokens
                .unwrap_or_else(|| quantize_static_token_bucket(max_n));
            let inputs = build_session_inputs(
                chunk,
                (&kp_q, &desc_q, m),
                (batch, pad_n, pad_m),
                self.shape.has_key_bias,
            )?;
            let mut guard = self.session.lock().map_err(|_| MatchError::Device {
                stage: "model session matcher lock",
                message: "session mutex poisoned".into(),
            })?;
            let outputs = guard.forward(&inputs).map_err(|err| MatchError::Device {
                stage: "model session matcher forward",
                message: err.to_string(),
            })?;
            drop(guard);
            let decoded = decode_session_outputs(
                &outputs,
                chunk,
                (&idx_q, &query.pixels, m),
                batch,
                config.score_threshold,
            )?;
            for ((out_idx, _, _, _, idx_r), matches) in chunk.iter().zip(decoded) {
                out[*out_idx] = Some((
                    matches,
                    FastMatchStats {
                        layers_executed: config.max_layers,
                        initial_ref_tokens: idx_r.len(),
                        initial_query_tokens: m,
                        pruned_ref_tokens: 0,
                        pruned_query_tokens: 0,
                        early_exited: false,
                    },
                ));
            }
        }
        Ok(out.into_iter().flatten().collect())
    }
}

type ActiveRefItem<'a> = (usize, &'a SparseFeatures, Vec<f32>, Vec<f32>, Vec<usize>);

fn build_session_inputs(
    chunk: &[ActiveRefItem<'_>],
    (kp_q, desc_q, m): (&[f32], &[f32], usize),
    (batch, pad_n, pad_m): (usize, usize, usize),
    has_key_bias: bool,
) -> Result<Vec<NamedTensor<HostTensor>>, MatchError> {
    let mut kp0 = vec![0.0_f32; batch * pad_n * 2];
    let mut desc0 = vec![0.0_f32; batch * pad_n * INPUT_DIM];
    let mut bias0 = vec![PAD_LOGIT_MASK; batch * pad_n];
    for (b, (_, _, kp_r, desc_r, idx_r)) in chunk.iter().enumerate() {
        let n = idx_r.len();
        kp0[b * pad_n * 2..b * pad_n * 2 + n * 2].copy_from_slice(kp_r);
        desc0[b * pad_n * INPUT_DIM..b * pad_n * INPUT_DIM + n * INPUT_DIM].copy_from_slice(desc_r);
        bias0[b * pad_n..b * pad_n + n].fill(0.0);
    }
    for b in chunk.len()..batch {
        bias0[b * pad_n] = 0.0;
    }

    let mut kp1 = vec![0.0_f32; batch * pad_m * 2];
    let mut desc1 = vec![0.0_f32; batch * pad_m * INPUT_DIM];
    let mut bias1 = vec![PAD_LOGIT_MASK; batch * pad_m];
    for b in 0..batch {
        kp1[b * pad_m * 2..b * pad_m * 2 + m * 2].copy_from_slice(kp_q);
        desc1[b * pad_m * INPUT_DIM..b * pad_m * INPUT_DIM + m * INPUT_DIM].copy_from_slice(desc_q);
        bias1[b * pad_m..b * pad_m + m].fill(0.0);
    }

    let mk = |name: &str, shape: Vec<usize>, values: Vec<f32>| {
        HostTensor::new(shape, values)
            .map(|t| NamedTensor::new(name, t))
            .map_err(|_| MatchError::Output("invalid ModelSessionMatcher input tensor"))
    };
    let mut inputs = vec![
        mk("keypoints0", vec![batch, pad_n, 2], kp0)?,
        mk("keypoints1", vec![batch, pad_m, 2], kp1)?,
        mk("descriptors0", vec![batch, pad_n, INPUT_DIM], desc0)?,
        mk("descriptors1", vec![batch, pad_m, INPUT_DIM], desc1)?,
    ];
    if has_key_bias {
        inputs.push(mk("key_bias0", vec![batch, 1, pad_n], bias0)?);
        inputs.push(mk("key_bias1", vec![batch, 1, pad_m], bias1)?);
    }
    Ok(inputs)
}

fn decode_session_outputs(
    outputs: &[NamedTensor<HostTensor>],
    chunk: &[ActiveRefItem<'_>],
    (idx_q, qry_pixels, m): (&[usize], &[[f64; 2]], usize),
    batch: usize,
    threshold: f32,
) -> Result<Vec<Vec<IndexedMatch>>, MatchError> {
    let out = outputs
        .iter()
        .find(|o| o.name == "assignment" || o.name == "log_assignment")
        .or_else(|| outputs.first())
        .ok_or(MatchError::Output(
            "ModelSessionMatcher produced no outputs",
        ))?;
    let is_prob = out.name == "assignment";
    let [b_dim, rows, cols] = match out.tensor.shape() {
        [b, r, c] => [*b, *r, *c],
        [r, c] if batch == 1 => [1, *r, *c],
        _ => return Err(MatchError::Output("unexpected assignment tensor rank")),
    };
    if b_dim < chunk.len() {
        return Err(MatchError::Output("assignment batch smaller than chunk"));
    }
    let vals = out.tensor.as_slice();
    let stride = rows * cols;
    let mut results = Vec::with_capacity(chunk.len());

    for (item, (_, reference, _, _, idx_r)) in chunk.iter().enumerate() {
        let n = idx_r.len().min(rows);
        let eff_m = m.min(cols);
        let mat = &vals[item * stride..(item + 1) * stride];
        let mut row_scores = vec![f32::NEG_INFINITY; n];
        let mut row_indices = vec![-1_i64; n];
        let mut col_scores = vec![f32::NEG_INFINITY; eff_m];
        let mut col_indices = vec![-1_i64; eff_m];
        for r in 0..n {
            let row = &mat[r * cols..r * cols + cols];
            for (c, &raw) in row.iter().enumerate() {
                let log_s = if is_prob { raw.max(1e-12).ln() } else { raw };
                if log_s > row_scores[r] {
                    row_scores[r] = log_s;
                    row_indices[r] = if c < eff_m { c as i64 } else { -1 };
                }
                if c < eff_m && log_s > col_scores[c] {
                    col_scores[c] = log_s;
                    col_indices[c] = r as i64;
                }
            }
        }
        // Check dustbin row (when `log_assignment` has shape `[B, N + 1, M + 1]`).
        if rows > n && !is_prob {
            let dustbin_row = &mat[(rows - 1) * cols..(rows - 1) * cols + eff_m];
            for (c, &raw) in dustbin_row.iter().enumerate() {
                if raw > col_scores[c] {
                    col_indices[c] = -1;
                }
            }
        }
        results.push(decode_mutual(
            &row_scores,
            &row_indices,
            &col_indices,
            (idx_r, &reference.pixels),
            (idx_q, qry_pixels),
            threshold,
        )?);
    }
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::EngineKind;

    struct MockStaticMatcherSession {
        seen_shapes: Vec<(String, Vec<usize>)>,
    }

    impl ModelInference for MockStaticMatcherSession {
        type Tensor = HostTensor;
        type Error = MatchError;

        fn engine_kind(&self) -> EngineKind {
            EngineKind::Ort
        }

        fn forward(
            &mut self,
            inputs: &[NamedTensor<HostTensor>],
        ) -> Result<Vec<NamedTensor<HostTensor>>, MatchError> {
            self.seen_shapes = inputs
                .iter()
                .map(|i| (i.name.clone(), i.tensor.shape().to_vec()))
                .collect();
            let [b, n, _] = inputs[0].tensor.shape() else {
                return Err(MatchError::Output("bad kp0 shape"));
            };
            let [_, m, _] = inputs[1].tensor.shape() else {
                return Err(MatchError::Output("bad kp1 shape"));
            };
            let mut probs = vec![0.01_f32; b * n * m];
            for item in 0..*b {
                for k in 0..8.min(*n).min(*m) {
                    probs[item * n * m + k * m + k] = 0.85;
                }
            }
            let out = HostTensor::new(vec![*b, *n, *m], probs)
                .map_err(|_| MatchError::Output("bad mock output"))?;
            Ok(vec![NamedTensor::new("assignment", out)])
        }
    }

    fn make_features(count: usize) -> SparseFeatures {
        let mut f = SparseFeatures::empty();
        for i in 0..count {
            f.pixels.push([i as f64 * 12.0, 24.0]);
            f.model_pixels.push([i as f32 * 12.0, 24.0]);
            f.scores.push(0.9);
            let mut d = [0.0_f32; 64];
            d[i % 64] = 1.0;
            f.descriptors.extend_from_slice(&d);
        }
        f
    }

    #[test]
    fn model_session_matcher_pads_static_shapes_and_decodes_probabilities() -> Result<(), MatchError>
    {
        let r0 = make_features(10);
        let r1 = make_features(12);
        let q = make_features(14);
        let matcher = ModelSessionMatcher::new(
            MockStaticMatcherSession {
                seen_shapes: Vec::new(),
            },
            StaticMatcherShape {
                static_batch: Some(4),
                static_ref_tokens: Some(512),
                static_query_tokens: Some(512),
                has_key_bias: true,
            },
            "coreml",
        );
        assert_eq!(matcher.runtime_name(), "coreml");
        let res =
            matcher.match_batch_indexed_blocking(&[&r0, &r1], &q, &FastMatchConfig::default())?;
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].0.len(), 8);
        assert_eq!(res[1].0.len(), 8);
        let guard = matcher
            .session
            .lock()
            .map_err(|_| MatchError::Output("lock"))?;
        assert_eq!(guard.seen_shapes.len(), 6);
        assert_eq!(guard.seen_shapes[0], ("keypoints0".into(), vec![4, 512, 2]));
        assert_eq!(guard.seen_shapes[4], ("key_bias0".into(), vec![4, 1, 512]));
        Ok(())
    }
}
