//! Adaptive confidence early-exit and mid-network token pruning for [`super::LighterGlue`].
use super::{INPUT_DIM, LAYER_COUNT, LighterGlue};
use crate::{MatchError, SparseFeatures, loftr::layers::upload};
use burn::tensor::{Device, Int, Tensor, TensorData, Transaction};
use navigate_visual::PixelMatch;

const MIN_EARLY_EXIT_MATCHES: usize = 16;
const MIN_RETAINED_TOKENS: usize = 16;

/// Mutual correspondence with source keypoint indices and log-assignment score.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IndexedMatch {
    /// Index of the matched keypoint in the input `reference` [`SparseFeatures`].
    pub ref_index: usize,
    /// Index of the matched keypoint in the input `query` [`SparseFeatures`].
    pub query_index: usize,
    /// Matched pixel coordinates in the reference and query images.
    pub pixel_match: PixelMatch,
    /// Mutual log-assignment score from [`LighterGlue`].
    pub log_score: f32,
}

/// Configuration for adaptive [`LighterGlue::match_features_fast`] execution.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FastMatchConfig {
    /// Maximum input keypoints kept per image before layer 0.
    pub max_keypoints: usize,
    /// Minimum transformer layers executed before early-exit checks (`1..=6`).
    pub min_layers: usize,
    /// Maximum transformer layers executed (`min_layers..=6`).
    pub max_layers: usize,
    /// Bitmask of 1-based layers (`bit (l - 1)`) eligible for early-exit head checks.
    pub exit_check_layers: u8,
    /// Minimum mutual-match fraction of `min(active_n, active_m)` required for early exit.
    pub early_exit_mutual_fraction: f32,
    /// Optional 1-based layer count after which the matcher prunes low-matchability tokens.
    pub prune_after_layer: Option<usize>,
    /// Minimum matchability logit retained during token pruning.
    pub min_matchability_logit: f32,
    /// Minimum log-assignment score required to accept a mutual match.
    pub score_threshold: f32,
}

impl Default for FastMatchConfig {
    fn default() -> Self {
        Self {
            max_keypoints: 512,
            min_layers: 2,
            max_layers: 6,
            exit_check_layers: 0b0001_1110,
            early_exit_mutual_fraction: 0.30,
            prune_after_layer: Some(2),
            min_matchability_logit: -1.5,
            score_threshold: 0.1_f32.ln(),
        }
    }
}

impl FastMatchConfig {
    /// Return the preset configuration for high-rate frame tracking.
    pub fn realtime_tracking() -> Self {
        Self {
            max_keypoints: 384,
            min_layers: 2,
            max_layers: 4,
            exit_check_layers: 1 << 1,
            early_exit_mutual_fraction: 0.25,
            prune_after_layer: Some(2),
            ..Self::default()
        }
    }

    /// Return the preset configuration for balanced map search.
    pub fn balanced_search() -> Self {
        Self {
            max_keypoints: 1024,
            min_layers: 4,
            max_layers: 6,
            exit_check_layers: 1 << 3,
            early_exit_mutual_fraction: 0.35,
            prune_after_layer: None,
            ..Self::default()
        }
    }

    pub(crate) fn validate(&self) -> Result<(), MatchError> {
        if self.max_keypoints == 0 {
            return Err(MatchError::Output(
                "FastMatchConfig max_keypoints must be at least 1",
            ));
        }
        if self.min_layers == 0
            || self.min_layers > LAYER_COUNT
            || self.max_layers < self.min_layers
            || self.max_layers > LAYER_COUNT
        {
            return Err(MatchError::Output(
                "FastMatchConfig layer bounds must satisfy 1 <= min_layers <= max_layers <= 6",
            ));
        }
        if !self.early_exit_mutual_fraction.is_finite() || self.early_exit_mutual_fraction < 0.0 {
            return Err(MatchError::Output(
                "FastMatchConfig early_exit_mutual_fraction must be finite and non-negative",
            ));
        }
        if let Some(layer) = self.prune_after_layer
            && (layer == 0 || layer > LAYER_COUNT)
        {
            return Err(MatchError::Output(
                "FastMatchConfig prune_after_layer must be 1 through 6",
            ));
        }
        if !self.min_matchability_logit.is_finite() || !self.score_threshold.is_finite() {
            return Err(MatchError::Output(
                "FastMatchConfig thresholds must be finite",
            ));
        }
        Ok(())
    }
}

/// Execution telemetry from [`LighterGlue::match_features_fast`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FastMatchStats {
    /// Number of transformer layers executed.
    pub layers_executed: usize,
    /// Number of reference tokens entering layer 0.
    pub initial_ref_tokens: usize,
    /// Number of query tokens entering layer 0.
    pub initial_query_tokens: usize,
    /// Number of reference tokens removed by mid-network token pruning.
    pub pruned_ref_tokens: usize,
    /// Number of query tokens removed by mid-network token pruning.
    pub pruned_query_tokens: usize,
    /// `true` when matching stopped before `max_layers` through confidence early exit.
    pub early_exited: bool,
}

struct TokenSide {
    tokens: Tensor<3>,
    cos: Tensor<3>,
    sin: Tensor<3>,
    indices: Vec<usize>,
    pruned: usize,
}

impl TokenSide {
    fn len(&self) -> usize {
        self.indices.len()
    }

    fn prune(&mut self, logits: &[f32], min_logit: f32, device: &Device) -> Result<(), MatchError> {
        let kept = retain_indices(logits, min_logit)?;
        let prev = self.indices.len();
        if kept.len() == prev {
            return Ok(());
        }
        let raw: Vec<i64> = kept.iter().map(|&i| i as i64).collect();
        let select_idx: Tensor<1, Int> =
            Tensor::from_data(TensorData::new(raw, [kept.len()]), device);
        self.tokens = self.tokens.clone().select(1, select_idx.clone());
        self.cos = self.cos.clone().select(1, select_idx.clone());
        self.sin = self.sin.clone().select(1, select_idx);
        self.indices = kept
            .iter()
            .filter_map(|&i| self.indices.get(i).copied())
            .collect();
        self.pruned += prev.saturating_sub(self.indices.len());
        Ok(())
    }
}

impl LighterGlue {
    /// Match two [`SparseFeatures`] sets with confidence early exit and token pruning.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when `config` or feature shapes are invalid or device read-back fails.
    pub async fn match_features_fast(
        &self,
        reference: &SparseFeatures,
        query: &SparseFeatures,
        config: &FastMatchConfig,
    ) -> Result<(Vec<PixelMatch>, FastMatchStats), MatchError> {
        let (indexed, stats) = self
            .match_features_indexed_fast(reference, query, config)
            .await?;
        Ok((indexed.into_iter().map(|m| m.pixel_match).collect(), stats))
    }

    /// Match two [`SparseFeatures`] sets and return [`IndexedMatch`] records with source indices.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when `config` or feature shapes are invalid or device read-back fails.
    pub async fn match_features_indexed_fast(
        &self,
        reference: &SparseFeatures,
        query: &SparseFeatures,
        config: &FastMatchConfig,
    ) -> Result<(Vec<IndexedMatch>, FastMatchStats), MatchError> {
        config.validate()?;
        let (kp0, desc0, idx_a) = select_input(reference, config.max_keypoints)?;
        let (kp1, desc1, idx_b) = select_input(query, config.max_keypoints)?;
        let (n0, m0) = (idx_a.len(), idx_b.len());
        if n0 == 0 || m0 == 0 {
            return Ok((
                Vec::new(),
                FastMatchStats {
                    initial_ref_tokens: n0,
                    initial_query_tokens: m0,
                    ..FastMatchStats::default()
                },
            ));
        }
        let mut ref_side = self.init_side(kp0, desc0, idx_a);
        let mut qry_side = self.init_side(kp1, desc1, idx_b);

        for (idx, layer) in self.layers.iter().take(config.max_layers).enumerate() {
            ref_side.tokens =
                layer
                    .self_attn
                    .forward(ref_side.tokens, &ref_side.cos, &ref_side.sin);
            qry_side.tokens =
                layer
                    .self_attn
                    .forward(qry_side.tokens, &qry_side.cos, &qry_side.sin);
            (ref_side.tokens, qry_side.tokens) =
                layer.cross_attn.forward(ref_side.tokens, qry_side.tokens);
            let done = idx + 1;
            let check_exit = done >= config.min_layers
                && done < config.max_layers
                && (config.exit_check_layers & (1u8 << (done - 1))) != 0;
            let do_prune = config.prune_after_layer == Some(done) && done < config.max_layers;
            if check_exit {
                let (matches, logits_a, logits_b) = self
                    .eval_head(
                        &ref_side,
                        &qry_side,
                        (&reference.pixels, &query.pixels),
                        config.score_threshold,
                        do_prune,
                    )
                    .await?;
                let min_active = ref_side.len().min(qry_side.len()) as f32;
                if matches.len() >= MIN_EARLY_EXIT_MATCHES
                    && (matches.len() as f32) >= config.early_exit_mutual_fraction * min_active
                {
                    return Ok((
                        matches,
                        build_stats(done, (n0, m0), &ref_side, &qry_side, true),
                    ));
                }
                if do_prune {
                    ref_side.prune(&logits_a, config.min_matchability_logit, &self.device)?;
                    qry_side.prune(&logits_b, config.min_matchability_logit, &self.device)?;
                }
            } else if do_prune {
                let (logits_a, logits_b) = self.read_logits(&ref_side, &qry_side).await?;
                ref_side.prune(&logits_a, config.min_matchability_logit, &self.device)?;
                qry_side.prune(&logits_b, config.min_matchability_logit, &self.device)?;
            }
        }

        let (matches, _, _) = self
            .eval_head(
                &ref_side,
                &qry_side,
                (&reference.pixels, &query.pixels),
                config.score_threshold,
                false,
            )
            .await?;
        Ok((
            matches,
            build_stats(config.max_layers, (n0, m0), &ref_side, &qry_side, false),
        ))
    }

    fn init_side(&self, kp: Vec<f32>, desc: Vec<f32>, indices: Vec<usize>) -> TokenSide {
        let len = indices.len();
        let tokens = self
            .input_proj
            .forward(upload(desc, [1, len, INPUT_DIM], &self.device));
        let (cos, sin) = self.encoding(upload(kp, [1, len, 2], &self.device), len);
        TokenSide {
            tokens,
            cos,
            sin,
            indices,
            pruned: 0,
        }
    }

    async fn eval_head(
        &self,
        ref_side: &TokenSide,
        qry_side: &TokenSide,
        (ref_pixels, qry_pixels): (&[[f64; 2]], &[[f64; 2]]),
        threshold: f32,
        with_logits: bool,
    ) -> Result<(Vec<IndexedMatch>, Vec<f32>, Vec<f32>), MatchError> {
        let (n, m) = (ref_side.len(), qry_side.len());
        let (scores, ma, mb) =
            self.assignment_with_logits(ref_side.tokens.clone(), qry_side.tokens.clone());
        let (row_max, row_argmax) = scores.clone().max_dim_with_indices(2);
        let (_, col_argmax) = scores.max_dim_with_indices(1);
        let mut tx = Transaction::default()
            .register(row_max.reshape([n]))
            .register(row_argmax.reshape([n]))
            .register(col_argmax.reshape([m]));
        if with_logits {
            tx = tx.register(ma.reshape([n])).register(mb.reshape([m]));
        }
        let mut data = tx
            .execute_async()
            .await
            .map_err(|error| MatchError::Device {
                stage: "lighterglue fast assignment read-back",
                message: format!("{error:?}"),
            })?;
        let (logits_a, logits_b) = if with_logits {
            let mb_data = data
                .pop()
                .ok_or(MatchError::Output("lighterglue read-back count"))?;
            let ma_data = data
                .pop()
                .ok_or(MatchError::Output("lighterglue read-back count"))?;
            (
                ma_data.iter::<f32>().collect(),
                mb_data.iter::<f32>().collect(),
            )
        } else {
            (Vec::new(), Vec::new())
        };
        let [rmax, ridx, cidx] = <[_; 3]>::try_from(data)
            .map_err(|_| MatchError::Output("lighterglue read-back count"))?;
        let matches = decode_mutual(
            &rmax.iter::<f32>().collect::<Vec<_>>(),
            &ridx.iter::<i64>().collect::<Vec<_>>(),
            &cidx.iter::<i64>().collect::<Vec<_>>(),
            (&ref_side.indices, ref_pixels),
            (&qry_side.indices, qry_pixels),
            threshold,
        )?;
        Ok((matches, logits_a, logits_b))
    }

    async fn read_logits(
        &self,
        ref_side: &TokenSide,
        qry_side: &TokenSide,
    ) -> Result<(Vec<f32>, Vec<f32>), MatchError> {
        let ma = self
            .matchability
            .forward(ref_side.tokens.clone())
            .reshape([ref_side.len()]);
        let mb = self
            .matchability
            .forward(qry_side.tokens.clone())
            .reshape([qry_side.len()]);
        let data = Transaction::default()
            .register(ma)
            .register(mb)
            .execute_async()
            .await
            .map_err(|error| MatchError::Device {
                stage: "lighterglue matchability read-back",
                message: format!("{error:?}"),
            })?;
        let [ma_data, mb_data] = <[_; 2]>::try_from(data)
            .map_err(|_| MatchError::Output("lighterglue matchability read-back count"))?;
        Ok((
            ma_data.iter::<f32>().collect(),
            mb_data.iter::<f32>().collect(),
        ))
    }
}

fn build_stats(
    layers_executed: usize,
    (initial_ref_tokens, initial_query_tokens): (usize, usize),
    ref_side: &TokenSide,
    qry_side: &TokenSide,
    early_exited: bool,
) -> FastMatchStats {
    FastMatchStats {
        layers_executed,
        initial_ref_tokens,
        initial_query_tokens,
        pruned_ref_tokens: ref_side.pruned,
        pruned_query_tokens: qry_side.pruned,
        early_exited,
    }
}

type SelectedInput = (Vec<f32>, Vec<f32>, Vec<usize>);

fn select_input(
    features: &SparseFeatures,
    max_keypoints: usize,
) -> Result<SelectedInput, MatchError> {
    let len = features.len();
    if features.model_pixels.len() != len
        || features.scores.len() != len
        || features.descriptors.len() != INPUT_DIM * len
    {
        return Err(MatchError::Output("LighterGlue input shape mismatch"));
    }
    if len == 0 {
        return Ok((Vec::new(), Vec::new(), Vec::new()));
    }
    let keep = max_keypoints.min(len);
    let indices: Vec<usize> =
        if len <= max_keypoints || features.scores.windows(2).all(|w| w[0] >= w[1]) {
            (0..keep).collect()
        } else {
            let mut order: Vec<usize> = (0..len).collect();
            order.sort_by(|&a, &b| {
                features.scores[b]
                    .total_cmp(&features.scores[a])
                    .then_with(|| a.cmp(&b))
            });
            order.truncate(keep);
            order
        };
    let mut keypoints = Vec::with_capacity(2 * keep);
    let mut descriptors = Vec::with_capacity(INPUT_DIM * keep);
    for &orig in &indices {
        let p = features.model_pixels[orig];
        keypoints.extend_from_slice(&[(p[0] - 400.0) / 400.0, (p[1] - 300.0) / 400.0]);
        descriptors
            .extend_from_slice(&features.descriptors[orig * INPUT_DIM..(orig + 1) * INPUT_DIM]);
    }
    Ok((keypoints, descriptors, indices))
}

fn decode_mutual(
    row_scores: &[f32],
    row_indices: &[i64],
    col_indices: &[i64],
    (idx_a, ref_pixels): (&[usize], &[[f64; 2]]),
    (idx_b, qry_pixels): (&[usize], &[[f64; 2]]),
    threshold: f32,
) -> Result<Vec<IndexedMatch>, MatchError> {
    if row_scores.iter().any(|v| !v.is_finite()) {
        return Err(MatchError::Output("non-finite LighterGlue row score"));
    }
    Ok(row_scores
        .iter()
        .copied()
        .zip(row_indices.iter().copied())
        .enumerate()
        .filter_map(|(i, (score, j))| {
            let j = usize::try_from(j).ok()?;
            let mutual = col_indices.get(j).copied() == Some(i as i64);
            if !(score > threshold && mutual) {
                return None;
            }
            let ref_index = *idx_a.get(i)?;
            let query_index = *idx_b.get(j)?;
            Some(IndexedMatch {
                ref_index,
                query_index,
                pixel_match: PixelMatch {
                    reference: (*ref_pixels.get(ref_index)?).into(),
                    query: (*qry_pixels.get(query_index)?).into(),
                },
                log_score: score,
            })
        })
        .collect())
}

fn retain_indices(logits: &[f32], min_logit: f32) -> Result<Vec<usize>, MatchError> {
    if logits.iter().any(|v| !v.is_finite()) {
        return Err(MatchError::Output(
            "non-finite LighterGlue matchability logit",
        ));
    }
    let len = logits.len();
    let floor = MIN_RETAINED_TOKENS.min(len);
    let kept: Vec<usize> = (0..len).filter(|&i| logits[i] >= min_logit).collect();
    if kept.len() >= floor {
        return Ok(kept);
    }
    let mut ranked: Vec<usize> = (0..len).collect();
    ranked.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]).then_with(|| a.cmp(&b)));
    ranked.truncate(floor);
    ranked.sort_unstable();
    Ok(ranked)
}
