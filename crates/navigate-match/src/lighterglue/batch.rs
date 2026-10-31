//! Batched multi-reference `[B, N, 96]` matching for [`super::LighterGlue`].
//!
//! [`LighterGlue::match_features_batch_indexed_fast`] evaluates multiple candidate
//! reference views against one query feature set in a single batched transformer pass.
//! Padded reference positions carry an additive `-1e9` key mask so attention weights
//! and mutual log-assignment scores match single-pair execution.

use super::{
    FastMatchConfig, FastMatchStats, INPUT_DIM, IndexedMatch, LighterGlue,
    fast::{decode_mutual, select_input},
};
use crate::{MatchError, SparseFeatures, loftr::layers::upload};
use burn::tensor::{Tensor, Transaction};
use navigate_visual::PixelMatch;

const MIN_EARLY_EXIT_MATCHES: usize = 16;
const PAD_LOGIT_MASK: f32 = -1.0e9;

struct BatchRefSlot<'a> {
    out_idx: usize,
    pixels: &'a [[f64; 2]],
    indices: Vec<usize>,
}

impl LighterGlue {
    /// Match `references` against `query` in one batched `[B, N, 96]` GPU pass.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when `config` or feature shapes are invalid or read-back fails.
    pub async fn match_features_batch_fast(
        &self,
        references: &[&SparseFeatures],
        query: &SparseFeatures,
        config: &FastMatchConfig,
    ) -> Result<Vec<(Vec<PixelMatch>, FastMatchStats)>, MatchError> {
        let batch = self
            .match_features_batch_indexed_fast(references, query, config)
            .await?;
        Ok(batch
            .into_iter()
            .map(|(m, s)| (m.into_iter().map(|x| x.pixel_match).collect(), s))
            .collect())
    }

    /// Match `references` against `query` in one batched `[B, N, 96]` GPU pass and return [`IndexedMatch`]es.
    ///
    /// # Errors
    ///
    /// Returns [`MatchError`] when `config` or feature shapes are invalid or read-back fails.
    pub async fn match_features_batch_indexed_fast(
        &self,
        references: &[&SparseFeatures],
        query: &SparseFeatures,
        config: &FastMatchConfig,
    ) -> Result<Vec<(Vec<IndexedMatch>, FastMatchStats)>, MatchError> {
        config.validate()?;
        if references.is_empty() {
            return Ok(Vec::new());
        }
        if let [single] = references {
            return Ok(vec![
                self.match_features_indexed_fast(single, query, config)
                    .await?,
            ]);
        }

        let (kp_q, desc_q, idx_q) = select_input(query, config.max_keypoints)?;
        let m = idx_q.len();
        let mut out: Vec<Option<(Vec<IndexedMatch>, FastMatchStats)>> =
            vec![None; references.len()];

        let mut slots = Vec::with_capacity(references.len());
        let mut selected_refs = Vec::with_capacity(references.len());
        let mut max_n = 0_usize;

        for (out_idx, &reference) in references.iter().enumerate() {
            let (kp_r, desc_r, idx_r) = select_input(reference, config.max_keypoints)?;
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
                max_n = max_n.max(n);
                slots.push(BatchRefSlot {
                    out_idx,
                    pixels: &reference.pixels,
                    indices: idx_r,
                });
                selected_refs.push((kp_r, desc_r));
            }
        }

        let batch = slots.len();
        if batch == 0 {
            return Ok(out.into_iter().flatten().collect());
        }

        let mut kp_batch = vec![0.0_f32; batch * max_n * 2];
        let mut desc_batch = vec![0.0_f32; batch * max_n * INPUT_DIM];
        let mut mask_vals = vec![0.0_f32; batch * max_n];
        let mut has_padding = false;

        for (b, ((kp_r, desc_r), slot)) in selected_refs.into_iter().zip(&slots).enumerate() {
            let n = slot.indices.len();
            kp_batch[b * max_n * 2..b * max_n * 2 + n * 2].copy_from_slice(&kp_r);
            desc_batch[b * max_n * INPUT_DIM..b * max_n * INPUT_DIM + n * INPUT_DIM]
                .copy_from_slice(&desc_r);
            if n < max_n {
                has_padding = true;
                mask_vals[b * max_n + n..(b + 1) * max_n].fill(PAD_LOGIT_MASK);
            }
        }

        let key_mask = has_padding.then(|| upload(mask_vals, [batch, 1, max_n], &self.device));
        let mut a =
            self.input_proj
                .forward(upload(desc_batch, [batch, max_n, INPUT_DIM], &self.device));
        let (cos_a, sin_a) =
            self.encoding(upload(kp_batch, [batch, max_n, 2], &self.device), max_n);

        let b0 = self
            .input_proj
            .forward(upload(desc_q, [1, m, INPUT_DIM], &self.device));
        let (cos_b0, sin_b0) = self.encoding(upload(kp_q, [1, m, 2], &self.device), m);
        let mut b_tok = b0.repeat_dim(0, batch);
        let cos_b = cos_b0.repeat_dim(0, batch);
        let sin_b = sin_b0.repeat_dim(0, batch);

        for (idx, layer) in self.layers.iter().take(config.max_layers).enumerate() {
            a = layer
                .self_attn
                .forward(a, &cos_a, &sin_a, key_mask.as_ref());
            b_tok = layer.self_attn.forward(b_tok, &cos_b, &sin_b, None);
            (a, b_tok) = layer.cross_attn.forward(a, b_tok, key_mask.as_ref());

            let done = idx + 1;
            let check_exit = done >= config.min_layers
                && done < config.max_layers
                && (config.exit_check_layers & (1u8 << (done - 1))) != 0;
            if check_exit {
                let decoded = self
                    .eval_batch_head(
                        &a,
                        &b_tok,
                        key_mask.as_ref(),
                        &slots,
                        (&idx_q, &query.pixels),
                        (max_n, m),
                        config.score_threshold,
                    )
                    .await?;
                for (slot, matches) in slots.iter().zip(decoded) {
                    if out[slot.out_idx].is_some() {
                        continue;
                    }
                    let n = slot.indices.len();
                    let min_active = n.min(m) as f32;
                    if matches.len() >= MIN_EARLY_EXIT_MATCHES
                        && (matches.len() as f32) >= config.early_exit_mutual_fraction * min_active
                    {
                        out[slot.out_idx] = Some((
                            matches,
                            FastMatchStats {
                                layers_executed: done,
                                initial_ref_tokens: n,
                                initial_query_tokens: m,
                                pruned_ref_tokens: 0,
                                pruned_query_tokens: 0,
                                early_exited: true,
                            },
                        ));
                    }
                }
                if slots.iter().all(|s| out[s.out_idx].is_some()) {
                    return Ok(out.into_iter().flatten().collect());
                }
            }
        }

        let decoded = self
            .eval_batch_head(
                &a,
                &b_tok,
                key_mask.as_ref(),
                &slots,
                (&idx_q, &query.pixels),
                (max_n, m),
                config.score_threshold,
            )
            .await?;
        for (slot, matches) in slots.iter().zip(decoded) {
            if out[slot.out_idx].is_none() {
                out[slot.out_idx] = Some((
                    matches,
                    FastMatchStats {
                        layers_executed: config.max_layers,
                        initial_ref_tokens: slot.indices.len(),
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

    #[allow(clippy::too_many_arguments)]
    async fn eval_batch_head(
        &self,
        a: &Tensor<3>,
        b: &Tensor<3>,
        key_mask: Option<&Tensor<3>>,
        slots: &[BatchRefSlot<'_>],
        (idx_q, qry_pixels): (&[usize], &[[f64; 2]]),
        (max_n, m): (usize, usize),
        threshold: f32,
    ) -> Result<Vec<Vec<IndexedMatch>>, MatchError> {
        let batch = slots.len();
        let (scores, _, _) = self.assignment_with_logits(a.clone(), b.clone(), key_mask);
        let (row_max, row_argmax) = scores.clone().max_dim_with_indices(2);
        let (_, col_argmax) = scores.max_dim_with_indices(1);
        let data = Transaction::default()
            .register(row_max.reshape([batch * max_n]))
            .register(row_argmax.reshape([batch * max_n]))
            .register(col_argmax.reshape([batch * m]))
            .execute_async()
            .await
            .map_err(|error| MatchError::Device {
                stage: "lighterglue batch assignment read-back",
                message: format!("{error:?}"),
            })?;
        let [rmax, ridx, cidx] = <[_; 3]>::try_from(data)
            .map_err(|_| MatchError::Output("lighterglue batch read-back count"))?;
        let all_row_scores: Vec<f32> = rmax.iter::<f32>().collect();
        let all_row_indices: Vec<i64> = ridx.iter::<i64>().collect();
        let all_col_indices: Vec<i64> = cidx.iter::<i64>().collect();

        let mut results = Vec::with_capacity(batch);
        for (item, slot) in slots.iter().enumerate() {
            let n = slot.indices.len();
            let r_off = item * max_n;
            let c_off = item * m;
            results.push(decode_mutual(
                &all_row_scores[r_off..r_off + n],
                &all_row_indices[r_off..r_off + n],
                &all_col_indices[c_off..c_off + m],
                (&slot.indices, slot.pixels),
                (idx_q, qry_pixels),
                threshold,
            )?);
        }
        Ok(results)
    }
}
