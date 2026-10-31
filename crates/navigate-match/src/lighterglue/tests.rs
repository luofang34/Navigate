use super::*;
use crate::{EngineKind, WeightTensor, XFeat, XFeatLighterGlueMatcher};
use image::{GrayImage, Luma};

fn insert_linear(weights: &mut Weights, name: &str, out: usize, input: usize, bias: bool) {
    let values = (0..out * input)
        .map(|i| (((i * 11 + 5) % 23) as f32 - 11.0) * 0.01)
        .collect();
    weights.insert(
        format!("{name}.weight"),
        WeightTensor {
            shape: vec![out, input],
            values,
        },
    );
    if bias {
        weights.insert(
            format!("{name}.bias"),
            WeightTensor {
                shape: vec![out],
                values: vec![0.0; out],
            },
        );
    }
}

fn insert_small_linear(
    weights: &mut Weights,
    name: &str,
    out: usize,
    input: usize,
    bias: bool,
    scale: f32,
) {
    let values = (0..out * input)
        .map(|i| (((i * 13 + 7) % 19) as f32 - 9.0) * scale)
        .collect();
    weights.insert(
        format!("{name}.weight"),
        WeightTensor {
            shape: vec![out, input],
            values,
        },
    );
    if bias {
        weights.insert(
            format!("{name}.bias"),
            WeightTensor {
                shape: vec![out],
                values: vec![0.0; out],
            },
        );
    }
}

fn insert_norm(weights: &mut Weights, name: &str, width: usize) {
    weights.insert(
        format!("{name}.weight"),
        WeightTensor {
            shape: vec![width],
            values: vec![1.0; width],
        },
    );
    weights.insert(
        format!("{name}.bias"),
        WeightTensor {
            shape: vec![width],
            values: vec![0.0; width],
        },
    );
}

fn synthetic_lighterglue_weights() -> Weights {
    let mut w = Weights::default();
    insert_linear(&mut w, "net.input_proj", 96, 64, true);
    insert_linear(&mut w, "net.posenc.Wr", 48, 2, false);
    for i in 0..6 {
        let s = format!("net.transformers.{i}.self_attn");
        insert_linear(&mut w, &format!("{s}.Wqkv"), 288, 96, true);
        insert_linear(&mut w, &format!("{s}.out_proj"), 96, 96, true);
        insert_linear(&mut w, &format!("{s}.ffn.0"), 192, 192, true);
        insert_norm(&mut w, &format!("{s}.ffn.1"), 192);
        insert_linear(&mut w, &format!("{s}.ffn.3"), 96, 192, true);
        let c = format!("net.transformers.{i}.cross_attn");
        insert_linear(&mut w, &format!("{c}.to_qk"), 96, 96, true);
        insert_linear(&mut w, &format!("{c}.to_v"), 96, 96, true);
        insert_linear(&mut w, &format!("{c}.to_out"), 96, 96, true);
        insert_linear(&mut w, &format!("{c}.ffn.0"), 192, 192, true);
        insert_norm(&mut w, &format!("{c}.ffn.1"), 192);
        insert_linear(&mut w, &format!("{c}.ffn.3"), 96, 192, true);
    }
    insert_linear(&mut w, "net.log_assignment.5.final_proj", 96, 96, true);
    insert_linear(&mut w, "net.log_assignment.5.matchability", 1, 96, true);
    w
}

/// Build LighterGlue weights where channel 0 controls matchability and channels 1..64 drive assignment.
fn high_agreement_lighterglue_weights() -> Weights {
    let mut w = Weights::default();
    let mut in_proj = vec![0.0_f32; 96 * 64];
    for c in 0..64 {
        in_proj[c * 64 + c] = 1.0;
    }
    w.insert(
        "net.input_proj.weight",
        WeightTensor {
            shape: vec![96, 64],
            values: in_proj,
        },
    );
    w.insert(
        "net.input_proj.bias",
        WeightTensor {
            shape: vec![96],
            values: vec![0.0; 96],
        },
    );
    insert_small_linear(&mut w, "net.posenc.Wr", 48, 2, false, 0.01);
    for i in 0..6 {
        let s = format!("net.transformers.{i}.self_attn");
        insert_small_linear(&mut w, &format!("{s}.Wqkv"), 288, 96, true, 0.002);
        insert_small_linear(&mut w, &format!("{s}.out_proj"), 96, 96, true, 0.002);
        insert_small_linear(&mut w, &format!("{s}.ffn.0"), 192, 192, true, 0.002);
        insert_norm(&mut w, &format!("{s}.ffn.1"), 192);
        insert_small_linear(&mut w, &format!("{s}.ffn.3"), 96, 192, true, 0.0001);
        let c = format!("net.transformers.{i}.cross_attn");
        insert_small_linear(&mut w, &format!("{c}.to_qk"), 96, 96, true, 0.002);
        insert_small_linear(&mut w, &format!("{c}.to_v"), 96, 96, true, 0.002);
        insert_small_linear(&mut w, &format!("{c}.to_out"), 96, 96, true, 0.002);
        insert_small_linear(&mut w, &format!("{c}.ffn.0"), 192, 192, true, 0.002);
        insert_norm(&mut w, &format!("{c}.ffn.1"), 192);
        insert_small_linear(&mut w, &format!("{c}.ffn.3"), 96, 192, true, 0.0001);
    }
    let mut final_proj = vec![0.0_f32; 96 * 96];
    for c in 1..64 {
        final_proj[c * 96 + c] = 12.0;
    }
    w.insert(
        "net.log_assignment.5.final_proj.weight",
        WeightTensor {
            shape: vec![96, 96],
            values: final_proj,
        },
    );
    w.insert(
        "net.log_assignment.5.final_proj.bias",
        WeightTensor {
            shape: vec![96],
            values: vec![0.0; 96],
        },
    );
    let mut match_w = vec![0.0_f32; 96];
    match_w[0] = 6.0;
    w.insert(
        "net.log_assignment.5.matchability.weight",
        WeightTensor {
            shape: vec![1, 96],
            values: match_w,
        },
    );
    w.insert(
        "net.log_assignment.5.matchability.bias",
        WeightTensor {
            shape: vec![1],
            values: vec![0.0],
        },
    );
    w
}

/// Build one L2-normalized 64-D descriptor with explicit channel-0 matchability sign and one-hot identity slot.
fn token_descriptor(matchable: bool, slot: usize) -> [f32; 64] {
    let mut d = [0.0_f32; 64];
    d[0] = if matchable { 0.5 } else { -0.5 };
    d[1 + (slot % 63)] = (1.0_f32 - 0.25).sqrt();
    d
}

fn synthetic_pair(count: usize) -> (SparseFeatures, SparseFeatures) {
    let mut ref_feat = SparseFeatures::empty();
    let mut qry_feat = SparseFeatures::empty();
    for i in 0..count {
        let rx = 20.0 + i as f64 * 7.5;
        let ry = 30.0 + (i % 9) as f64 * 11.0;
        let qx = rx + 3.0;
        let qy = ry - 2.0;
        let desc = token_descriptor(true, i);
        ref_feat.pixels.push([rx, ry]);
        ref_feat.model_pixels.push([rx as f32, ry as f32]);
        ref_feat.scores.push(0.95 - i as f32 * 0.01);
        ref_feat.descriptors.extend_from_slice(&desc);

        qry_feat.pixels.push([qx, qy]);
        qry_feat.model_pixels.push([qx as f32, qy as f32]);
        qry_feat.scores.push(0.94 - i as f32 * 0.01);
        qry_feat.descriptors.extend_from_slice(&desc);
    }
    (ref_feat, qry_feat)
}

fn pair_coords(pairs: &[PixelMatch]) -> Vec<[f64; 4]> {
    pairs
        .iter()
        .map(|p| [p.reference.x, p.reference.y, p.query.x, p.query.y])
        .collect()
}

#[test]
fn lighterglue_produces_finite_log_assignments_and_matches() {
    let device = Device::flex();
    let model = LighterGlue::load(synthetic_lighterglue_weights(), &device).expect("load model");
    let kp0 = vec![-0.5, -0.2, 0.1, 0.3, 0.4, -0.1];
    let kp1 = vec![-0.4, -0.2, 0.2, 0.3, 0.5, 0.0, -0.1, 0.4];
    let d0: Vec<f32> = (0..3 * 64).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();
    let d1: Vec<f32> = (0..4 * 64).map(|i| ((i % 5) as f32 - 2.0) * 0.1).collect();
    let scores = pollster::block_on(model.log_assignment(&kp0, &kp1, &d0, &d1)).expect("scores");
    assert_eq!(scores.len(), 3 * 4);
    assert!(scores.iter().all(|v| v.is_finite() && *v <= 0.0));
}

#[test]
fn match_features_fast_early_exits_at_min_layers_on_high_agreement() {
    let device = Device::flex();
    let model =
        LighterGlue::load(high_agreement_lighterglue_weights(), &device).expect("load model");
    let (reference, query) = synthetic_pair(24);

    let zero_frac = FastMatchConfig {
        min_layers: 2,
        max_layers: 6,
        early_exit_mutual_fraction: 0.0,
        ..FastMatchConfig::default()
    };
    let (matches, stats) =
        pollster::block_on(model.match_features_fast(&reference, &query, &zero_frac))
            .expect("fast match");
    assert!(stats.early_exited);
    assert_eq!(stats.layers_executed, 2);
    assert_eq!(stats.initial_ref_tokens, 24);
    assert_eq!(stats.initial_query_tokens, 24);
    assert_eq!(stats.pruned_ref_tokens, 0);
    assert_eq!(stats.pruned_query_tokens, 0);
    assert_eq!(matches.len(), 24);
    for (i, m) in matches.iter().enumerate() {
        assert_eq!([m.reference.x, m.reference.y], reference.pixels[i]);
        assert_eq!([m.query.x, m.query.y], query.pixels[i]);
    }

    let tracking = FastMatchConfig::realtime_tracking();
    let (rt_matches, rt_stats) =
        pollster::block_on(model.match_features_fast(&reference, &query, &tracking))
            .expect("realtime tracking match");
    assert!(rt_stats.early_exited);
    assert_eq!(rt_stats.layers_executed, tracking.min_layers);
    assert_eq!(pair_coords(&rt_matches), pair_coords(&matches));

    let masked_exit = FastMatchConfig {
        min_layers: 2,
        max_layers: 6,
        exit_check_layers: 1 << 3,
        early_exit_mutual_fraction: 0.0,
        prune_after_layer: None,
        ..FastMatchConfig::default()
    };
    let (_, masked_stats) =
        pollster::block_on(model.match_features_fast(&reference, &query, &masked_exit))
            .expect("masked exit check");
    assert!(masked_stats.early_exited);
    assert_eq!(masked_stats.layers_executed, 4);
}

#[test]
fn match_features_fast_prunes_tokens_and_preserves_original_pixel_mapping() {
    let device = Device::flex();
    let model =
        LighterGlue::load(high_agreement_lighterglue_weights(), &device).expect("load model");

    let mut reference = SparseFeatures::empty();
    for d in 0..10 {
        reference.pixels.push([500.0 + d as f64, 510.0 + d as f64]);
        reference
            .model_pixels
            .push([500.0 + d as f32, 510.0 + d as f32]);
        reference.scores.push(0.99);
        reference
            .descriptors
            .extend_from_slice(&token_descriptor(false, 40 + d));
    }
    for k in 0..20 {
        let px = 40.0 + k as f64 * 9.0;
        let py = 60.0 + k as f64 * 5.0;
        reference.pixels.push([px, py]);
        reference.model_pixels.push([px as f32, py as f32]);
        reference.scores.push(0.80 - k as f32 * 0.01);
        reference
            .descriptors
            .extend_from_slice(&token_descriptor(true, k));
    }

    let mut query = SparseFeatures::empty();
    for d in 0..8 {
        query.pixels.push([600.0 + d as f64, 610.0 + d as f64]);
        query
            .model_pixels
            .push([600.0 + d as f32, 610.0 + d as f32]);
        query.scores.push(0.99);
        query
            .descriptors
            .extend_from_slice(&token_descriptor(false, 52 + d));
    }
    for rev in 0..20 {
        let slot = 19 - rev;
        let qx = 120.0 + slot as f64 * 6.0;
        let qy = 90.0 + slot as f64 * 4.0;
        query.pixels.push([qx, qy]);
        query.model_pixels.push([qx as f32, qy as f32]);
        query.scores.push(0.80 - rev as f32 * 0.01);
        query
            .descriptors
            .extend_from_slice(&token_descriptor(true, slot));
    }

    let config = FastMatchConfig {
        max_keypoints: 512,
        min_layers: 4,
        max_layers: 4,
        exit_check_layers: 0,
        early_exit_mutual_fraction: 1.0,
        prune_after_layer: Some(2),
        min_matchability_logit: -1.5,
        score_threshold: 0.1_f32.ln(),
    };
    let (indexed, stats) =
        pollster::block_on(model.match_features_indexed_fast(&reference, &query, &config))
            .expect("pruned indexed match");
    let (matches, fast_stats) =
        pollster::block_on(model.match_features_fast(&reference, &query, &config))
            .expect("pruned match");
    assert_eq!(stats, fast_stats);
    assert_eq!(stats.layers_executed, 4);
    assert!(!stats.early_exited);
    assert_eq!(stats.initial_ref_tokens, 30);
    assert_eq!(stats.initial_query_tokens, 28);
    assert_eq!(stats.pruned_ref_tokens, 10);
    assert_eq!(stats.pruned_query_tokens, 8);
    assert_eq!(matches.len(), 20);
    assert_eq!(indexed.len(), 20);
    for (slot, (m, idx_m)) in matches.iter().zip(&indexed).enumerate() {
        let expected_ref_idx = 10 + slot;
        let expected_qry_idx = 8 + (19 - slot);
        assert_eq!(idx_m.ref_index, expected_ref_idx);
        assert_eq!(idx_m.query_index, expected_qry_idx);
        assert!(idx_m.log_score.is_finite());
        assert_eq!(idx_m.pixel_match, *m);
        assert_eq!(
            [m.reference.x, m.reference.y],
            reference.pixels[expected_ref_idx]
        );
        assert_eq!([m.query.x, m.query.y], query.pixels[expected_qry_idx]);
    }

    // Verify the min(16, len) floor when every token falls below min_matchability_logit.
    let mut low_only = SparseFeatures::empty();
    for i in 0..20 {
        low_only.pixels.push([10.0 + i as f64, 20.0 + i as f64]);
        low_only
            .model_pixels
            .push([10.0 + i as f32, 20.0 + i as f32]);
        low_only.scores.push(0.5);
        low_only
            .descriptors
            .extend_from_slice(&token_descriptor(false, i));
    }
    let (_, floor_stats) =
        pollster::block_on(model.match_features_fast(&low_only, &low_only, &config))
            .expect("floor match");
    assert_eq!(floor_stats.pruned_ref_tokens, 4);
    assert_eq!(floor_stats.pruned_query_tokens, 4);
}

#[test]
fn match_features_fast_matches_full_six_layer_output_without_pruning() {
    let device = Device::flex();
    let full_cfg = FastMatchConfig {
        max_keypoints: 512,
        min_layers: 6,
        max_layers: 6,
        prune_after_layer: None,
        ..FastMatchConfig::default()
    };
    let (reference, query) = synthetic_pair(22);

    for weights in [
        synthetic_lighterglue_weights(),
        high_agreement_lighterglue_weights(),
    ] {
        let model = LighterGlue::load(weights, &device).expect("load model");
        let expected =
            pollster::block_on(model.match_features(&reference, &query)).expect("full 6-layer");
        let (actual, stats) =
            pollster::block_on(model.match_features_fast(&reference, &query, &full_cfg))
                .expect("fast 6-layer");
        assert_eq!(pair_coords(&actual), pair_coords(&expected));
        assert_eq!(stats.layers_executed, 6);
        assert!(!stats.early_exited);
        assert_eq!(stats.pruned_ref_tokens, 0);
        assert_eq!(stats.pruned_query_tokens, 0);
    }
}

#[test]
fn staged_matcher_runs_masked_extraction_and_fast_matching() {
    let device = Device::flex();
    let detector =
        XFeat::load(crate::xfeat::tests::synthetic_xfeat_weights(), &device).expect("xfeat");
    let lg = LighterGlue::load(high_agreement_lighterglue_weights(), &device).expect("lg");
    let staged = XFeatLighterGlueMatcher::new(
        detector,
        "detsha",
        Some((lg, "lgsha")),
        128,
        EngineKind::BurnCpu,
    )
    .expect("staged");

    let mut image = GrayImage::new(160, 120);
    for (x, y, p) in image.enumerate_pixels_mut() {
        p.0[0] = (((x * 7 + y * 13) ^ (x * y)) & 0xff) as u8;
    }
    let mask = GrayImage::from_fn(
        160,
        120,
        |_, y| if y < 60 { Luma([0]) } else { Luma([255]) },
    );

    let unmasked = pollster::block_on(staged.extract_features(&image)).expect("unmasked");
    let masked =
        pollster::block_on(staged.extract_features_masked(&image, Some(&mask))).expect("masked");
    assert_ne!(unmasked, masked);
    assert!(masked.sparse().pixels.iter().all(|p| p[1] >= 60.0));

    let search_cfg = FastMatchConfig::balanced_search();
    let (pairs, stats) =
        pollster::block_on(staged.match_extracted_fast(&unmasked, &masked, &search_cfg))
            .expect("match_extracted_fast");
    assert!(stats.layers_executed >= search_cfg.min_layers);
    assert!(pairs.iter().all(|m| m.query.y >= 60.0));
}

#[test]
fn batched_match_features_indexed_fast_matches_single_pair_across_padded_lengths() {
    let device = Device::flex();
    let model = LighterGlue::load(high_agreement_lighterglue_weights(), &device).expect("lg");
    let cfg = FastMatchConfig::balanced_search();
    let (ref0, query) = synthetic_pair(28);
    let ref1 = ref0.truncate_top(19);
    let ref2 = SparseFeatures::empty();
    let (ref3, _) = synthetic_pair(34);
    let refs = [&ref0, &ref1, &ref2, &ref3];

    let batched = pollster::block_on(model.match_features_batch_indexed_fast(&refs, &query, &cfg))
        .expect("batched");
    assert_eq!(batched.len(), refs.len());
    for (reference, (batch_matches, batch_stats)) in refs.iter().zip(&batched) {
        let (single_matches, single_stats) =
            pollster::block_on(model.match_features_indexed_fast(reference, &query, &cfg))
                .expect("single");
        assert_eq!(batch_stats.layers_executed, single_stats.layers_executed);
        assert_eq!(batch_stats.early_exited, single_stats.early_exited);
        let b_idx: Vec<(usize, usize)> = batch_matches
            .iter()
            .map(|m| (m.ref_index, m.query_index))
            .collect();
        let s_idx: Vec<(usize, usize)> = single_matches
            .iter()
            .map(|m| (m.ref_index, m.query_index))
            .collect();
        assert_eq!(b_idx, s_idx);
    }
}
