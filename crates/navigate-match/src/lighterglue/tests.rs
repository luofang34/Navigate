use super::*;
use crate::WeightTensor;

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
