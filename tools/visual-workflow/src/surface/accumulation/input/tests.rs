use super::*;
fn frozen() -> Frozen {
    let asset = serde_json::json!({"path":"fixture.png","sha256":"a".repeat(64)});
    let job = serde_json::json!({"query":asset,"camera":{"size":[128,128],"focal":[100.0,100.0],"principal":[64.0,64.0]},"prior":{"position":[0.0,0.0,100.0],"rotation":[0.0,0.0,0.0,1.0]},"position_radius_m":1000.0,"attitude_radius_rad":3.0,"reference":{"image":asset,"heights":asset,"origin":[0.0,0.0],"column":[1.0,0.0],"row":[0.0,1.0],"map_release":"fixture","map_sha256":"b".repeat(64)},"reference_quarter_turns":[0,1,2,3],"record_matches":true});
    Frozen {
        backend: "fixture".into(),
        job: serde_json::from_value(job).unwrap(),
        frames: vec![Sample {
            image: serde_json::from_value(asset).unwrap(),
            frame_id: 1,
            capture_time_ns: 33_333_333,
            observation_sha256: "c".repeat(64),
            map_candidates: (0..4)
                .map(|rotation| Candidate {
                    reference_quarter_turns: rotation,
                    matches: 1,
                    correspondences: vec![[16.0, 16.0, 32.0, 32.0]],
                })
                .collect(),
        }],
    }
}
#[test]
fn frozen_input_rejects_missing_rotations_counts_and_nonfinite_pairs() {
    let mut input = frozen();
    assert!(input.validate().is_ok());
    input.frames[0].map_candidates[0].matches = 2;
    assert!(input.validate().is_err());
    input.frames[0].map_candidates[0].matches = 1;
    input.frames[0].map_candidates[0].correspondences[0][0] = f64::NAN;
    assert!(input.validate().is_err());
    input.frames[0].map_candidates.remove(0);
    assert!(input.validate().is_err());
}
#[test]
fn image_bounds_and_frame_order_are_required() {
    let mut input = frozen();
    assert!(input.frames[0].evidence((128, 128), (128, 128)).is_ok());
    input.frames[0].map_candidates[0].correspondences[0][2] = 128.0;
    assert!(input.frames[0].evidence((128, 128), (128, 128)).is_err());
    input.frames.extend(frozen().frames);
    assert!(input.validate().is_err());
}
