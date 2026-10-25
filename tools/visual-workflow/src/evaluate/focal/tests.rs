use super::*;
use nalgebra::Vector2;
use navigate_visual::{FrameStamp, LocalizerConfig};

#[test]
fn flat_ground_can_hide_focal_error_as_height_with_perfect_heldout_pixels() {
    let (camera, reference) = super::super::tests::fixture();
    let verifier = PoseVerifier::new(LocalizerConfig::default()).unwrap();
    let frame = Frame {
        camera,
        image: reference.image.clone(),
        stamp: FrameStamp {
            sequence: 0,
            capture_time_ns: 0,
        },
    };
    let (training, controls) = pairs();
    let probe = Probe {
        verifier: &verifier,
        reference: (&reference, &camera),
        prior: PosePrior {
            pose: reference.pose,
            position_radius_m: 100.0,
            attitude_radius_rad: 0.5,
        },
        identity: "focal-test",
    };
    let result = probe.run(&frame, &training, &controls, &[0.9, 1.0, 1.1], None);
    let hypotheses = result["hypotheses"].as_array().unwrap();
    assert_eq!(hypotheses.len(), 3);
    for (h, expected_z) in hypotheses.iter().zip([-10.0, 0.0, 10.0]) {
        assert_eq!(h["accepted"], true);
        assert!(h["heldout"]["median_px"].as_f64().unwrap() < 0.00001);
        assert!((h["pose"]["position"][2].as_f64().unwrap() - expected_z).abs() < 0.00001);
    }
    let baseline = verifier
        .verify(&frame, &reference, &probe.prior, &training, "focal-test")
        .unwrap();
    assert_eq!(hypotheses[1]["pose"], json!(Pose::from(baseline.pose)));
    let check = PositionCheck {
        source: "unrelated external check".into(),
        latitude_longitude: [0.0, 0.0],
        reported_horizontal_error_m: None,
        associated_capture_time_ns: 0,
    };
    let checked = probe.run(&frame, &training, &controls, &[0.9, 1.0, 1.1], Some(&check));
    for (a, b) in hypotheses
        .iter()
        .zip(checked["hypotheses"].as_array().unwrap())
    {
        assert_eq!(a["pose"], b["pose"]);
        assert_eq!(a["heldout"], b["heldout"]);
    }
    assert_eq!(result["training_sha256"], checked["training_sha256"]);
    assert_eq!(result["heldout_sha256"], checked["heldout_sha256"]);
    assert_ne!(result["training_sha256"], result["heldout_sha256"]);
}

#[test]
fn probe_work_is_bounded_and_non_finite_scales_are_refused() {
    assert!(valid_scales(&[0.8, 1.0, 1.2]));
    for scales in [
        vec![],
        vec![1.0; 22],
        vec![f64::NAN],
        vec![f64::INFINITY],
        vec![0.79],
        vec![1.21],
    ] {
        assert!(!valid_scales(&scales));
    }
}

#[cfg(test)]
fn pairs() -> (Vec<PixelMatch>, Vec<PixelMatch>) {
    let pairs: Vec<_> = (10..230)
        .step_by(10)
        .flat_map(|y| {
            (10..310).step_by(10).map(move |x| {
                let p = Vector2::new(f64::from(x), f64::from(y));
                PixelMatch {
                    reference: p,
                    query: p,
                }
            })
        })
        .collect();
    let training = pairs
        .iter()
        .filter(|p| metrics::partition(p, 0) == Some(false))
        .copied()
        .collect();
    let controls = pairs
        .iter()
        .filter(|p| metrics::partition(p, 0) == Some(true))
        .copied()
        .collect();
    (training, controls)
}
