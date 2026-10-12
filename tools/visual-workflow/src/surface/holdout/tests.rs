use super::*;
use nalgebra::{UnitQuaternion, Vector2};
use navigate_visual::{CameraModel, CameraPose, MapRevision, PixelMatch, RasterGeometry};
fn fixture() -> (Frame, OrthographicReference, PosePrior, Vec<PixelMatch>) {
    let camera = CameraModel {
        width: 160,
        height: 128,
        fx: 120.0,
        fy: 120.0,
        cx: 80.0,
        cy: 64.0,
    };
    let frame = Frame {
        camera,
        stamp: FrameStamp {
            sequence: 0,
            capture_time_ns: 0,
        },
        image: image::GrayImage::new(160, 128),
    };
    let reference = OrthographicReference {
        image: image::GrayImage::new(160, 128),
        heights_m: vec![0.0; 160 * 128],
        map: MapRevision {
            release_id: "sequence-test".into(),
            manifest_sha256: "a".repeat(64),
        },
        geometry: RasterGeometry {
            origin: [-80.0, 64.0].into(),
            column: [1.0, 0.0].into(),
            row: [0.0, -1.0].into(),
        },
    };
    let prior = PosePrior {
        pose: CameraPose {
            position: [0.0, 0.0, 120.0].into(),
            orientation: UnitQuaternion::identity(),
        },
        position_radius_m: 1000.0,
        attitude_radius_rad: 3.0,
    };
    let pairs = (16..145)
        .step_by(16)
        .flat_map(|x| {
            (16..113).step_by(16).map(move |y| PixelMatch {
                reference: [f64::from(x), f64::from(y)].into(),
                query: [f64::from(x), f64::from(y)].into(),
            })
        })
        .collect();
    (frame, reference, prior, pairs)
}

#[test]
fn full_replay_is_exact_and_folds_fit_through_the_same_geometry_path() {
    let (frame, reference, prior, pairs) = fixture();
    let runner = Replay {
        frame,
        reference,
        prior,
        verifier: PoseVerifier::new(LocalizerConfig::default()).unwrap(),
        limit: 1024 * 1024,
    };
    let matches = Matches {
        rotation: 0,
        pairs,
        matching_ms: 0.0,
    };
    let (original, _) = report::verify_with_estimate(
        &runner.verifier,
        &runner.frame,
        &runner.reference,
        &runner.prior,
        &matches,
        "test",
        true,
    );
    assert_eq!(original["accepted"], true);
    let mut candidate = Candidate {
        stage: "coarse",
        window: None,
        original,
        matches,
    };
    let control = runner.control(&candidate).unwrap();
    assert_eq!(control["exact_original_decision"], true);
    let rows = runner.folds(&candidate).unwrap();
    let folds = rows["folds"].as_array().unwrap();
    assert_eq!(folds.len(), 5);
    assert!(
        folds
            .iter()
            .any(|f| f["decision"]["accepted"] == true && f["heldout_count"].as_u64().unwrap() > 0)
    );
    for fold in folds {
        assert_eq!(
            fold["training_count"].as_u64().unwrap()
                + fold["heldout_count"].as_u64().unwrap()
                + fold["boundary_count"].as_u64().unwrap()
                + fold["invalid_reference_count"].as_u64().unwrap(),
            candidate.matches.pairs.len() as u64
        );
        if fold["decision"]["accepted"] == true {
            let held = &fold["decision"]["heldout"];
            assert_eq!(held["within_2_5_px"], held["valid_3d"]);
        }
    }
    candidate.matches.pairs.clear();
    let changed = runner.control(&candidate).unwrap();
    assert_eq!(changed["exact_original_decision"], false);
    assert_eq!(changed["replayed"]["accepted"], false);
}
#[test]
fn shared_global_patch_membership_keeps_original_crop_coordinates() {
    let a = PixelMatch {
        reference: [76.0, 176.0].into(),
        query: [20.0, 20.0].into(),
    };
    let b = PixelMatch {
        reference: [1100.0, 1200.0].into(),
        query: [80.0, 80.0].into(),
    };
    for fold in 0..5 {
        let ca = scoring::split(&[a], Some([1024, 1024, 2048, 2048]), fold);
        let cb = scoring::split(&[b], None, fold);
        assert_eq!(ca.heldout.len(), cb.heldout.len());
        assert_eq!(ca.training.len(), cb.training.len());
        let kept = ca.heldout.iter().chain(&ca.training).next().unwrap();
        assert_eq!(kept.reference, a.reference);
    }
    let edge = PixelMatch {
        reference: [64.0, 100.0].into(),
        ..a
    };
    assert_eq!(scoring::split(&[edge], None, 0).boundary, 1);
    let invalid = PixelMatch {
        reference: [f64::NAN, 100.0].into(),
        ..a
    };
    assert_eq!(scoring::split(&[invalid], None, 0).invalid, 1);
    assert_ne!(scoring::digest(&[a, b]), scoring::digest(&[b, a]));
}
#[test]
fn heldout_scores_keep_outliers_missing_heights_and_behind_camera_failures() {
    let (frame, reference, prior, source) = fixture();
    let base = source[0];
    let pairs = [
        base,
        PixelMatch {
            query: base.query + Vector2::new(3.0, 0.0),
            ..base
        },
        PixelMatch {
            query: base.query + Vector2::new(60.0, 0.0),
            ..base
        },
        PixelMatch {
            reference: [-1.0, 10.0].into(),
            ..base
        },
        PixelMatch {
            query: [-1.0, 10.0].into(),
            ..base
        },
    ];
    let score = scoring::score(&pairs, &frame, &reference, &prior.pose);
    assert_eq!(score["raw_matches"], 5);
    assert_eq!(score["valid_3d"], 3);
    assert_eq!(score["missing_surface"], 1);
    assert_eq!(score["invalid_query"], 1);
    assert_eq!(score["within_2_5_px"], 1);
    assert_eq!(score["median_px"], 3.0);
    assert_eq!(score["p90_px"], 60.0);
    let behind = CameraPose {
        position: [0.0, 0.0, -120.0].into(),
        ..prior.pose
    };
    let score = scoring::score(&pairs, &frame, &reference, &behind);
    assert_eq!(score["behind_camera"], 3);
    assert_eq!(score["within_2_5_px"], 0);
    assert_eq!(score["median_is_infinite"], true);
    assert_eq!(score["median_px"], Value::Null);
}
