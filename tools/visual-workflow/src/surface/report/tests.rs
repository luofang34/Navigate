use super::*;
use navigate_visual::{
    CameraModel, CameraPose, FrameStamp, LocalizerConfig, MapRevision, PixelMatch, RasterGeometry,
};
fn scene() -> (Frame, OrthographicReference, PosePrior, Matches) {
    let frame = Frame {
        camera: CameraModel {
            width: 160,
            height: 128,
            fx: 120.0,
            fy: 120.0,
            cx: 80.0,
            cy: 64.0,
        },
        stamp: FrameStamp {
            sequence: 11,
            capture_time_ns: 123,
        },
        image: image::GrayImage::new(160, 128),
    };
    let reference = OrthographicReference {
        image: image::GrayImage::new(160, 128),
        heights_m: vec![0.0; 160 * 128],
        geometry: RasterGeometry {
            origin: [-80.0, 64.0].into(),
            column: [1.0, 0.0].into(),
            row: [0.0, -1.0].into(),
        },
        map: MapRevision {
            release_id: "verified-handoff".into(),
            manifest_sha256: "a".repeat(64),
        },
    };
    let prior = PosePrior {
        pose: CameraPose {
            position: [0.0, 0.0, 120.0].into(),
            orientation: nalgebra::UnitQuaternion::identity(),
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
    let candidate = Matches {
        rotation: 2,
        pairs,
        matching_ms: 0.0,
    };
    (frame, reference, prior, candidate)
}
#[test]
fn report_carries_the_same_verified_estimate_and_rejection() {
    let (frame, reference, prior, mut candidate) = scene();
    let verifier = PoseVerifier::new(LocalizerConfig::default()).unwrap();
    let (row, result) = verify_with_estimate(
        &verifier, &frame, &reference, &prior, &candidate, "test", false,
    );
    let estimate = result.unwrap();
    assert_eq!(row["accepted"], true);
    assert_eq!(row["pose"], json!(Pose::from(estimate.pose)));
    assert_eq!(estimate.map, reference.map);
    assert_eq!(row["quality"]["inliers"], estimate.quality.inliers);
    assert_eq!(row["reference_quarter_turns"], 2);
    candidate.pairs.clear();
    let (row, result) = verify_with_estimate(
        &verifier, &frame, &reference, &prior, &candidate, "test", false,
    );
    assert!(result.is_err());
    assert_eq!(row["accepted"], false);
    assert!(row.get("pose").is_none());
}

#[test]
fn raw_gate_failure_cannot_be_selected_as_an_accepted_map_candidate() {
    let (frame, reference, prior, mut candidate) = scene();
    for (i, pair) in candidate.pairs.iter_mut().enumerate() {
        pair.query.x += if i % 2 == 0 { 1.4 } else { -1.4 };
    }
    let source = CameraModel {
        width: 641,
        height: 513,
        fx: 480.0,
        fy: 480.0,
        cx: 320.0,
        cy: 256.0,
    };
    let map = navigate_visual::RectificationMap::new(source, frame.camera, [0.0; 5], 160 * 128 * 8)
        .unwrap();
    let verifier = PoseVerifier::new(LocalizerConfig::default()).unwrap();
    let (row, result) = Verification {
        verifier: &verifier,
        rectification: Some(&map),
    }
    .run(&frame, &reference, &prior, &candidate, "test", true);
    assert_eq!(row["pinhole_candidate_accepted"], true);
    assert_eq!(row["raw_pixel_gate"]["evaluated"], true);
    assert_eq!(row["raw_pixel_gate"]["accepted"], false);
    assert_eq!(row["accepted"], false);
    assert!(result.is_err());
    assert!(row.get("pose").is_none());
    assert!(row.get("quality").is_none());
    assert_eq!(
        row["correspondences"].as_array().unwrap().len(),
        candidate.pairs.len()
    );
}
