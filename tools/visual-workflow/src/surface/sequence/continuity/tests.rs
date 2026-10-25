use super::*;
use nalgebra::UnitQuaternion;
use navigate_visual::{
    FrameStamp, LocalizerConfig, MapRevision, OrthographicReference, PixelMatch, PosePrior,
    PoseVerifier, RasterGeometry,
};
fn observation(index: u64, position_x: f64) -> (Frame, OrthographicEstimate) {
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
            sequence: index,
            capture_time_ns: index * 33333333,
        },
        image: image::GrayImage::new(160, 128),
    };
    let reference = OrthographicReference {
        map: MapRevision {
            release_id: "continuity-test".into(),
            manifest_sha256: "a".repeat(64),
        },
        image: image::GrayImage::new(160, 128),
        heights_m: vec![0.0; 160 * 128],
        geometry: RasterGeometry {
            origin: [-80.0, 64.0].into(),
            column: [1.0, 0.0].into(),
            row: [0.0, -1.0].into(),
        },
    };
    let pose = CameraPose {
        position: [position_x, 0.0, 120.0].into(),
        orientation: UnitQuaternion::identity(),
    };
    let mut pairs = Vec::new();
    for y in (16..113).step_by(16) {
        for x in (32..145).step_by(16) {
            let p = nalgebra::Vector2::new(f64::from(x), f64::from(y));
            let query = camera
                .project(&pose, reference.world_at(p).unwrap())
                .unwrap();
            pairs.push(PixelMatch {
                reference: p,
                query,
            });
        }
    }
    let prior = PosePrior {
        pose,
        position_radius_m: 1000.0,
        attitude_radius_rad: 3.0,
    };
    let estimate = PoseVerifier::new(LocalizerConfig::default())
        .unwrap()
        .verify_orthographic(&frame, &reference, &prior, &pairs, "synthetic-control")
        .unwrap();
    (frame, estimate)
}
#[test]
fn map_reacquisition_never_becomes_a_motion_step() {
    let (a, first) = observation(0, 0.0);
    let mut continuity = Continuity::new(a.camera).unwrap();
    let start = continuity
        .step(&a, &a.evidence_sha256(), Some(&first), true)
        .unwrap();
    let (b, tracked) = observation(1, 1.0);
    let step = continuity
        .step(&b, &b.evidence_sha256(), Some(&tracked), false)
        .unwrap();
    assert!(step["motion_applied"].as_bool().unwrap());
    assert_eq!(start["odometry"]["segment"], step["odometry"]["segment"]);
    assert!((step["odometry"]["path_m"].as_f64().unwrap() - 1.0).abs() < 1e-8);
    let (c, relocated) = observation(2, 9.0);
    let reset = continuity
        .step(&c, &c.evidence_sha256(), Some(&relocated), true)
        .unwrap();
    assert_eq!(reset["motion_applied"], false);
    assert_eq!(reset["odometry"]["path_m"], 0.0);
    assert_ne!(step["odometry"]["segment"], reset["odometry"]["segment"]);
    assert_eq!(reset["map_anchors_submitted"], 0);
    let (d, moved) = observation(3, 10.0);
    let resumed = continuity
        .step(&d, &d.evidence_sha256(), Some(&moved), false)
        .unwrap();
    assert!((resumed["odometry"]["path_m"].as_f64().unwrap() - 1.0).abs() < 1e-8);
    let (e, _) = observation(4, 11.0);
    let missing = continuity
        .step(&e, &e.evidence_sha256(), None, true)
        .unwrap();
    assert!(missing["odometry"].is_null());
    assert_eq!(missing["motion_applied"], false);
}
