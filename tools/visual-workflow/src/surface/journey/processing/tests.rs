use super::*;
use nalgebra::UnitQuaternion;
use navigate_visual::{
    CameraModel, CameraPose, FrameStamp, LocalizerConfig, MapRevision, PosePrior, PoseVerifier,
    RasterGeometry,
};
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
fn retained_recovery_rejects_discontinuous_tracks() {
    let (previous, reference, prior, pairs) = fixture();
    let estimate = PoseVerifier::new(LocalizerConfig::default())
        .unwrap()
        .verify_orthographic(&previous, &reference, &prior, &pairs, "test")
        .unwrap();
    let tracks = Tracks::new(previous, &reference, &estimate, pairs, None, true);
    let mut current = fixture().0;
    current.stamp = FrameStamp {
        sequence: 1,
        capture_time_ns: 33_333_333,
    };
    let area = Some([0, 0, 160, 128]);
    assert_eq!(retained_window(&tracks, &current, area), area);
    assert_eq!(retained_window(&tracks, &current, None), None);
    assert!(!tracks.eligible(&current, 1));
    assert_eq!(retained_window(&tracks, &current, area), area);
    current.stamp.capture_time_ns = MAX_TRACK_GAP_NS + 1;
    assert_eq!(retained_window(&tracks, &current, area), None);
    current.stamp.capture_time_ns = 33_333_333;
    current.stamp.sequence = 0;
    assert_eq!(retained_window(&tracks, &current, area), None);
    current.stamp.sequence = 1;
    current.stamp.capture_time_ns = 0;
    assert_eq!(retained_window(&tracks, &current, area), None);
    current.stamp.capture_time_ns = 33_333_333;
    current.camera.fx += 1.0;
    assert_eq!(retained_window(&tracks, &current, area), None);
}
