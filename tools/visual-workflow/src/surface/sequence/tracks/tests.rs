use super::*;
use nalgebra::{UnitQuaternion, Vector2};
use navigate_visual::{
    CameraModel, CameraPose, FrameStamp, MapRevision, PosePrior, PoseVerifier, RasterGeometry,
    VisualError,
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
struct Shift;
impl navigate_visual::ImageMatcher for Shift {
    fn identity(&self) -> &str {
        "fixed-shift-test"
    }
    fn match_images_blocking(
        &mut self,
        _: &image::GrayImage,
        _: &image::GrayImage,
    ) -> Result<Vec<PixelMatch>, VisualError> {
        Ok(Vec::new())
    }
}
impl PointTracker for Shift {
    fn features_blocking(
        &mut self,
        _: &image::GrayImage,
    ) -> Result<Vec<Vector2<f64>>, VisualError> {
        Ok(Vec::new())
    }

    fn track_points_blocking(
        &mut self,
        _: &image::GrayImage,
        _: &image::GrayImage,
        points: &[Vector2<f64>],
    ) -> Result<Vec<Option<Vector2<f64>>>, VisualError> {
        Ok(points
            .iter()
            .map(|p| Some(*p - Vector2::new(1.0, 0.0)))
            .collect())
    }
}
#[test]
fn propagation_preserves_map_coordinates_and_anchor_evidence() {
    let (frame, reference, prior, pairs) = fixture();
    let verifier = PoseVerifier::new(LocalizerConfig::default()).unwrap();
    let estimate = verifier
        .verify_orthographic(&frame, &reference, &prior, &pairs, "test")
        .unwrap();
    let tracks = Tracks::new(frame, &reference, &estimate, pairs, None, true);
    let mut next = fixture().0;
    next.stamp = FrameStamp {
        sequence: 1,
        capture_time_ns: 33333333,
    };
    assert!(tracks.eligible(&next, 30));
    assert!(!tracks.eligible(&next, 1));
    let found = tracks.propagate_blocking(&next, &mut Shift).unwrap();
    assert!(
        tracks
            .pairs
            .iter()
            .zip(&found)
            .all(|(a, b)| a.reference == b.reference)
    );
    let estimate = verifier
        .verify_orthographic(&next, &reference, &prior, &found, "test-tracker")
        .unwrap();
    assert!((estimate.pose.position.x - 1.0).abs() < 1e-6);
    let update = Tracks::new(next, &reference, &estimate, found, Some(&tracks), false);
    assert_eq!(update.origin, tracks.origin);
    assert_eq!(update.age, 2);
    let mut later = fixture().0;
    later.stamp = FrameStamp {
        sequence: 9,
        capture_time_ns: 300000000,
    };
    assert!(!update.eligible(&later, 30));
}
