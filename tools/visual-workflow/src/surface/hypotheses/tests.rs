use super::*;
use crate::surface::sequence::tracks::Tracks;
use nalgebra::{UnitQuaternion, Vector2};
use navigate_visual::{
    CameraModel, CameraPose, FrameStamp, MapRevision, PosePrior, PoseVerifier, RasterGeometry,
    VisualError,
};
use navigate_visual::{PixelMatch, PointTracker};
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
struct Shift {
    drop_after: usize,
}
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
            .enumerate()
            .map(|(i, p)| (i < self.drop_after).then_some(*p - Vector2::new(1.0, 0.0)))
            .collect())
    }
}

fn runner() -> Runner {
    let (frame, reference, prior, _) = fixture();
    Runner {
        reference,
        prior,
        previous: frame,
        verifier: PoseVerifier::new(LocalizerConfig::default()).unwrap(),
        tracker: Tracker {
            identity: "test-tracker".into(),
            backend: Box::new(Shift {
                drop_after: usize::MAX,
            }),
        },
        hypotheses: Vec::new(),
        reference_limit: 1024 * 1024,
    }
}
fn candidates(runner: &Runner, n: usize) -> Vec<acquire::Acquired> {
    (0..n)
        .map(|_| {
            let pairs = fixture().3;
            let estimate = runner
                .verifier
                .verify_orthographic(
                    &runner.previous,
                    &runner.reference,
                    &runner.prior,
                    &pairs,
                    "test",
                )
                .unwrap();
            acquire::Acquired {
                estimate,
                pairs,
                window: None,
                rotation: 0,
            }
        })
        .collect()
}
#[test]
fn shared_tracking_matches_single_path_and_does_not_revive_rejected_hypothesis() {
    let mut r = runner();
    r.initialize(candidates(&r, 2)).unwrap();
    let (frame, reference, prior, pairs) = fixture();
    let estimate = r
        .verifier
        .verify_orthographic(&frame, &reference, &prior, &pairs, "test")
        .unwrap();
    let baseline = Tracks::new(frame, &reference, &estimate, pairs, None, true);
    r.tracker.backend = Box::new(Shift {
        drop_after: baseline.pairs.len(),
    });
    let mut next = fixture().0;
    next.stamp = FrameStamp {
        sequence: 1,
        capture_time_ns: 33_333_333,
    };
    let baseline_pairs = baseline
        .propagate_blocking(
            &next,
            &mut Shift {
                drop_after: usize::MAX,
            },
        )
        .unwrap();
    let baseline_estimate = r
        .verifier
        .verify_orthographic(&next, &reference, &prior, &baseline_pairs, "test-tracker")
        .unwrap();
    let rows = r.advance_blocking(&next).unwrap();
    assert_eq!(rows["tracker_calls"], 1);
    assert_eq!(
        rows["hypotheses"][0]["pose"],
        json!(navigate_recording::Pose::from(baseline_estimate.pose))
    );
    assert_eq!(
        rows["hypotheses"][0]["quality"]["rms_px"],
        json!(baseline_estimate.quality.reprojection_rms_px)
    );
    assert_eq!(rows["hypotheses"][1]["accepted"], false);
    assert!(r.hypotheses[1].pairs.is_empty());
    assert!(
        r.hypotheses[0]
            .pairs
            .iter()
            .zip(&baseline_pairs)
            .all(|(a, b)| a.reference == b.reference && a.query == b.query)
    );
    r.previous = next;
    r.tracker.backend = Box::new(Shift {
        drop_after: usize::MAX,
    });
    let mut next = fixture().0;
    next.stamp = FrameStamp {
        sequence: 2,
        capture_time_ns: 66_666_667,
    };
    let rows = r.advance_blocking(&next).unwrap();
    assert_eq!(rows["hypotheses"][0]["accepted"], true);
    assert_eq!(
        rows["hypotheses"][1]["rejection"],
        "hypothesis_already_lost"
    );
}
#[test]
fn a_time_gap_ends_every_hypothesis_without_tracking_or_recovery() {
    let mut r = runner();
    r.initialize(candidates(&r, 2)).unwrap();
    let mut frame = fixture().0;
    frame.stamp = FrameStamp {
        sequence: 8,
        capture_time_ns: 266_666_667,
    };
    let rows = r.advance_blocking(&frame).unwrap();
    assert_eq!(rows["tracker_calls"], 0);
    for h in rows["hypotheses"].as_array().unwrap() {
        assert_eq!(h["accepted"], false);
        assert_eq!(h["rejection"], "ineligible_frame_gap");
    }
    r.previous = frame;
    let mut next = fixture().0;
    next.stamp = FrameStamp {
        sequence: 9,
        capture_time_ns: 300_000_000,
    };
    let rows = r.advance_blocking(&next).unwrap();
    assert_eq!(rows["tracker_calls"], 0);
    assert!(r.hypotheses.iter().all(|h| h.estimate.is_none()));
}
