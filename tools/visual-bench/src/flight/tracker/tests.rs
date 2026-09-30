use super::*;
use image::GrayImage;
use nalgebra::{UnitQuaternion, Vector2, Vector3};
use navigate_visual::{
    CameraModel, FrameStamp, LocalFrame, MapRevision, PixelMatch, RendererIdentity, VisualError,
};
struct Surface {
    camera: CameraModel,
    valid: bool,
}
impl ReferenceRenderer for Surface {
    type Error = BenchError;
    fn identity(&self) -> RendererIdentity {
        RendererIdentity {
            revision: "test-surface".into(),
            style_sha256: "a".repeat(64),
        }
    }
    fn render_blocking(&mut self, pose: CameraPose) -> Result<ReferenceView, BenchError> {
        Ok(ReferenceView {
            pose,
            map: MapRevision {
                release_id: "test".into(),
                manifest_sha256: "b".repeat(64),
            },
            frame: LocalFrame::anchor_mercator(40.0, -74.0)?,
            image: GrayImage::new(self.camera.width, self.camera.height),
            depth_m: vec![
                if self.valid { 100.0 } else { 0.0 };
                (self.camera.width * self.camera.height) as usize
            ],
        })
    }
}
struct Correspondences {
    calls: usize,
    empty: bool,
    shift_map_checks: bool,
}
impl ImageMatcher for Correspondences {
    fn identity(&self) -> &str {
        "test-correspondences"
    }
    fn match_images_blocking(
        &mut self,
        _: &GrayImage,
        _: &GrayImage,
    ) -> Result<Vec<PixelMatch>, VisualError> {
        let shift = if self.shift_map_checks && self.calls > 0 {
            2.0
        } else {
            0.0
        };
        self.calls = self.calls.wrapping_add(1);
        if self.empty {
            return Ok(Vec::new());
        }
        Ok((12..85)
            .step_by(12)
            .flat_map(|y| {
                (12..117).step_by(13).map(move |x| PixelMatch {
                    reference: Vector2::new(f64::from(x), f64::from(y)),
                    query: Vector2::new(f64::from(x) + shift, f64::from(y)),
                })
            })
            .collect())
    }
}
fn camera() -> CameraModel {
    CameraModel {
        width: 128,
        height: 96,
        fx: 100.0,
        fy: 100.0,
        cx: 63.5,
        cy: 47.5,
    }
}
fn pose() -> CameraPose {
    CameraPose {
        position: Vector3::new(0.0, 0.0, 100.0),
        orientation: UnitQuaternion::identity(),
    }
}
fn frame(sequence: u64) -> Frame {
    Frame {
        camera: camera(),
        stamp: FrameStamp {
            sequence,
            capture_time_ns: sequence * 1_000_000_000,
        },
        image: GrayImage::from_pixel(128, 96, image::Luma([sequence as u8 + 1])),
    }
}
fn tracker(valid: bool, empty_fast: bool, poses: Vec<CameraPose>) -> Result<Tracker, BenchError> {
    Tracker::new(
        Box::new(Surface {
            camera: camera(),
            valid,
        }),
        Box::new(Correspondences {
            calls: 0,
            empty: false,
            shift_map_checks: true,
        }),
        Box::new(Correspondences {
            calls: 0,
            empty: empty_fast,
            shift_map_checks: false,
        }),
        PosePrior {
            pose: pose(),
            position_radius_m: 500.0,
            attitude_radius_rad: 3.0,
        },
        poses,
        Policy {
            dense_only: false,
            map_interval_ns: 1_000_000_000,
            reanchor: false,
            fixed_tilt: false,
            surface_tracks: false,
            keyframe_interval_ns: 2_000_000_000,
        },
    )
}
#[test]
fn diagnostic_map_check_does_not_move_relative_track() -> Result<(), BenchError> {
    let mut t = tracker(true, false, vec![pose()])?;
    let (first, _) = t.observe_blocking(&frame(0))?;
    assert_eq!(first["candidate_hypotheses"][0]["accepted"], true);
    assert!(
        first["candidate_hypotheses"][0]["spatial_support"]
            .as_u64()
            .is_some_and(|n| n >= 20)
    );
    let anchor = t.branches[0].anchor.clone();
    let before = t.branches[0].pose;
    let (second, timing) = t.observe_blocking(&frame(1))?;
    let h = &second["candidate_hypotheses"][0];
    assert_eq!(h["tracking_supported"], true);
    assert_eq!(h["accepted"], false);
    assert_eq!(h["map_check"]["accepted"], true);
    assert!((t.branches[0].pose.position - before.position).norm() < 0.01);
    assert_eq!(t.branches[0].anchor, anchor);
    assert_eq!(timing.fast_runs, 1);
    assert!(timing.dense_runs > 0);
    Ok(())
}
#[test]
fn dense_fallback_requires_full_geometry_and_unknown_depth_still_rejects() -> Result<(), BenchError>
{
    let mut t = tracker(true, true, vec![pose()])?;
    t.observe_blocking(&frame(0))?;
    let (next, timing) = t.observe_blocking(&frame(1))?;
    assert_eq!(next["candidate_hypotheses"][0]["tracking_supported"], true);
    assert_eq!(timing.fast_runs, 1);
    assert!(timing.dense_runs > 0);
    let (missing, _) = tracker(false, false, vec![pose()])?.observe_blocking(&frame(0))?;
    assert_eq!(missing["candidate_hypotheses"][0]["accepted"], false);
    assert_eq!(
        missing["candidate_hypotheses"][0]["tracking_supported"],
        false
    );
    Ok(())
}
#[test]
fn separate_candidates_remain_separate() -> Result<(), BenchError> {
    let mut other = pose();
    other.position.x = 50.0;
    let mut t = tracker(true, false, vec![pose(), other])?;
    let (report, _) = t.observe_blocking(&frame(0))?;
    assert_eq!(
        report["candidate_hypotheses"].as_array().map(Vec::len),
        Some(2)
    );
    assert!((t.branches[0].pose.position.x - t.branches[1].pose.position.x).abs() > 40.0);
    Ok(())
}

#[test]
fn explicit_map_restart_retains_the_relative_alternative() -> Result<(), BenchError> {
    let mut t = tracker(true, false, vec![pose()])?;
    t.policy.reanchor = true;
    t.observe_blocking(&frame(0))?;
    let previous_anchor = t.branches[0].anchor.clone();
    let (report, _) = t.observe_blocking(&frame(1))?;
    let h = &report["candidate_hypotheses"][0];
    assert_eq!(h["accepted"], true);
    assert_eq!(h["continuity_break"], true);
    assert_eq!(h["relative_alternative"]["tracking_supported"], true);
    assert_eq!(
        h["relative_alternative"]["anchor_observation_sha256"],
        json!(previous_anchor)
    );
    assert_ne!(t.branches[0].anchor, previous_anchor);
    Ok(())
}

struct FailedMatcher;
impl ImageMatcher for FailedMatcher {
    fn identity(&self) -> &str {
        "failed-test-backend"
    }
    fn match_images_blocking(
        &mut self,
        _: &GrayImage,
        _: &GrayImage,
    ) -> Result<Vec<PixelMatch>, VisualError> {
        Err(VisualError::Backend {
            backend: "failed-test-backend".into(),
            source: Box::new(std::io::Error::other("injected inference failure")),
        })
    }
}
#[test]
fn map_backend_failure_preserves_relative_tracking_and_future_frames() -> Result<(), BenchError> {
    let mut t = tracker(true, false, vec![pose()])?;
    t.observe_blocking(&frame(0))?;
    let anchor = t.branches[0].anchor.clone();
    t.dense = Box::new(FailedMatcher);
    for sequence in [1, 2] {
        let (report, _) = t.observe_blocking(&frame(sequence))?;
        let h = &report["candidate_hypotheses"][0];
        assert_eq!(h["tracking_supported"], true);
        assert_eq!(h["accepted"], false);
        assert_eq!(h["map_check"]["accepted"], false);
        assert_eq!(h["map_check"]["acceptance_stage"], "image_matching_error");
        assert_eq!(
            h["map_check"]["error"]["causes"][0],
            "injected inference failure"
        );
        assert_eq!(t.branches[0].anchor, anchor);
    }
    Ok(())
}
#[test]
fn all_matcher_failures_are_observation_failures_with_no_pose() -> Result<(), BenchError> {
    let mut t = tracker(true, false, vec![pose()])?;
    t.observe_blocking(&frame(0))?;
    t.fast = Box::new(FailedMatcher);
    t.dense = Box::new(FailedMatcher);
    for sequence in [1, 2] {
        let (report, _) = t.observe_blocking(&frame(sequence))?;
        let h = &report["candidate_hypotheses"][0];
        assert_eq!(h["tracking_supported"], false);
        assert_eq!(h["accepted"], false);
        assert_eq!(h["matching_errors"].as_array().map(Vec::len), Some(2));
        assert!(h.get("position_enu_m").is_none());
    }
    Ok(())
}

#[test]
fn reference_observation_and_pose_stay_bound_until_keyframe_refresh() -> Result<(), BenchError> {
    let mut t = tracker(true, false, vec![pose()])?;
    t.policy.map_interval_ns = 0;
    t.observe_blocking(&frame(0))?;
    let first = frame(0).evidence_sha256();
    for sequence in [1, 2] {
        let (report, _) = t.observe_blocking(&frame(sequence))?;
        let h = &report["candidate_hypotheses"][0];
        assert_eq!(h["tracking_supported"], true);
        assert_eq!(h["reference_observation_sha256"], first);
        assert_eq!(h["motion_assumption"], "free pose");
    }
    let (report, _) = t.observe_blocking(&frame(3))?;
    assert_eq!(
        report["candidate_hypotheses"][0]["reference_observation_sha256"],
        frame(2).evidence_sha256()
    );
    Ok(())
}

impl navigate_visual::PointTracker for Correspondences {
    fn features_blocking(&mut self, image: &GrayImage) -> Result<Vec<Vector2<f64>>, VisualError> {
        self.match_images_blocking(image, image)
            .map(|pairs| pairs.into_iter().map(|pair| pair.reference).collect())
    }
    fn track_points_blocking(
        &mut self,
        _: &GrayImage,
        _: &GrayImage,
        points: &[Vector2<f64>],
    ) -> Result<Vec<Option<Vector2<f64>>>, VisualError> {
        Ok(points.iter().copied().map(Some).collect())
    }
}
impl navigate_visual::PointTracker for FailedMatcher {
    fn features_blocking(&mut self, image: &GrayImage) -> Result<Vec<Vector2<f64>>, VisualError> {
        self.match_images_blocking(image, image).map(|_| Vec::new())
    }
    fn track_points_blocking(
        &mut self,
        reference: &GrayImage,
        query: &GrayImage,
        _: &[Vector2<f64>],
    ) -> Result<Vec<Option<Vector2<f64>>>, VisualError> {
        self.match_images_blocking(reference, query)
            .map(|_| Vec::new())
    }
}

#[test]
fn surface_tracks_restart_at_map_anchors_and_keep_relative_evidence() -> Result<(), BenchError> {
    let mut t = tracker(true, false, vec![pose()])?;
    t.policy.surface_tracks = true;
    t.policy.keyframe_interval_ns = 0;
    t.policy.map_interval_ns = 0;
    t.observe_blocking(&frame(0))?;
    t.observe_blocking(&frame(1))?;
    assert!(t.branches[0].surface_tracks.is_some());
    t.policy.reanchor = true;
    t.policy.map_interval_ns = 1;
    let (report, _) = t.observe_blocking(&frame(2))?;
    let h = &report["candidate_hypotheses"][0];
    assert_eq!(h["accepted"], true);
    assert_eq!(h["relative_alternative"]["tracking_supported"], true);
    assert!(t.branches[0].surface_tracks.is_none());
    t.policy.map_interval_ns = 0;
    let (report, _) = t.observe_blocking(&frame(3))?;
    let h = &report["candidate_hypotheses"][0];
    assert_eq!(h["tracking_supported"], true);
    assert_eq!(
        h["reference_observation_sha256"],
        frame(2).evidence_sha256()
    );
    assert_eq!(
        h["depth_observation_sha256s"],
        json!([frame(2).evidence_sha256()])
    );
    Ok(())
}

mod map_refinement;

mod matching_alternatives;
