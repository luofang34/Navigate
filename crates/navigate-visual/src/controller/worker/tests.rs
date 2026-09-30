use super::*;
use crate::{
    CameraModel, CameraPose, CandidateDecision, CandidateId, CandidateResults, FrameStamp,
    LocalFrame, MapRevision, PixelMatch,
};
use image::GrayImage;
use nalgebra::{UnitQuaternion, Vector2, Vector3};
use std::{cell::Cell, rc::Rc};

fn ms(value: u64) -> Duration {
    Duration::from_millis(value)
}
struct Matcher {
    clock: Rc<Cell<Duration>>,
    delay: Duration,
    calls: u64,
    fail: bool,
}
impl ImageMatcher for Matcher {
    fn identity(&self) -> &str {
        "synthetic-correspondences"
    }
    fn match_images_blocking(
        &mut self,
        _: &GrayImage,
        _: &GrayImage,
    ) -> Result<Vec<PixelMatch>, VisualError> {
        self.calls = self.calls.wrapping_add(1);
        self.clock.set(self.clock.get() + self.delay);
        if self.fail {
            return Err(VisualError::Backend {
                backend: self.identity().into(),
                source: Box::new(std::io::Error::other("device fault")),
            });
        }
        Ok((12..85)
            .step_by(12)
            .flat_map(|y| {
                (12..117).step_by(13).map(move |x| {
                    let pixel = Vector2::new(f64::from(x), f64::from(y));
                    PixelMatch {
                        reference: pixel,
                        query: pixel,
                    }
                })
            })
            .collect())
    }
}
fn setup() -> (CandidateWorker<Matcher>, Rc<Cell<Duration>>) {
    let profile = ExecutionProfile {
        id: super::super::ProfileId(1),
        device: super::super::DeviceId(1),
        work: WorkKind::MapCheck,
        initial_cost: ms(100),
        initial_call: ms(100),
        peak_bytes: 1000,
    };
    let clock = Rc::new(Cell::new(ms(0)));
    let matcher = Matcher {
        clock: Rc::clone(&clock),
        delay: ms(100),
        calls: 0,
        fail: false,
    };
    let worker = CandidateWorker::new(
        matcher,
        ControllerConfig {
            maximum_capture_age: ms(500),
            maximum_result_age: ms(1000),
            reserve: ms(10),
            minimum_interval: ms(1),
        },
        profile,
        LocalizerConfig::default(),
    )
    .expect("worker");
    (worker, clock)
}
fn grant(until: u64, call: u64) -> ResourceGrant {
    ResourceGrant {
        device: super::super::DeviceId(1),
        until: ms(until),
        maximum_call: ms(call),
        memory_bytes: 1000,
    }
}
struct Scene {
    observation: Frame,
    reference: ReferenceView,
    prior: PosePrior,
}
impl Scene {
    fn new() -> Self {
        let pose = CameraPose {
            position: Vector3::new(0.0, 0.0, 100.0),
            orientation: UnitQuaternion::identity(),
        };
        Self {
            observation: Frame {
                stamp: FrameStamp {
                    sequence: 7,
                    capture_time_ns: 0,
                },
                camera: CameraModel {
                    width: 128,
                    height: 96,
                    fx: 100.0,
                    fy: 100.0,
                    cx: 63.5,
                    cy: 47.5,
                },
                image: GrayImage::new(128, 96),
            },
            reference: ReferenceView {
                pose,
                frame: LocalFrame::anchor_mercator(40.0, -74.0).expect("frame"),
                map: MapRevision {
                    release_id: "fixture".into(),
                    manifest_sha256: "a".repeat(64),
                },
                image: GrayImage::new(128, 96),
                depth_m: vec![100.0; 128 * 96],
            },
            prior: PosePrior {
                pose,
                position_radius_m: 500.0,
                attitude_radius_rad: 1.0,
            },
        }
    }
    fn input(&self) -> CandidateInput<'_> {
        CandidateInput {
            observation: &self.observation,
            reference: &self.reference,
            prior: &self.prior,
        }
    }
}
fn completed(result: CandidateWork) -> CandidateCompletion {
    match result {
        CandidateWork::Completed(value) => *value,
        CandidateWork::Deferred(_) => panic!("expected execution"),
    }
}

#[test]
fn unavailable_memory_and_call_budget_do_not_invoke_the_backend() {
    let (mut worker, clock) = setup();
    let scene = Scene::new();
    let mut low_memory = grant(1000, 500);
    low_memory.memory_bytes = 999;
    for grants in [vec![], vec![grant(1000, 99)], vec![low_memory]] {
        assert!(matches!(
            worker
                .evaluate_blocking(scene.input(), None, &grants, || clock.get())
                .expect("admission"),
            CandidateWork::Deferred(Admission::NoResources)
        ));
    }
    assert_eq!(worker.matcher.calls, 0);
    assert!(matches!(
        worker
            .evaluate_blocking(scene.input(), Some(ms(1000)), &[grant(2000, 500)], || clock
                .get())
            .expect("due time"),
        CandidateWork::Deferred(Admission::NotDue(_))
    ));
    assert_eq!(worker.matcher.calls, 0);
}

#[test]
fn backend_failure_keeps_its_source_and_updates_cost_before_the_next_request() {
    let (mut worker, clock) = setup();
    worker.matcher.fail = true;
    worker.matcher.delay = ms(600);
    let mut scene = Scene::new();
    let result = completed(
        worker
            .evaluate_blocking(scene.input(), None, &[grant(500, 200)], || clock.get())
            .expect("call"),
    );
    assert!(!result.within_grant);
    assert!(result.within_age);
    let error = result.evaluation.acceptance.err().expect("backend error");
    assert_eq!(
        std::error::Error::source(&error)
            .expect("source")
            .to_string(),
        "device fault"
    );
    scene.observation.stamp.capture_time_ns = 600_000_000;
    let next = worker
        .evaluate_blocking(scene.input(), None, &[grant(2000, 500)], || clock.get())
        .expect("next");
    assert!(matches!(
        next,
        CandidateWork::Deferred(Admission::NoResources)
    ));
    assert_eq!(worker.matcher.calls, 1);
}

#[test]
fn stale_completion_is_distinct_from_valid_geometry_and_keeps_capture_identity() {
    let (mut worker, clock) = setup();
    worker.matcher.delay = ms(1200);
    let scene = Scene::new();
    let result = completed(
        worker
            .evaluate_blocking(scene.input(), None, &[grant(2000, 1500)], || clock.get())
            .expect("call"),
    );
    assert!(!result.within_age);
    assert!(result.within_grant);
    let estimate = result.evaluation.acceptance.expect("geometry");
    assert_eq!(estimate.stamp, scene.observation.stamp);
    assert_eq!(result.ticket.observation(), scene.observation.stamp);
    assert_eq!(
        estimate.observation_sha256,
        scene.observation.evidence_sha256()
    );
}

#[test]
fn candidate_refinements_replace_results_and_geographic_alternatives_stay_unresolved() {
    let (mut worker, clock) = setup();
    let mut scene = Scene::new();
    let mut results = CandidateResults::new(&scene.observation);
    let mut original_covariance = None;
    for id in [1, 1, 2] {
        scene.reference.pose.position.x = if id == 2 { 100.0 } else { 0.0 };
        let completion = completed(
            worker
                .evaluate_blocking(scene.input(), None, &[grant(2000, 500)], || clock.get())
                .expect("call"),
        );
        let estimate = completion.evaluation.acceptance.expect("geometry");
        assert!((estimate.pose.position.x - scene.reference.pose.position.x).abs() < 1e-7);
        if id == 1 {
            if let Some(covariance) = original_covariance {
                assert_eq!(estimate.geometry_covariance, covariance);
            }
            original_covariance = Some(estimate.geometry_covariance);
        }
        results
            .record(CandidateId(id), Ok(estimate))
            .expect("same evidence");
    }
    assert_eq!(results.iter().count(), 2);
    assert_eq!(
        results.decision(),
        CandidateDecision::Unresolved(vec![CandidateId(1), CandidateId(2)])
    );
}

#[test]
fn missing_depth_remains_a_geometric_rejection_and_stale_input_does_not_run() {
    let (mut worker, clock) = setup();
    let mut scene = Scene::new();
    scene.reference.depth_m.fill(0.0);
    let result = completed(
        worker
            .evaluate_blocking(scene.input(), None, &[grant(2000, 500)], || clock.get())
            .expect("call"),
    );
    assert!(result.evaluation.acceptance.is_err());
    clock.set(ms(600));
    let next = worker
        .evaluate_blocking(scene.input(), None, &[grant(2000, 500)], || clock.get())
        .expect("admission");
    assert!(matches!(
        next,
        CandidateWork::Deferred(Admission::StaleObservation)
    ));
    assert_eq!(worker.matcher.calls, 1);
}
