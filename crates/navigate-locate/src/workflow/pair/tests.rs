use super::super::test_scene::{limits, query, reference};
use super::*;
use image::GrayImage;
use navigate_visual::{AsyncImageMatcher, PixelMatch};

struct Matcher {
    calls: usize,
    fail: bool,
}
impl ImageMatcher for Matcher {
    fn identity(&self) -> &str {
        "pair-test"
    }
    fn match_images_blocking(
        &mut self,
        _: &GrayImage,
        _: &GrayImage,
    ) -> Result<Vec<PixelMatch>, VisualError> {
        self.calls = self.calls.wrapping_add(1);
        if self.fail {
            Err(VisualError::Invalid { field: "device" })
        } else {
            Ok(Vec::new())
        }
    }
}

fn workflow(config: ResourceLimits) -> PairLocalizer<Matcher> {
    PairLocalizer::new(
        Matcher {
            calls: 0,
            fail: false,
        },
        LocalizerConfig::default(),
        config,
    )
    .unwrap()
}

#[test]
fn rejected_geometry_is_a_result_and_runs_each_candidate() {
    let mut workflow = workflow(limits());
    let (frame, prior) = query();
    let report = workflow
        .estimate_blocking(&frame, &prior, &[reference(), reference()])
        .unwrap();
    assert_eq!(workflow.matcher.calls, 2);
    assert_eq!(report.extraction, Duration::ZERO);
    assert!(report.candidates.iter().all(|c| c.decision.is_err()));
}

#[test]
fn invalid_depth_stops_before_any_inference() {
    let mut workflow = workflow(limits());
    let (frame, prior) = query();
    let mut invalid = reference();
    invalid.depth_m.pop();
    assert!(
        workflow
            .estimate_blocking(&frame, &prior, &[reference(), invalid])
            .is_err()
    );
    assert_eq!(workflow.matcher.calls, 0);
}

#[test]
fn aggregate_resident_limit_stops_before_inference() {
    let mut config = limits();
    config.reference_bytes = 60_000;
    let mut workflow = workflow(config);
    let (frame, prior) = query();
    assert!(matches!(
        workflow.estimate_blocking(&frame, &prior, &[reference(), reference()]),
        Err(WorkflowError::ResourceLimit {
            resource: "reference bytes",
            ..
        })
    ));
    assert_eq!(workflow.matcher.calls, 0);
}

#[test]
fn backend_failure_is_not_a_geometry_rejection() {
    let mut workflow = workflow(limits());
    workflow.matcher.fail = true;
    let (frame, prior) = query();
    assert!(matches!(
        workflow.estimate_blocking(&frame, &prior, &[reference()]),
        Err(WorkflowError::Backend {
            stage: "match image pair",
            ..
        })
    ));
    assert_eq!(workflow.matcher.calls, 1);
}

#[test]
fn separate_camera_is_validated_before_pair_inference() {
    let mut workflow = workflow(limits());
    let (mut frame, prior) = query();
    let camera = frame.camera;
    frame.camera.width = 120;
    frame.image = GrayImage::new(120, 80);
    let view = reference();
    assert!(
        workflow
            .estimate_calibrated_blocking(&frame, &prior, &[(&view, &frame.camera)])
            .is_err()
    );
    assert_eq!(workflow.matcher.calls, 0);
    let report = workflow
        .estimate_calibrated_blocking(&frame, &prior, &[(&view, &camera)])
        .unwrap();
    assert_eq!(workflow.matcher.calls, 1);
    assert!(report.candidates[0].decision.is_err());
}

impl AsyncImageMatcher for Matcher {
    fn identity(&self) -> &str {
        "pair-test"
    }
    async fn match_images(
        &mut self,
        reference: &GrayImage,
        query: &GrayImage,
    ) -> Result<Vec<PixelMatch>, VisualError> {
        self.match_images_blocking(reference, query)
    }
}

#[test]
fn the_async_path_gives_the_blocking_decisions() {
    let (frame, prior) = query();
    let view = reference();
    let calibrated = [(&view, &frame.camera), (&view, &frame.camera)];
    let mut blocking = workflow(limits());
    let expected = blocking
        .estimate_calibrated_blocking(&frame, &prior, &calibrated)
        .unwrap();
    let mut asynchronous = workflow(limits());
    let actual =
        pollster::block_on(asynchronous.estimate_calibrated(&frame, &prior, &calibrated)).unwrap();
    assert_eq!(asynchronous.matcher.calls, blocking.matcher.calls);
    assert_eq!(actual.candidates.len(), expected.candidates.len());
    for (a, e) in actual.candidates.iter().zip(&expected.candidates) {
        assert_eq!(a.reference_index, e.reference_index);
        assert_eq!(a.matches, e.matches);
        assert_eq!(a.decision.is_ok(), e.decision.is_ok());
    }
}

#[test]
fn the_async_path_reports_a_backend_failure() {
    let mut workflow = workflow(limits());
    workflow.matcher.fail = true;
    let (frame, prior) = query();
    let view = reference();
    let result =
        pollster::block_on(workflow.estimate_calibrated(&frame, &prior, &[(&view, &frame.camera)]));
    assert!(matches!(
        result,
        Err(WorkflowError::Backend {
            stage: "match image pair",
            ..
        })
    ));
}
