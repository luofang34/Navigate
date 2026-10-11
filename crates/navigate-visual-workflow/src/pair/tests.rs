use super::*;
use crate::test_scene::{limits, query, reference};
use image::GrayImage;
use navigate_visual::PixelMatch;

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
