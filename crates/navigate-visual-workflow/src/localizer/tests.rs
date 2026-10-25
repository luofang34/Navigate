use super::*;
use crate::FeatureMetadata;
use crate::test_scene::{limits, query, reference};
use image::GrayImage;
use navigate_visual::PixelMatch;

struct Backend {
    extractions: usize,
    matches: usize,
}
impl FeatureBackend for Backend {
    type Features = GrayImage;
    fn identity(&self) -> &str {
        "test-grid"
    }
    fn extract_features_blocking(
        &mut self,
        image: &GrayImage,
    ) -> Result<Self::Features, VisualError> {
        self.extractions = self.extractions.wrapping_add(1);
        Ok(image.clone())
    }
    fn validate_features(
        &self,
        features: &Self::Features,
        image: &GrayImage,
    ) -> Result<FeatureMetadata, VisualError> {
        if features != image {
            return Err(VisualError::Invalid {
                field: "test source",
            });
        }
        Ok(FeatureMetadata {
            resident_bytes: features.as_raw().len(),
            count: 24,
        })
    }
    fn equivalent_features(&self, first: &GrayImage, second: &GrayImage) -> bool {
        first == second
    }
    fn match_features_blocking(
        &mut self,
        _: &Self::Features,
        query: &Self::Features,
    ) -> Result<Vec<PixelMatch>, VisualError> {
        self.matches = self.matches.wrapping_add(1);
        let x_scale = f64::from(query.width()) / 100.0;
        Ok((0..4)
            .flat_map(move |y| {
                (0..6).map(move |x| {
                    let p = [10.0 + f64::from(x) * 14.0, 10.0 + f64::from(y) * 20.0];
                    PixelMatch {
                        reference: p.into(),
                        query: [p[0] * x_scale, p[1]].into(),
                    }
                })
            })
            .collect())
    }
}
#[test]
fn cached_references_extract_one_query_for_all_candidates() {
    let mut workflow = FeatureLocalizer::new(
        Backend {
            extractions: 0,
            matches: 0,
        },
        LocalizerConfig::default(),
        limits(),
    )
    .unwrap();
    let refs = vec![
        workflow.prepare_reference_blocking(reference()).unwrap(),
        workflow.prepare_reference_blocking(reference()).unwrap(),
    ];
    let (frame, prior) = query();
    let report = workflow.estimate_blocking(&frame, &prior, &refs).unwrap();
    assert_eq!(workflow.backend().extractions, 3);
    assert_eq!(workflow.backend().matches, 1);
    assert_eq!(report.candidates.len(), 2);
    assert!(report.candidates.iter().all(|r| r.matches == 24));
    assert!(report.candidates.iter().all(|r| r.decision.is_ok()));
}
#[test]
fn rejected_resource_request_does_not_extract_query() {
    let mut config = limits();
    config.candidates = 1;
    let mut workflow = FeatureLocalizer::new(
        Backend {
            extractions: 0,
            matches: 0,
        },
        LocalizerConfig::default(),
        config,
    )
    .unwrap();
    let refs = vec![
        workflow.prepare_reference_blocking(reference()).unwrap(),
        workflow.prepare_reference_blocking(reference()).unwrap(),
    ];
    let (frame, prior) = query();
    assert!(matches!(
        workflow.estimate_blocking(&frame, &prior, &refs),
        Err(WorkflowError::ResourceLimit {
            resource: "reference candidates",
            ..
        })
    ));
    assert_eq!(workflow.backend().extractions, 2);
    assert_eq!(workflow.backend().matches, 0);
}
#[test]
fn imported_reference_requires_matching_source_pixels() {
    let workflow = FeatureLocalizer::new(
        Backend {
            extractions: 0,
            matches: 0,
        },
        LocalizerConfig::default(),
        limits(),
    )
    .unwrap();
    let features = GrayImage::new(100, 80);
    assert!(workflow.import_reference(reference(), features).is_err());
}
#[test]
fn no_candidates_does_not_run_extraction() {
    let mut workflow = FeatureLocalizer::new(
        Backend {
            extractions: 0,
            matches: 0,
        },
        LocalizerConfig::default(),
        limits(),
    )
    .unwrap();
    let (frame, prior) = query();
    assert!(
        workflow
            .estimate_blocking(&frame, &prior, &[])
            .unwrap()
            .candidates
            .is_empty()
    );
    assert_eq!(workflow.backend().extractions, 0);
}

#[test]
fn borrowed_references_reuse_features_across_queries() {
    let mut workflow = FeatureLocalizer::new(
        Backend {
            extractions: 0,
            matches: 0,
        },
        LocalizerConfig::default(),
        limits(),
    )
    .unwrap();
    let first = std::sync::Arc::new(workflow.prepare_reference_blocking(reference()).unwrap());
    let second = std::sync::Arc::new(workflow.prepare_reference_blocking(reference()).unwrap());
    let (frame, prior) = query();
    for refs in [
        [first.as_ref(), second.as_ref()],
        [second.as_ref(), first.as_ref()],
    ] {
        let report = workflow
            .estimate_selected_blocking(&frame, &prior, &refs)
            .unwrap();
        assert!(
            report
                .candidates
                .iter()
                .all(|candidate| candidate.decision.is_ok())
        );
    }
    assert_eq!(workflow.backend().extractions, 4);
    assert_eq!(workflow.backend().matches, 2);
    assert_eq!(std::sync::Arc::strong_count(&first), 1);
}

#[test]
fn borrowed_references_enforce_aggregate_memory_before_query_extraction() {
    let mut config = limits();
    config.reference_bytes = 100 * 80 * 6;
    let mut workflow = FeatureLocalizer::new(
        Backend {
            extractions: 0,
            matches: 0,
        },
        LocalizerConfig::default(),
        config,
    )
    .unwrap();
    let prepared = workflow.prepare_reference_blocking(reference()).unwrap();
    let (frame, prior) = query();
    assert!(matches!(
        workflow.estimate_selected_blocking(&frame, &prior, &[&prepared, &prepared]),
        Err(WorkflowError::ResourceLimit {
            resource: "reference bytes",
            ..
        })
    ));
    assert_eq!(workflow.backend().extractions, 1);
}

#[test]
fn equivalent_features_reuse_matches_but_keep_distinct_geometry_decisions() {
    let mut workflow = FeatureLocalizer::new(
        Backend {
            extractions: 0,
            matches: 0,
        },
        LocalizerConfig::default(),
        limits(),
    )
    .unwrap();
    let mut bad_geometry = reference();
    bad_geometry.depth_m.fill(0.0);
    let mut different_image = reference();
    different_image.image.put_pixel(0, 0, image::Luma([101]));
    let refs = vec![
        workflow.prepare_reference_blocking(reference()).unwrap(),
        workflow.prepare_reference_blocking(bad_geometry).unwrap(),
        workflow
            .prepare_reference_blocking(different_image)
            .unwrap(),
    ];
    let (frame, prior) = query();
    let report = workflow.estimate_blocking(&frame, &prior, &refs).unwrap();
    assert_eq!(workflow.backend().matches, 2);
    assert_eq!(workflow.backend().extractions, 4);
    assert_eq!(report.candidates.len(), 3);
    assert!(report.candidates[0].decision.is_ok());
    assert!(report.candidates[1].decision.is_err());
    assert!(report.candidates[2].decision.is_ok());
    assert_eq!(report.candidates[1].matching_reused_from, Some(0));
    assert_eq!(report.candidates[1].times.matching, Duration::ZERO);
    assert_eq!(report.candidates[2].matching_reused_from, None);
    assert_eq!(
        report
            .candidates
            .iter()
            .map(|c| c.reference_index)
            .collect::<Vec<_>>(),
        [0, 1, 2]
    );
}

#[test]
fn calibrated_cached_reference_keeps_storage_and_metric_pose() {
    let mut workflow = FeatureLocalizer::new(
        Backend {
            extractions: 0,
            matches: 0,
        },
        LocalizerConfig::default(),
        limits(),
    )
    .unwrap();
    let (mut frame, prior) = query();
    let camera = frame.camera;
    let prepared = workflow.prepare_reference_blocking(reference()).unwrap();
    let pixels = prepared.view.image.as_raw().as_ptr();
    let features = prepared.features.as_raw().as_ptr();
    let prepared = prepared.with_camera(camera).unwrap();
    assert_eq!(pixels, prepared.view.image.as_raw().as_ptr());
    assert_eq!(features, prepared.features.as_raw().as_ptr());
    frame.camera.width = 120;
    frame.camera.fx *= 1.2;
    frame.camera.cx *= 1.2;
    frame.image = GrayImage::new(120, 80);
    let result = workflow
        .estimate_selected_blocking(&frame, &prior, &[&prepared, &prepared])
        .unwrap();
    assert_eq!(workflow.backend().extractions, 2);
    assert_eq!(workflow.backend().matches, 1);
    for candidate in result.candidates {
        let estimate = candidate.decision.unwrap();
        assert!((estimate.pose.position - prior.pose.position).norm() < 1e-6);
    }
}

#[test]
fn reference_camera_binding_rejects_mismatched_dimensions() {
    let workflow = FeatureLocalizer::new(
        Backend {
            extractions: 0,
            matches: 0,
        },
        LocalizerConfig::default(),
        limits(),
    )
    .unwrap();
    let (frame, _) = query();
    let mut camera = frame.camera;
    camera.width = 120;
    let view = reference();
    let features = view.image.clone();
    let prepared = workflow.import_reference(view, features).unwrap();
    assert!(prepared.with_camera(camera).is_err());
    assert_eq!(workflow.backend().extractions, 0);
}
