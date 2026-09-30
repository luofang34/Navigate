use super::*;

struct CountedSurface {
    source: Surface,
    renders: usize,
}
impl ReferenceRenderer for CountedSurface {
    type Error = BenchError;
    fn identity(&self) -> RendererIdentity {
        self.source.identity()
    }
    fn render_blocking(&mut self, pose: CameraPose) -> Result<ReferenceView, BenchError> {
        self.renders = self.renders.wrapping_add(1);
        self.source.render_blocking(pose)
    }
}
fn renderer() -> CountedSurface {
    CountedSurface {
        source: Surface {
            camera: camera(),
            valid: true,
        },
        renders: 0,
    }
}
fn prior() -> PosePrior {
    PosePrior {
        pose: pose(),
        position_radius_m: 500.0,
        attitude_radius_rad: 3.0,
    }
}
fn shifted_matches() -> Correspondences {
    Correspondences {
        calls: 1,
        empty: false,
        shift_map_checks: true,
    }
}
fn attempts(report: &Value) -> Result<&Vec<Value>, BenchError> {
    report["refinement_attempts"]
        .as_array()
        .ok_or_else(|| BenchError::Record {
            reason: "map check did not retain attempts".into(),
        })
}

#[test]
fn bounded_refinement_renders_only_views_that_are_matched() -> Result<(), BenchError> {
    let mut renderer = renderer();
    let reference = renderer.source.render_blocking(pose())?;
    let frame = frame(0);
    let mut timing = Timing::default();
    let (report, fitted) = map_check(
        &mut renderer,
        &mut shifted_matches(),
        &PoseVerifier::new(LocalizerConfig::default())?,
        &frame,
        reference,
        &prior(),
        &mut timing,
    )?;
    let attempts = attempts(&report)?;
    assert_eq!(attempts.len(), 3);
    assert_eq!(renderer.renders, 2);
    assert_eq!(timing.dense_runs, 3);
    assert!(fitted.is_some());
    for attempt in attempts {
        assert_eq!(attempt["accepted"], true);
        assert_eq!(attempt["observation_sha256"], frame.evidence_sha256());
        assert_eq!(attempt["map_manifest_sha256"], "b".repeat(64));
        assert!(
            attempt["image_correspondences"]
                .as_u64()
                .is_some_and(|n| n >= 20)
        );
        assert_eq!(
            attempt["reference_depth_sha256"].as_str().map(str::len),
            Some(64)
        );
    }
    assert_ne!(attempts[0]["reference_pose"], attempts[1]["reference_pose"]);
    assert_eq!(report["position_enu_m"], attempts[2]["position_enu_m"]);
    assert_eq!(
        report["refinement_evidence"],
        "same observation; attempts are not independent measurements"
    );
    Ok(())
}

struct RejectRefinement {
    matcher: Correspondences,
    called: bool,
    device_error: bool,
}
impl ImageMatcher for RejectRefinement {
    fn identity(&self) -> &str {
        "reject-refinement-test"
    }
    fn match_images_blocking(
        &mut self,
        a: &GrayImage,
        b: &GrayImage,
    ) -> Result<Vec<PixelMatch>, VisualError> {
        if !self.called {
            self.called = true;
            return self.matcher.match_images_blocking(a, b);
        }
        if self.device_error {
            FailedMatcher.match_images_blocking(a, b)
        } else {
            Ok(Vec::new())
        }
    }
}

#[test]
fn failed_refinement_cannot_reuse_an_earlier_acceptance() -> Result<(), BenchError> {
    for device_error in [false, true] {
        let mut renderer = renderer();
        let reference = renderer.source.render_blocking(pose())?;
        let mut matcher = RejectRefinement {
            matcher: shifted_matches(),
            called: false,
            device_error,
        };
        let mut timing = Timing::default();
        let (report, fitted) = map_check(
            &mut renderer,
            &mut matcher,
            &PoseVerifier::new(LocalizerConfig::default())?,
            &frame(0),
            reference,
            &prior(),
            &mut timing,
        )?;
        assert!(fitted.is_none());
        assert_eq!(report["accepted"], false);
        assert!(report.get("position_enu_m").is_none());
        assert_eq!(renderer.renders, 1);
        assert_eq!(timing.dense_runs, 2);
        let attempts = attempts(&report)?;
        assert_eq!(attempts.len(), 2);
        assert_eq!(attempts[0]["accepted"], true);
        assert_eq!(attempts[1]["accepted"], false);
        assert_eq!(
            attempts[0]["observation_sha256"],
            attempts[1]["observation_sha256"]
        );
        assert_eq!(report["reason"], attempts[1]["reason"]);
        if device_error {
            assert_eq!(
                attempts[1]["error"]["causes"][0],
                "injected inference failure"
            );
        } else {
            assert_eq!(attempts[1]["image_correspondences"], 0);
        }
    }
    Ok(())
}

#[test]
fn converged_map_check_stops_without_another_render() -> Result<(), BenchError> {
    let mut renderer = renderer();
    let reference = renderer.source.render_blocking(pose())?;
    let mut matcher = Correspondences {
        calls: 0,
        empty: false,
        shift_map_checks: false,
    };
    let mut timing = Timing::default();
    let (report, fitted) = map_check(
        &mut renderer,
        &mut matcher,
        &PoseVerifier::new(LocalizerConfig::default())?,
        &frame(0),
        reference,
        &prior(),
        &mut timing,
    )?;
    assert!(fitted.is_some());
    assert_eq!(attempts(&report)?.len(), 1);
    assert_eq!(renderer.renders, 0);
    assert_eq!(timing.dense_runs, 1);
    Ok(())
}
