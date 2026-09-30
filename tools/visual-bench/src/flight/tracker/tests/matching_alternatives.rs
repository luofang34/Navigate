use super::*;

struct Alternatives {
    base: Correspondences,
    valid_alternative: bool,
    alternatives: Vec<u32>,
}
impl ImageMatcher for Alternatives {
    fn identity(&self) -> &str {
        "test-alternatives"
    }
    fn match_images_blocking(
        &mut self,
        a: &GrayImage,
        b: &GrayImage,
    ) -> Result<Vec<PixelMatch>, VisualError> {
        self.base.match_images_blocking(a, b)
    }
    fn match_alternative_blocking(
        &mut self,
        a: &GrayImage,
        b: &GrayImage,
        attempt: u32,
    ) -> Result<Option<Vec<PixelMatch>>, VisualError> {
        self.alternatives.push(attempt);
        let mut source = Correspondences {
            calls: 0,
            empty: !self.valid_alternative,
            shift_map_checks: false,
        };
        source.match_images_blocking(a, b).map(Some)
    }
}
fn run(matcher: &mut Alternatives, valid_depth: bool) -> Result<(Value, Timing), BenchError> {
    let reference = frame(0);
    let query = frame(1);
    let surface = Surface {
        camera: camera(),
        valid: valid_depth,
    }
    .render_blocking(pose())?;
    let verifier = PoseVerifier::new(LocalizerConfig::default())?;
    let prior = PosePrior {
        pose: pose(),
        position_radius_m: 500.0,
        attitude_radius_rad: 3.0,
    };
    let mut timing = Timing::default();
    let (report, _) = relative(
        &mut FailedMatcher,
        matcher,
        &query,
        TrackingReference {
            observation: &reference,
            surface: &surface,
        },
        Check {
            verifier: &verifier,
            prior: &prior,
            dense_only: true,
            fixed_tilt: false,
        },
        &mut timing,
    )?;
    Ok((report, timing))
}
fn matcher(base_empty: bool, valid_alternative: bool) -> Alternatives {
    Alternatives {
        base: Correspondences {
            calls: 0,
            empty: base_empty,
            shift_map_checks: false,
        },
        valid_alternative,
        alternatives: Vec::new(),
    }
}
#[test]
fn passing_base_geometry_does_not_run_alternatives() -> Result<(), BenchError> {
    let mut matcher = matcher(false, true);
    let (report, timing) = run(&mut matcher, true)?;
    assert_eq!(report["tracking_supported"], true);
    assert!(matcher.alternatives.is_empty());
    assert_eq!(timing.dense_runs, 1);
    Ok(())
}
#[test]
fn first_supported_alternative_stops_and_retains_shared_evidence() -> Result<(), BenchError> {
    let mut matcher = matcher(true, true);
    let (report, timing) = run(&mut matcher, true)?;
    assert_eq!(report["accepted"], false);
    assert_eq!(report["tracking_supported"], true);
    assert_eq!(
        report["reference_observation_sha256"],
        frame(0).evidence_sha256()
    );
    assert_eq!(report["matcher_attempt"], 1);
    assert_eq!(
        report["matching_attempts"].as_array().map(Vec::len),
        Some(2)
    );
    assert_eq!(report["matching_attempts"][0]["tracking_supported"], false);
    assert_eq!(
        report["matching_evidence"],
        "same image pair; attempts are not independent measurements"
    );
    assert_eq!(matcher.alternatives, [0]);
    assert_eq!(timing.dense_runs, 2);
    Ok(())
}
#[test]
fn alternatives_cannot_bypass_depth_or_exceed_the_attempt_budget() -> Result<(), BenchError> {
    for (valid_depth, valid_alternative) in [(false, true), (true, false)] {
        let mut matcher = matcher(true, valid_alternative);
        let (report, timing) = run(&mut matcher, valid_depth)?;
        assert_eq!(report["tracking_supported"], false);
        assert_eq!(report["accepted"], false);
        assert!(report.get("position_enu_m").is_none());
        assert_eq!(matcher.alternatives, [0, 1, 2]);
        assert_eq!(timing.dense_runs, 4);
    }
    Ok(())
}
