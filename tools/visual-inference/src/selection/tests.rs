#![allow(clippy::panic)]

use super::*;
use std::{error::Error, path::PathBuf};

fn invalid_candidate(provider: Provider) -> MatcherCandidate {
    MatcherCandidate {
        files: MatcherFiles::XFeat {
            model: PathBuf::from("unused.onnx"),
        },
        execution: ExecutionConfig {
            provider,
            ..ExecutionConfig::default()
        },
    }
}

#[test]
fn empty_preferences_do_not_initialize_runtime() {
    assert!(matches!(
        OnnxMatcher::load_preferred_blocking(Vec::new(), 1024),
        Err(SelectionError::Empty)
    ));
}

#[test]
fn all_candidate_errors_keep_order_provider_and_source() {
    let result = OnnxMatcher::load_preferred_blocking(
        vec![
            invalid_candidate(Provider::Cuda),
            invalid_candidate(Provider::Cpu),
        ],
        0,
    );
    match result {
        Err(error @ SelectionError::AllFailed { .. }) => {
            assert!(error.source().is_some());
            let SelectionError::AllFailed { previous, last } = error else {
                return;
            };
            assert_eq!(previous.len(), 1);
            assert_eq!(previous[0].index, 0);
            assert!(matches!(previous[0].requested_provider, Provider::Cuda));
            assert_eq!(last.index, 1);
            assert!(matches!(last.requested_provider, Provider::Cpu));
            assert!(matches!(last.source, InferenceError::Invalid(_)));
            assert!(last.source().is_some());
        }
        _ => panic!("invalid keypoint limits must reject every candidate"),
    }
}

#[test]
#[ignore = "requires NAVIGATE_TEST_ORT, NAVIGATE_TEST_XFEAT, and NAVIGATE_TEST_IMAGE"]
fn cpu_fallback_runs_real_image_matching() -> Result<(), Box<dyn Error>> {
    use navigate_visual::ImageMatcher;
    let path = |name| std::env::var_os(name).map(PathBuf::from).ok_or(name);
    crate::initialize_blocking(&path("NAVIGATE_TEST_ORT")?)?;
    let model = path("NAVIGATE_TEST_XFEAT")?;
    let image = image::open(path("NAVIGATE_TEST_IMAGE")?)?.to_luma8();
    let candidate = |provider| MatcherCandidate {
        files: MatcherFiles::XFeat {
            model: model.clone(),
        },
        execution: ExecutionConfig {
            provider,
            ..ExecutionConfig::default()
        },
    };
    let (mut matcher, selected) = OnnxMatcher::load_preferred_blocking(
        vec![candidate(Provider::Cuda), candidate(Provider::Cpu)],
        512,
    )?;
    assert_eq!(selected.index, 1);
    assert!(matches!(selected.requested_provider, Provider::Cpu));
    assert_eq!(selected.rejected.len(), 1);
    let matches = matcher.match_images_blocking(&image, &image)?;
    assert!(matches.len() >= 20);
    for pair in matches {
        assert!((pair.reference - pair.query).norm() < 0.5);
    }
    Ok(())
}
