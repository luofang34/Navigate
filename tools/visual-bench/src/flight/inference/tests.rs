use super::*;
struct Command {
    inference: InferenceArgs,
}

impl Command {
    fn try_parse_from<'a>(args: impl IntoIterator<Item = &'a str>) -> Result<Self, BenchError> {
        let mut matches =
            InferenceArgs::args(clap::Command::new("flight")).try_get_matches_from(args)?;
        Ok(Self {
            inference: InferenceArgs::from_matches(&mut matches)?,
        })
    }
}

#[test]
fn model_selection_rejects_missing_or_unused_detector_before_runtime_io() -> Result<(), BenchError>
{
    let base = [
        "flight",
        "--model",
        "matcher.onnx",
        "--runtime",
        "runtime.dylib",
    ];
    for name in ["lightglue", "superglue"] {
        assert!(Command::try_parse_from(base.into_iter().chain(["--matcher", name])).is_err());
        let command = Command::try_parse_from(base.into_iter().chain([
            "--matcher",
            name,
            "--detector",
            "detector.onnx",
        ]))?;
        let files = command.inference.sparse_files()?;
        match name {
            "lightglue" => assert!(matches!(
                files,
                Some(MatcherFiles::LightGlue {
                    image_size: [640, 360],
                    ..
                })
            )),
            _ => assert!(matches!(files, Some(MatcherFiles::SuperGlue { .. }))),
        }
    }
    let command = Command::try_parse_from(base.into_iter().chain(["--detector", "unused.onnx"]))?;
    assert!(command.inference.load_blocking().is_err());
    let command = Command::try_parse_from(base)?;
    assert!(command.inference.sparse_files()?.is_none());
    Ok(())
}

#[test]
fn feature_profile_rejects_other_models_before_runtime_io() -> Result<(), BenchError> {
    for matcher in ["loftr", "superglue"] {
        let mut args = vec![
            "flight",
            "--model",
            "missing.onnx",
            "--runtime",
            "missing.dylib",
            "--profile-keypoints",
            "--matcher",
            matcher,
        ];
        if matcher == "superglue" {
            args.extend(["--detector", "missing-detector.onnx"]);
        }
        let command = Command::try_parse_from(args)?;
        assert!(
            matches!(command.inference.load_blocking(),Err(BenchError::Record {reason}) if reason.contains("require LightGlue"))
        );
    }
    let command = Command::try_parse_from([
        "flight",
        "--model",
        "missing.onnx",
        "--runtime",
        "missing.dylib",
        "--matcher",
        "lightglue",
        "--detector",
        "missing-detector.onnx",
        "--profile-keypoints",
    ])?;
    assert!(command.inference.sparse_files()?.is_some());
    Ok(())
}

#[test]
fn refinement_patches_are_opt_in_and_reject_sparse_models_before_runtime_io()
-> Result<(), BenchError> {
    let base = [
        "flight",
        "--model",
        "missing.onnx",
        "--runtime",
        "missing.dylib",
    ];
    assert!(!Command::try_parse_from(base)?.inference.refinement_patches);
    let command = Command::try_parse_from(base.into_iter().chain(["--refinement-patches"]))?;
    assert!(command.inference.refinement_patches);
    assert!(command.inference.sparse_files()?.is_none());
    for matcher in ["lightglue", "superglue"] {
        let command = Command::try_parse_from(base.into_iter().chain([
            "--matcher",
            matcher,
            "--detector",
            "missing-detector.onnx",
            "--refinement-patches",
        ]))?;
        assert!(
            matches!(command.inference.load_blocking(), Err(BenchError::Record { reason }) if reason == "refinement patches require LoFTR")
        );
    }
    Ok(())
}

#[test]
fn quarter_turns_are_opt_in_and_reject_sparse_models_before_runtime_io() -> Result<(), BenchError> {
    let base = [
        "flight",
        "--model",
        "missing.onnx",
        "--runtime",
        "missing.dylib",
    ];
    assert!(!Command::try_parse_from(base)?.inference.quarter_turns);
    let command = Command::try_parse_from(base.into_iter().chain(["--quarter-turns"]))?;
    assert!(command.inference.quarter_turns);
    assert!(command.inference.sparse_files()?.is_none());
    for matcher in ["lightglue", "superglue"] {
        let command = Command::try_parse_from(base.into_iter().chain([
            "--matcher",
            matcher,
            "--detector",
            "missing-detector.onnx",
            "--quarter-turns",
        ]))?;
        assert!(
            matches!(command.inference.load_blocking(), Err(BenchError::Record { reason }) if reason == "quarter turns require LoFTR")
        );
    }
    Ok(())
}

#[test]
fn host_default_and_explicit_provider_overrides_survive_cli_parsing() -> Result<(), BenchError> {
    let base = [
        "flight",
        "--model",
        "model.onnx",
        "--runtime",
        "runtime.dylib",
    ];
    let default = Command::try_parse_from(base)?.inference.provider();
    if cfg!(target_os = "macos") {
        assert!(matches!(default, Provider::CoreMlGpu));
    } else {
        assert!(matches!(default, Provider::Cpu));
    }
    for (argument, expected) in [
        ("cpu", Provider::Cpu),
        ("coreml-gpu", Provider::CoreMlGpu),
        ("coreml-ane", Provider::CoreMlAne),
        ("cuda", Provider::Cuda),
        ("tensor-rt", Provider::TensorRt),
    ] {
        let selected = Command::try_parse_from(base.into_iter().chain(["--device", argument]))?
            .inference
            .provider();
        assert_eq!(
            std::mem::discriminant(&selected),
            std::mem::discriminant(&expected)
        );
    }
    Ok(())
}

#[test]
fn coreml_cache_is_opt_in_and_keeps_the_selected_directory() -> Result<(), BenchError> {
    let base = [
        "flight",
        "--model",
        "model.onnx",
        "--runtime",
        "runtime.dylib",
    ];
    assert!(
        Command::try_parse_from(base)?
            .inference
            .coreml_cache
            .is_none()
    );
    let parsed =
        Command::try_parse_from(base.into_iter().chain(["--coreml-cache", "model-cache"]))?;
    assert_eq!(
        parsed.inference.coreml_cache,
        Some(PathBuf::from("model-cache"))
    );
    Ok(())
}
