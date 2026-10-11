#[test]
fn command_requires_explicit_model_files_and_feature_budget() {
    super::command().debug_assert();
    assert!(
        super::command()
            .try_get_matches_from(["navigate-model", "in.onnx", "out.onnx"])
            .is_err()
    );
    assert!(
        super::command()
            .try_get_matches_from(["navigate-model", "in.onnx", "out.onnx", "--features", "512"])
            .is_ok()
    );
    assert!(
        super::command()
            .try_get_matches_from([
                "navigate-model",
                "in.onnx",
                "out.onnx",
                "--recipe",
                "xfeat-dense"
            ])
            .is_ok()
    );
}

#[test]
fn masked_recipe_requires_a_capacity() {
    let args = [
        "navigate-model",
        "in.onnx",
        "out.onnx",
        "--recipe",
        "lighterglue-masked",
    ];
    assert!(super::command().try_get_matches_from(args).is_err());
    assert!(
        super::command()
            .try_get_matches_from(args.into_iter().chain(["--features", "2048"]))
            .is_ok()
    );
}
