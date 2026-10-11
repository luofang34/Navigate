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
}
