#[test]
fn command_is_consistent() {
    super::command().debug_assert();
}
#[test]
fn paths_are_explicit() {
    assert!(
        super::command()
            .try_get_matches_from(["navigate-workflow", "pack"])
            .is_err()
    );
}
