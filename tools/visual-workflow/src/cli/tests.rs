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

#[test]
fn gpu_timing_is_opt_in_and_global() {
    let args = [
        "navigate-workflow",
        "evaluate-acquisition",
        "config.json",
        "manifest.json",
        "output.json",
    ];
    let normal = super::command().try_get_matches_from(args).unwrap();
    assert!(!normal.get_flag("gpu-timings"));
    let timed = super::command()
        .try_get_matches_from(args.into_iter().chain(["--gpu-timings"]))
        .unwrap();
    assert!(timed.get_flag("gpu-timings"));
}
