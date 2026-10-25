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

#[test]
fn gpu_image_reuse_is_opt_in_for_acquisition() {
    let args = [
        "navigate-workflow",
        "evaluate-acquisition",
        "config.json",
        "manifest.json",
        "output.json",
    ];
    let normal = super::command().try_get_matches_from(args).unwrap();
    assert!(
        !normal
            .subcommand_matches("evaluate-acquisition")
            .unwrap()
            .get_flag("gpu-image-reuse")
    );
    let enabled = super::command()
        .try_get_matches_from(args.into_iter().chain(["--gpu-image-reuse"]))
        .unwrap();
    assert!(
        enabled
            .subcommand_matches("evaluate-acquisition")
            .unwrap()
            .get_flag("gpu-image-reuse")
    );
}

#[test]
fn rectification_is_explicit_and_not_available_for_sequence_tracking() {
    let args = super::command()
        .try_get_matches_from([
            "navigate-workflow",
            "acquire-surface",
            "config",
            "job",
            "output",
            "--rectification",
            "camera.json",
        ])
        .unwrap();
    assert_eq!(
        args.subcommand_matches("acquire-surface")
            .unwrap()
            .get_one::<std::path::PathBuf>("rectification")
            .unwrap(),
        &std::path::PathBuf::from("camera.json")
    );
    assert!(
        super::command()
            .try_get_matches_from([
                "navigate-workflow",
                "evaluate-acquisition",
                "config",
                "manifest",
                "output",
                "--rectification",
                "camera.json"
            ])
            .is_err()
    );
}

#[test]
fn grid_probe_requires_explicit_scale_and_accepts_raw_camera_binding() {
    let args = [
        "navigate-workflow",
        "probe-acquisition-grid",
        "config",
        "job",
        "output",
    ];
    assert!(super::command().try_get_matches_from(args).is_err());
    let parsed = super::command()
        .try_get_matches_from(args.into_iter().chain([
            "--side",
            "512",
            "--rectification",
            "camera.json",
        ]))
        .unwrap();
    let grid = parsed.subcommand_matches("probe-acquisition-grid").unwrap();
    assert_eq!(grid.get_one::<u32>("side"), Some(&512));
    assert_eq!(
        grid.get_one::<std::path::PathBuf>("rectification").unwrap(),
        &std::path::PathBuf::from("camera.json")
    );
}

#[test]
fn pyramid_requires_a_complete_explicit_search_budget() {
    let args = [
        "navigate-workflow",
        "probe-acquisition-pyramid",
        "config",
        "job",
        "output",
        "--side",
        "1024",
    ];
    assert!(super::command().try_get_matches_from(args).is_err());
    let parsed = super::command()
        .try_get_matches_from(args.into_iter().chain([
            "--levels",
            "3",
            "--beam",
            "8",
            "--rectification",
            "camera.json",
        ]))
        .unwrap();
    let args = parsed
        .subcommand_matches("probe-acquisition-pyramid")
        .unwrap();
    assert_eq!(args.get_one::<usize>("beam"), Some(&8));
    assert_eq!(args.get_one::<usize>("levels"), Some(&3));
}

#[test]
fn offline_pyramid_coverage_is_explicit_and_separate_from_acquisition_fine() {
    let base = [
        "navigate-workflow",
        "prepare-acquisition",
        "config",
        "surface",
        "output",
    ];
    for suffix in [
        vec!["--pyramid-side", "1024"],
        vec!["--pyramid-levels", "2"],
        vec![
            "--pyramid-side",
            "1024",
            "--pyramid-levels",
            "2",
            "--include-fine",
        ],
    ] {
        assert!(
            super::command()
                .try_get_matches_from(base.into_iter().chain(suffix))
                .is_err()
        );
    }
    let parsed = super::command()
        .try_get_matches_from(base.into_iter().chain([
            "--pyramid-side",
            "1024",
            "--pyramid-levels",
            "2",
        ]))
        .unwrap();
    let args = parsed.subcommand_matches("prepare-acquisition").unwrap();
    assert!(matches!(
        super::feature_coverage(args).unwrap(),
        crate::surface::acquire::Coverage::Pyramid {
            side: 1024,
            levels: 2
        }
    ));
    let parsed = super::command()
        .try_get_matches_from([
            "navigate-workflow",
            "probe-acquisition-pyramid",
            "config",
            "job",
            "output",
            "--side",
            "1024",
            "--levels",
            "3",
            "--beam",
            "8",
            "--reference-feature-catalog",
            "catalog.json",
        ])
        .unwrap();
    let args = parsed
        .subcommand_matches("probe-acquisition-pyramid")
        .unwrap();
    assert_eq!(
        args.get_one::<std::path::PathBuf>("reference-feature-catalog")
            .unwrap(),
        &std::path::PathBuf::from("catalog.json")
    );
}

#[test]
fn retained_window_recovery_is_opt_in_for_acquisition() {
    let args = [
        "navigate-workflow",
        "evaluate-acquisition",
        "config.json",
        "manifest.json",
        "output.json",
    ];
    let normal = super::command().try_get_matches_from(args).unwrap();
    assert!(
        !normal
            .subcommand_matches("evaluate-acquisition")
            .unwrap()
            .get_flag("retained-window-recovery")
    );
    let enabled = super::command()
        .try_get_matches_from(args.into_iter().chain(["--retained-window-recovery"]))
        .unwrap();
    assert!(
        enabled
            .subcommand_matches("evaluate-acquisition")
            .unwrap()
            .get_flag("retained-window-recovery")
    );
}
