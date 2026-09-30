use super::*;

#[test]
fn command_is_consistent() {
    command().debug_assert();
}

#[test]
fn global_backend_and_subcommand_defaults_parse() {
    let mut matches = command()
        .try_get_matches_from([
            "visual-bench",
            "video",
            "map",
            "in.mp4",
            "--prior",
            "p.json",
            "--output",
            "o.jsonl",
            "--backend",
            "gpu",
        ])
        .expect("valid arguments");
    let backend: BackendKind = take(&mut matches, "backend").expect("backend");
    assert!(matches!(backend, BackendKind::Gpu));
    let (name, mut sub) = matches.remove_subcommand().expect("subcommand");
    assert_eq!(name, "video");
    let trial = TrialArgs::from_matches(&mut sub).expect("trial arguments");
    assert_eq!(trial.input, PathBuf::from("in.mp4"));
    assert_eq!(trial.track, None);
    assert!(command().try_get_matches_from(["visual-bench"]).is_err());
}
