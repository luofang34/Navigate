use super::*;

#[test]
fn command_is_consistent() {
    command().debug_assert();
}

#[test]
fn defaults_and_value_names_match_the_documented_interface() {
    let paths = [
        "--library",
        "l",
        "--model",
        "m",
        "--inputs",
        "i",
        "--output",
        "o",
    ];
    let matches = command()
        .try_get_matches_from(["probe"].into_iter().chain(paths))
        .expect("valid arguments");
    let args = Args::from_matches(matches).expect("arguments");
    assert_eq!(args.device_id, 0);
    assert_eq!(args.workspace_mib, 256);
    assert!(!args.save_outputs);
    assert!(matches!(args.provider, Provider::Cpu));
    assert_eq!((args.repetitions, args.threads), (10, 4));
    let matches = command()
        .try_get_matches_from(["probe"].into_iter().chain(paths).chain([
            "--provider",
            "tensor-rt",
            "--save-outputs",
            "--device-id",
            "1",
        ]))
        .expect("valid arguments");
    let args = Args::from_matches(matches).expect("arguments");
    assert!(matches!(args.provider, Provider::TensorRt));
    assert!(args.save_outputs);
    assert_eq!(args.device_id, 1);
    assert!(
        command()
            .try_get_matches_from(["probe", "--threads", "0"])
            .is_err()
    );
}
