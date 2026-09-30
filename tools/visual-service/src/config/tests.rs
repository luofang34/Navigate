use super::*;

#[test]
fn command_is_consistent() {
    command().debug_assert();
}

#[test]
fn defaults_match_the_documented_values() {
    let matches = command()
        .try_get_matches_from(["service", "--state", "/data", "--origin", "https://a.test"])
        .expect("valid arguments");
    let config = Config::from_matches(matches).expect("config");
    assert_eq!(config.state, PathBuf::from("/data"));
    assert_eq!(config.webapp, PathBuf::from("tools/visual-bench/webapp"));
    assert_eq!(config.origin, "https://a.test");
    assert_eq!(config.catalog, None);
    assert_eq!(
        config.bind,
        "127.0.0.1:8080".parse::<SocketAddr>().expect("address")
    );
}
