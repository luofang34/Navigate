use super::DataUri;
#[test]
fn storage_names_cannot_escape_the_root() {
    for name in [
        "file:///etc/passwd",
        "pilotage://../secret",
        "pilotage://maps//tile",
        "pilotage://maps/%2e%2e",
        "pilotage://maps/\\tile",
    ] {
        assert!(DataUri::parse(name).is_err());
    }
    assert_eq!(
        DataUri::parse("pilotage://chunks/abc.bin")
            .map(|uri| uri.relative_path().to_owned())
            .ok(),
        Some("chunks/abc.bin".into())
    );
}
