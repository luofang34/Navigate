#![allow(clippy::expect_used)]
use super::DataUri;
#[test]
fn storage_names_cannot_escape_the_root() {
    for name in [
        "file:///etc/passwd",
        "pilotage://../secret",
        "pilotage://maps//tile",
        "pilotage://maps/%2e%2e",
        "pilotage://maps/\\tile",
        "file://chunks/abc.bin",
        "Pilotage://chunks/abc.bin",
        "://chunks/abc.bin",
        "chunks/abc.bin",
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

#[test]
fn any_host_scheme_names_a_storage_root() {
    let uri = DataUri::parse("navdata://chunks/abc.bin").expect("host scheme");
    assert_eq!(uri.scheme(), "navdata");
    assert_eq!(uri.relative_path(), "chunks/abc.bin");
    let uri = DataUri::parse("pilotage://chunks/abc.bin").expect("host scheme");
    assert_eq!(uri.scheme(), "pilotage");
}
