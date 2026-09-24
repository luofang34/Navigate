#![allow(clippy::expect_used)]
use super::StorageClass;
use crate::DataUri;

#[test]
fn every_class_round_trips_through_its_scheme() {
    for class in StorageClass::ALL {
        assert_eq!(StorageClass::from_scheme(class.scheme()), Some(class));
        let uri = DataUri::in_class(class, "packages/terrain/manifest.json").expect("valid");
        assert_eq!(uri.storage_class(), Some(class));
        assert_eq!(uri.relative_path(), "packages/terrain/manifest.json");
    }
    assert_eq!(StorageClass::from_scheme("pilotage"), None);
    let host = DataUri::parse("pilotage://chunks/a.bin").expect("host scheme");
    assert_eq!(host.storage_class(), None);
}

#[test]
fn a_class_uri_refuses_an_unsafe_path() {
    assert!(DataUri::in_class(StorageClass::Offline, "../escape").is_err());
    assert!(DataUri::in_class(StorageClass::Offline, "").is_err());
}
