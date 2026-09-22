#![allow(clippy::expect_used)]

use super::*;

#[test]
fn digest_changes_when_source_content_changes() {
    assert_ne!(digest(b"one"), digest(b"two"));
    assert_eq!(
        digest(b"abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}

#[test]
fn archive_escape_is_rejected_before_file_access() {
    for path in ["../other.png", "/other.png", "./other.png"] {
        let artifact = Artifact {
            path: path.into(),
            sha256: "0".repeat(64),
        };
        assert!(matches!(
            load_image_blocking(Path::new("unused"), &artifact),
            Err(BenchError::Package { .. })
        ));
    }
}

#[test]
fn verified_selection_detects_changed_assets_and_manifest() {
    let directory = tempfile::tempdir().expect("temporary directory");
    crate::fixture::prepare_blocking(directory.path()).expect("write fixture");
    let first = MapPackage::open_blocking(directory.path()).expect("valid package");
    assert_eq!(first.tiles.len(), 25);
    let manifest_path = directory.path().join("map.json");
    let mut manifest = first.manifest;
    manifest.release_id = "another-selection".into();
    std::fs::write(
        &manifest_path,
        serde_json::to_vec(&manifest).expect("serialize"),
    )
    .expect("write manifest");
    let second = MapPackage::open_blocking(directory.path()).expect("changed selection");
    assert_ne!(
        first.revision.manifest_sha256,
        second.revision.manifest_sha256
    );
    let path = directory.path().join(&manifest.tiles[0].imagery.path);
    std::fs::write(&path, b"changed contents").expect("change source bytes");
    assert!(
        matches!(MapPackage::open_blocking(directory.path()), Err(BenchError::Package { reason })
        if reason.contains("digest mismatch"))
    );
}
