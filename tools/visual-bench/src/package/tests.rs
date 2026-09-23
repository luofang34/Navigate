#![allow(clippy::expect_used)]

use super::*;

fn fixture() -> tempfile::TempDir {
    let directory = tempfile::tempdir().expect("temporary directory");
    crate::fixture::prepare_blocking(directory.path()).expect("write fixture");
    directory
}

fn source(directory: &Path) -> SourceManifest {
    serde_json::from_slice(&std::fs::read(directory.join("map.json")).expect("read"))
        .expect("parse")
}

fn write_source(directory: &Path, manifest: &SourceManifest) {
    std::fs::write(
        directory.join("map.json"),
        serde_json::to_vec(manifest).expect("serialize"),
    )
    .expect("write manifest");
}

#[test]
fn a_source_folder_reports_its_package_identity() {
    let directory = fixture();
    let first = MapPackage::open_blocking(directory.path()).expect("valid package");
    let imagery = first
        .tiles
        .iter()
        .filter(|tile| tile.imagery.is_some())
        .count();
    assert_eq!(imagery, 25);
    assert_eq!(first.revision.manifest_sha256, first.manifest.pack_id);
    assert_eq!(
        first.manifest.compute_pack_id().expect("digest"),
        first.manifest.pack_id
    );
    let mut manifest = source(directory.path());
    manifest.release_id = "another-selection".into();
    write_source(directory.path(), &manifest);
    let second = MapPackage::open_blocking(directory.path()).expect("changed selection");
    assert_ne!(
        first.revision.manifest_sha256,
        second.revision.manifest_sha256
    );
}

#[test]
fn changed_source_bytes_and_escaping_paths_are_refused() {
    let directory = fixture();
    let mut manifest = source(directory.path());
    let path = manifest.tiles[0]
        .imagery
        .as_ref()
        .or(manifest.tiles[0].elevation.as_ref())
        .expect("asset")
        .path
        .clone();
    std::fs::write(directory.path().join(&path), b"changed contents").expect("change bytes");
    assert!(MapPackage::open_blocking(directory.path()).is_err());
    let directory = fixture();
    if let Some(asset) = manifest.tiles[0].imagery.as_mut() {
        asset.path = "../other.png".into();
    }
    write_source(directory.path(), &manifest);
    assert!(MapPackage::open_blocking(directory.path()).is_err());
}

#[test]
fn independent_sources_require_version_two() {
    let directory = fixture();
    let mut manifest = source(directory.path());
    manifest.schema_version = 1;
    write_source(directory.path(), &manifest);
    assert!(MapPackage::open_blocking(directory.path()).is_err());
    manifest.schema_version = 2;
    write_source(directory.path(), &manifest);
    let package = MapPackage::open_blocking(directory.path()).expect("independent sources");
    assert!(
        package
            .tiles
            .iter()
            .any(|t| t.imagery.is_some() && t.elevation.is_none())
    );
    assert!(
        package
            .tiles
            .iter()
            .any(|t| t.imagery.is_none() && t.elevation.is_some())
    );
}

#[test]
fn a_published_package_opens_with_the_same_identity_as_its_source() {
    let directory = fixture();
    let built = MapPackage::open_blocking(directory.path()).expect("source package");
    let published = tempfile::tempdir().expect("published");
    let chunks = published.path().join("chunks");
    std::fs::create_dir_all(&chunks).expect("chunks");
    let source = source(directory.path());
    let root = directory.path().canonicalize().expect("root");
    source
        .build(
            LOCAL_SOURCE_REGION,
            |_, _, asset| read_source_asset_blocking(&root, &asset.path),
            |sha, bytes| {
                std::fs::write(chunks.join(format!("{sha}.bin")), bytes).expect("chunk");
                Ok::<(), BenchError>(())
            },
        )
        .expect("build");
    let manifest = published.path().join("package.json");
    std::fs::write(
        &manifest,
        serde_json::to_vec(&built.manifest).expect("json"),
    )
    .expect("write");
    let opened = MapPackage::open_blocking(&manifest).expect("published package");
    assert_eq!(opened.revision, built.revision);
    let mut tampered = built.manifest.clone();
    tampered.attribution = "changed".into();
    std::fs::write(&manifest, serde_json::to_vec(&tampered).expect("json")).expect("write");
    assert!(MapPackage::open_blocking(&manifest).is_err());
}
