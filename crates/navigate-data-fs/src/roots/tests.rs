#![allow(clippy::expect_used)]
use super::*;

fn roots(base: &Path) -> StorageRoots {
    StorageRoots::new(
        base.join("data"),
        base.join("offline"),
        base.join("cache"),
        base.join("config"),
        base.join("temp"),
    )
}

#[tokio::test]
async fn each_class_reads_from_its_own_root() {
    let base = tempfile::tempdir().expect("temp");
    let roots = roots(base.path());
    let store = ClassStore::new(&roots).await.expect("store");
    for class in StorageClass::ALL {
        std::fs::write(roots.root(class).join("item.bin"), class.scheme()).expect("write");
    }
    for class in StorageClass::ALL {
        let uri = DataUri::in_class(class, "item.bin").expect("uri");
        let file = store.open(&uri).await.expect("open");
        let bytes = file.read_at(0, file.len() as usize).await.expect("read");
        assert_eq!(bytes, class.scheme().as_bytes());
    }
}

#[tokio::test]
async fn a_host_scheme_is_not_a_storage_class() {
    let base = tempfile::tempdir().expect("temp");
    let store = ClassStore::new(&roots(base.path())).await.expect("store");
    let uri = DataUri::parse("pilotage://item.bin").expect("uri");
    assert!(matches!(
        store.open(&uri).await,
        Err(DataError::InvalidUri { .. })
    ));
}

#[test]
fn platform_roots_keep_offline_apart_from_data_and_cache() {
    let roots = StorageRoots::for_platform("Luofang", "Pilotage").expect("home directory");
    let offline = roots.root(StorageClass::Offline);
    assert!(offline.ends_with("Offline"));
    assert_ne!(offline, roots.root(StorageClass::Cache));
    assert_ne!(offline, roots.root(StorageClass::Data));
}
