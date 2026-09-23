#![allow(clippy::expect_used)]
use super::FileStore;
use navigate_data::{DataStore, DataUri};
#[tokio::test]
async fn exact_ranges_do_not_share_a_cursor() {
    let dir = tempfile::tempdir().expect("temporary directory");
    tokio::fs::write(dir.path().join("test"), b"0123456789")
        .await
        .expect("write fixture");
    let store = FileStore::new(dir.path()).await.expect("open store");
    let uri = DataUri::parse("pilotage://test").expect("logical URI");
    let file = store.open(&uri).await.expect("open resource");
    let (a, b) = tokio::join!(file.read_at(5, 3), file.read_at(1, 2));
    assert_eq!(a.expect("first range"), b"567");
    assert_eq!(b.expect("second range"), b"12");
    assert!(file.read_at(10, 0).await.expect("empty range").is_empty());
    assert!(file.read_at(10, 1).await.is_err());
    tokio::fs::write(dir.path().join("test"), b"0")
        .await
        .expect("truncate fixture");
    assert!(file.read_at(5, 3).await.is_err());
}
#[cfg(unix)]
#[tokio::test]
async fn escaping_symlinks_are_rejected() {
    let root = tempfile::tempdir().expect("root");
    let other = tempfile::tempdir().expect("other");
    tokio::fs::write(other.path().join("secret"), b"unrelated")
        .await
        .expect("fixture");
    std::os::unix::fs::symlink(other.path().join("secret"), root.path().join("link"))
        .expect("symlink");
    let store = FileStore::new(root.path()).await.expect("store");
    let uri = DataUri::parse("pilotage://link").expect("URI");
    assert!(store.open(&uri).await.is_err());
}
