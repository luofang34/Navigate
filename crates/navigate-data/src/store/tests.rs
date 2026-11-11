use super::*;
use alloc::string::ToString;

#[test]
fn paths_that_leave_the_store_are_refused() {
    for bad in ["", "/abs", "a//b", "a/./b", "../a", "a/..", "a\\b"] {
        assert!(checked_path(bad).is_err(), "{bad}");
    }
    assert!(checked_path("chunks/ab").is_ok());
}

#[test]
fn memory_store_keeps_the_first_write_and_reads_ranges() {
    let mut store = MemoryStore::default();
    pollster::block_on(store.write("chunks/a", b"one two")).unwrap();
    pollster::block_on(store.write("chunks/a", b"other")).unwrap();
    let file = pollster::block_on(store.open("chunks/a")).unwrap().unwrap();
    assert_eq!(file.len(), 7);
    assert_eq!(pollster::block_on(file.read_at(4, 3)).unwrap(), b"two");
    assert!(pollster::block_on(file.read_at(5, 3)).is_err());
    assert!(
        pollster::block_on(store.open("chunks/b"))
            .unwrap()
            .is_none()
    );
    store.replace("chunks/a", b"changed");
    assert_eq!(store.get("chunks/a"), Some(b"changed".as_slice()));
    assert!(store.remove("chunks/a"));
}

#[test]
fn directory_store_reads_back_what_it_wrote() {
    let dir = tempfile::tempdir().unwrap();
    let mut store = DirectoryStore::new(dir.path());
    let digest = Sha256Digest::of(b"tile");
    let path = blob_path(digest);
    assert!(pollster::block_on(store.open(&path)).unwrap().is_none());
    pollster::block_on(store.write(&path, b"tile")).unwrap();
    assert!(pollster::block_on(store.contains(&path)).unwrap());
    let file = pollster::block_on(store.open(&path)).unwrap().unwrap();
    assert_eq!(pollster::block_on(file.read_at(1, 2)).unwrap(), b"il");
    assert_eq!(
        pollster::block_on(read_all(&store, &path))
            .unwrap()
            .as_deref(),
        Some(b"tile".as_slice())
    );
    assert!(dir.path().join("chunks").join(digest.to_string()).is_file());
    assert!(pollster::block_on(store.open("../x")).is_err());
}

#[test]
fn digests_have_one_text_form() {
    let d = Sha256Digest::of(b"abc");
    let text = d.to_string();
    assert_eq!(
        text,
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
    assert_eq!(text.parse::<Sha256Digest>().unwrap(), d);
    assert!(text.to_uppercase().parse::<Sha256Digest>().is_err());
}
