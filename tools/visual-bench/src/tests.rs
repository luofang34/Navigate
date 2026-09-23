#[test]
fn renderer_revision_is_one_full_commit_hash() {
    let revision = crate::RENDERER_REVISION.trim();
    assert_eq!(revision.len(), 40, "{revision:?}");
    assert!(
        revision
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
    );
}
