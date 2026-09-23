use super::*;
#[test]
fn range_and_digest_must_both_match() {
    let chunks = BTreeMap::from([("part".into(), b"prefix-pixels-tail".to_vec())]);
    let mut asset = Asset {
        chunk: "part".into(),
        offset: 7,
        length: 6,
        sha256: digest(b"pixels"),
    };
    assert!(verify_asset(&asset, &chunks).is_ok());
    asset.offset = 6;
    assert!(verify_asset(&asset, &chunks).is_err());
    asset.offset = u64::MAX;
    assert!(verify_asset(&asset, &chunks).is_err());
    asset.offset = 7;
    asset.length = usize::MAX;
    assert!(verify_asset(&asset, &chunks).is_err());
}
