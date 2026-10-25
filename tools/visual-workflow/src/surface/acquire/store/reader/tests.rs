use super::*;
use serde_json::json;
fn packet(contract: [u8; 32]) -> Vec<u8> {
    let mut bytes = b"NAVXFT01".to_vec();
    bytes.extend(contract);
    bytes.extend([0; 32]);
    for n in [32_u32, 32, 0] {
        bytes.extend(n.to_le_bytes());
    }
    bytes.extend(Sha256::digest(&bytes));
    bytes
}
fn fixture(root: &Path, bytes: &[u8]) -> PathBuf {
    let index = root.join("catalog.json");
    let source = [1_u8; 32];
    let contract = [2_u8; 32];
    let key = [3_u8; 32];
    let catalog = json!({"version":1,"source":source,"contract":contract,"include_fine":false,
        "entries":[{"key":key,"sha256":Sha256::digest(bytes).to_vec(),"bytes":bytes.len()}]});
    std::fs::write(&index, serde_json::to_vec(&catalog).unwrap()).unwrap();
    std::fs::write(root.join(format!("{}.navfeat", hex(&[3; 32]))), bytes).unwrap();
    index
}
#[test]
fn real_packet_read_and_absent_key_have_distinct_outcomes() {
    let temp = tempfile::tempdir().unwrap();
    let bytes = packet([2; 32]);
    let index = fixture(temp.path(), &bytes);
    let reader = Reader::open_blocking(&index, [1; 32], [2; 32]).unwrap();
    assert!(reader.load_blocking([4; 32], 32).unwrap().is_none());
    let loaded = reader.load_blocking([3; 32], 32).unwrap().unwrap();
    assert!(loaded.features.is_empty());
    assert_eq!(loaded.file_bytes, bytes.len());
    assert_eq!(loaded.buffer_bytes, bytes.len());
    assert!(reader.resident_bytes() > size_of::<Reader>());
    std::fs::remove_file(loaded.path).unwrap();
    assert!(matches!(
        reader.load_blocking([3; 32], 32),
        Err(Error::Io { .. })
    ));
}
#[test]
fn corrupt_truncated_trailing_and_foreign_packets_are_errors_not_misses() {
    let temp = tempfile::tempdir().unwrap();
    let bytes = packet([2; 32]);
    let index = fixture(temp.path(), &bytes);
    let reader = Reader::open_blocking(&index, [1; 32], [2; 32]).unwrap();
    let path = temp.path().join(format!("{}.navfeat", hex(&[3; 32])));
    for bad in [
        vec![0; bytes.len()],
        bytes[..bytes.len() - 1].to_vec(),
        [bytes.clone(), vec![0]].concat(),
    ] {
        std::fs::write(&path, bad).unwrap();
        assert!(reader.load_blocking([3; 32], 32).is_err());
    }
    fixture(temp.path(), &packet([9; 32]));
    let reader = Reader::open_blocking(&index, [1; 32], [2; 32]).unwrap();
    assert!(reader.load_blocking([3; 32], 32).is_err());
}
#[test]
fn index_rejects_foreign_identity_duplicate_keys_and_oversized_entries() {
    let temp = tempfile::tempdir().unwrap();
    let index = fixture(temp.path(), &packet([2; 32]));
    assert!(Reader::open_blocking(&index, [9; 32], [2; 32]).is_err());
    assert!(Reader::open_blocking(&index, [1; 32], [9; 32]).is_err());
    let base: Value = serde_json::from_slice(&std::fs::read(&index).unwrap()).unwrap();
    let mut bad = base.clone();
    bad["entries"] = json!([base["entries"][0].clone(), base["entries"][0].clone()]);
    std::fs::write(&index, serde_json::to_vec(&bad).unwrap()).unwrap();
    assert!(Reader::open_blocking(&index, [1; 32], [2; 32]).is_err());
    let mut bad = base;
    bad["entries"][0]["bytes"] = json!(usize::MAX);
    std::fs::write(&index, serde_json::to_vec(&bad).unwrap()).unwrap();
    assert!(Reader::open_blocking(&index, [1; 32], [2; 32]).is_err());
}
