use super::*;
use crate::MemoryStore;
use crate::{ReadPort, WritePort};

#[cfg(test)]
fn sample() -> BTreeMap<IndexKey, IndexEntry> {
    let mut entries = BTreeMap::new();
    for i in 0..3_000_u32 {
        let a = TileAddress::new(17, 40_000 + i % 300, 48_000 + i / 300).unwrap();
        let entry = if i % 7 == 0 {
            IndexEntry::Gap
        } else {
            IndexEntry::Tile {
                blob: Sha256Digest::of(&i.to_le_bytes()),
                bytes: 100 + u64::from(i),
            }
        };
        entries.insert(key(a), entry);
    }
    entries
}

#[test]
fn an_index_verifies_by_ranges_and_finds_each_record() {
    let entries = sample();
    let bytes = encode_index(&entries);
    assert_eq!(bytes.len(), 8 + entries.len() * RECORD_BYTES);
    let mut store = MemoryStore::default();
    pollster::block_on(store.write("chunks/i", &bytes)).unwrap();
    let file = pollster::block_on(store.open("chunks/i")).unwrap().unwrap();
    let counts = pollster::block_on(verify_index(&file, Sha256Digest::of(&bytes), "i")).unwrap();
    assert_eq!(counts.gaps, 3_000_u64.div_ceil(7));
    assert_eq!(counts.tiles + counts.gaps, 3_000);
    for (&(zoom, y, x), entry) in entries.iter().step_by(97) {
        let a = TileAddress::new(zoom, x, y).unwrap();
        let found = pollster::block_on(find(&file, 3_000, a, "i")).unwrap();
        assert_eq!(found, Some(*entry));
    }
    let outside = TileAddress::new(17, 1, 1).unwrap();
    assert_eq!(
        pollster::block_on(find(&file, 3_000, outside, "i")).unwrap(),
        None
    );
}

#[test]
fn a_changed_or_unordered_index_is_refused() {
    let bytes = encode_index(&sample());
    let mut store = MemoryStore::default();
    let mut changed = bytes.clone();
    if let Some(b) = changed.get_mut(100) {
        *b ^= 1;
    }
    store.replace("chunks/changed", &changed);
    let file = pollster::block_on(store.open("chunks/changed"))
        .unwrap()
        .unwrap();
    assert!(pollster::block_on(verify_index(&file, Sha256Digest::of(&bytes), "c")).is_err());
    let mut swapped = bytes[..8].to_vec();
    swapped.extend_from_slice(&bytes[8 + RECORD_BYTES..8 + 2 * RECORD_BYTES]);
    swapped.extend_from_slice(&bytes[8..8 + RECORD_BYTES]);
    store.replace("chunks/swapped", &swapped);
    let file = pollster::block_on(store.open("chunks/swapped"))
        .unwrap()
        .unwrap();
    let refused = pollster::block_on(verify_index(&file, Sha256Digest::of(&swapped), "s"));
    assert!(
        matches!(refused, Err(DataError::Integrity { .. })),
        "{refused:?}"
    );
}
