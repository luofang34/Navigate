use std::vec::Vec;

use super::HistoryRing;

fn contents<const N: usize>(ring: &HistoryRing<u32, N>) -> Vec<u32> {
    ring.iter().copied().collect()
}

#[test]
fn the_oldest_entries_go_at_the_bound() {
    let mut ring = HistoryRing::<u32, 4>::new();
    for value in 0..10 {
        ring.push_back(value, 3);
    }
    assert_eq!(contents(&ring), [7, 8, 9]);
    assert_eq!(ring.front(), Some(&7));
    // A bound above the inline capacity is clamped to it.
    for value in 10..20 {
        ring.push_back(value, 100);
    }
    assert_eq!(contents(&ring), [16, 17, 18, 19]);
    assert_eq!(ring.len(), 4);
}

#[test]
fn a_zero_bound_keeps_nothing() {
    let mut ring = HistoryRing::<u32, 4>::new();
    ring.push_back(1, 0);
    ring.insert(0, 2, 0);
    assert_eq!(ring.len(), 0);
    assert_eq!(ring.front(), None);
}

#[test]
fn an_insert_moves_the_later_entries_across_the_wrap() {
    let mut ring = HistoryRing::<u32, 5>::new();
    for value in [10, 20, 30, 40, 50, 60, 70] {
        ring.push_back(value, 5);
    }
    assert_eq!(contents(&ring), [30, 40, 50, 60, 70]);
    // A full ring drops its oldest entry to make room.
    ring.insert(2, 55, 5);
    assert_eq!(contents(&ring), [40, 55, 50, 60, 70]);
    // At position zero of a full ring the new entry is the oldest, so it
    // goes at once.
    ring.insert(0, 1, 5);
    assert_eq!(contents(&ring), [40, 55, 50, 60, 70]);
    ring.clear();
    ring.insert(0, 3, 5);
    ring.insert(0, 2, 5);
    ring.insert(5, 4, 5);
    assert_eq!(contents(&ring), [2, 3, 4]);
    if let Some(entry) = ring.get_mut(1) {
        *entry = 33;
    }
    assert_eq!(contents(&ring), [2, 33, 4]);
    assert_eq!(ring.partition_point(|value| *value < 10), 1);
    assert_eq!(ring.get(3), None);
}
