use super::*;

#[test]
fn lookup_retains_storage_and_refreshes_eviction_order() {
    let mut cache = Cache::new(2, 10);
    let first = cache.insert("a".into(), vec![1_u8; 4], 4, &[]).unwrap();
    cache.insert("b".into(), vec![2_u8; 4], 4, &[]).unwrap();
    let hit = cache.get_cached("a").unwrap();
    assert!(Arc::ptr_eq(&first, &hit));
    cache.insert("c".into(), vec![3_u8; 4], 4, &[]).unwrap();
    assert!(cache.get_cached("b").is_none());
    assert_eq!(cache.bytes(), 8);
}

#[test]
fn byte_budget_evicts_until_the_new_reference_fits() {
    let mut cache = Cache::new(8, 10);
    cache.insert("a".into(), 1, 4, &[]).unwrap();
    cache.insert("b".into(), 2, 4, &[]).unwrap();
    cache.insert("c".into(), 3, 7, &[]).unwrap();
    assert!(cache.get_cached("a").is_none());
    assert!(cache.get_cached("b").is_none());
    assert_eq!(cache.bytes(), 7);
}

#[test]
fn current_frame_references_cannot_be_evicted() {
    let mut cache = Cache::new(2, 8);
    cache.insert("a".into(), 1, 4, &[]).unwrap();
    cache.insert("b".into(), 2, 4, &[]).unwrap();
    let protected = ["a".into(), "b".into(), "c".into()];
    assert!(cache.insert("c".into(), 3, 4, &protected).is_err());
    assert_eq!(cache.bytes(), 8);
    assert_eq!(*cache.get_cached("a").unwrap(), 1);
    assert_eq!(*cache.get_cached("b").unwrap(), 2);
}

#[test]
fn oversized_entry_does_not_discard_existing_references() {
    let mut cache = Cache::new(2, 8);
    cache.insert("a".into(), 1, 4, &[]).unwrap();
    assert!(cache.insert("b".into(), 2, 9, &[]).is_err());
    assert_eq!(*cache.get_cached("a").unwrap(), 1);
}
