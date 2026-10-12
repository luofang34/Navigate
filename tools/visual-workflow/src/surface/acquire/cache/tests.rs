use super::*;
#[test]
fn access_promotes_entries_and_total_budget_can_evict_below_cache_limit() {
    let mut cache = Cache::new(12);
    cache.insert([1; 32], vec![1_u8; 4], 4);
    cache.insert([2; 32], vec![2; 4], 4);
    cache.insert([3; 32], vec![3; 4], 4);
    let (data, bytes) = cache.take_cached(&[1; 32]).unwrap();
    assert_eq!(data, vec![1; 4]);
    cache.insert([1; 32], data, bytes);
    cache.trim(8);
    assert!(cache.take_cached(&[2; 32]).is_none());
    assert_eq!(cache.evictions, 1);
    cache.trim(4);
    assert!(cache.take_cached(&[3; 32]).is_none());
    assert_eq!(cache.take_cached(&[1; 32]).unwrap().0, vec![1; 4]);
    assert_eq!(cache.bytes, 0);
}
#[test]
fn oversized_entries_are_not_retained_and_exact_keys_do_not_alias() {
    let mut cache = Cache::new(4);
    cache.insert([1; 32], vec![9_u8; 5], 5);
    assert_eq!(cache.bytes, 0);
    cache.insert([2; 32], vec![2; 4], 4);
    assert!(cache.take_cached(&[3; 32]).is_none());
    assert_eq!(cache.take_cached(&[2; 32]).unwrap().0, vec![2; 4]);
    assert!(Cache::<Vec<u8>>::entry_bytes(4) > 4);
}

#[test]
fn feature_keys_bind_source_contract_crop_rotation_and_actual_pixels() {
    let image = image::GrayImage::new(32, 40);
    let window = Some([10, 20, 32, 40]);
    let key = fingerprint([1; 32], [2; 32], &image, window);
    assert_eq!(key, fingerprint([1; 32], [2; 32], &image, window));
    assert_ne!(key, fingerprint([3; 32], [2; 32], &image, window));
    assert_ne!(key, fingerprint([1; 32], [3; 32], &image, window));
    assert_eq!(rotation_key(key, 0), rotation_key(key, 0));
    for rotation in 1..4 {
        assert_ne!(rotation_key(key, 0), rotation_key(key, rotation));
    }
    assert_ne!(
        key,
        fingerprint([1; 32], [2; 32], &image, Some([11, 20, 32, 40]))
    );
    let mut changed = image.clone();
    changed.put_pixel(4, 8, image::Luma([1]));
    assert_ne!(key, fingerprint([1; 32], [2; 32], &changed, window));
    assert_ne!(
        key,
        fingerprint([1; 32], [2; 32], &image::GrayImage::new(40, 32), window)
    );
}
