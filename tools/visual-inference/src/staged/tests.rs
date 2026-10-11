use super::*;

fn feature_packet() -> XFeatFeatures {
    let mut f = Features::empty(64);
    f.push([10.0, 12.0], [20.0, 24.0], 0.8, &[0.125; 64])
        .unwrap();
    XFeatFeatures::new([1; 32], [2; 32], [100, 80], f)
}
#[test]
fn packet_round_trip_preserves_descriptor_bits() {
    let original = feature_packet();
    let bytes = original.encode();
    let restored = XFeatFeatures::decode(&bytes, 512).unwrap();
    assert_eq!(restored.encode(), bytes);
    assert_eq!(restored.pixels(), original.pixels());
    assert_eq!(restored.descriptors(), original.descriptors());
    assert_eq!(restored.contract(), original.contract());
}
#[test]
fn packet_rejects_truncation_corruption_and_excess_count() {
    let mut bytes = feature_packet().encode();
    assert!(XFeatFeatures::decode(&bytes[..bytes.len() - 1], 512).is_err());
    bytes[100] ^= 1;
    assert!(XFeatFeatures::decode(&bytes, 512).is_err());
    let mut f = feature_packet();
    for _ in 0..32 {
        f.features
            .push([10.0, 12.0], [20.0, 24.0], 0.8, &[0.125; 64])
            .unwrap();
    }
    assert!(XFeatFeatures::decode(&f.encode(), 32).is_err());
}
#[test]
fn checksum_does_not_admit_non_finite_descriptors() {
    let mut f = feature_packet();
    f.features.descriptors[0] = f32::NAN;
    assert!(XFeatFeatures::decode(&f.encode(), 512).is_err());
}
#[test]
fn empty_packet_is_a_valid_absence_of_evidence() {
    let f = XFeatFeatures::new([1; 32], [2; 32], [100, 80], Features::empty(64));
    assert!(XFeatFeatures::decode(&f.encode(), 512).unwrap().is_empty());
}
