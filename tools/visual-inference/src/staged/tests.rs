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

#[test]
fn equivalence_checks_feature_payload_not_only_image_and_encoder() {
    let original = feature_packet();
    let mut second = feature_packet();
    assert!(original == second);
    second.features.descriptors[0] = 0.25;
    assert!(original != second);
    second = feature_packet();
    second.features.pixels[0][0] += 1.0;
    assert!(original != second);
}

#[cfg(target_os = "macos")]
mod masked;

#[test]
fn window_selection_preserves_parent_identity_and_all_record_bits() {
    let mut source = feature_packet();
    source
        .features
        .push([80.0, 60.0], [640.0, 450.0], 0.7, &[0.125; 64])
        .unwrap();
    source = XFeatFeatures::new(
        source.contract,
        source.image_digest,
        source.image_size,
        source.features,
    );
    let selected = source.select_window([0, 0, 50, 40]).unwrap();
    assert_eq!(selected.len(), 1);
    assert_eq!(selected.encode(), feature_packet().encode());
    assert_eq!(selected.image_digest, source.image_digest);
    assert_eq!(selected.image_size, source.image_size);
    assert_eq!(
        XFeatFeatures::decode(&selected.encode(), 2048)
            .unwrap()
            .encode(),
        selected.encode()
    );
    assert!(source.select_window([0, 0, 10, 12]).unwrap().is_empty());
    assert_eq!(
        source.select_window([0, 0, 100, 80]).unwrap().encode(),
        source.encode()
    );
    for window in [
        [0, 0, 0, 20],
        [99, 0, 2, 2],
        [0, 79, 1, 2],
        [u32::MAX, 0, 2, 2],
    ] {
        assert!(source.select_window(window).is_err());
    }
}
