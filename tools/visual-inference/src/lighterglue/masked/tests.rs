use super::*;

#[test]
fn padding_has_zero_descriptors_and_excludes_only_unused_keys() {
    let mut features = Features::empty(64);
    features
        .push([1.0, 2.0], [3.0, 4.0], 1.0, &[0.125; 64])
        .unwrap();
    let pixels = [0.2, 0.3];
    let exact = Padded::new(&features, &pixels, 1).unwrap();
    assert!(matches!(exact.pixels, Cow::Borrowed(_)));
    assert!(matches!(exact.descriptors, Cow::Borrowed(_)));
    let padded = Padded::new(&features, &pixels, 3).unwrap();
    assert_eq!(padded.bias, [0.0, -10000.0, -10000.0]);
    assert_eq!(&padded.pixels[..2], &pixels);
    assert!(padded.descriptors[64..].iter().all(|v| *v == 0.0));
    assert!(Padded::new(&features, &pixels, 0).is_err());
    assert!(Padded::new(&features, &pixels[..1], 3).is_err());
}
