use super::*;
fn features(points: &[[f64; 2]]) -> Features {
    let mut f = Features::empty(2);
    for p in points {
        f.push(*p, [0.0, 0.0], 0.9, &[1.0, 0.0])
            .expect("valid descriptor");
    }
    f
}
#[test]
fn assignments_return_original_image_pixels_and_skip_unmatched_points() {
    let a = features(&[[12.5, 40.0], [50.0, 60.0], [90.0, 80.0]]);
    let b = features(&[[20.0, 30.0], [100.0, 110.0]]);
    let pairs = decode(&[1, -1, 0], &a, &b).expect("valid assignments");
    assert_eq!(pairs.len(), 2);
    assert_eq!(pairs[0].reference.x, 12.5);
    assert_eq!(pairs[0].query.y, 110.0);
    assert_eq!(pairs[1].reference.y, 80.0);
    assert_eq!(pairs[1].query.x, 20.0);
}
#[test]
fn malformed_and_non_mutual_assignments_are_errors() {
    let a = features(&[[12.0, 40.0], [90.0, 80.0]]);
    let b = features(&[[20.0, 30.0]]);
    for indices in [&[0][..], &[-2, -1], &[1, -1], &[0, 0]] {
        assert!(decode(indices, &a, &b).is_err());
    }
}
