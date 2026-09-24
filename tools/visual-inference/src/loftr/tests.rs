#![allow(clippy::expect_used)]
use super::*;
#[test]
fn padding_and_backend_scores_do_not_become_geometry_evidence() {
    let image = GrayImage::new(960, 540);
    let input = preprocessing::prepare(&image, 640, 480, 1).expect("input");
    let points = [100.0, 10.0, 320.0, 240.0, 350.0, 250.0];
    let pairs = decode(
        &points,
        &points,
        &[0.9, 0.8, 0.1],
        [&input, &input],
        [&image, &image],
    )
    .expect("matches");
    assert_eq!(pairs.len(), 1);
    assert!((pairs[0].reference.x - 480.25).abs() < 1e-9);
    assert!((pairs[0].query.y - 270.25).abs() < 1e-9);
    assert!(decode(&[1.0], &[], &[], [&input, &input], [&image, &image]).is_err());
    assert!(
        decode(
            &[f32::NAN, 1.0],
            &[1.0, 1.0],
            &[1.0],
            [&input, &input],
            [&image, &image]
        )
        .is_err()
    );
}
