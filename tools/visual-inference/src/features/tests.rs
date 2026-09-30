use super::*;
#[test]
fn mutual_matches_retain_pixel_coordinates_and_reject_ambiguity() -> Result<(), InferenceError> {
    let mut a = Features::empty(2);
    let mut b = Features::empty(2);
    a.push([12.0, 20.0], [12.0, 20.0], 1.0, &[1.0, 0.0])?;
    a.push([30.0, 40.0], [30.0, 40.0], 1.0, &[0.0, 1.0])?;
    b.push([17.0, 27.0], [17.0, 27.0], 1.0, &[1.0, 0.0])?;
    b.push([35.0, 47.0], [35.0, 47.0], 1.0, &[0.0, 1.0])?;
    let pairs = mutual(&a, &b);
    assert_eq!(pairs.len(), 2);
    for p in pairs {
        assert_eq!(p.query - p.reference, nalgebra::Vector2::new(5.0, 7.0));
    }
    b.push([80.0, 80.0], [80.0, 80.0], 1.0, &[1.0, 0.0])?;
    assert_eq!(mutual(&a, &b).len(), 1);
    assert!(mutual(&a, &Features::empty(2)).is_empty());
    assert!(
        a.push([1.0, 1.0], [1.0, 1.0], 1.0, &[f32::NAN, 0.0])
            .is_err()
    );
    Ok(())
}

#[test]
fn invalid_device_descriptors_do_not_partially_append_a_feature() -> Result<(), InferenceError> {
    let mut features = Features::empty(2);
    features.push([12.0, 20.0], [12.0, 20.0], 1.0, &[3.0, 4.0])?;
    for descriptor in [[0.0, 0.0], [f32::NAN, 0.0], [f32::INFINITY, 0.0]] {
        assert!(
            features
                .push([30.0, 40.0], [30.0, 40.0], 0.5, &descriptor)
                .is_err()
        );
        assert_eq!(features.pixels, [[12.0, 20.0]]);
        assert_eq!(features.model_pixels, [12.0, 20.0]);
        assert_eq!(features.scores, [1.0]);
        assert_eq!(features.descriptors, [0.6, 0.8]);
    }
    features.push([30.0, 40.0], [30.0, 40.0], 0.5, &[0.0, 2.0])?;
    assert_eq!(features.descriptors, [0.6, 0.8, 0.0, 1.0]);
    assert_eq!(features.pixels.len(), 2);
    Ok(())
}
