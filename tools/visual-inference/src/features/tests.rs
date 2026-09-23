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
