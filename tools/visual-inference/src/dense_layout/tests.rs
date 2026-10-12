use super::*;
#[test]
fn layout_rounds_internal_height_down_to_multiple_of_32() {
    assert_eq!(
        DenseLayout::from_shape(&[1, 1, 600, 800]).unwrap().grid(),
        [72, 100]
    );
    assert_eq!(
        DenseLayout::from_shape(&[1, 3, 1200, 1600]).unwrap().grid(),
        [148, 200]
    );
    for shape in [
        [1, 1, 1184, 1600],
        [1, 2, 1200, 1600],
        [2, 1, 1200, 1600],
        [1, 1, 2400, 3200],
    ] {
        assert!(DenseLayout::from_shape(&shape).is_err());
    }
}
