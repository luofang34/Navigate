use super::*;

#[test]
fn cells_do_not_depend_on_a_mission_anchor() {
    let a = GridCell::containing([41.7001, -0.9], 16).unwrap();
    let b = GridCell::containing([41.7002, -0.9001], 16).unwrap();
    assert_eq!(a, b);
    let fine = TileAddress::containing([41.7001, -0.9], 19).unwrap();
    assert_eq!(GridCell::of_tile(fine, 16), Some(a));
    assert_eq!(GridCell::of_tile(fine.ancestor(4), 16), None);
    assert!(GridCell::containing([85.2, 0.0], 16).is_err());
}

#[test]
fn a_cell_shrinks_with_latitude() {
    let side = |lat| GridCell::side_m(lat, DEFAULT_CORRELATION_ZOOM);
    assert!((side(0.0) - 611.5).abs() < 0.1, "{}", side(0.0));
    assert!((440.0..480.0).contains(&side(41.0)));
    assert!((side(60.0) - 305.7).abs() < 0.1, "{}", side(60.0));
}
