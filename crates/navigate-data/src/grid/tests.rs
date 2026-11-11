use super::*;

#[test]
fn cells_do_not_depend_on_a_mission_anchor() {
    let a = GridCell::containing([41.7001, -0.9]).unwrap();
    let b = GridCell::containing([41.7002, -0.9001]).unwrap();
    assert_eq!(a, b);
    assert_eq!(a.0.zoom, CORRELATION_ZOOM);
    let fine = TileAddress::containing([41.7001, -0.9], 19).unwrap();
    assert_eq!(GridCell::of_tile(fine), Some(a));
    assert_eq!(GridCell::of_tile(fine.ancestor(4)), None);
    assert_eq!(a.id(), alloc::format!("16/{}/{}", a.0.x, a.0.y));
}

#[test]
fn a_cell_is_near_five_hundred_metres_at_mid_latitude() {
    let side = crate::tile::ground_sample_distance_m(41.0, CORRELATION_ZOOM, 1);
    assert!((440.0..480.0).contains(&side), "{side}");
}
