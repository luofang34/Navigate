use super::*;

#[test]
fn the_oryol_area_matches_the_catalog_tiles() {
    // The Oryol catalog held z18 imagery columns 157320 through 157338 and rows
    // 85407 through 85424 around the rooftops.
    let grid = TileGrid::around([52.982, 36.0595], 800.0, 18);
    assert!(grid.x[0] <= 157_325 && grid.x[1] >= 157_334, "{grid:?}");
    assert!(grid.y[0] <= 85_410 && grid.y[1] >= 85_421, "{grid:?}");
    assert_eq!(grid.count(), grid.tiles().count());
}

#[test]
fn tile_coordinates_follow_web_mercator() {
    let [x, y] = tile_xy([0.0, 0.0], 1);
    assert!((x - 1.0).abs() < 1e-12 && (y - 1.0).abs() < 1e-12);
    let [x, _] = tile_xy([0.0, 180.0], 3);
    assert!((x - 8.0).abs() < 1e-9);
}
