use super::*;

#[test]
fn a_circle_contains_its_centre_and_not_a_far_point() {
    let region = Region::Circle {
        centre: [40.9, -75.1],
        radius_m: 10_000.0,
    };
    assert!(region.contains([40.9, -75.1]));
    assert!(region.contains([40.95, -75.1]));
    assert!(!region.contains([41.1, -75.1]));
    assert!((region.area_km2() - 314.159).abs() < 0.01);
}

#[test]
fn a_polygon_contains_points_inside_its_outline_only() {
    let region = Region::Polygon {
        points: vec![[40.0, -75.0], [40.0, -74.0], [41.0, -74.0], [41.0, -75.0]],
    };
    assert!(region.contains([40.5, -74.5]));
    assert!(!region.contains([41.5, -74.5]));
    assert!(!region.contains([40.5, -73.5]));
    assert!(region.area_km2() > 9_000.0 && region.area_km2() < 10_000.0);
}

#[test]
fn the_cells_of_a_region_cover_its_bounds() {
    let region = Region::Circle {
        centre: [40.9, -75.1],
        radius_m: 30_000.0,
    };
    let cells = cells(&region);
    let [s, w, n, e] = region.bounds();
    for corner in [[s, w], [s, e], [n, w], [n, e]] {
        let [x, y] = tile_xy(corner, CELL_ZOOM);
        assert!(
            cells.contains(&Cell {
                x: x as u32,
                y: y as u32
            }),
            "{corner:?}"
        );
    }
    assert!(cells.len() <= 9, "{}", cells.len());
}
