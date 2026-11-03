use navigate_contract::GeodeticPosition;
use navigate_fusion::ScalarFieldMap;

use super::{BilinearGridMap, GridMapError};

#[test]
fn the_grid_map_interpolates_inside_and_refuses_outside() {
    let step = 1e-4;
    let values = [0.0, 1.0, 2.0, 10.0, 11.0, 12.0];
    let map = BilinearGridMap::new(0.8, step, 0.1, step, 2, 3, &values).expect("valid grid");
    let at = |lat: f64, lon: f64| map.sample(&GeodeticPosition::new(lat, lon, 0.0));
    let near =
        |value: Option<f64>, expected: f64| value.is_some_and(|v| (v - expected).abs() < 1e-9);
    assert!(near(at(0.8, 0.1), 0.0));
    assert!(near(at(0.8 + step, 0.1 + 2.0 * step), 12.0));
    let middle = at(0.8 + 0.5 * step, 0.1 + 1.5 * step).expect("inside");
    assert!((middle - 6.5).abs() < 1e-9, "{middle}");
    assert_eq!(at(0.8 - step, 0.1), None);
    assert_eq!(at(0.8, 0.1 + 3.0 * step), None);
    assert_eq!(
        BilinearGridMap::new(0.8, step, 0.1, step, 2, 3, &[0.0; 5]),
        Err(GridMapError { rows: 2, cols: 3 })
    );
    assert!(BilinearGridMap::new(0.8, -step, 0.1, step, 2, 2, &[0.0; 4]).is_err());
    assert!(BilinearGridMap::new(0.8, step, 0.1, step, 2, 2, &[0.0, f64::NAN, 0.0, 0.0]).is_err());
}

#[test]
fn a_borrowed_grid_drives_the_particle_filter_by_reference() {
    // The filter reads a map through `&impl ScalarFieldMap`, and a
    // reference to a map is also a map.
    let values = [5.0; 4];
    let map = BilinearGridMap::new(-0.1, 0.2, -0.1, 0.2, 2, 2, &values).expect("valid grid");
    let by_reference: &dyn ScalarFieldMap = &map;
    let origin = GeodeticPosition::new(0.0, 0.0, 0.0);
    assert_eq!((&by_reference).sample(&origin), Some(5.0));
}
