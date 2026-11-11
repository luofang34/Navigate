use super::*;
use alloc::vec::Vec;

/// Mean radius of the Earth in metres. A distance on this sphere is an
/// angle smaller than the same distance on the polar-radius sphere.
const MEAN_RADIUS_M: f64 = 6_371_008.8;

/// A small deterministic generator, so the property test needs no crate.
#[cfg(test)]
struct Lcg(u64);

#[cfg(test)]
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 11) as f64 / (1_u64 << 53) as f64
    }
}

/// The point at angular distance `angle` from `p` along `bearing` (radians).
#[cfg(test)]
fn destination([lat, lon]: [f64; 2], bearing: f64, angle: f64) -> [f64; 2] {
    let (lat, lon) = (lat.to_radians(), lon.to_radians());
    let lat2 = (lat.sin() * angle.cos() + lat.cos() * angle.sin() * bearing.cos()).asin();
    let lon2 =
        lon + (bearing.sin() * angle.sin() * lat.cos()).atan2(angle.cos() - lat.sin() * lat2.sin());
    [lat2.to_degrees(), lon2.to_degrees()]
}

#[cfg(test)]
fn bearing([lat1, lon1]: [f64; 2], [lat2, lon2]: [f64; 2]) -> f64 {
    let (p1, p2, dl) = (
        lat1.to_radians(),
        lat2.to_radians(),
        (lon2 - lon1).to_radians(),
    );
    (dl.sin() * p2.cos()).atan2(p1.cos() * p2.sin() - p1.sin() * p2.cos() * dl.cos())
}

#[cfg(test)]
fn select(route: &[[f64; 2]], radius_m: f64, zoom: u32) -> Vec<TileAddress> {
    let mut tiles = BTreeSet::new();
    corridor_tiles(route, radius_m, zoom, &mut tiles).unwrap();
    tiles.into_iter().collect()
}

#[cfg(test)]
fn contains(tiles: &[TileAddress], point: [f64; 2], zoom: u32) -> bool {
    tiles
        .binary_search(&TileAddress::containing(point, zoom).unwrap())
        .is_ok()
}

/// Every random point within the radius of a leg is in a selected tile.
#[cfg(test)]
fn assert_covers(route: &[[f64; 2]], radius_m: f64, zoom: u32) {
    let tiles = select(route, radius_m, zoom);
    let mut random = Lcg(7);
    for pair in route.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        for _ in 0..2_000 {
            let on_leg = point_along(a, b, random.next());
            let side = if random.next() < 0.5 { 1.0 } else { -1.0 };
            let course = bearing(on_leg, b);
            let offset = 0.999 * radius_m * random.next() / MEAN_RADIUS_M;
            let point = destination(on_leg, course + side * core::f64::consts::FRAC_PI_2, offset);
            assert!(contains(&tiles, point, zoom), "{point:?} off {route:?}");
        }
    }
}

#[test]
fn a_corridor_with_a_turn_covers_every_point_within_its_radius() {
    let route = [
        [40.86, -75.18],
        [40.86, -74.94],
        [40.98, -74.94],
        [40.98, -75.10],
    ];
    assert_covers(&route, 1_500.0, 15);
    assert_covers(&route, 300.0, 17);
}

#[test]
fn a_corridor_across_the_antimeridian_wraps_its_columns() {
    let route = [[-17.0, 179.8], [-17.2, -179.7]];
    assert_covers(&route, 2_000.0, 14);
    let tiles = select(&route, 2_000.0, 14);
    assert!(tiles.iter().any(|t| t.x == 0) && tiles.iter().any(|t| t.x == (1 << 14) - 1));
}

#[test]
fn a_high_latitude_leg_is_covered_and_a_polar_one_refused() {
    assert_covers(&[[80.0, 10.0], [80.5, 25.0]], 3_000.0, 12);
    let mut tiles = BTreeSet::new();
    let refused = corridor_tiles(&[[84.9, 0.0], [84.9, 10.0]], 20_000.0, 10, &mut tiles);
    assert!(matches!(
        refused,
        Err(DataError::Invalid {
            field: "latitude",
            ..
        })
    ));
}

#[test]
fn far_points_are_not_selected() {
    let tiles = select(&[[40.9, -75.1], [40.9, -75.0]], 1_000.0, 15);
    assert!(!contains(&tiles, [40.9 - 0.05, -75.05], 15));
    assert!(tiles.len() < 200, "{}", tiles.len());
}

#[test]
fn unusable_requests_are_refused() {
    let mut tiles = BTreeSet::new();
    assert!(corridor_tiles(&[], 100.0, 15, &mut tiles).is_err());
    assert!(corridor_tiles(&[[0.0, 0.0]], -1.0, 15, &mut tiles).is_err());
    assert!(corridor_tiles(&[[f64::NAN, 0.0]], 100.0, 15, &mut tiles).is_err());
    assert_eq!(select(&[[0.0, 0.0]], 10.0, 3).len(), 4);
}
