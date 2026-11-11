use super::*;

#[test]
fn containing_tile_matches_the_published_tile_of_a_known_place() {
    // Delaware Water Gap, zoom 15, from the XYZ equations in double precision.
    let t = TileAddress::containing([40.97, -75.14], 15).unwrap();
    assert_eq!((t.x, t.y), (9544, 12289));
    let [lat, lon] = t.north_west();
    assert!(lat >= 40.97 && lon <= -75.14, "{lat} {lon}");
    let [x, y] = tile_xy([lat, lon], 15);
    assert!((x - 9544.0).abs() < 1e-9 && (y - 12289.0).abs() < 1e-9);
}

#[test]
fn columns_wrap_at_the_antimeridian_and_rows_stop_at_the_limit() {
    assert_eq!(TileAddress::containing([0.0, 180.0], 3).unwrap().x, 0);
    assert_eq!(TileAddress::containing([0.0, -180.0], 3).unwrap().x, 0);
    assert_eq!(TileAddress::containing([0.0, 179.99], 3).unwrap().x, 7);
    assert_eq!(
        TileAddress::containing([MAX_TILE_LATITUDE_DEG, 0.0], 3)
            .unwrap()
            .y,
        0
    );
    assert!(TileAddress::containing([85.1, 0.0], 3).is_err());
    assert!(TileAddress::containing([f64::NAN, 0.0], 3).is_err());
    assert!(TileAddress::new(3, 8, 0).is_err());
    assert!(TileAddress::new(MAX_ZOOM + 1, 0, 0).is_err());
}

#[test]
fn ancestor_and_ground_sample_distance() {
    let t = TileAddress::new(16, 19089, 24578).unwrap();
    assert_eq!(t.ancestor(1), TileAddress::new(15, 9544, 12289).unwrap());
    assert_eq!(t.ancestor(40).zoom, 0);
    let gsd = ground_sample_distance_m(0.0, 15, 256);
    assert!((gsd - 4.777).abs() < 1e-3, "{gsd}");
    assert!((ground_sample_distance_m(60.0, 15, 256) - gsd / 2.0).abs() < 1e-9);
    assert!((ground_sample_distance_m(0.0, 14, 512) - gsd).abs() < 1e-9);
}

#[test]
fn terrarium_decoding() {
    assert_eq!(decode_terrarium_m([128, 0, 0]), 0.0);
    assert_eq!(decode_terrarium_m([129, 56, 128]), 312.5);
    assert_eq!(decode_terrarium_m([0, 0, 0]), -32_768.0);
}
