use alloc::vec::Vec;

use super::*;
use crate::{TileAddress, tile::decode_terrarium_m};

/// The same ground everywhere.
struct Flat(f64);

impl GroundHeight for Flat {
    fn ground_m(&self, _point: [f64; 2]) -> Option<f64> {
        Some(self.0)
    }
}

/// Zaragoza's closed land circuit at 2000 m above mean sea level.
const CIRCUIT: [[f64; 3]; 5] = [
    [41.7, -1.05, 2000.0],
    [41.7, -0.8, 2000.0],
    [41.8, -0.8, 2000.0],
    [41.8, -1.0, 2000.0],
    [41.74, -1.05, 2000.0],
];

/// A 400 by 300 nadir camera of 67.4° horizontal field of view.
const CAMERA: PlanCamera = PlanCamera {
    width_px: 400,
    height_px: 300,
    horizontal_fov_deg: 67.4,
    depression_deg: 90.0,
};

fn circuit_plan(cameras: &[PlanCamera]) -> PackagePlan {
    let request = route_request(
        &CIRCUIT,
        cameras,
        PlanBounds::ONBOARD,
        LayerLimits::imagery(),
        LayerLimits::terrarium(),
    );
    PackagePlan::new(request, &Flat(300.0)).unwrap()
}

fn specs() -> [LayerSpec; 2] {
    [
        layer_spec(
            LayerKind::Imagery,
            SourceKey::EsriWorldImagery,
            Revision::Unversioned,
        ),
        layer_spec(
            LayerKind::Elevation,
            SourceKey::MapzenTerrarium,
            Revision::Unversioned,
        ),
    ]
}

/// A square Terrarium tile with every pixel `height_m` above mean sea level.
fn terrarium_rgb(side: u32, height_m: f64) -> Vec<u8> {
    let scaled = height_m + 32_768.0;
    let pixel = [
        (scaled / 256.0) as u8,
        (scaled % 256.0) as u8,
        ((scaled % 1.0) * 256.0) as u8,
    ];
    pixel.repeat((side * side) as usize)
}

#[test]
fn the_values_an_identity_depends_on_are_fixed() {
    assert_eq!(
        (GROUND_ZOOM, GROUND_CORRIDOR_M, ELEVATION_OFFSET),
        (10, 2_000.0, 3)
    );
    assert_eq!(
        PlanBounds::ONBOARD,
        PlanBounds {
            search_radius_m: 500.0,
            deviation_m: 500.0,
            max_range_m: 4_000.0
        }
    );
    let [imagery, elevation] = specs();
    assert_eq!(
        (imagery.encoding, imagery.vertical_datum),
        (Encoding::Jpeg, None)
    );
    assert_eq!(
        (elevation.encoding, elevation.vertical_datum),
        (
            Encoding::TerrariumPng,
            Some(VerticalDatum::Orthometric {
                geoid: GeoidModel::Egm96
            })
        )
    );
    for spec in [imagery, elevation] {
        assert_eq!(spec.horizontal_datum, HorizontalDatum::Wgs84);
        assert_eq!(spec.tile_pixels, 256);
        assert_eq!(spec.accuracy, Accuracy::unknown());
        assert_eq!(spec.correlation_zoom, 16);
    }
}

#[test]
fn the_focal_length_comes_from_the_horizontal_field_of_view() {
    let wide = PlanCamera {
        width_px: 800,
        height_px: 600,
        horizontal_fov_deg: 90.0,
        depression_deg: 90.0,
    }
    .planned();
    assert!((wide.focal_px - 400.0).abs() < 1e-9);
    assert_eq!((wide.width_px, wide.height_px), (800, 600));
    assert_eq!(wide.depression_rad, core::f64::consts::FRAC_PI_2);
    assert!((CAMERA.planned().focal_px - 299.85).abs() < 0.05);
}

#[test]
fn the_request_carries_the_route_cameras_bounds_and_offset() {
    let request = route_request(
        &CIRCUIT,
        &[CAMERA],
        PlanBounds::ONBOARD,
        LayerLimits::imagery(),
        LayerLimits::terrarium(),
    );
    assert_eq!(request.legs.len(), 4);
    assert_eq!(request.legs[0].from, CIRCUIT[0]);
    assert_eq!(request.legs[3].to, CIRCUIT[4]);
    assert_eq!(request.cameras, [CAMERA.planned()]);
    assert_eq!(
        (
            request.search_radius_m,
            request.deviation_m,
            request.max_range_m,
            request.elevation_offset
        ),
        (500.0, 500.0, 4_000.0, 3)
    );
    assert!(route_legs(&CIRCUIT[..1]).is_empty());
}

#[test]
fn the_circuit_at_two_thousand_metres_needs_zoom_fifteen_imagery() {
    let plan = circuit_plan(&[CAMERA]);
    for leg in &plan.resources.legs {
        assert_eq!(leg.agl_m, 1_700.0);
        let zooms: Vec<(LayerKind, u32)> = leg.bands.iter().map(|b| (b.layer, b.zoom)).collect();
        assert_eq!(
            zooms,
            [(LayerKind::Imagery, 15), (LayerKind::Elevation, 12)]
        );
    }
    let imagery = plan.tiles[&LayerKind::Imagery].len();
    let elevation = plan.tiles[&LayerKind::Elevation].len();
    assert_eq!((imagery, elevation), (IMAGERY_TILES, ELEVATION_TILES));
    assert_eq!(plan.tile_count(), imagery + elevation);
    assert_eq!(
        plan.size.upper_bytes,
        imagery as u64 * 20_000 + elevation as u64 * 110_000
    );
    let keys = plan.keys(&specs()).unwrap();
    assert_eq!(keys.len(), plan.tile_count());
    assert!(keys.iter().all(|key| key.revision == Revision::Unversioned));
    assert!(keys.iter().all(
        |key| (key.layer == LayerKind::Imagery) == (key.source == SourceKey::EsriWorldImagery)
    ));
}

/// Tiles of the circuit plan over flat ground at 300 m.
const IMAGERY_TILES: usize = 510;
const ELEVATION_TILES: usize = 15;

#[test]
fn a_larger_image_needs_finer_and_more_tiles() {
    let small = circuit_plan(&[CAMERA]);
    let large = circuit_plan(&[PlanCamera {
        width_px: 800,
        height_px: 600,
        ..CAMERA
    }]);
    assert!(
        large
            .resources
            .legs
            .iter()
            .all(|leg| leg.bands[0].zoom == 16)
    );
    assert!(large.tile_count() > 3 * small.tile_count());
}

#[test]
fn a_planned_layer_must_be_declared() {
    let plan = circuit_plan(&[CAMERA]);
    let [imagery, _] = specs();
    let refused = plan.keys(&[imagery]).unwrap_err();
    assert!(
        matches!(
            refused,
            DataError::Invalid {
                field: "layers",
                ..
            }
        ),
        "{refused}"
    );
}

#[test]
fn the_planning_tiles_are_a_corridor_at_the_ground_zoom() {
    let track: Vec<[f64; 2]> = CIRCUIT.iter().map(|p| [p[0], p[1]]).collect();
    let tiles = ground_tiles(&track).unwrap();
    assert!(tiles.iter().all(|tile| tile.zoom == GROUND_ZOOM));
    // A zoom-10 tile is about 29 km wide here; the circuit spans 21 km by
    // 11 km and its corridor crosses a column edge.
    assert!((1..=6).contains(&tiles.len()), "{tiles:?}");
    assert!(tiles.contains(&TileAddress::containing([41.7, -1.05], GROUND_ZOOM).unwrap()));
    assert!(ground_tiles(&[]).is_err());
}

#[test]
fn a_decoded_tile_gives_its_heights_and_nothing_outside_it() {
    let mut ground = TerrariumGround::default();
    assert!(ground.is_empty());
    let address = TileAddress::containing([41.7, -1.05], GROUND_ZOOM).unwrap();
    ground
        .insert_rgb(address, 8, &terrarium_rgb(8, 307.5))
        .unwrap();
    assert_eq!(ground.len(), 1);
    assert_eq!(ground.ground_m([41.7, -1.05]), Some(307.5));
    assert_eq!(ground.ground_m([10.0, 10.0]), None);
    ground
        .insert_rgb(address, 8, &terrarium_rgb(8, -12.25))
        .unwrap();
    assert_eq!(ground.ground_m([41.7, -1.05]), Some(-12.25));
}

#[test]
fn each_pixel_is_its_own_height() {
    let address = TileAddress::containing([41.7, -1.05], GROUND_ZOOM).unwrap();
    // Height rises by one metre a column and ten a row.
    let side = 4_u32;
    let mut rgb = Vec::new();
    for row in 0..side {
        for column in 0..side {
            let scaled = 32_768.0 + f64::from(column) + 10.0 * f64::from(row);
            rgb.extend([(scaled / 256.0) as u8, (scaled % 256.0) as u8, 0]);
        }
    }
    let mut ground = TerrariumGround::default();
    ground.insert_rgb(address, side, &rgb).unwrap();
    let pixel = |column: f64, row: f64| {
        let n = f64::from(1_u32 << GROUND_ZOOM);
        let x = f64::from(address.x) + (column + 0.5) / f64::from(side);
        let y = f64::from(address.y) + (row + 0.5) / f64::from(side);
        let lon = x / n * 360.0 - 180.0;
        let lat = (core::f64::consts::PI * (1.0 - 2.0 * y / n))
            .sinh()
            .atan()
            .to_degrees();
        [lat, lon]
    };
    assert_eq!(ground.ground_m(pixel(0.0, 0.0)), Some(0.0));
    assert_eq!(ground.ground_m(pixel(3.0, 0.0)), Some(3.0));
    assert_eq!(ground.ground_m(pixel(1.0, 2.0)), Some(21.0));
    assert_eq!(decode_terrarium_m([128, 21, 0]), 21.0);
}

#[test]
fn a_tile_of_another_zoom_or_size_is_refused() {
    let mut ground = TerrariumGround::default();
    let fine = TileAddress::containing([41.7, -1.05], 11).unwrap();
    assert!(ground.insert_rgb(fine, 8, &terrarium_rgb(8, 0.0)).is_err());
    let address = TileAddress::containing([41.7, -1.05], GROUND_ZOOM).unwrap();
    assert!(
        ground
            .insert_rgb(address, 8, &terrarium_rgb(7, 0.0))
            .is_err()
    );
    assert!(ground.insert_rgb(address, 0, &[]).is_err());
    assert!(ground.is_empty());
}
