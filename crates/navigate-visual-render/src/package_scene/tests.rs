use super::*;
use crate::test_package::{CENTRE, ELEVATION, IMAGERY, synthetic_package};
use navigate_contract::Ellipsoid;
use navigate_geodesy::ZeroUndulation;

#[cfg(test)]
fn request() -> SceneRequest {
    SceneRequest {
        imagery_layer: IMAGERY,
        imagery_zoom: 15,
        elevation_layer: ELEVATION,
        elevation_zoom: 12,
        centre: CENTRE,
        half_side_m: 1_500.0,
    }
}

#[test]
fn a_scene_window_carries_the_package_identity_and_the_datum_bound() {
    let (store, package) = synthetic_package(&[]);
    let window = pollster::block_on(scene_around(&package, &store, &request(), None)).unwrap();
    assert_eq!(
        window.scene.map.manifest_sha256,
        package.identity().to_string()
    );
    assert_eq!(window.imagery_coverage, 1.0);
    assert_eq!(
        window.undulation_bound_m,
        navigate_data::UNKNOWN_UNDULATION_BOUND_M
    );
    let zero = ZeroUndulation(Ellipsoid::WGS84);
    let named = pollster::block_on(scene_around(
        &package,
        &store,
        &request(),
        Some((GeoidModel::Egm96, &zero)),
    ))
    .unwrap();
    assert_eq!(named.undulation_bound_m, 0.0);
    assert!(
        named
            .scene
            .terrain
            .heights_m
            .iter()
            .zip(&named.scene.terrain.coverage)
            .all(|(h, c)| *c == 0 || (*h - 250.0).abs() < 0.01)
    );
}

#[test]
fn gaps_leave_pixels_uncovered_and_a_changed_tile_is_refused() {
    let centre_tile = TileAddress::containing(CENTRE, 15).unwrap();
    let (store, package) = synthetic_package(&[centre_tile]);
    let window = pollster::block_on(scene_around(&package, &store, &request(), None)).unwrap();
    assert!(
        window.imagery_coverage < 1.0 && window.imagery_coverage > 0.5,
        "{}",
        window.imagery_coverage
    );
    let (mut store, package) = synthetic_package(&[]);
    let tiles: Vec<String> = store
        .paths()
        .filter(|p| p.starts_with("chunks/"))
        .map(String::from)
        .collect();
    let index_blobs: Vec<String> = package
        .manifest()
        .layers
        .iter()
        .map(|l| navigate_data::blob_path(l.index.blob))
        .collect();
    for path in tiles.iter().filter(|p| !index_blobs.contains(p)) {
        store.replace(path, b"changed");
    }
    assert!(pollster::block_on(scene_around(&package, &store, &request(), None)).is_err());
}

#[test]
fn a_window_across_the_antimeridian_is_refused() {
    let (store, package) = synthetic_package(&[]);
    let across = SceneRequest {
        centre: [10.0, 179.999],
        ..request()
    };
    assert!(matches!(
        pollster::block_on(scene_around(&package, &store, &across, None)),
        Err(RenderError::Invalid(_))
    ));
}
