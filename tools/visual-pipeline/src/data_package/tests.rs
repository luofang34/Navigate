use super::*;
use navigate_data::{
    FetchPort, GroundHeight, MemoryStore, Package, TileAddress, TileKey, TileRead,
};
use std::collections::BTreeSet;
use std::path::Path;

#[cfg(test)]
fn terrarium_png(height_m: f64) -> Vec<u8> {
    let v = height_m + 32_768.0;
    let pixel = image::Rgb([
        (v / 256.0).floor() as u8,
        (v % 256.0).floor() as u8,
        (v.fract() * 256.0) as u8,
    ]);
    let mut out = std::io::Cursor::new(Vec::new());
    image::RgbImage::from_pixel(8, 8, pixel)
        .write_to(&mut out, image::ImageFormat::Png)
        .unwrap();
    out.into_inner()
}

#[test]
fn terrarium_ground_reads_the_planning_tiles() {
    let point = [41.7, -1.0];
    let address = TileAddress::containing(point, GROUND_ZOOM).unwrap();
    let mut ground = TerrariumGround::default();
    insert_png(&mut ground, address, &terrarium_png(312.5)).unwrap();
    assert_eq!(ground.ground_m(point), Some(312.5));
    assert_eq!(ground.ground_m([10.0, 10.0]), None);
    assert!(insert_png(&mut ground, address, b"not a png").is_err());
}

#[test]
fn the_template_cache_is_keyed_by_the_tile_key_not_the_url() {
    let provider = tempfile::tempdir().unwrap();
    let cache = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(provider.path().join("15/9")).unwrap();
    std::fs::write(provider.path().join("15/9/4.jpg"), b"tile").unwrap();
    let spec = layer_spec(
        LayerKind::Imagery,
        SourceKey::EsriWorldImagery,
        Revision::Release(1),
    );
    let layer = |dir: &Path| TemplateLayer {
        url: format!("file://{}/{{z}}/{{x}}/{{y}}.jpg", dir.display()),
        spec,
        attribution: String::new(),
    };
    let fetch =
        TemplateFetch::new(cache.path()).with_layer(LayerKind::Imagery, layer(provider.path()));
    let key = TileKey {
        source: SourceKey::EsriWorldImagery,
        revision: Revision::Release(1),
        layer: LayerKind::Imagery,
        address: TileAddress::new(15, 9, 4).unwrap(),
    };
    assert_eq!(
        fetch.fetch_blocking(&key).unwrap(),
        navigate_data::Fetched::Tile(b"tile".to_vec())
    );
    assert!(
        cache
            .path()
            .join("esri_world_imagery/release-1/15/9/4")
            .is_file()
    );
    let moved = TemplateFetch::new(cache.path())
        .with_layer(LayerKind::Imagery, layer(Path::new("/nonexistent")));
    assert_eq!(
        moved.fetch_blocking(&key).unwrap(),
        navigate_data::Fetched::Tile(b"tile".to_vec())
    );
    let other = TileKey {
        address: TileAddress::new(15, 9, 5).unwrap(),
        ..key
    };
    assert_eq!(
        moved.fetch_blocking(&other).unwrap(),
        navigate_data::Fetched::Absent
    );
}

/// Writes a catalog in the layout of the `reference` subcommand.
#[cfg(test)]
fn write_catalog(root: &Path, route: &[[f64; 2]]) {
    let mut tiles = Vec::new();
    for (kind, zoom, ext) in [("imagery", 15, "jpg"), ("dem", 12, "png")] {
        let mut set = BTreeSet::new();
        navigate_data::corridor::corridor_tiles(route, 5_000.0, zoom, &mut set).unwrap();
        for a in set {
            let path = format!("{kind}/{zoom}/{}/{}.{ext}", a.x, a.y);
            let bytes = format!("{kind} {zoom} {} {}", a.x, a.y).into_bytes();
            std::fs::create_dir_all(root.join(&path).parent().unwrap()).unwrap();
            std::fs::write(root.join(&path), &bytes).unwrap();
            tiles.push(json!({ "kind": kind, "xyz": [zoom, a.x, a.y], "path": path, "sha256": "", "bytes": bytes.len() }));
        }
    }
    let catalog = json!({
        "format": "photo-reference-catalog", "version": 1, "complete": true,
        "attribution": { "imagery": "test imagery", "dem": "test elevation" },
        "configuration": { "bbox": [0, 0, 0, 0], "dem_zoom": 12, "imagery_zoom": 15 },
        "tiles": tiles,
    });
    std::fs::write(
        root.join("catalog.json"),
        serde_json::to_vec(&catalog).unwrap(),
    )
    .unwrap();
}

/// Packages every tile of a catalog and compares the package with it.
#[cfg(test)]
fn assert_catalog_round_trip(root: &Path) -> usize {
    let catalog = CatalogFetch::open_blocking(root).unwrap();
    let specs = [
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
    ];
    let mut tiles = navigate_data::PlannedTiles::new();
    for kind in [LayerKind::Imagery, LayerKind::Elevation] {
        let set = tiles.entry(kind).or_default();
        for [z, x, y] in catalog.tiles(kind) {
            set.insert(TileAddress::new(z, x, y).unwrap());
        }
    }
    let mut store = MemoryStore::default();
    let summary = pollster::block_on(prepare(
        &tiles,
        &specs,
        &Descriptor::default(),
        100_000,
        &catalog,
        &mut store,
    ))
    .unwrap();
    assert_eq!(summary.gaps, 0);
    let package = pollster::block_on(Package::open(&store, summary.identity)).unwrap();
    let mut compared = 0;
    for (&kind, addresses) in &tiles {
        let layer = specs.iter().find(|s| s.key.kind == kind).unwrap().key;
        for &a in addresses {
            let key = TileKey {
                source: layer.source,
                revision: Revision::Unversioned,
                layer: kind,
                address: a,
            };
            let expected = pollster::block_on(catalog.fetch(&key)).unwrap();
            let read = pollster::block_on(package.tile(&store, layer, a)).unwrap();
            match (read, expected) {
                (TileRead::Present(got), navigate_data::Fetched::Tile(want)) => {
                    assert_eq!(got, want)
                }
                other => panic!("{a:?}: {other:?}"),
            }
            compared += 1;
        }
    }
    compared
}

#[test]
fn a_package_from_a_catalog_equals_the_catalog_tile_for_tile() {
    let root = tempfile::tempdir().unwrap();
    write_catalog(root.path(), &[[40.86, -75.18], [40.86, -75.17]]);
    assert!(assert_catalog_round_trip(root.path()) > 4);
}

#[test]
#[ignore = "reads the Delaware catalog under ~/Downloads/delaware-videos/claw"]
fn a_package_from_the_delaware_catalog_equals_it_tile_for_tile() {
    let home = std::env::var_os("HOME").unwrap();
    let root = Path::new(&home).join("Downloads/delaware-videos/claw/reference-watergap");
    assert!(assert_catalog_round_trip(&root) > 300);
}

#[test]
fn sources_and_revisions_parse_from_the_registry_and_one_spelling() {
    assert_eq!(
        parse_source("esri_world_imagery").unwrap(),
        SourceKey::EsriWorldImagery
    );
    assert!(parse_source("someone").is_err());
    assert_eq!(
        parse_revision("unversioned").unwrap(),
        Revision::Unversioned
    );
    assert_eq!(parse_revision("release:3").unwrap(), Revision::Release(3));
    assert_eq!(
        parse_revision("collected:2024-06-01").unwrap(),
        Revision::Collected(navigate_data::DayNumber::from_ymd(2024, 6, 1).unwrap())
    );
    assert!(parse_revision("2024").is_err());
    assert!(parse_revision("collected:2024-02-30").is_err());
}
