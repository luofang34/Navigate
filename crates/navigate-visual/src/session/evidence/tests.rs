use super::*;
use crate::session::{ContinuityEpoch, StreamId};
use navigate_data::{
    Accuracy, DEFAULT_CORRELATION_ZOOM, Descriptor, Encoding, HorizontalDatum, LayerKey, LayerKind,
    LayerSpec, Manifest, MemoryStore, Package, PackageWriter, Revision, SourceKey, TileAddress,
};

#[cfg(test)]
fn frame(index: u64) -> FrameKey {
    FrameKey {
        stream: StreamId(1),
        continuity: ContinuityEpoch(0),
        index,
    }
}

/// The centre of one zoom 16 cell, `[lat, lon]` in degrees, so that a point
/// tens of metres away is in the same cell.
#[cfg(test)]
fn centre() -> [f64; 2] {
    let cell = TileAddress::containing([41.70, -1.00], DEFAULT_CORRELATION_ZOOM).unwrap();
    navigate_data::tile::tile_lat_lon(f64::from(cell.x) + 0.5, f64::from(cell.y) + 0.5, cell.zoom)
}

#[cfg(test)]
const IMAGERY: LayerKey = LayerKey {
    kind: LayerKind::Imagery,
    source: SourceKey::EsriWorldImagery,
};

/// The manifest of a sealed package that holds the imagery tiles `tiles`
/// of one source and revision.
#[cfg(test)]
fn package(tiles: &[TileAddress]) -> Manifest {
    let mut store = MemoryStore::default();
    let mut writer = PackageWriter::new();
    writer
        .add_layer(LayerSpec {
            key: IMAGERY,
            revision: Revision::Release(7),
            horizontal_datum: HorizontalDatum::Wgs84,
            vertical_datum: None,
            encoding: Encoding::Jpeg,
            tile_pixels: 256,
            accuracy: Accuracy::unknown(),
            correlation_zoom: DEFAULT_CORRELATION_ZOOM,
        })
        .unwrap();
    for tile in tiles {
        // The blob of a tile is the same in each package that holds it.
        let blob = format!("{}/{}/{}", tile.zoom, tile.x, tile.y);
        pollster::block_on(writer.add_tile(&mut store, IMAGERY, *tile, blob.as_bytes())).unwrap();
    }
    let identity = pollster::block_on(writer.seal(&mut store, &Descriptor::default())).unwrap();
    let opened = pollster::block_on(Package::open(&store, identity)).unwrap();
    assert_eq!(opened.identity(), identity);
    opened.manifest().clone()
}

#[cfg(test)]
fn cell_of(manifest: &Manifest) -> MapCell {
    MapCell::of_layer(manifest.layer(IMAGERY).unwrap(), centre()).unwrap()
}

#[test]
fn a_second_anchor_in_one_cell_shares_the_map_error() {
    let tile = TileAddress::containing(centre(), 17).unwrap();
    let cell = cell_of(&package(&[tile]));
    let mut ledger = MapCellLedger::new(16);
    assert_eq!(
        ledger.label(&cell, frame(1)),
        FusionEligibility::Independent
    );
    ledger.commit(cell.clone(), frame(1));
    assert_eq!(
        ledger.label(&cell, frame(1)),
        FusionEligibility::Independent
    );
    assert_eq!(
        ledger.label(&cell, frame(2)),
        FusionEligibility::SharedMapError {
            cell: cell.clone(),
            earlier: frame(1)
        }
    );
}

/// The map error of a cell belongs to the source of the imagery, not to the
/// package that carries it (ADR-0022 §7). Two packages that hold the same
/// tile of one source have different identities and the same cell.
#[test]
fn two_packages_of_one_source_share_the_map_error_of_a_cell() {
    let shared = TileAddress::containing(centre(), 17).unwrap();
    let east = TileAddress::new(17, shared.x + 40, shared.y).unwrap();
    let west = TileAddress::new(17, shared.x - 40, shared.y).unwrap();
    let first_package = package(&[shared, east]);
    let second_package = package(&[shared, west]);
    assert_ne!(
        first_package.identity().unwrap(),
        second_package.identity().unwrap(),
        "the packages differ in content, so they differ in identity"
    );
    let mut ledger = MapCellLedger::new(16);
    let first = cell_of(&first_package);
    assert_eq!(
        ledger.label(&first, frame(1)),
        FusionEligibility::Independent
    );
    ledger.commit(first.clone(), frame(1));
    assert!(matches!(
        ledger.label(&cell_of(&second_package), frame(2)),
        FusionEligibility::SharedMapError { earlier, .. } if earlier == frame(1)
    ));
}

#[test]
fn another_source_revision_or_cell_is_independent() {
    let tile = TileAddress::containing(centre(), 17).unwrap();
    let manifest = package(&[tile]);
    let layer = *manifest.layer(IMAGERY).unwrap();
    let mut ledger = MapCellLedger::new(16);
    let cell = cell_of(&manifest);
    ledger.commit(cell.clone(), frame(1));
    let mut revised = layer;
    revised.revision = Revision::Release(8);
    let mut other_source = layer;
    other_source.key.source = SourceKey::Synthetic;
    // One correlation cell to the east: 360° / 2^16 of longitude.
    let [lat, lon] = centre();
    let next_cell = [lat, lon + 360.0 / 65_536.0];
    for other in [
        MapCell::of_layer(&revised, centre()).unwrap(),
        MapCell::of_layer(&other_source, centre()).unwrap(),
        MapCell::of_layer(&layer, next_cell).unwrap(),
    ] {
        assert_ne!(other, cell);
        assert_eq!(
            ledger.label(&other, frame(2)),
            FusionEligibility::Independent
        );
    }
}

#[test]
fn a_label_does_not_commit_and_a_full_ledger_does_not_track() {
    let tile = TileAddress::containing(centre(), 17).unwrap();
    let layer = *package(&[tile]).layer(IMAGERY).unwrap();
    let [lat, lon] = centre();
    let cell = MapCell::of_layer(&layer, [lat, lon]).unwrap();
    let mut ledger = MapCellLedger::new(1);
    // A fix that the filter refuses is labelled and not committed: the next
    // frame in the cell is still the first.
    assert_eq!(
        ledger.label(&cell, frame(1)),
        FusionEligibility::Independent
    );
    assert!(ledger.is_empty());
    assert_eq!(
        ledger.label(&cell, frame(2)),
        FusionEligibility::Independent
    );
    ledger.commit(cell.clone(), frame(2));
    assert_eq!(ledger.len(), 1);
    let beyond = MapCell::of_layer(&layer, [lat, lon + 0.5]).unwrap();
    assert_eq!(
        ledger.label(&beyond, frame(3)),
        FusionEligibility::Untracked
    );
    ledger.commit(beyond.clone(), frame(3));
    assert_eq!(ledger.len(), 1, "a full ledger takes no new cell");
}

#[test]
fn a_map_without_a_correlation_key_cuts_cells_in_its_own_frame() {
    let a = MapCell::local("map-a", 120.0, -30.0, 500.0);
    assert_eq!(a.local_index(), Some([0, -1]));
    assert_eq!(a, MapCell::local("map-a", 499.0, -1.0, 500.0));
    assert_ne!(a, MapCell::local("map-a", 501.0, -1.0, 500.0));
    assert_ne!(a, MapCell::local("map-b", 120.0, -30.0, 500.0));
    let tile = TileAddress::containing(centre(), 17).unwrap();
    let source = cell_of(&package(&[tile]));
    assert_eq!(source.local_index(), None);
    let mut ledger = MapCellLedger::new(4);
    ledger.commit(a.clone(), frame(1));
    assert_eq!(
        ledger.label(&source, frame(2)),
        FusionEligibility::Independent
    );
    assert!(matches!(
        ledger.label(&a, frame(2)),
        FusionEligibility::SharedMapError { .. }
    ));
}
