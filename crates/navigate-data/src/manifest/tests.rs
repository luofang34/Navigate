use super::*;
use crate::package::PackageWriter;
use crate::{Accuracy, MemoryStore};

#[cfg(test)]
pub(crate) fn imagery_layer() -> Layer {
    Layer {
        id: "imagery".into(),
        kind: LayerKind::Imagery,
        source: SourceRef {
            id: "test-imagery".into(),
            revision: "2026-10".into(),
            attribution: "Test imagery".into(),
        },
        encoding: Encoding::Jpeg,
        tile_pixels: 256,
        vertical_datum: None,
        accuracy: Accuracy::unknown(),
        zooms: Vec::new(),
    }
}

#[cfg(test)]
pub(crate) fn elevation_layer() -> Layer {
    Layer {
        id: "elevation".into(),
        kind: LayerKind::Elevation,
        source: SourceRef {
            id: "test-terrarium".into(),
            revision: "unknown".into(),
            attribution: "Test elevation".into(),
        },
        encoding: Encoding::TerrariumPng,
        tile_pixels: 256,
        vertical_datum: Some(VerticalDatum::Orthometric {
            geoid_model: "egm96".into(),
        }),
        accuracy: Accuracy {
            vertical: Sigma::Known(5.0),
            ..Accuracy::unknown()
        },
        zooms: Vec::new(),
    }
}

#[cfg(test)]
fn sample() -> Manifest {
    let mut writer = PackageWriter::new("r1");
    writer.add_layer(imagery_layer()).unwrap();
    writer.add_layer(elevation_layer()).unwrap();
    let mut store = MemoryStore::default();
    for (x, y) in [(5, 3), (4, 3), (4, 2)] {
        let a = TileAddress::new(4, x, y).unwrap();
        pollster::block_on(writer.add_tile(&mut store, "imagery", a, &[x as u8, y as u8])).unwrap();
    }
    writer
        .add_gap("imagery", TileAddress::new(4, 5, 2).unwrap())
        .unwrap();
    writer.manifest().unwrap()
}

#[test]
fn canonical_bytes_round_trip_and_sort_the_content() {
    let manifest = sample();
    assert_eq!(manifest.layers[0].id, "elevation");
    let zoom = &manifest.layer("imagery").unwrap().zooms[0];
    let order: Vec<_> = zoom.tiles.iter().map(|t| (t.x, t.y)).collect();
    assert_eq!(order, [(4, 2), (4, 3), (5, 3)]);
    let bytes = manifest.canonical_bytes().unwrap();
    assert_eq!(Manifest::from_canonical(&bytes, "m").unwrap(), manifest);
    assert_eq!(manifest.identity().unwrap(), Sha256Digest::of(&bytes));
}

#[test]
fn non_canonical_bytes_are_refused() {
    let bytes = sample().canonical_bytes().unwrap();
    let pretty = serde_json::to_vec_pretty(&sample()).unwrap();
    assert!(matches!(
        Manifest::from_canonical(&pretty, "m"),
        Err(DataError::Manifest { .. })
    ));
    let mut extra: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    extra["note"] = "fetched at noon".into();
    let extra = serde_json::to_vec(&extra).unwrap();
    assert!(Manifest::from_canonical(&extra, "m").is_err());
}

#[test]
fn declared_coverage_has_tiles_gaps_and_an_outside() {
    let manifest = sample();
    let layer = manifest.layer("imagery").unwrap();
    let at = |x, y| layer.declared(TileAddress::new(4, x, y).unwrap());
    assert!(matches!(at(4, 2), Declared::Tile(t) if t.bytes == 2));
    assert_eq!(at(5, 2), Declared::Gap);
    assert_eq!(at(6, 2), Declared::Outside);
    assert_eq!(
        layer.declared(TileAddress::new(5, 8, 4).unwrap()),
        Declared::Outside
    );
}

#[test]
fn rule_breaks_are_refused() {
    let mut m = sample();
    m.layers.swap(0, 1);
    assert!(m.validate().is_err());
    let mut m = sample();
    m.layers[1].zooms[0].tiles.swap(0, 1);
    assert!(m.validate().is_err());
    let mut m = sample();
    m.layers[1].zooms[0].gaps.push([4, 3]);
    m.layers[1].zooms[0].gaps.sort_by_key(|&[x, y]| (y, x));
    assert!(m.validate().is_err());
    let mut m = sample();
    m.layers[0].vertical_datum = None;
    assert!(m.validate().is_err());
    let mut m = sample();
    m.layers[1].accuracy.horizontal = Sigma::Known(f64::NAN);
    assert!(m.validate().is_err());
    let mut m = sample();
    m.grid.correlation_zoom = 15;
    assert!(m.validate().is_err());
}

#[test]
fn floats_survive_the_canonical_round_trip() {
    let mut writer = PackageWriter::new("r1");
    let mut layer = imagery_layer();
    layer.accuracy.horizontal = Sigma::Known(0.1 + 0.2);
    layer.accuracy.age_growth_per_year = Sigma::Known(1.0 / 3.0);
    writer.add_layer(layer).unwrap();
    let manifest = writer.manifest().unwrap();
    let bytes = manifest.canonical_bytes().unwrap();
    assert_eq!(Manifest::from_canonical(&bytes, "m").unwrap(), manifest);
}
