use super::*;
use crate::{
    DEFAULT_CORRELATION_ZOOM, Descriptor, GeoidModel, LayerSpec, MemoryStore, PackageWriter,
    SurfaceModel, TileAddress,
};

pub(crate) const IMAGERY: LayerKey = LayerKey {
    kind: LayerKind::Imagery,
    source: SourceKey::EsriWorldImagery,
};
pub(crate) const ELEVATION: LayerKey = LayerKey {
    kind: LayerKind::Elevation,
    source: SourceKey::MapzenTerrarium,
};

#[cfg(test)]
pub(crate) fn imagery_spec() -> LayerSpec {
    LayerSpec {
        key: IMAGERY,
        revision: Revision::Unversioned,
        horizontal_datum: HorizontalDatum::Wgs84,
        vertical_datum: None,
        encoding: Encoding::Jpeg,
        tile_pixels: 256,
        accuracy: Accuracy::unknown(),
        correlation_zoom: DEFAULT_CORRELATION_ZOOM,
    }
}

#[cfg(test)]
pub(crate) fn elevation_spec() -> LayerSpec {
    LayerSpec {
        key: ELEVATION,
        vertical_datum: Some(VerticalDatum::Orthometric {
            geoid: GeoidModel::Egm96,
        }),
        encoding: Encoding::TerrariumPng,
        accuracy: Accuracy {
            vertical_mm: Some(5_000),
            surface: SurfaceModel::FirstReturn,
            ..Accuracy::unknown()
        },
        correlation_zoom: 12,
        ..imagery_spec()
    }
}

#[cfg(test)]
fn sample() -> Manifest {
    let mut writer = PackageWriter::new();
    writer.add_layer(imagery_spec()).unwrap();
    writer.add_layer(elevation_spec()).unwrap();
    let mut store = MemoryStore::default();
    for (x, y) in [(5, 3), (4, 3), (4, 2)] {
        let a = TileAddress::new(4, x, y).unwrap();
        pollster::block_on(writer.add_tile(&mut store, IMAGERY, a, &[x as u8, y as u8])).unwrap();
    }
    writer
        .add_gap(IMAGERY, TileAddress::new(4, 5, 2).unwrap())
        .unwrap();
    writer.manifest().unwrap().0
}

#[test]
fn canonical_bytes_round_trip_and_hold_no_host_strings_or_floats() {
    let manifest = sample();
    assert_eq!(manifest.layers[0].key, IMAGERY);
    assert_eq!(manifest.layers[0].index.tiles, 3);
    assert_eq!(manifest.layers[0].index.gaps, 1);
    let bytes = manifest.canonical_bytes().unwrap();
    assert_eq!(Manifest::from_canonical(&bytes, "m").unwrap(), manifest);
    assert_eq!(manifest.identity().unwrap(), Sha256Digest::of(&bytes));
    let text = core::str::from_utf8(&bytes).unwrap();
    assert!(
        !text.replace(PACKAGE_FORMAT, "").contains('.'),
        "a float: {text}"
    );
    assert!(bytes.len() < 2_048, "{}", bytes.len());
}

#[test]
fn the_descriptor_does_not_change_the_identity() {
    let mut store = MemoryStore::default();
    let mut ids = Vec::new();
    for release in ["first", "second label"] {
        let mut writer = PackageWriter::new();
        writer.add_layer(imagery_spec()).unwrap();
        let a = TileAddress::new(4, 1, 1).unwrap();
        pollster::block_on(writer.add_tile(&mut store, IMAGERY, a, b"tile")).unwrap();
        let descriptor = Descriptor {
            release_id: release.into(),
            ..Descriptor::default()
        };
        ids.push(pollster::block_on(writer.seal(&mut store, &descriptor)).unwrap());
    }
    assert_eq!(ids[0], ids[1]);
}

#[test]
fn non_canonical_bytes_are_refused() {
    let pretty = serde_json::to_vec_pretty(&sample()).unwrap();
    assert!(matches!(
        Manifest::from_canonical(&pretty, "m"),
        Err(DataError::Document { .. })
    ));
    let mut extra: serde_json::Value =
        serde_json::from_slice(&sample().canonical_bytes().unwrap()).unwrap();
    extra["note"] = "fetched at noon".into();
    assert!(Manifest::from_canonical(&serde_json::to_vec(&extra).unwrap(), "m").is_err());
    let mut float: serde_json::Value =
        serde_json::from_slice(&sample().canonical_bytes().unwrap()).unwrap();
    float["layers"][1]["accuracy"]["vertical_mm"] = serde_json::json!(5000.5);
    assert!(Manifest::from_canonical(&serde_json::to_vec(&float).unwrap(), "m").is_err());
}

#[test]
fn other_versions_and_unknown_values_are_refused_with_typed_errors() {
    let edit = |field: &str, value: serde_json::Value, at_layer: bool| {
        let mut doc: serde_json::Value =
            serde_json::from_slice(&sample().canonical_bytes().unwrap()).unwrap();
        if at_layer {
            doc["layers"][0][field] = value;
        } else {
            doc[field] = value;
        }
        Manifest::from_canonical(&serde_json::to_vec(&doc).unwrap(), "m")
    };
    assert!(matches!(
        edit("version", 2.into(), false),
        Err(DataError::Version { found: 2, .. })
    ));
    assert_eq!(
        edit("body", "mars".into(), false),
        Err(DataError::Unsupported { field: "body" })
    );
    assert_eq!(
        edit("grid", "polar_stereographic".into(), false),
        Err(DataError::Unsupported { field: "grid" })
    );
    assert_eq!(
        edit("encoding", "avif".into(), true),
        Err(DataError::Unsupported { field: "encoding" })
    );
    let kind = serde_json::json!({"kind": "lidar", "source": "esri_world_imagery"});
    assert_eq!(
        edit("key", kind, true),
        Err(DataError::Unsupported {
            field: "layer kind"
        })
    );
    let source = serde_json::json!({"kind": "imagery", "source": "someone"});
    assert_eq!(
        edit("key", source, true),
        Err(DataError::Unsupported { field: "source" })
    );
}

#[test]
fn rule_breaks_are_refused() {
    let mut m = sample();
    m.layers.swap(0, 1);
    assert!(m.validate().is_err());
    let mut m = sample();
    m.layers[1].vertical_datum = None;
    assert!(m.validate().is_err());
    let mut m = sample();
    m.layers[0].index.tiles += 1;
    assert!(m.validate().is_err());
    let mut m = sample();
    m.layers.clear();
    assert!(m.validate().is_err());
}
