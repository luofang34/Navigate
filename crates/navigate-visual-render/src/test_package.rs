//! A synthetic package for the tests of this crate.

use image::{ImageFormat, Rgb, RgbImage};
use navigate_data::{
    Accuracy, Encoding, Layer, LayerKind, MemoryStore, Package, PackageWriter, SourceRef,
    TileAddress, VerticalDatum, corridor::corridor_tiles,
};
use std::collections::BTreeSet;
use std::io::Cursor;

/// Centre of the synthetic package, `[lat, lon]` in degrees.
pub(crate) const CENTRE: [f64; 2] = [41.70, -1.00];

fn png(image: &RgbImage) -> Vec<u8> {
    let mut out = Cursor::new(Vec::new());
    image.write_to(&mut out, ImageFormat::Png).unwrap();
    out.into_inner()
}

fn layer(id: &str, kind: LayerKind, encoding: Encoding, datum: Option<VerticalDatum>) -> Layer {
    Layer {
        id: id.into(),
        kind,
        source: SourceRef {
            id: format!("synthetic-{id}"),
            revision: "1".into(),
            attribution: String::new(),
        },
        encoding,
        tile_pixels: 256,
        vertical_datum: datum,
        accuracy: Accuracy::unknown(),
        zooms: Vec::new(),
    }
}

/// A package around [`CENTRE`]: textured imagery at zoom 15 and flat
/// Terrarium elevation of 250 m at zoom 12. Imagery tiles in `gaps` are
/// gaps.
pub(crate) fn synthetic_package(gaps: &[TileAddress]) -> (MemoryStore, Package) {
    let mut store = MemoryStore::default();
    let mut writer = PackageWriter::new("synthetic");
    writer
        .add_layer(layer("imagery", LayerKind::Imagery, Encoding::Png, None))
        .unwrap();
    let datum = VerticalDatum::Orthometric {
        geoid_model: "egm96".into(),
    };
    writer
        .add_layer(layer(
            "elevation",
            LayerKind::Elevation,
            Encoding::TerrariumPng,
            Some(datum),
        ))
        .unwrap();
    let mut imagery = BTreeSet::new();
    corridor_tiles(&[CENTRE], 6_000.0, 15, &mut imagery).unwrap();
    for a in imagery {
        if gaps.contains(&a) {
            writer.add_gap("imagery", a).unwrap();
            continue;
        }
        let tile = RgbImage::from_fn(256, 256, |x, y| {
            let gx = a.x * 256 + x;
            let gy = a.y * 256 + y;
            let v = (((gx / 9) ^ (gy / 13)) % 7) as u8 * 30 + ((gx * 7 + gy * 3) % 23) as u8;
            Rgb([v, v, v])
        });
        pollster::block_on(writer.add_tile(&mut store, "imagery", a, &png(&tile))).unwrap();
    }
    let mut terrain = BTreeSet::new();
    corridor_tiles(&[CENTRE], 12_000.0, 12, &mut terrain).unwrap();
    let flat = png(&RgbImage::from_pixel(256, 256, Rgb([128, 250, 0])));
    for a in terrain {
        pollster::block_on(writer.add_tile(&mut store, "elevation", a, &flat)).unwrap();
    }
    let identity = pollster::block_on(writer.seal(&mut store)).unwrap();
    let package = pollster::block_on(Package::open(&store, identity)).unwrap();
    (store, package)
}
