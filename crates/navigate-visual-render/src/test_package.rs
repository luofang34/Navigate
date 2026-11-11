//! A synthetic package for the tests of this crate.

use image::{ImageFormat, Rgb, RgbImage};
use navigate_data::{
    Accuracy, DEFAULT_CORRELATION_ZOOM, Descriptor, Encoding, GeoidModel, HorizontalDatum,
    LayerKey, LayerKind, LayerSpec, MemoryFile, MemoryStore, Package, PackageWriter, Revision,
    SourceKey, TileAddress, VerticalDatum, corridor::corridor_tiles,
};
use std::collections::BTreeSet;
use std::io::Cursor;

/// Centre of the synthetic package, `[lat, lon]` in degrees.
pub(crate) const CENTRE: [f64; 2] = [41.70, -1.00];
/// The imagery layer.
pub(crate) const IMAGERY: LayerKey = LayerKey {
    kind: LayerKind::Imagery,
    source: SourceKey::Synthetic,
};
/// The elevation layer.
pub(crate) const ELEVATION: LayerKey = LayerKey {
    kind: LayerKind::Elevation,
    source: SourceKey::Synthetic,
};

fn png(image: &RgbImage) -> Vec<u8> {
    let mut out = Cursor::new(Vec::new());
    image.write_to(&mut out, ImageFormat::Png).unwrap();
    out.into_inner()
}

fn spec(key: LayerKey, encoding: Encoding, datum: Option<VerticalDatum>) -> LayerSpec {
    LayerSpec {
        key,
        revision: Revision::Unversioned,
        horizontal_datum: HorizontalDatum::Wgs84,
        vertical_datum: datum,
        encoding,
        tile_pixels: 256,
        accuracy: Accuracy::unknown(),
        correlation_zoom: DEFAULT_CORRELATION_ZOOM,
    }
}

/// A package around [`CENTRE`]: textured imagery at zoom 15 and flat
/// Terrarium elevation of 250 m at zoom 12. Imagery tiles in `gaps` are
/// gaps.
pub(crate) fn synthetic_package(gaps: &[TileAddress]) -> (MemoryStore, Package<MemoryFile>) {
    let mut store = MemoryStore::default();
    let mut writer = PackageWriter::new();
    writer
        .add_layer(spec(IMAGERY, Encoding::Png, None))
        .unwrap();
    let datum = VerticalDatum::Orthometric {
        geoid: GeoidModel::Egm96,
    };
    writer
        .add_layer(spec(ELEVATION, Encoding::TerrariumPng, Some(datum)))
        .unwrap();
    let mut imagery = BTreeSet::new();
    corridor_tiles(&[CENTRE], 6_000.0, 15, &mut imagery).unwrap();
    for a in imagery {
        if gaps.contains(&a) {
            writer.add_gap(IMAGERY, a).unwrap();
            continue;
        }
        let tile = RgbImage::from_fn(256, 256, |x, y| {
            let gx = a.x * 256 + x;
            let gy = a.y * 256 + y;
            let v = (((gx / 9) ^ (gy / 13)) % 7) as u8 * 30 + ((gx * 7 + gy * 3) % 23) as u8;
            Rgb([v, v, v])
        });
        pollster::block_on(writer.add_tile(&mut store, IMAGERY, a, &png(&tile))).unwrap();
    }
    let mut terrain = BTreeSet::new();
    corridor_tiles(&[CENTRE], 12_000.0, 12, &mut terrain).unwrap();
    let flat = png(&RgbImage::from_pixel(256, 256, Rgb([128, 250, 0])));
    for a in terrain {
        pollster::block_on(writer.add_tile(&mut store, ELEVATION, a, &flat)).unwrap();
    }
    let identity = pollster::block_on(writer.seal(&mut store, &Descriptor::default())).unwrap();
    let package = pollster::block_on(Package::open(&store, identity)).unwrap();
    (store, package)
}
