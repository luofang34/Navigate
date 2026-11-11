//! Derived products bind their core package (ADR-0022): the `NVKTILE1`
//! corridor bank built on a package surface records the package identity
//! and its complete recipe.

use image::{ImageFormat, Rgb, RgbImage};
use navigate_data::product::{Product, ProductKind};
use navigate_data::{
    Accuracy, DirectoryStore, Encoding, Layer, LayerKind, Package, PackageWriter, Sha256Digest,
    SourceRef, TileAddress, VerticalDatum,
};
use navigate_match::{CorridorLandmarkBank, WeightTensor, Weights};
use navigate_visual_flight::canonical::PriorWaypoint;
use navigate_visual_flight::surface::{Surface, SurfaceLayers};
use navigate_visual_flight::{
    CorridorBuildSpec, CorridorProductRecipe, XFeatRuntimeConfig, build_corridor_bank_on_surface,
    write_corridor_product,
};
use std::io::Cursor;
use std::path::Path;

const LAT: f64 = 41.11;
const LON: f64 = -75.35;

#[cfg(test)]
fn synthetic_xfeat_weights() -> Weights {
    let specs: &[(&str, [usize; 3])] = &[
        ("block1.0", [4, 1, 3]),
        ("block1.1", [8, 4, 3]),
        ("block1.2", [8, 8, 3]),
        ("block1.3", [24, 8, 3]),
        ("skip1.1", [24, 1, 1]),
        ("block2.0", [24, 24, 3]),
        ("block2.1", [24, 24, 3]),
        ("block3.0", [64, 24, 3]),
        ("block3.1", [64, 64, 3]),
        ("block3.2", [64, 64, 1]),
        ("block4.0", [64, 64, 3]),
        ("block4.1", [64, 64, 3]),
        ("block4.2", [64, 64, 3]),
        ("block5.0", [128, 64, 3]),
        ("block5.1", [128, 128, 3]),
        ("block5.2", [128, 128, 3]),
        ("block5.3", [64, 128, 1]),
        ("block_fusion.0", [64, 64, 3]),
        ("block_fusion.1", [64, 64, 3]),
        ("block_fusion.2", [64, 64, 1]),
        ("heatmap_head.0", [64, 64, 1]),
        ("heatmap_head.1", [64, 64, 1]),
        ("heatmap_head.2", [1, 64, 1]),
        ("keypoint_head.0", [64, 64, 1]),
        ("keypoint_head.1", [64, 64, 1]),
        ("keypoint_head.2", [64, 64, 1]),
        ("keypoint_head.3", [65, 64, 1]),
    ];
    let mut weights = Weights::default();
    for &(name, [out, input, k]) in specs {
        let len = out * input * k * k;
        let scale = ((2.0 / ((input * k * k) as f32)).sqrt() / 0.09).min(1.0);
        let values = (0..len)
            .map(|i| (((i * 23 + 7) % 37) as f32 - 18.0) * 0.015 * scale)
            .collect();
        weights.insert(
            format!("{name}.weight"),
            WeightTensor {
                shape: vec![out, input, k, k],
                values,
            },
        );
        let mut bias = vec![0.005; out];
        if name == "keypoint_head.3" {
            bias.fill(0.0);
            bias[27] = 3.5;
        }
        weights.insert(
            format!("{name}.bias"),
            WeightTensor {
                shape: vec![out],
                values: bias,
            },
        );
    }
    weights
}

#[cfg(test)]
fn png(image: &RgbImage) -> Vec<u8> {
    let mut out = Cursor::new(Vec::new());
    image.write_to(&mut out, ImageFormat::Png).unwrap();
    out.into_inner()
}

#[cfg(test)]
fn layer(id: &str, kind: LayerKind, datum: Option<VerticalDatum>) -> Layer {
    Layer {
        id: id.into(),
        kind,
        source: SourceRef {
            id: format!("synthetic-{id}"),
            revision: "1".into(),
            attribution: String::new(),
        },
        encoding: Encoding::Png,
        tile_pixels: 256,
        vertical_datum: datum,
        accuracy: Accuracy::unknown(),
        zooms: Vec::new(),
    }
}

/// The synthetic terrain of the corridor suite, as a package.
#[cfg(test)]
fn write_package(store: &mut DirectoryStore) -> Package {
    let mut writer = PackageWriter::new("corridor");
    writer
        .add_layer(layer("imagery", LayerKind::Imagery, None))
        .unwrap();
    let datum = VerticalDatum::Orthometric {
        geoid_model: "egm96".into(),
    };
    writer
        .add_layer(layer("elevation", LayerKind::Elevation, Some(datum)))
        .unwrap();
    let dem = RgbImage::from_fn(256, 256, |x, y| {
        Rgb([128, (((x * 3 + y * 5) % 40) as u8).saturating_add(160), 0])
    });
    let centre12 = TileAddress::containing([LAT, LON], 12).unwrap();
    let centre14 = TileAddress::containing([LAT, LON], 14).unwrap();
    for oy in -1..=1_i32 {
        for ox in -1..=1_i32 {
            let a = TileAddress::new(
                12,
                centre12.x.wrapping_add_signed(ox),
                centre12.y.wrapping_add_signed(oy),
            )
            .unwrap();
            pollster::block_on(writer.add_tile(store, "elevation", a, &png(&dem))).unwrap();
            let salt = ((ox + 2) * 53 + (oy + 2) * 97) as u32;
            let img = RgbImage::from_fn(256, 256, |x, y| {
                let h1 = ((x.wrapping_mul(37) ^ y.wrapping_mul(73) ^ salt) & 0x7f) as u8;
                let patch = ((((x / 16) * 29 + (y / 16) * 43 + salt) & 0x3f) as u8) + 40;
                let road = if (x + salt) % 47 < 3 || (y + salt) % 53 < 3 {
                    95_u8
                } else {
                    0
                };
                Rgb([
                    patch.saturating_add(road).saturating_add(h1 / 3),
                    patch.wrapping_add(h1 / 2),
                    patch / 2 + road / 2,
                ])
            });
            let a = TileAddress::new(
                14,
                centre14.x.wrapping_add_signed(ox),
                centre14.y.wrapping_add_signed(oy),
            )
            .unwrap();
            pollster::block_on(writer.add_tile(store, "imagery", a, &png(&img))).unwrap();
        }
    }
    let identity = pollster::block_on(writer.seal(store)).unwrap();
    pollster::block_on(Package::open(store, identity)).unwrap()
}

#[cfg(test)]
fn spec(dir: &Path, weights: &Path) -> CorridorBuildSpec {
    CorridorBuildSpec {
        catalog_path: dir.join("unused-catalog.json"),
        priors: vec![PriorWaypoint {
            lat_deg: LAT,
            lon_deg: LON,
            alt_msl_m: None,
            heading_deg: None,
        }],
        xfeat_weights_path: weights.to_path_buf(),
        output_bank_path: dir.join("corridor.nvktile"),
        tile_size_m: 15_000.0,
        corridor_radius_m: 0.0,
        nominal_agl_m: 900.0,
        max_landmarks_per_tile: 512,
        adaptive_stride: 4,
        include_obliques: false,
        xfeat_runtime: XFeatRuntimeConfig::default(),
    }
}

#[test]
fn a_corridor_bank_built_on_a_package_is_a_product_of_that_package() {
    let dir = tempfile::tempdir().unwrap();
    let weights = dir.path().join("xfeat.safetensors");
    let weight_bytes = synthetic_xfeat_weights().to_safetensors(false).unwrap();
    std::fs::write(&weights, &weight_bytes).unwrap();
    let mut store = DirectoryStore::new(dir.path().join("store"));
    let package = write_package(&mut store);
    let layers = SurfaceLayers {
        imagery_layer: "imagery",
        imagery_zoom: 14,
        elevation_layer: "elevation",
        elevation_zoom: 12,
    };
    let surface = pollster::block_on(Surface::from_package(&package, &store, &layers)).unwrap();
    assert_eq!(surface.catalog_sha256, package.identity().to_string());

    let spec = spec(dir.path(), &weights);
    let stats = build_corridor_bank_on_surface(&spec, &surface, package.identity().0).unwrap();
    assert!(stats.tiles >= 1, "{stats:?}");
    let bank_bytes = std::fs::read(&spec.output_bank_path).unwrap();
    let extra = CorridorProductRecipe {
        weights_sha256: Sha256Digest::of(&weight_bytes),
        tile_frame_origin: [surface.origin_lat, surface.origin_lon],
    };
    let id = pollster::block_on(write_corridor_product(
        &mut store,
        package.identity(),
        &spec,
        &extra,
        &bank_bytes,
    ))
    .unwrap();

    let product = pollster::block_on(Product::open(&store, id, &package)).unwrap();
    let manifest = product.manifest();
    assert_eq!(manifest.kind, ProductKind::CorridorLandmarks);
    assert_eq!(
        manifest.recipe.weights,
        Some(Sha256Digest::of(&weight_bytes))
    );
    for key in [
        "tile_size_m",
        "nominal_agl_m",
        "kashin_seed",
        "encoder_contract",
        "tile_frame",
        "xfeat_runtime",
    ] {
        assert!(manifest.recipe.parameters.contains_key(key), "{key}");
    }
    assert_eq!(manifest.altitude_band, Some(3));
    let entry = manifest.entries[0].clone();
    let bank = CorridorLandmarkBank::from_bytes(
        &pollster::block_on(product.entry_bytes(&store, &entry)).unwrap(),
    )
    .unwrap();
    assert_eq!(bank.catalog_sha256, package.identity().0);
    assert!(bank.total_landmarks() >= 32);

    let other = Sha256Digest::of(b"another package");
    assert!(
        pollster::block_on(write_corridor_product(
            &mut store,
            other,
            &spec,
            &extra,
            &bank_bytes
        ))
        .is_err()
    );
}
