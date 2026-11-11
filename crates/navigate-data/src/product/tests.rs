use super::*;
use crate::manifest::tests::imagery_layer;
use crate::{MemoryStore, PackageWriter, TileAddress};
use pollster::block_on;

#[cfg(test)]
fn core(store: &mut MemoryStore, release: &str) -> Package {
    let mut writer = PackageWriter::new(release);
    writer.add_layer(imagery_layer()).unwrap();
    let a = TileAddress::new(12, 1, 2).unwrap();
    block_on(writer.add_tile(store, "imagery", a, b"tile")).unwrap();
    let id = block_on(writer.seal(store)).unwrap();
    block_on(Package::open(store, id)).unwrap()
}

#[cfg(test)]
fn recipe(step: &str) -> Recipe {
    Recipe {
        algorithm: "test-render".into(),
        version: "1".into(),
        weights: None,
        precision: "f32".into(),
        parameters: BTreeMap::from([("step_m".into(), step.into())]),
    }
}

#[cfg(test)]
fn write(store: &mut MemoryStore, core: &Package, step: &str, order: &[&str]) -> Sha256Digest {
    let mut writer = ProductWriter::new(ProductKind::Keyframes, core.identity(), recipe(step))
        .with_altitude_band(3)
        .with_envelope(ValidityEnvelope {
            lateral_m: 150.0,
            vertical_m: 100.0,
            heading_rad: 0.2,
        });
    for key in order {
        let pose = PlannedPose {
            position: [41.7, -1.0, 1300.0],
            heading_rad: 1.0,
        };
        block_on(writer.add_entry(store, key, Some(pose), key.as_bytes())).unwrap();
    }
    block_on(writer.seal(store)).unwrap()
}

#[test]
fn a_product_binds_its_core_and_its_recipe() {
    let mut store = MemoryStore::default();
    let first = core(&mut store, "r1");
    let id = write(&mut store, &first, "500", &["k0001", "k0000"]);
    assert_eq!(id, write(&mut store, &first, "500", &["k0000", "k0001"]));
    assert_ne!(id, write(&mut store, &first, "250", &["k0000", "k0001"]));
    let product = block_on(Product::open(&store, id, &first)).unwrap();
    assert_eq!(product.manifest().entries.len(), 2);
    let entry = product.manifest().entries[1].clone();
    assert_eq!(
        block_on(product.entry_bytes(&store, &entry)).unwrap(),
        b"k0001"
    );
    let second = core(&mut store, "r2");
    assert!(matches!(
        block_on(Product::open(&store, id, &second)),
        Err(DataError::Manifest { .. })
    ));
}

#[test]
fn a_changed_entry_blob_is_found() {
    let mut store = MemoryStore::default();
    let first = core(&mut store, "r1");
    let id = write(&mut store, &first, "500", &["k0000"]);
    let product = block_on(Product::open(&store, id, &first)).unwrap();
    let entry = product.manifest().entries[0].clone();
    store
        .files
        .insert(blob_path(entry.blob), b"changed".to_vec());
    assert!(matches!(
        block_on(product.entry_bytes(&store, &entry)),
        Err(DataError::Integrity { .. })
    ));
}

#[test]
fn product_rule_breaks_are_refused() {
    let mut store = MemoryStore::default();
    let first = core(&mut store, "r1");
    let mut writer = ProductWriter::new(
        ProductKind::CorridorLandmarks,
        first.identity(),
        recipe("1"),
    );
    block_on(writer.add_entry(&mut store, "a", None, b"x")).unwrap();
    assert!(block_on(writer.add_entry(&mut store, "a", None, b"y")).is_err());
    let mut manifest = writer.manifest().unwrap();
    manifest.recipe.algorithm.clear();
    assert!(manifest.validate().is_err());
    let nan = ProductWriter::new(ProductKind::Keyframes, first.identity(), recipe("1"))
        .with_envelope(ValidityEnvelope {
            lateral_m: f64::NAN,
            vertical_m: 1.0,
            heading_rad: 0.1,
        });
    assert!(nan.manifest().is_err());
}
