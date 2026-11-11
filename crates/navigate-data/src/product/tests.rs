use super::*;
use crate::manifest::tests::{IMAGERY, imagery_spec};
use crate::{Descriptor, MemoryFile, MemoryStore, Package, PackageWriter, TileAddress, blob_path};
use pollster::block_on;

#[cfg(test)]
fn package(store: &mut MemoryStore, tiles: &[&[u8]]) -> Package<MemoryFile> {
    let mut writer = PackageWriter::new();
    writer.add_layer(imagery_spec()).unwrap();
    for (i, bytes) in tiles.iter().enumerate() {
        let a = TileAddress::new(12, 100 + i as u32, 200).unwrap();
        block_on(writer.add_tile(store, IMAGERY, a, bytes)).unwrap();
    }
    let id = block_on(writer.seal(store, &Descriptor::default())).unwrap();
    block_on(Package::open(store, id)).unwrap()
}

#[cfg(test)]
fn recipe(spacing_mm: i64) -> Recipe {
    Recipe::new("test-render", "1", None, Precision::F32).with("spacing_mm", spacing_mm)
}

#[cfg(test)]
fn binding(inputs: &[&[u8]], band: u32) -> ChunkBinding {
    ChunkBinding {
        cell: None,
        altitude_band: Some(band),
        inputs: inputs.iter().map(|b| Sha256Digest::of(b)).collect(),
        route: Some(route_digest(&[[41.7, -1.0, 1300.0], [41.7, -0.9, 1300.0]])),
        envelope: Some(ValidityEnvelope {
            lateral_mm: 150_000,
            vertical_mm: 100_000,
            heading_urad: 200_000,
        }),
        calibration: None,
    }
}

#[cfg(test)]
fn chunk(store: &mut MemoryStore, spacing_mm: i64, band: u32, keys: &[u32]) -> Sha256Digest {
    let mut writer = ProductWriter::new(
        ProductKind::Keyframes,
        recipe(spacing_mm),
        binding(&[b"b", b"a"], band),
    );
    for &key in keys {
        let pose = PlannedPose::from_si([41.7, -1.0 + f64::from(key) * 0.01], 1300.0, 1.5);
        block_on(writer.add_entry(store, key, Some(pose), &key.to_le_bytes())).unwrap();
    }
    block_on(writer.seal(store)).unwrap()
}

#[test]
fn a_chunk_is_identified_by_its_recipe_and_binding_not_by_a_package() {
    let mut store = MemoryStore::default();
    let id = chunk(&mut store, 500_000, 3, &[1, 0]);
    assert_eq!(id, chunk(&mut store, 500_000, 3, &[0, 1]));
    assert_ne!(id, chunk(&mut store, 250_000, 3, &[0, 1]));
    assert_ne!(id, chunk(&mut store, 500_000, 4, &[0, 1]));
    let opened = block_on(open_chunk(&store, id)).unwrap();
    let mut expected = [Sha256Digest::of(b"a"), Sha256Digest::of(b"b")];
    expected.sort_unstable();
    assert_eq!(opened.binding.inputs, expected);
    assert_eq!(
        block_on(read_recipe(&store, opened.recipe)).unwrap(),
        recipe(500_000)
    );
    let text = core::str::from_utf8(store.get(&chunk_path(id)).unwrap()).unwrap();
    assert!(
        !text.replace(CHUNK_FORMAT, "").contains('.'),
        "a float: {text}"
    );
}

#[test]
fn two_missions_with_the_same_inputs_use_one_chunk() {
    let mut store = MemoryStore::default();
    let first = package(&mut store, &[b"a", b"b"]);
    let second = package(&mut store, &[b"a", b"b", b"c"]);
    let third = package(&mut store, &[b"a"]);
    let id = chunk(&mut store, 500_000, 3, &[0]);
    let opened = block_on(open_chunk(&store, id)).unwrap();
    assert!(opened.applies_to(&block_on(first.blobs()).unwrap()));
    assert!(opened.applies_to(&block_on(second.blobs()).unwrap()));
    assert!(!opened.applies_to(&block_on(third.blobs()).unwrap()));
    let list_a = block_on(write_list(
        &mut store,
        first.identity(),
        &[(KindName::Known(ProductKind::Keyframes), id)],
    ))
    .unwrap();
    let list_b = block_on(write_list(
        &mut store,
        second.identity(),
        &[(KindName::Known(ProductKind::Keyframes), id)],
    ))
    .unwrap();
    assert_ne!(list_a, list_b);
    let list = block_on(open_list(&store, list_a, first.identity())).unwrap();
    assert_eq!(list.chunks[0].chunk, id);
    assert!(block_on(open_list(&store, list_a, second.identity())).is_err());
}

#[test]
fn unknown_kinds_are_carried_but_never_read() {
    let mut store = MemoryStore::default();
    let core = package(&mut store, &[b"a"]);
    let known = chunk(&mut store, 500_000, 3, &[0]);
    let future = Sha256Digest::of(b"a chunk of a future kind");
    let kinds = [
        (KindName::Known(ProductKind::Keyframes), known),
        (KindName::Unknown("gravity_gradient".into()), future),
    ];
    let id = block_on(write_list(&mut store, core.identity(), &kinds)).unwrap();
    let list = block_on(open_list(&store, id, core.identity())).unwrap();
    let unknown: Vec<_> = list.unknown_kinds().collect();
    assert_eq!(unknown.len(), 1);
    assert_eq!(
        unknown[0].kind,
        KindName::Unknown("gravity_gradient".into())
    );
    let mut forged = block_on(open_chunk(&store, known)).unwrap();
    forged.kind = KindName::Unknown("gravity_gradient".into());
    let (bytes, digest) = forged.canonical().unwrap();
    store.replace(&chunk_path(digest), &bytes);
    assert_eq!(
        block_on(open_chunk(&store, digest)),
        Err(DataError::Unsupported {
            field: "product kind"
        })
    );
}

#[test]
fn changed_documents_and_other_versions_are_refused() {
    let mut store = MemoryStore::default();
    let id = chunk(&mut store, 500_000, 3, &[0]);
    let mut opened = block_on(open_chunk(&store, id)).unwrap();
    let entry = opened.entries[0];
    store.replace(&blob_path(entry.blob), b"changed");
    assert!(matches!(
        block_on(opened.entry_bytes(&store, &entry)),
        Err(DataError::Integrity { .. })
    ));
    opened.version = 2;
    assert!(matches!(
        opened.validate(),
        Err(DataError::Version { found: 2, .. })
    ));
    store.replace(&chunk_path(id), b"{}");
    assert!(matches!(
        block_on(open_chunk(&store, id)),
        Err(DataError::Integrity { .. })
    ));
}

#[test]
fn poses_round_to_integer_units() {
    let pose = PlannedPose::from_si([41.123_456_789, -1.5], 1_300.25, 0.5);
    assert_eq!(pose.lat_ndeg, 41_123_456_789);
    assert_eq!(pose.height_mm, 1_300_250);
    let ([lat, _], h, heading) = pose.to_si();
    assert!(
        (lat - 41.123_456_789).abs() < 1e-9
            && (h - 1_300.25).abs() < 1e-9
            && (heading - 0.5).abs() < 1e-6
    );
}
