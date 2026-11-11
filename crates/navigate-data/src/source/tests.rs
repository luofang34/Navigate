use super::*;
use crate::manifest::tests::imagery_spec;
use crate::{Descriptor, LayerKey, LayerSpec, MemoryStore, Package, PackageWriter, TileAddress};
use pollster::block_on;

#[cfg(test)]
fn declaration(sheet: &str) -> SourceDeclaration {
    SourceDeclaration {
        version: SOURCE_FORMAT_VERSION,
        kind: LayerKind::Imagery,
        ground_sample_mm: 250,
        horizontal_mm: Some(500),
        vertical_mm: None,
        collected: Some(DayNumber(9_000)),
        provider_id: Sha256Digest::of(sheet.as_bytes()),
    }
}

#[test]
fn keys_have_one_text_form() {
    let declared = declaration("agency:ortho-2024:sheet-17").key().unwrap();
    for key in [
        SourceKey::EsriWorldImagery,
        SourceKey::MapzenTerrarium,
        SourceKey::Synthetic,
        declared,
    ] {
        let text = serde_json::to_string(&key).unwrap();
        assert_eq!(
            serde_json::from_str::<SourceKey>(&text).unwrap(),
            key,
            "{text}"
        );
    }
    assert!(alloc::format!("{declared}").starts_with("declared:"));
    assert_eq!(SourceKey::parse("someone"), SourceKey::Unsupported);
    assert_eq!(SourceKey::parse("declared:XYZ"), SourceKey::Unsupported);
}

#[test]
fn a_declaration_is_canonical_and_checked() {
    let a = declaration("sheet-17");
    assert_eq!(a.key().unwrap(), a.key().unwrap());
    assert_ne!(a.key().unwrap(), declaration("sheet-18").key().unwrap());
    assert!(SourceDeclaration { version: 2, ..a }.key().is_err());
    assert!(
        SourceDeclaration {
            ground_sample_mm: 0,
            ..a
        }
        .key()
        .is_err()
    );
    assert!(
        !core::str::from_utf8(&a.canonical_bytes().unwrap())
            .unwrap()
            .contains('.')
    );
}

#[cfg(test)]
fn package_with(
    store: &mut MemoryStore,
    source: SourceKey,
    tiles: &[&[u8]],
) -> Package<crate::MemoryFile> {
    let key = LayerKey {
        kind: LayerKind::Imagery,
        source,
    };
    let mut writer = PackageWriter::new();
    writer
        .add_layer(LayerSpec {
            key,
            ..imagery_spec()
        })
        .unwrap();
    for (i, bytes) in tiles.iter().enumerate() {
        let a = TileAddress::new(12, 100 + i as u32, 200).unwrap();
        block_on(writer.add_tile(store, key, a, bytes)).unwrap();
    }
    let id = block_on(writer.seal(store, &Descriptor::default())).unwrap();
    block_on(Package::open(store, id)).unwrap()
}

#[test]
fn packages_of_one_declared_source_share_correlation_keys() {
    let mut store = MemoryStore::default();
    let source = declaration("sheet-17").key().unwrap();
    let other = declaration("sheet-18").key().unwrap();
    let a = package_with(&mut store, source, &[b"a", b"b"]);
    let b = package_with(&mut store, source, &[b"a"]);
    let c = package_with(&mut store, other, &[b"a", b"b"]);
    assert_ne!(a.identity(), b.identity());
    let point = [41.7, -1.0];
    let key =
        |p: &Package<crate::MemoryFile>| p.manifest().layers[0].correlation_key(point).unwrap();
    assert_eq!(key(&a), key(&b));
    assert_ne!(key(&a), key(&c));
}

#[test]
fn keys_and_digests_decode_from_owned_values() {
    let declared = declaration("sheet-17").key().unwrap();
    for key in [SourceKey::EsriWorldImagery, declared] {
        let value = serde_json::to_value(key).unwrap();
        assert_eq!(serde_json::from_value::<SourceKey>(value).unwrap(), key);
    }
    let digest = Sha256Digest::of(b"x");
    assert_eq!(
        serde_json::from_value::<Sha256Digest>(serde_json::to_value(digest).unwrap()).unwrap(),
        digest
    );
}
