use super::*;
use navigate_visual::MapRevision;

fn catalog() -> ReferenceCatalog {
    ReferenceCatalog {
        map: MapRevision {
            release_id: "test-map".into(),
            manifest_sha256: "a".repeat(64),
        },
        manifest_sha256: "b".repeat(64),
    }
}
fn basis(axis: usize) -> Vec<f32> {
    let mut v = vec![0.0; 1024];
    v[axis] = 1.0;
    v
}
fn index() -> Result<CampIndex, InferenceError> {
    CampIndex::new(
        catalog(),
        "c".repeat(64),
        vec![ReferenceId(7), ReferenceId(2), ReferenceId(5)],
        [basis(0), basis(1), basis(0)].concat(),
    )
}

#[test]
fn ranking_uses_only_eligible_ids_and_keeps_ties_deterministic()
-> Result<(), Box<dyn std::error::Error>> {
    let index = index()?;
    let rows = index.eligible_rows(&[ReferenceId(7), ReferenceId(2), ReferenceId(5)], 2)?;
    assert_eq!(
        index.rank(&basis(0), &rows, 2),
        [ReferenceId(5), ReferenceId(7)]
    );
    assert_eq!(index.rank(&basis(1), &rows, 1), [ReferenceId(2)]);
    let rows = index.eligible_rows(&[ReferenceId(7)], 3)?;
    assert_eq!(index.rank(&basis(1), &rows, 3), [ReferenceId(7)]);
    assert!(index.rank(&basis(1), &rows, 0).is_empty());
    Ok(())
}

#[test]
fn mismatched_and_repeated_reference_ids_are_errors() -> Result<(), Box<dyn std::error::Error>> {
    let index = index()?;
    assert!(
        matches!(index.eligible_rows(&[ReferenceId(99)], 1), Err(RetrievalError::Invalid(message)) if message.contains("99"))
    );
    assert!(
        index
            .eligible_rows(&[ReferenceId(7), ReferenceId(7)], 2)
            .is_err()
    );
    assert!(index.eligible_rows(&[], 4097).is_err());
    assert!(
        CampIndex::new(
            catalog(),
            "c".repeat(64),
            vec![ReferenceId(7), ReferenceId(7)],
            [basis(0), basis(1)].concat()
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn map_catalog_and_model_content_change_the_index_identity() -> Result<(), InferenceError> {
    let build = |catalog, model| CampIndex::new(catalog, model, vec![ReferenceId(7)], basis(0));
    let first = build(catalog(), "c".repeat(64))?;
    let mut changed = catalog();
    changed.map.release_id = "other-map".into();
    assert_ne!(first.digest, build(changed, "c".repeat(64))?.digest);
    let mut changed = catalog();
    changed.manifest_sha256 = "d".repeat(64);
    assert_ne!(first.digest, build(changed, "c".repeat(64))?.digest);
    assert_ne!(first.digest, build(catalog(), "e".repeat(64))?.digest);
    Ok(())
}

#[test]
fn invalid_descriptor_rows_cannot_enter_the_index() {
    for values in [
        vec![0.0; 1024],
        vec![f32::NAN; 1024],
        vec![1.0; 1024],
        vec![0.0; 1023],
    ] {
        assert!(CampIndex::new(catalog(), "c".repeat(64), vec![ReferenceId(1)], values).is_err());
    }
    assert!(
        CampIndex::new(
            catalog(),
            "not-a-hash".into(),
            vec![ReferenceId(1)],
            basis(0)
        )
        .is_err()
    );
}

#[test]
fn wrong_model_is_rejected_before_runtime_initialization() -> Result<(), InferenceError> {
    let model = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/assets/add.onnx");
    let result =
        crate::CampRetriever::load_blocking(&model, index()?, &crate::ExecutionConfig::default());
    assert!(
        matches!(result, Err(InferenceError::Invalid(message)) if message.contains("does not match index model"))
    );
    Ok(())
}
