use super::*;
#[test]
fn stage_recipe_rejects_unknown_models_and_invalid_batch_limits() {
    assert!(matches!(
        prepare(b"unknown", None),
        Err(Error::Digest { .. })
    ));
    for batch in [0, 1025, usize::MAX] {
        assert!(matches!(
            prepare(b"unknown", Some(batch)),
            Err(Error::Invalid(_))
        ));
    }
}
#[test]
fn stage_boundaries_cut_image_dependencies_and_keep_cross_patch_inputs() -> Result<(), Error> {
    let mut model = GraphProto {
        input: vec![graph::value("image0", 1, &[1, 1, 480, 640])],
        node: vec![
            graph::node("Identity", &["image0"], PATCHES[0]),
            graph::node("Identity", &["image0"], PATCHES[1]),
            graph::node("Identity", &["image0"], QUERY),
            graph::node("Identity", &["image0"], "keypoints0"),
            graph::node("Identity", &["image0"], "confidence"),
            graph::node("Add", &PATCHES, OFFSETS),
        ],
        ..Default::default()
    };
    transform(&mut model, Some(64))?;
    assert_eq!(model.node.len(), 2);
    assert_eq!(model.node[0].input, ["patches0", "patches1"]);
    assert_eq!(model.output[0], graph::value("offsets", 1, &[64, 2]));
    assert_eq!(model.input[0], graph::value("patches0", 1, &[64, 25, 128]));
    Ok(())
}
