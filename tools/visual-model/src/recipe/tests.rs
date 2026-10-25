use super::*;

#[test]
fn preparation_refuses_unknown_exports_and_invalid_budgets() {
    assert!(matches!(
        prepare(b"foreign model", 512, true),
        Err(Error::Digest { .. })
    ));
    assert!(matches!(
        prepare(b"foreign model", 4097, true),
        Err(Error::Invalid(_))
    ));
}

#[test]
fn recipe_changes_the_score_domain_and_removes_unused_log_nodes() -> Result<(), Error> {
    let mut softmax = graph::node("LogSoftmax", &["scores"], "normalized");
    softmax.name = "/LogSoftmax".into();
    let mut multiply = graph::node("Add", &["normalized", "matchability"], "log_assignment");
    multiply.name = "/Add_52".into();
    let mut model = GraphProto {
        input: vec![graph::value("keypoints0", 1, &[1, 1, 2])],
        node: vec![softmax, graph::node("Log", &["orphan"], "unused"), multiply],
        ..Default::default()
    };
    transform(&mut model, 512, true)?;
    assert_eq!(
        model
            .node
            .iter()
            .map(|n| n.op_type.as_str())
            .collect::<Vec<_>>(),
        ["Softmax", "Mul", "ArgMax", "ArgMax", "ReduceMax"]
    );
    assert_eq!(
        model
            .output
            .iter()
            .map(|v| v.name.as_str())
            .collect::<Vec<_>>(),
        ["row_indices", "column_indices", "row_scores"]
    );
    assert_eq!(
        model.input[0].r#type,
        graph::value("keypoints0", 1, &[1, 512, 2]).r#type
    );
    Ok(())
}
