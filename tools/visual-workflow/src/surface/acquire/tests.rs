use super::*;
#[test]
fn insufficient_search_budget_fails_before_loading_images_or_models() {
    let root = tempfile::tempdir().unwrap();
    let input = root.path().join("job.json");
    let output = root.path().join("result.json");
    let job = json!({
        "query":{"path":"missing-query.png","sha256":"a".repeat(64)},
        "camera":{"size":[64,64],"focal":[55.0,55.0],"principal":[31.5,31.5]},
        "prior":{"position":[0.0,0.0,100.0],"rotation":[0.0,0.0,0.0,1.0]},
        "position_radius_m":1000.0,"attitude_radius_rad":std::f64::consts::PI,
        "reference_quarter_turns":[0,1,2,3],
        "reference":{"image":{"path":"r.png","sha256":"a".repeat(64)},
        "heights":{"path":"h.bin","sha256":"b".repeat(64)},
        "origin":[0.0,0.0],"column":[1.0,0.0],"row":[0.0,-1.0],
        "map_release":"test","map_sha256":"c".repeat(64)}});
    assets::write_json_blocking(&input, &job).unwrap();
    let mut config: Config = serde_json::from_value(json!({
        "runtime":"missing","detector":"missing","graph":"dense_image",
        "extraction":{"provider":"cpu","threads":1,"allow_cpu_fallback":false},
        "matching":{"provider":"cpu","threads":1,"allow_cpu_fallback":false},
        "keypoints":2048,"limits":{"image_pixels":41000000,"features_per_image":2048,
            "candidates":76,"reference_bytes":268435456,"query_feature_bytes":4194304}
    }))
    .unwrap();
    for (limit, probe_fine_after_pass) in [0, 4, 75]
        .into_iter()
        .flat_map(|limit| [false, true].map(|probe| (limit, probe)))
    {
        config.limits.candidates = limit;
        let error = run_blocking(
            &config,
            root.path(),
            &input,
            &output,
            probe_fine_after_pass,
            None,
        )
        .unwrap_err();
        assert!(
            matches!(error,Error::Input(ref message) if message.contains("budget of at least 76"))
        );
        assert!(!output.exists());
    }
    config.limits.candidates = 76;
    assert!(matches!(
        run_blocking(&config, root.path(), &input, &output, false, None),
        Err(Error::Io { .. })
    ));
}
