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

#[derive(serde::Deserialize)]
struct SearchFixture {
    config: std::path::PathBuf,
    job: std::path::PathBuf,
    failed_query: crate::job::Image,
}

#[test]
#[ignore = "set NAVIGATE_ACQUISITION_FIXTURE to a model, map and image fixture JSON"]
fn requested_hypotheses_preserve_selection_and_failed_search_coverage() {
    let path = std::env::var_os("NAVIGATE_ACQUISITION_FIXTURE").unwrap();
    let fixture: SearchFixture = assets::read_json_blocking(Path::new(&path)).unwrap();
    let config: Config = assets::read_json_blocking(&fixture.config).unwrap();
    let job: SurfaceJob = assets::read_json_blocking(&fixture.job).unwrap();
    validate(&config, &job).unwrap();
    let (mut frame, reference, prior) = job
        .load_blocking(fixture.job.parent().unwrap(), &config)
        .unwrap();
    let mut engine = Engine::new(
        config
            .load_blocking(fixture.config.parent().unwrap())
            .unwrap(),
    )
    .unwrap();
    for failed in [false, true] {
        if failed {
            frame.image = assets::image_verified_blocking(
                &fixture.failed_query.path,
                config.limits.image_pixels,
                Some(&fixture.failed_query.sha256),
            )
            .unwrap();
        }
        engine.query_blocking(&frame.image, &config).unwrap();
        let diagnostic = Trial {
            frame: &frame,
            reference: &reference,
            prior: &prior,
            verifier: PoseVerifier::new(LocalizerConfig::default()).unwrap(),
            rectification: None,
            record_matches: true,
            hypothesis_limit: 4,
            stop_after_hypotheses: false,
            probe_fine_after_pass: false,
            config: &config,
        }
        .run_blocking(&mut engine)
        .unwrap();
        let expected: Vec<_> = diagnostic
            .selected
            .iter()
            .chain(&diagnostic.alternatives)
            .collect();
        let all_rows = geometry_rows(&diagnostic.record);
        assert_eq!(expected.len(), if failed { 0 } else { 4 });
        if failed {
            assert_eq!(all_rows.len(), 76);
        }
        for limit in [1, 2, 4] {
            engine.query_blocking(&frame.image, &config).unwrap();
            let actual = search_hypotheses_blocking(
                &frame,
                &reference,
                &prior,
                &mut engine,
                &config,
                true,
                limit,
            )
            .unwrap();
            let selected: Vec<_> = actual.selected.iter().chain(&actual.alternatives).collect();
            assert_eq!(selected.len(), expected.len().min(limit));
            for (a, b) in selected.into_iter().zip(&expected) {
                assert_selected_equal(a, b);
            }
            let rows = geometry_rows(&actual.record);
            assert_eq!(rows, all_rows[..rows.len()]);
            if failed {
                assert_eq!(rows.len(), all_rows.len());
            } else {
                assert!(rows.len() < all_rows.len());
            }
        }
    }
}

fn geometry_rows(record: &Value) -> Vec<Value> {
    let mut rows = record["coarse"].as_array().unwrap().clone();
    for field in ["windows", "fine_windows"] {
        for window in record[field].as_array().unwrap() {
            rows.extend(window["candidates"].as_array().unwrap().iter().cloned());
        }
    }
    for row in &mut rows {
        // Compare evidence and geometry; preparation depends on execution history.
        row.as_object_mut().unwrap().retain(|key, _| {
            !key.ends_with("_ms") && key != "reference_packet" && key != "reference_cache_hit"
        });
    }
    rows
}

fn assert_selected_equal(a: &Acquired, b: &Acquired) {
    assert_eq!(a.window, b.window);
    assert_eq!(a.rotation, b.rotation);
    assert_eq!(a.estimate.pose.position, b.estimate.pose.position);
    assert_eq!(a.estimate.pose.orientation, b.estimate.pose.orientation);
    assert_eq!(
        a.estimate.geometry_covariance,
        b.estimate.geometry_covariance
    );
    assert_eq!(a.pairs.len(), b.pairs.len());
    for (a, b) in a.pairs.iter().zip(&b.pairs) {
        assert_eq!(a.query, b.query);
        assert_eq!(a.reference, b.reference);
    }
}
