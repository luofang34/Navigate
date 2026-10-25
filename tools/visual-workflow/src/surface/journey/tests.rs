use super::*;

#[test]
fn optional_recovery_budget_is_checked_before_manifest_or_model_io() {
    let config: Config = serde_json::from_value(json!({
        "runtime":"missing","detector":"missing","graph":"dense_image",
        "extraction":{"provider":"cpu","threads":1,"allow_cpu_fallback":false},
        "matching":{"provider":"cpu","threads":1,"allow_cpu_fallback":false},
        "keypoints":2048,"limits":{"image_pixels":41000000,"features_per_image":2048,
            "candidates":76,"reference_bytes":268435456,"query_feature_bytes":4194304}
    }))
    .unwrap();
    let root = tempfile::tempdir().unwrap();
    let manifest = root.path().join("missing.json");
    let output = root.path().join("out.json");
    for enabled in [false, true] {
        let options = Options {
            retained_window_recovery: enabled,
            reload_model_on_recovery: false,
            reference_cache_mib: 0,
            gpu_image_reuse: false,
            arrival: arrival::Options::new("offline", 64, "ready").unwrap(),
            feature_catalog: None,
        };
        let error = run_blocking(&config, root.path(), &manifest, &output, options).unwrap_err();
        if enabled {
            assert!(matches!(error, Error::Input(ref s) if s.contains("80 candidates")));
        } else {
            assert!(matches!(error, Error::Io { .. }));
        }
        assert!(!output.exists());
    }
    assert!(validate_limits(32, true, 80).is_ok());
    assert!(validate_limits(65, true, 80).is_err());
}
