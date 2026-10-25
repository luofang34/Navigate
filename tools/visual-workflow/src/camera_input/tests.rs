use super::*;

fn fixture_blocking(root: &Path) -> (std::path::PathBuf, serde_json::Value) {
    let image = GrayImage::from_fn(64, 64, |x, y| image::Luma([(x + 2 * y) as u8]));
    image.save(root.join("source.png")).unwrap();
    let bytes = std::fs::read(root.join("source.png")).unwrap();
    let spec = json!({
        "image":{"path":"source.png","sha256":hex::encode(Sha256::digest(bytes))},
        "source_camera":{"size":[64,64],"focal":[50.0,50.0],"principal":[31.5,31.5]},
        "target_camera":{"size":[32,32],"focal":[50.0,50.0],"principal":[15.5,15.5]},
        "distortion":[0.0,0.0,0.0,0.0,0.0],"max_map_bytes":8192
    });
    let path = root.join("spec.json");
    std::fs::write(&path, serde_json::to_vec(&spec).unwrap()).unwrap();
    (path, spec)
}
#[test]
fn prepared_pixels_camera_and_provenance_agree() {
    let dir = tempfile::tempdir().unwrap();
    let (path, spec) = fixture_blocking(dir.path());
    let out = dir.path().join("prepared");
    run_blocking(&path, &out, 3).unwrap();
    let report: serde_json::Value =
        assets::read_json_blocking(&out.join("rectified.json")).unwrap();
    assert_eq!(report["camera"], spec["target_camera"]);
    assert_eq!(report["map_bytes"], 8192);
    assert_eq!(report["warp_ms"].as_array().unwrap().len(), 3);
    let input = assets::image_blocking(&dir.path().join("source.png"), 4096).unwrap();
    let image = assets::image_verified_blocking(
        &out.join("rectified.png"),
        1024,
        report["image"]["sha256"].as_str(),
    )
    .unwrap();
    assert_eq!(
        image,
        image::imageops::crop_imm(&input, 16, 16, 32, 32).to_image()
    );
    assert_eq!(
        report["source"]["image"]["path"],
        dir.path()
            .join("source.png")
            .canonicalize()
            .unwrap()
            .to_str()
            .unwrap()
    );
    assert_eq!(
        report["specification_sha256"],
        hex::encode(Sha256::digest(std::fs::read(&path).unwrap()))
    );
}
#[test]
fn invalid_image_binding_or_coverage_creates_no_output() {
    let dir = tempfile::tempdir().unwrap();
    let (path, spec) = fixture_blocking(dir.path());
    for invalid in ["hash", "coverage", "budget"] {
        let mut changed = spec.clone();
        match invalid {
            "hash" => changed["image"]["sha256"] = json!("0".repeat(64)),
            "coverage" => changed["target_camera"]["principal"] = json!([500.0, 500.0]),
            _ => changed["max_map_bytes"] = json!(1),
        }
        std::fs::write(&path, serde_json::to_vec(&changed).unwrap()).unwrap();
        let out = dir.path().join(invalid);
        assert!(run_blocking(&path, &out, 1).is_err());
        assert!(!out.exists());
    }
}

#[test]
fn acquisition_preparation_binds_pixels_calibration_and_limits() {
    let dir = tempfile::tempdir().unwrap();
    let (path, spec) = fixture_blocking(dir.path());
    let image: Image = serde_json::from_value(spec["image"].clone()).unwrap();
    let camera: Camera = serde_json::from_value(spec["source_camera"].clone()).unwrap();
    let frame = || navigate_visual::Frame {
        image: assets::image_blocking(&dir.path().join("source.png"), 4096).unwrap(),
        camera: camera.model(),
        stamp: navigate_visual::FrameStamp {
            sequence: 1,
            capture_time_ns: 7,
        },
    };
    let (prepared, map, report) =
        prepare_frame_blocking(&path, frame(), &image, dir.path(), 4096).unwrap();
    assert_eq!(prepared.camera, map.camera());
    assert_eq!(prepared.image.dimensions(), (32, 32));
    assert_eq!(report["raw_observation_sha256"], frame().evidence_sha256());
    assert_eq!(
        report["rectified_observation_sha256"],
        prepared.evidence_sha256()
    );
    assert_eq!(prepared.stamp, frame().stamp);
    for invalid in ["hash", "camera", "path", "budget", "pixels"] {
        let mut changed = spec.clone();
        match invalid {
            "hash" => changed["image"]["sha256"] = json!("f".repeat(64)),
            "camera" => changed["source_camera"]["focal"] = json!([51.0, 50.0]),
            "path" => changed["image"]["path"] = json!("missing.png"),
            "budget" => changed["max_map_bytes"] = json!(1),
            _ => {}
        }
        std::fs::write(&path, serde_json::to_vec(&changed).unwrap()).unwrap();
        let limit = if invalid == "pixels" { 64 } else { 4096 };
        assert!(prepare_frame_blocking(&path, frame(), &image, dir.path(), limit).is_err());
    }
}
