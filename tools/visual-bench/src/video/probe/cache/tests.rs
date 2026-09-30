use super::*;
fn camera() -> CameraModel {
    CameraModel {
        width: 128,
        height: 96,
        fx: 100.0,
        fy: 100.0,
        cx: 63.5,
        cy: 47.5,
    }
}
#[test]
fn reuse_requires_source_digest_dimensions_and_monotonic_timestamps() -> Result<(), BenchError> {
    let root = tempfile::tempdir().map_err(|source| BenchError::Io {
        path: "<timeline-test>".into(),
        source,
    })?;
    let video = root.path().join("video.bin");
    let cache = root.path().join("timeline.json");
    crate::write_blocking(&video, b"source-content")?;
    let mut record = Timeline {
        video_sha256: digest_blocking(&video)?,
        source_size: [128, 96],
        timestamps: vec![0, 100, 250],
    };
    let write = |record: &Timeline| -> Result<(), BenchError> {
        let bytes = serde_json::to_vec(record).map_err(|source| BenchError::Json {
            path: cache.clone(),
            source,
        })?;
        crate::write_blocking(&cache, &bytes)
    };
    write(&record)?;
    assert_eq!(
        timestamps_blocking(&video, camera(), &cache)?,
        [0, 100, 250]
    );
    let mut wrong = camera();
    wrong.width = 64;
    assert!(timestamps_blocking(&video, wrong, &cache).is_err());
    record.timestamps = vec![0, 250, 100];
    write(&record)?;
    assert!(timestamps_blocking(&video, camera(), &cache).is_err());
    record.timestamps = vec![0, 100, 250];
    write(&record)?;
    crate::write_blocking(&video, b"changed-content")?;
    assert!(timestamps_blocking(&video, camera(), &cache).is_err());
    Ok(())
}
