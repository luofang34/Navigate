use super::*;

#[cfg(test)]
fn write_array(path: &Path, dtype: &str, shape: &str, bytes: &[u8]) {
    let mut header = format!("{{'descr': '{dtype}', 'fortran_order': False, 'shape': {shape}, }}");
    while (10 + header.len() + 1) % 64 != 0 {
        header.push(' ');
    }
    header.push('\n');
    let mut contents = b"\x93NUMPY\x01\x00".to_vec();
    contents.extend_from_slice(&(header.len() as u16).to_le_bytes());
    contents.extend_from_slice(header.as_bytes());
    contents.extend_from_slice(bytes);
    std::fs::write(path, contents).unwrap();
}
#[cfg(test)]
fn inputs(root: &Path, warp: bool) {
    for name in ["image_query.npy", "image_dop.npy"] {
        write_array(
            &root.join(name),
            "|u1",
            "(32, 32, 3)",
            &vec![100; 32 * 32 * 3],
        );
    }
    let camera = [30.0_f32, 0.0, 15.5, 0.0, 31.0, 15.5, 0.0, 0.0, 1.0];
    write_array(
        &root.join("intrinsics.npy"),
        "<f4",
        "(3, 3)",
        &camera
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    );
    let mut surface = Vec::new();
    for y in 0..32 {
        for x in 0..32 {
            let dx = if warp && x == 15 && y == 15 { 1.0 } else { 0.0 };
            surface.extend([x as f32 * 0.3 + dx, -(y as f32) * 0.3, -2.0_f32]);
        }
    }
    write_array(
        &root.join("dsm.npy"),
        "<f4",
        "(32, 32, 3)",
        &surface
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect::<Vec<_>>(),
    );
}
#[test]
fn preparation_uses_measured_grid_and_does_not_read_query_pose_truth() {
    let root = tempfile::tempdir().unwrap();
    inputs(root.path(), false);
    std::fs::write(root.path().join("extrinsics.npy"), b"not an array").unwrap();
    run_blocking(root.path(), &root.path().join("a")).unwrap();
    std::fs::write(
        root.path().join("extrinsics.npy"),
        b"different hidden truth",
    )
    .unwrap();
    run_blocking(root.path(), &root.path().join("b")).unwrap();
    let a = std::fs::read(root.path().join("a/job.json")).unwrap();
    assert_eq!(a, std::fs::read(root.path().join("b/job.json")).unwrap());
    let job: SurfaceJob = serde_json::from_slice(&a).unwrap();
    assert!((job.reference.column[0] - 0.3).abs() < 1e-6);
    assert!((job.reference.row[1] + 0.3).abs() < 1e-6);
    assert_eq!(job.prior.position[2], 98.0);
    assert_eq!(
        std::fs::metadata(root.path().join("a/heights.bin"))
            .unwrap()
            .len(),
        4096
    );
}
#[test]
fn warped_grid_and_oversized_array_header_are_rejected() {
    let root = tempfile::tempdir().unwrap();
    inputs(root.path(), true);
    assert!(run_blocking(root.path(), &root.path().join("invalid")).is_err());
    assert!(!root.path().join("invalid").exists());
    let path = root.path().join("large.npy");
    write_array(&path, "<f4", "(1000000000,)", &[]);
    assert!(arrays::read_blocking::<f32>(&path).is_err());
}
