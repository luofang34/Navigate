//! `.npz` serialization and deserialization for [`super::PoseCandidate`].

use super::PoseCandidate;
use crate::PipelineError;
use nalgebra::Matrix3;
use navigate_segment::{NpyArray, read_npz, write_npz};
use std::{collections::BTreeMap, path::Path};

/// Loaded pose tuple `(eye_enu, world_to_cam, intrinsics_k, image_size)`.
pub type LoadedPose = ([f64; 3], Matrix3<f64>, Matrix3<f64>, [u32; 2]);

/// Write a [`PoseCandidate`] to `candidate.npz` matching the 7-array Delaware layout.
///
/// # Errors
///
/// Returns [`PipelineError::Segment`] when `.npz` serialization fails.
pub fn save_candidate_npz(path: &Path, cand: &PoseCandidate) -> Result<(), PipelineError> {
    let w2c = &cand.rotation;
    let rot_gl_vec: Vec<f64> = (0..3)
        .flat_map(|r| [w2c[(0, r)], -w2c[(1, r)], -w2c[(2, r)]])
        .collect();
    let k_vec: Vec<f64> = (0..3)
        .flat_map(|r| (0..3).map(move |c| cand.k[(r, c)]))
        .collect();
    let pts_flat: Vec<f64> = cand.points.iter().flat_map(|p| p.iter().copied()).collect();
    let pix_flat: Vec<f64> = cand.pixels.iter().flat_map(|p| p.iter().copied()).collect();
    let n = cand.points.len();

    let mut entries = BTreeMap::new();
    entries.insert("eye".into(), NpyArray::from_f64(vec![3], &cand.eye)?);
    entries.insert(
        "rotation".into(),
        NpyArray::from_f64(vec![3, 3], &rot_gl_vec)?,
    );
    entries.insert("K".into(), NpyArray::from_f64(vec![3, 3], &k_vec)?);
    entries.insert(
        "size".into(),
        NpyArray::from_i64(vec![2], &[i64::from(cand.size[0]), i64::from(cand.size[1])])?,
    );
    entries.insert("points".into(), NpyArray::from_f64(vec![n, 3], &pts_flat)?);
    entries.insert("pixels".into(), NpyArray::from_f64(vec![n, 2], &pix_flat)?);
    entries.insert(
        "inliers".into(),
        NpyArray::from_bool(vec![n], &cand.inliers)?,
    );
    write_npz(path, &entries)?;
    Ok(())
}

/// Load `(eye, rotation, K, size)` from a saved `candidate.npz`.
///
/// # Errors
///
/// Returns [`PipelineError`] when `path` cannot be read or lacks required arrays.
pub fn load_candidate_pose(path: &Path) -> Result<LoadedPose, PipelineError> {
    let map = read_npz(path)?;
    let get = |name: &str| {
        map.get(name)
            .ok_or_else(|| PipelineError::Invalid(format!("missing {name} in {}", path.display())))
    };
    let eye_v = get("eye")?.to_f64();
    let rot_v = get("rotation")?.to_f64();
    let k_v = get("K")?.to_f64();
    let sz_v = get("size")?.to_f64();
    if eye_v.len() < 3 || rot_v.len() < 9 || k_v.len() < 9 || sz_v.len() < 2 {
        return Err(PipelineError::Invalid(
            "truncated arrays in candidate.npz".into(),
        ));
    }
    let gl = Matrix3::from_row_slice(&rot_v[..9]);
    let rot = Matrix3::new(
        gl[(0, 0)],
        gl[(1, 0)],
        gl[(2, 0)],
        -gl[(0, 1)],
        -gl[(1, 1)],
        -gl[(2, 1)],
        -gl[(0, 2)],
        -gl[(1, 2)],
        -gl[(2, 2)],
    );
    let k = Matrix3::from_row_slice(&k_v[..9]);
    Ok((
        [eye_v[0], eye_v[1], eye_v[2]],
        rot,
        k,
        [sz_v[0].max(1.0) as u32, sz_v[1].max(1.0) as u32],
    ))
}
