//! Multi-view DEM-regularized terrain bundle adjustment (`run_terrain_bundle_stage`).

use super::super::collect_parent_anchors;
use crate::{
    PipelineError,
    pose::LoadedPose,
    surface::Surface,
    util::{file_sha256, write_json},
};
use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector2, Vector3};
use navigate_segment::read_npz;
use navigate_visual::{
    CameraModel, LocalScene, LocalSceneCamera, LocalScenePoint, LocalScenePose,
    ScenePointObservation, ScenePointPrior, refine_regularized_local_scene,
};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::{Path, PathBuf},
};

const VOXEL_CELL_M: f64 = 15.0;
const RAY_CELL_M: f64 = 25.0;
const MAX_BUNDLE_POINTS: usize = 16_384;
const MAX_FREE_CAMERAS: usize = 96;

struct CellAccumulator {
    sum_enu: [f64; 3],
    count: usize,
    obs_by_cam: BTreeMap<usize, ([f64; 2], f64)>,
}

impl CellAccumulator {
    fn add(&mut self, cam_idx: usize, world_enu: [f64; 3], pixel: [f64; 2], cell_center: [f64; 2]) {
        self.sum_enu[0] += world_enu[0];
        self.sum_enu[1] += world_enu[1];
        self.sum_enu[2] += world_enu[2];
        self.count += 1;
        let dist_sq =
            (world_enu[0] - cell_center[0]).powi(2) + (world_enu[1] - cell_center[1]).powi(2);
        let replace = self
            .obs_by_cam
            .get(&cam_idx)
            .is_none_or(|&(_, prev_d2)| dist_sq < prev_d2);
        if replace {
            self.obs_by_cam.insert(cam_idx, (pixel, dist_sq));
        }
    }
}

/// Multi-view DEM-regularized bundle adjustment via [`refine_regularized_local_scene`].
///
/// # Errors
///
/// Returns [`PipelineError`] when scene construction or writing fails.
pub fn run_terrain_bundle_stage(
    catalog_path: &Path,
    parent_dirs: &[PathBuf],
    output_dir: &Path,
) -> Result<Value, PipelineError> {
    let surface = Surface::from_catalog(catalog_path)?;
    fs::create_dir_all(output_dir).map_err(|source| PipelineError::Io {
        path: output_dir.to_path_buf(),
        source,
    })?;
    let anchors = collect_parent_anchors(parent_dirs, &surface)?;
    if anchors.len() < 2 {
        let summary = json!({
            "catalog_sha256": surface.catalog_sha256,
            "cameras": anchors.len(),
            "refined_cameras": 0,
            "points": 0,
            "refinement": Value::Null,
        });
        write_json(&output_dir.join("run.json"), &summary)?;
        return Ok(summary);
    }

    let ordered: Vec<(&String, &LoadedPose)> = anchors.iter().collect();
    let flip = Matrix3::from_diagonal(&Vector3::new(1.0, -1.0, -1.0));
    let base_sha = file_sha256(catalog_path)?;
    let mut calibrations = Vec::with_capacity(ordered.len());
    let mut cameras = Vec::with_capacity(ordered.len());

    for (idx, (_id, (eye, rot, k, sz))) in ordered.iter().enumerate() {
        calibrations.push(CameraModel {
            width: sz[0].max(1),
            height: sz[1].max(1),
            fx: k[(0, 0)].max(1.0),
            fy: k[(1, 1)].max(1.0),
            cx: k[(0, 2)],
            cy: k[(1, 2)],
        });
        let eye_to_scene = orthonormalize_so3(&(rot.transpose() * flip));
        let q =
            UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(eye_to_scene));
        let mut sha = base_sha.clone();
        sha.replace_range(0..4, &format!("{idx:04x}"));
        cameras.push(LocalSceneCamera {
            observation_sha256: sha,
            pose: LocalScenePose {
                position: Vector3::from(*eye),
                orientation: q,
            },
            fixed: true,
        });
    }

    let mut cells = collect_npz_inlier_cells(&ordered, parent_dirs, &calibrations);
    let shared_npz = cells.values().filter(|c| c.obs_by_cam.len() >= 2).count();
    if shared_npz < 2 {
        add_raycast_tie_cells(&ordered, &calibrations, &surface, &mut cells);
    }

    let (points, priors) = build_scene_points_and_priors(&cells, &surface);
    let free_cameras = assign_component_gauges(&mut cameras, &points);
    let stats = if !points.is_empty() {
        let mut scene = LocalScene { cameras, points };
        let res = refine_regularized_local_scene(&calibrations, &mut scene, &priors, 15)
            .ok()
            .map(|r| {
                json!({
                    "initial_cost": r.initial_cost,
                    "final_cost": r.final_cost,
                    "steps": r.steps,
                })
            });
        let num_pts = scene.points.len();
        (num_pts, res)
    } else {
        (0_usize, None)
    };

    let refined_cameras = if stats.1.is_some() { free_cameras } else { 0 };
    let summary = json!({
        "catalog_sha256": surface.catalog_sha256,
        "cameras": anchors.len(),
        "refined_cameras": refined_cameras,
        "points": stats.0,
        "refinement": stats.1,
    });
    write_json(&output_dir.join("run.json"), &summary)?;
    Ok(summary)
}

fn collect_npz_inlier_cells(
    ordered: &[(&String, &LoadedPose)],
    parent_dirs: &[PathBuf],
    calibrations: &[CameraModel],
) -> BTreeMap<(i64, i64), CellAccumulator> {
    let mut cells: BTreeMap<(i64, i64), CellAccumulator> = BTreeMap::new();
    for (cam_idx, (case_id, _)) in ordered.iter().enumerate() {
        let Some(npz_path) = parent_dirs
            .iter()
            .rev()
            .map(|d| d.join(case_id.as_str()).join("candidate.npz"))
            .find(|p| p.is_file())
        else {
            continue;
        };
        let Ok(map) = read_npz(&npz_path) else {
            continue;
        };
        let (Some(pts_arr), Some(pix_arr)) = (map.get("points"), map.get("pixels")) else {
            continue;
        };
        let pts = pts_arr.to_f64();
        let pix = pix_arr.to_f64();
        let target_eye = ordered[cam_idx].1.0;
        let eye_npz = map
            .get("eye")
            .map(|a| a.to_f64())
            .filter(|v| v.len() >= 3)
            .map_or(target_eye, |v| [v[0], v[1], v[2]]);
        let shift = [
            target_eye[0] - eye_npz[0],
            target_eye[1] - eye_npz[1],
            target_eye[2] - eye_npz[2],
        ];
        let inliers: Vec<bool> = map
            .get("inliers")
            .map(|a| a.to_f64().into_iter().map(|v| v > 0.5).collect())
            .unwrap_or_else(|| vec![true; pts.len() / 3]);
        let n = (pts.len() / 3).min(pix.len() / 2).min(inliers.len());
        let cal = &calibrations[cam_idx];
        let (w_lim, h_lim) = (f64::from(cal.width), f64::from(cal.height));

        for i in 0..n {
            if !inliers[i] {
                continue;
            }
            let w_raw = [pts[3 * i], pts[3 * i + 1], pts[3 * i + 2]];
            let p = [pix[2 * i], pix[2 * i + 1]];
            if !w_raw.iter().all(|v| v.is_finite())
                || !(0.0..w_lim).contains(&p[0])
                || !(0.0..h_lim).contains(&p[1])
            {
                continue;
            }
            let w = [
                w_raw[0] + shift[0],
                w_raw[1] + shift[1],
                w_raw[2] + shift[2],
            ];
            let gx = (w[0] / VOXEL_CELL_M).round() as i64;
            let gy = (w[1] / VOXEL_CELL_M).round() as i64;
            let center = [(gx as f64) * VOXEL_CELL_M, (gy as f64) * VOXEL_CELL_M];
            cells
                .entry((gx, gy))
                .or_insert_with(|| CellAccumulator {
                    sum_enu: [0.0; 3],
                    count: 0,
                    obs_by_cam: BTreeMap::new(),
                })
                .add(cam_idx, w, p, center);
        }
    }
    cells
}

type RayTieSample = ([f64; 3], usize, [f64; 2]);

fn add_raycast_tie_cells(
    ordered: &[(&String, &LoadedPose)],
    calibrations: &[CameraModel],
    surface: &Surface,
    cells: &mut BTreeMap<(i64, i64), CellAccumulator>,
) {
    let mut ray_cells: BTreeMap<(i64, i64), Vec<RayTieSample>> = BTreeMap::new();
    for (cam_idx, (_, (eye, rot, k, _))) in ordered.iter().enumerate() {
        let cal = &calibrations[cam_idx];
        let mut grid_px = Vec::with_capacity(25);
        for iy in 0..5_u32 {
            let py = (f64::from(iy) + 0.5) * 0.2 * f64::from(cal.height);
            for ix in 0..5_u32 {
                let px = (f64::from(ix) + 0.5) * 0.2 * f64::from(cal.width);
                grid_px.push([px, py]);
            }
        }
        let hits = surface.raycast_pixels(*eye, rot, k, &grid_px);
        for (px, hit) in grid_px.into_iter().zip(hits) {
            let Some(h) = hit else {
                continue;
            };
            let gx = (h.world_enu[0] / RAY_CELL_M).round() as i64;
            let gy = (h.world_enu[1] / RAY_CELL_M).round() as i64;
            ray_cells
                .entry((gx, gy))
                .or_default()
                .push((h.world_enu, cam_idx, px));
        }
    }
    for ((gx, gy), hits) in ray_cells {
        let mean_enu = {
            let n = hits.len() as f64;
            let mut s = [0.0; 3];
            for (w, _, _) in &hits {
                s[0] += w[0];
                s[1] += w[1];
                s[2] += w[2];
            }
            [s[0] / n, s[1] / n, s[2] / n]
        };
        let key = (gx.wrapping_mul(1_000_003), gy.wrapping_mul(1_000_003));
        let acc = cells.entry(key).or_insert_with(|| CellAccumulator {
            sum_enu: [0.0; 3],
            count: 0,
            obs_by_cam: BTreeMap::new(),
        });
        for &(_, cam_idx, _) in &hits {
            let (_, (eye, rot, k, _)) = ordered[cam_idx];
            let cal = &calibrations[cam_idx];
            let rel = rot * (Vector3::from(mean_enu) - Vector3::from(*eye));
            if rel.z <= 0.5 {
                continue;
            }
            let u = k[(0, 0)] * (rel.x / rel.z) + k[(0, 2)];
            let v = k[(1, 1)] * (rel.y / rel.z) + k[(1, 2)];
            if (0.0..f64::from(cal.width)).contains(&u) && (0.0..f64::from(cal.height)).contains(&v)
            {
                acc.add(cam_idx, mean_enu, [u, v], [mean_enu[0], mean_enu[1]]);
            }
        }
    }
}

fn build_scene_points_and_priors(
    cells: &BTreeMap<(i64, i64), CellAccumulator>,
    surface: &Surface,
) -> (Vec<LocalScenePoint>, Vec<ScenePointPrior>) {
    let mut ranked: Vec<&CellAccumulator> = cells
        .values()
        .filter(|c| c.obs_by_cam.len() >= 2 && c.count > 0)
        .collect();
    ranked.sort_by(|a, b| {
        b.obs_by_cam
            .len()
            .cmp(&a.obs_by_cam.len())
            .then_with(|| b.count.cmp(&a.count))
    });
    ranked.truncate(MAX_BUNDLE_POINTS);

    let mut points = Vec::with_capacity(ranked.len());
    let mut priors = Vec::with_capacity(ranked.len());
    for (idx, cell) in ranked.into_iter().enumerate() {
        let fid = (idx as u64) + 1;
        let inv = 1.0 / (cell.count as f64);
        let mean_e = cell.sum_enu[0] * inv;
        let mean_n = cell.sum_enu[1] * inv;
        let mean_u = cell.sum_enu[2] * inv;
        let [lat, lon, _] = surface.geodetic([mean_e, mean_n, mean_u]);
        let dem_enu = surface
            .sample_dem(lat, lon)
            .map(|h| surface.local(lat, lon, h))
            .unwrap_or([mean_e, mean_n, mean_u]);
        let observations = cell
            .obs_by_cam
            .iter()
            .map(|(&camera_index, &(px, _))| ScenePointObservation {
                camera_index,
                pixel: Vector2::new(px[0], px[1]),
            })
            .collect();
        points.push(LocalScenePoint {
            feature_id: fid,
            position: Vector3::new(mean_e, mean_n, mean_u),
            observations,
        });
        priors.push(ScenePointPrior {
            feature_id: fid,
            position: Vector3::from(dem_enu),
            scale: 15.0,
        });
    }
    (points, priors)
}

fn assign_component_gauges(cameras: &mut [LocalSceneCamera], points: &[LocalScenePoint]) -> usize {
    let mut parents: Vec<usize> = (0..cameras.len()).collect();
    let mut used = BTreeSet::new();
    let root = |p: &[usize], mut i: usize| {
        while p[i] != i {
            i = p[i];
        }
        i
    };
    for pt in points {
        let first = pt.observations[0].camera_index;
        used.insert(first);
        for obs in &pt.observations[1..] {
            used.insert(obs.camera_index);
            let a = root(&parents, first);
            let b = root(&parents, obs.camera_index);
            parents[b] = a;
        }
    }
    let mut comps: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for &cam_idx in &used {
        comps
            .entry(root(&parents, cam_idx))
            .or_default()
            .push(cam_idx);
    }

    let mut free_count = 0_usize;
    for comp in comps.values() {
        if comp.len() < 2 {
            continue;
        }
        let anchor_a = comp[0];
        let Some(&anchor_b) = comp.iter().skip(1).max_by(|&&i, &&j| {
            let di = (cameras[i].pose.position - cameras[anchor_a].pose.position).norm_squared();
            let dj = (cameras[j].pose.position - cameras[anchor_a].pose.position).norm_squared();
            di.total_cmp(&dj)
        }) else {
            continue;
        };
        if (cameras[anchor_b].pose.position - cameras[anchor_a].pose.position).norm() <= 1e-4 {
            continue;
        }
        for &cam_idx in comp {
            if cam_idx != anchor_a && cam_idx != anchor_b && free_count < MAX_FREE_CAMERAS {
                cameras[cam_idx].fixed = false;
                free_count += 1;
            }
        }
    }
    free_count
}

fn orthonormalize_so3(m: &Matrix3<f64>) -> Matrix3<f64> {
    let svd = m.svd(true, true);
    if let (Some(u), Some(v_t)) = (svd.u, svd.v_t) {
        let mut r = u * v_t;
        if r.determinant() < 0.0 {
            let mut u_fix = u;
            u_fix.set_column(2, &(-u.column(2)));
            r = u_fix * v_t;
        }
        r
    } else {
        Matrix3::identity()
    }
}
