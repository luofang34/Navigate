//! Jointly refine a bounded selection while retaining source identities.
use super::*;
use crate::{
    LocalScene, LocalSceneCamera, LocalScenePoint, SceneCoordinateGauge,
    refine_local_scene_with_gauge,
};
pub(super) fn run(
    camera: &CameraModel,
    graph: &ImageTracks,
    state: &mut State,
    seed: [usize; 2],
    iterations: usize,
) -> Result<Reconstruction, ReconstructionError> {
    let indices: Vec<_> = state.poses.keys().copied().collect();
    let lookup: BTreeMap<_, _> = indices.iter().enumerate().map(|(i, &id)| (id, i)).collect();
    let mut occupied = BTreeMap::<usize, BTreeSet<(i32, i32)>>::new();
    let mut selected = Vec::new();
    let mut points = Vec::new();
    for (&id, &world) in &state.points {
        let support = structure::observations(camera, &graph.tracks[id], &state.poses, world);
        if support.len() < 2
            || !support.iter().any(|o| {
                !occupied
                    .entry(o.camera_index)
                    .or_default()
                    .contains(&((o.pixel.x / 25.0) as i32, (o.pixel.y / 25.0) as i32))
            })
        {
            continue;
        }
        for o in &support {
            occupied
                .entry(o.camera_index)
                .or_default()
                .insert(((o.pixel.x / 25.0) as i32, (o.pixel.y / 25.0) as i32));
        }
        points.push(LocalScenePoint {
            feature_id: graph.tracks[id].feature_id,
            position: world,
            observations: support
                .into_iter()
                .map(|o| ScenePointObservation {
                    camera_index: lookup[&o.camera_index],
                    pixel: o.pixel,
                })
                .collect(),
        });
        selected.push(id);
        if points.len() >= 16000 {
            break;
        }
    }
    let coordinate_gauge = SceneCoordinateGauge {
        origin_camera: lookup[&seed[0]],
        scale_camera: lookup[&seed[1]],
    };
    let mut scene = LocalScene {
        cameras: indices
            .iter()
            .map(|&id| LocalSceneCamera {
                observation_sha256: graph.observation_sha256[id].clone(),
                pose: state.poses[&id].to_scene(),
                fixed: id == seed[0],
            })
            .collect(),
        points,
    };
    let refinement =
        refine_local_scene_with_gauge(camera, &mut scene, coordinate_gauge, iterations)
            .map_err(|source| ReconstructionError::Refinement { source })?;
    for (&id, p) in indices.iter().zip(&scene.cameras) {
        state.poses.insert(id, Pose::from_scene(p.pose));
    }
    for (id, p) in selected.into_iter().zip(&scene.points) {
        state.points.insert(id, p.position);
    }
    Ok(Reconstruction {
        scene,
        source_camera_indices: indices,
        unresolved_camera_indices: (0..graph.observation_sha256.len())
            .filter(|i| !state.poses.contains_key(i))
            .collect(),
        coordinate_gauge,
        refinement,
    })
}
