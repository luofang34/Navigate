//! Validate identities and image bounds before geometry work.
use super::*;
pub(super) fn graph(camera: &CameraModel, graph: &ImageTracks) -> Result<(), ReconstructionError> {
    camera
        .validate()
        .map_err(|source| ReconstructionError::Camera { source })?;
    let count = graph
        .tracks
        .iter()
        .fold(0usize, |n, t| n.saturating_add(t.observations.len()));
    if !(2..=129).contains(&graph.observation_sha256.len())
        || graph.tracks.len() > 65536
        || count > 4_000_000
    {
        return Err(ReconstructionError::Limits {
            cameras: graph.observation_sha256.len(),
            tracks: graph.tracks.len(),
            observations: count,
        });
    }
    let mut identities = BTreeSet::new();
    for (index, id) in graph.observation_sha256.iter().enumerate() {
        if id.len() != 64
            || !id.bytes().all(|c| c.is_ascii_hexdigit())
            || !identities.insert(id.to_ascii_lowercase())
        {
            return Err(ReconstructionError::Observation {
                index,
                reason: "invalid or repeated source digest",
            });
        }
    }
    let mut features = BTreeSet::new();
    for track in &graph.tracks {
        let fail = |reason| ReconstructionError::Track {
            feature_id: track.feature_id,
            reason,
        };
        if !features.insert(track.feature_id) || track.observations.len() < 2 {
            return Err(fail(
                "duplicate feature identity or fewer than two observations",
            ));
        }
        let mut seen = BTreeSet::new();
        for o in &track.observations {
            if o.camera_index >= graph.observation_sha256.len() || !seen.insert(o.camera_index) {
                return Err(fail("missing or repeated camera observation"));
            }
            if !(0.0..f64::from(camera.width)).contains(&o.pixel.x)
                || !(0.0..f64::from(camera.height)).contains(&o.pixel.y)
            {
                return Err(fail("pixel outside the calibrated image"));
            }
        }
    }
    Ok(())
}
pub(super) fn pair(
    graph: &ImageTracks,
    camera_indices: [usize; 2],
) -> Result<(), ReconstructionError> {
    if camera_indices[0] == camera_indices[1]
        || camera_indices
            .iter()
            .any(|&i| i >= graph.observation_sha256.len())
    {
        return Err(ReconstructionError::Seed {
            camera_indices,
            reason: "two distinct source cameras are required",
        });
    }
    Ok(())
}
pub(super) fn seed(
    graph: &ImageTracks,
    seed: ReconstructionSeed,
) -> Result<(), ReconstructionError> {
    pair(graph, seed.camera_indices)?;
    let p = seed.second_pose;
    if !p
        .position
        .iter()
        .chain(p.orientation.coords.iter())
        .all(|x| x.is_finite())
        || (p.orientation.norm_squared() - 1.0).abs() > 1e-8
        || p.position.norm() <= 1e-6
    {
        return Err(ReconstructionError::Seed {
            camera_indices: seed.camera_indices,
            reason: "finite pose, unit rotation, and nonzero baseline are required",
        });
    }
    Ok(())
}
