//! Bounded geometric proposals without a geographic acceptance decision.
use super::*;
use std::collections::BTreeSet;
fn inliers(
    points: &[SceneMapAssociation],
    transform: SceneTransform,
    threshold: f64,
) -> Vec<usize> {
    points
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            ((transform.point(p.scene_point) - p.map_point).norm() <= threshold).then_some(i)
        })
        .collect()
}
fn candidate(
    camera: &CameraModel,
    points: &[SceneMapAssociation],
    mut ids: Vec<usize>,
    config: RegistrationConfig,
) -> Option<SceneRegistration> {
    let mut transform = fit::fit(points, &ids)?;
    for _ in 0..3 {
        ids = inliers(points, transform, config.inlier_threshold_m);
        if ids.len() < config.min_inliers {
            return None;
        }
        transform = fit::fit(points, &ids)?;
    }
    ids = inliers(points, transform, config.inlier_threshold_m);
    let cells: BTreeSet<_> = ids
        .iter()
        .map(|&i| {
            (
                (points[i].query_pixel.x * 4.0 / f64::from(camera.width)) as u32,
                (points[i].query_pixel.y * 3.0 / f64::from(camera.height)) as u32,
            )
        })
        .collect();
    if ids.len() < config.min_inliers || cells.len() < config.min_occupied_cells {
        return None;
    }
    let error: f64 = ids
        .iter()
        .map(|&i| (transform.point(points[i].scene_point) - points[i].map_point).norm_squared())
        .sum();
    Some(SceneRegistration {
        transform,
        inlier_rms_m: (error / ids.len() as f64).sqrt(),
        inlier_indices: ids,
        occupied_cells: cells.len(),
    })
}
pub(super) fn propose(
    camera: &CameraModel,
    points: &[SceneMapAssociation],
    config: RegistrationConfig,
) -> (Vec<SceneRegistration>, bool) {
    let mut seed = 0x1289319_u64;
    let mut candidates = Vec::<SceneRegistration>::new();
    let mut exhausted = false;
    for _ in 0..config.trials {
        let mut sample = BTreeSet::new();
        while sample.len() < 3 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            sample.insert((seed >> 32) as usize % points.len());
        }
        let ids: Vec<_> = sample.into_iter().collect();
        let Some(transform) = fit::fit(points, &ids) else {
            continue;
        };
        let support = inliers(points, transform, config.inlier_threshold_m);
        if support.len() < config.min_inliers {
            continue;
        }
        let Some(proposal) = candidate(camera, points, support, config) else {
            continue;
        };
        if candidates
            .iter()
            .any(|c| c.inlier_indices == proposal.inlier_indices)
        {
            continue;
        }
        candidates.push(proposal);
        candidates.sort_by(|a, b| {
            b.inlier_indices
                .len()
                .cmp(&a.inlier_indices.len())
                .then_with(|| a.inlier_rms_m.total_cmp(&b.inlier_rms_m))
        });
        if candidates.len() > config.max_candidates {
            exhausted = true;
            candidates.truncate(config.max_candidates);
        }
    }
    (candidates, exhausted)
}
