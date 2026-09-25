//! Bounded scale consensus without changing the shared camera rotations.
use super::{PointAlignmentConfig, PointLink};
use nalgebra::Vector3;
pub(super) struct Proposals {
    pub candidates: Vec<(f64, Vec<usize>)>,
    pub exhausted: bool,
}
fn select(points: &[PointLink], scale: f64, ratio: f64) -> Vec<usize> {
    points
        .iter()
        .enumerate()
        .filter_map(|(i, p)| {
            ((p.source * scale - p.target).norm() <= p.target.norm() * ratio).then_some(i)
        })
        .collect()
}
fn extent(points: &[PointLink], ids: &[usize], source: bool) -> f64 {
    let mut low = Vector3::repeat(f64::INFINITY);
    let mut high = Vector3::repeat(f64::NEG_INFINITY);
    for &i in ids {
        let p = if source {
            points[i].source
        } else {
            points[i].target
        };
        low = low.inf(&p);
        high = high.sup(&p);
    }
    (high - low).norm()
}
fn supported_extent(points: &[PointLink], ids: &[usize], ratio: f64) -> bool {
    [false, true].into_iter().all(|source| {
        let distances = ids
            .iter()
            .map(|&i| {
                if source {
                    points[i].source.norm()
                } else {
                    points[i].target.norm()
                }
            })
            .collect();
        super::fit::median(distances)
            .is_some_and(|distance| extent(points, ids, source) >= distance * ratio)
    })
}
pub(super) fn propose(points: &[PointLink], config: PointAlignmentConfig) -> Proposals {
    let mut seeds: Vec<_> = points
        .iter()
        .map(|p| p.source.dot(&p.target) / p.source.norm_squared())
        .filter(|v| v.is_finite() && *v > 1e-8)
        .collect();
    seeds.sort_by(f64::total_cmp);
    seeds.dedup_by(|a, b| (*a - *b).abs() <= a.abs() * 1e-10);
    let exhausted = seeds.len() > config.max_scale_proposals;
    let count = seeds.len().min(config.max_scale_proposals);
    let mut candidates = Vec::new();
    for i in 0..count {
        let index = if count == 1 {
            seeds.len() / 2
        } else {
            i * (seeds.len() - 1) / (count - 1)
        };
        let mut scale = seeds[index];
        let mut ids = select(points, scale, config.max_point_error_ratio);
        for _ in 0..4 {
            if ids.len() < config.min_shared_points {
                break;
            }
            scale = ids
                .iter()
                .map(|&i| points[i].source.dot(&points[i].target))
                .sum::<f64>()
                / ids
                    .iter()
                    .map(|&i| points[i].source.norm_squared())
                    .sum::<f64>();
            let next = select(points, scale, config.max_point_error_ratio);
            if next == ids {
                break;
            }
            ids = next;
        }
        if ids.len() >= config.min_shared_points
            && scale.is_finite()
            && scale > 1e-8
            && supported_extent(points, &ids, config.min_point_spread_ratio)
        {
            candidates.push((scale, ids));
        }
    }
    Proposals {
        candidates,
        exhausted,
    }
}
