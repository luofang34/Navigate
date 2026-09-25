//! Estimate a similarity using shared camera rotations and baselines.
use super::*;
use nalgebra::{Quaternion, Vector4};
pub(super) fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() || values.iter().any(|v| !v.is_finite()) {
        return None;
    }
    values.sort_by(f64::total_cmp);
    Some(values[(values.len() - 1) / 2])
}
pub(super) fn rotation(
    links: &[Link],
    ids: &[usize],
    reference: UnitQuaternion<f64>,
) -> Option<UnitQuaternion<f64>> {
    let mut sum = Vector4::zeros();
    for &i in ids {
        let q = links[i].target.orientation * links[i].source.orientation.inverse();
        sum += q.coords
            * if q.coords.dot(&reference.coords) < 0.0 {
                -1.0
            } else {
                1.0
            };
    }
    (sum.norm() > 1e-8).then(|| UnitQuaternion::new_normalize(Quaternion::from(sum)))
}
pub(super) fn baseline(links: &[Link], ids: &[usize], target: bool) -> Option<f64> {
    median(
        ids.iter()
            .enumerate()
            .flat_map(|(i, &a)| {
                ids[i + 1..].iter().map(move |&b| {
                    if target {
                        (links[a].target.position - links[b].target.position).norm()
                    } else {
                        (links[a].source.position - links[b].source.position).norm()
                    }
                })
            })
            .collect(),
    )
}
pub(super) fn initial(
    links: &[Link],
    ids: &[usize],
    rotation: UnitQuaternion<f64>,
) -> Option<SceneTransform> {
    let ratios = ids
        .iter()
        .enumerate()
        .flat_map(|(i, &a)| {
            ids[i + 1..].iter().filter_map(move |&b| {
                let source = (links[a].source.position - links[b].source.position).norm();
                let target = (links[a].target.position - links[b].target.position).norm();
                (source > 1e-8 && target > 1e-8).then_some(target / source)
            })
        })
        .collect();
    let scale = median(ratios)?;
    let offsets: Vec<_> = ids
        .iter()
        .map(|&i| links[i].target.position - rotation * links[i].source.position * scale)
        .collect();
    let translation = Vector3::new(
        median(offsets.iter().map(|v| v.x).collect())?,
        median(offsets.iter().map(|v| v.y).collect())?,
        median(offsets.iter().map(|v| v.z).collect())?,
    );
    (scale > 1e-8).then_some(SceneTransform {
        scale,
        rotation,
        translation,
    })
}
pub(super) fn refine(
    links: &[Link],
    ids: &[usize],
    reference: UnitQuaternion<f64>,
) -> Option<SceneTransform> {
    let rotation = rotation(links, ids, reference)?;
    let count = ids.len() as f64;
    let source = ids
        .iter()
        .map(|&i| links[i].source.position)
        .sum::<Vector3<f64>>()
        / count;
    let target = ids
        .iter()
        .map(|&i| links[i].target.position)
        .sum::<Vector3<f64>>()
        / count;
    let numerator = ids
        .iter()
        .map(|&i| {
            (links[i].target.position - target)
                .dot(&(rotation * (links[i].source.position - source)))
        })
        .sum::<f64>();
    let denominator = ids
        .iter()
        .map(|&i| (links[i].source.position - source).norm_squared())
        .sum::<f64>();
    let scale = numerator / denominator;
    let translation = target - rotation * source * scale;
    (scale.is_finite() && scale > 1e-8 && translation.iter().all(|v| v.is_finite())).then_some(
        SceneTransform {
            scale,
            rotation,
            translation,
        },
    )
}
