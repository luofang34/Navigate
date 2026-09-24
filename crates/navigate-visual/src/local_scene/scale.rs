//! Keep an arbitrary scene scale without fixing the second camera orientation.
use super::{LocalScene, LocalSceneError, SceneCoordinateGauge, bundle::Landmark, pose::Pose};
pub(super) fn validate(
    scene: &LocalScene,
    gauge: SceneCoordinateGauge,
) -> Result<(), LocalSceneError> {
    let invalid = |reason| LocalSceneError::Gauge {
        origin_camera: gauge.origin_camera,
        scale_camera: gauge.scale_camera,
        reason,
    };
    let origin = scene
        .cameras
        .get(gauge.origin_camera)
        .ok_or_else(|| invalid("missing origin camera"))?;
    let scale = scene
        .cameras
        .get(gauge.scale_camera)
        .ok_or_else(|| invalid("missing scale camera"))?;
    if !origin.fixed || scale.fixed || scene.cameras.iter().filter(|c| c.fixed).count() != 1 {
        return Err(invalid("only the origin camera must be fixed"));
    }
    if (origin.pose.position - scale.pose.position).norm() <= 1e-6 {
        return Err(invalid("the arbitrary baseline has zero length"));
    }
    Ok(())
}
pub(super) fn normalize(
    before: &[Pose],
    after: &mut [Pose],
    points: &mut [Landmark],
    gauge: SceneCoordinateGauge,
) -> Option<()> {
    let origin = before.get(gauge.origin_camera)?.center();
    let length = (before.get(gauge.scale_camera)?.center() - origin).norm();
    let next = (after.get(gauge.scale_camera)?.center() - origin).norm();
    let ratio = length / next;
    if !ratio.is_finite() || length <= 1e-6 || next <= 1e-6 {
        return None;
    }
    for (index, pose) in after.iter_mut().enumerate() {
        if index != gauge.origin_camera {
            let center = origin + (pose.center() - origin) * ratio;
            pose.t = -pose.r * center;
        }
    }
    for point in points {
        point.world = origin + (point.world - origin) * ratio;
    }
    Some(())
}
