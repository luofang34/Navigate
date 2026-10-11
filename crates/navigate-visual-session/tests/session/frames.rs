//! Local-frame changes must preserve one physical camera and its uncertainty.
use crate::support::*;
use nalgebra::{Matrix6, Vector3};
use navigate_visual::LocalFrame;
use navigate_visual_session::{AnchorDecision, MapPose, Pose};

#[test]
fn spherical_anchor_transport_preserves_attitude_and_position_rotation_covariance() {
    let a = solve(None);
    let b = solve(Some(LocalFrame::spherical_enu(48.0, 12.0).unwrap()));
    assert!((a.0.translation.vector - b.0.translation.vector).norm() < 1e-6);
    assert!(a.0.rotation.angle_to(&b.0.rotation) < 1e-9);
    assert!((a.1 - b.1).abs() < 1e-6);
}

#[cfg(test)]
fn solve(other: Option<LocalFrame>) -> (Pose, f64) {
    let mut flight = Flight::standard();
    for i in 0..12 {
        flight.fly_to(nadir(Vector3::new(f64::from(i) * 6.0, 0.0, 100.0), 0.0));
    }
    let base = LocalFrame::spherical_enu(47.0, 11.0).unwrap();
    let (first, last) = (flight.truth[0].0, flight.truth[11].0);
    let mut initial = anchor(&flight, &first, Vector3::zeros());
    initial.local_frame = base;
    flight.session.submit_anchor(initial).unwrap();
    let mut second = anchor(&flight, &last, Vector3::new(2.0, -1.0, 0.0));
    second.local_frame = base;
    if let Some(other) = other {
        let rotation = other.local_rotation(base.anchor()).unwrap();
        let position = other
            .local(base.geodetic(second.pose.translation.vector))
            .unwrap();
        second.pose = Pose::from_parts(position.into(), rotation * second.pose.rotation);
        let mut transform = Matrix6::identity();
        transform
            .fixed_view_mut::<3, 3>(0, 0)
            .copy_from(rotation.to_rotation_matrix().matrix());
        second.geometry_covariance = transform * second.geometry_covariance * transform.transpose();
        second.local_frame = other;
    }
    let decision = flight.session.submit_anchor(second).unwrap();
    assert!(
        matches!(decision, AnchorDecision::Accepted { .. }),
        "{decision:?}"
    );
    match flight.session.pose_at(&last).unwrap().map {
        MapPose::Located {
            pose,
            position_bound_m,
            ..
        } => (pose, position_bound_m),
        other => panic!("expected map pose: {other:?}"),
    }
}

#[test]
fn mixed_flat_and_curved_frames_are_refused_without_changing_the_session() {
    let mut flight = Flight::standard();
    for i in 0..2 {
        flight.fly_to(nadir(Vector3::new(f64::from(i) * 30.0, 0.0, 100.0), 0.0));
    }
    let (first, last) = (flight.truth[0].0, flight.truth[1].0);
    let mut initial = anchor(&flight, &first, Vector3::zeros());
    initial.local_frame = LocalFrame::spherical_enu(47.0, 11.0).unwrap();
    flight.session.submit_anchor(initial).unwrap();
    let (usage, revision) = (flight.session.usage(), flight.session.revision());
    let before = flight.session.pose_at(&last).unwrap();
    let second = anchor(&flight, &last, Vector3::zeros());
    assert!(matches!(
        flight.session.submit_anchor(second),
        Err(navigate_visual_session::SessionError::Invalid { .. })
    ));
    assert_eq!(flight.session.usage(), usage);
    assert_eq!(flight.session.revision(), revision);
    assert_eq!(flight.session.pose_at(&last).unwrap(), before);
}
