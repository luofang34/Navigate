use super::*;
use crate::{ContinuityEpoch, StreamId};
use nalgebra::{SMatrix, Translation3, UnitQuaternion};

#[test]
fn a_covariance_follows_the_conversion_between_local_frames() {
    let from = LocalFrame::anchor_mercator(47.0, 11.0).unwrap();
    let to = LocalFrame::anchor_mercator(47.3, 11.2).unwrap();
    let position = Vector3::new(120.0, -80.0, 500.0);
    let convert = |p: Vector3<f64>| to.local(from.geodetic(p)).unwrap();
    let mut jacobian = Matrix6::identity();
    for i in 0..3 {
        let step = Vector3::ith(i, 1.0);
        let column = (convert(position + step) - convert(position - step)) * 0.5;
        jacobian.fixed_view_mut::<3, 1>(0, i).copy_from(&column);
    }
    // Position, rotation, and cross terms.
    let covariance = Matrix6::from_fn(|i, j| match (i, j) {
        (i, j) if i == j => 4.0,
        (i, j) if i + j == 5 => 0.5,
        _ => 0.0,
    });
    let expected = jacobian * covariance * jacobian.transpose();
    let ratio = to.mercator_scale_m() / from.mercator_scale_m();
    assert!(ratio < 0.995, "the frames differ in scale: {ratio}");
    let moved = rescaled(&covariance, ratio);
    assert!(
        (moved - expected).norm() < 1e-6 * expected.norm(),
        "{moved} vs {expected}"
    );
}

/// The image-geometry covariance of a camera over flat ground, from the same
/// reprojection Jacobian that a map match uses: world position, then camera
/// rotation `R Exp(theta)`, with one pixel of noise.
fn plane_view(pose: &Pose) -> Matrix6<f64> {
    let f = 460.0;
    let r = pose.rotation.to_rotation_matrix().into_inner();
    let mut information = Matrix6::zeros();
    for i in -4..=4 {
        for j in -3..=3 {
            let world = Vector3::new(f64::from(i) * 14.0, f64::from(j) * 9.0, 0.0);
            let eye = r.transpose() * (world - pose.translation.vector);
            let depth = -eye.z;
            let projection = SMatrix::<f64, 2, 3>::new(
                f / depth,
                0.0,
                f * eye.x / depth.powi(2),
                0.0,
                -f / depth,
                -f * eye.y / depth.powi(2),
            );
            let mut motion = SMatrix::<f64, 3, 6>::zeros();
            motion
                .fixed_view_mut::<3, 3>(0, 0)
                .copy_from(&(-r.transpose()));
            motion
                .fixed_view_mut::<3, 3>(0, 3)
                .copy_from(&eye.cross_matrix());
            let jacobian = projection * motion;
            information += jacobian.transpose() * jacobian;
        }
    }
    information.try_inverse().unwrap()
}

/// Horizontal motion of the bare-earth point under the image centre, per pose error.
fn centre_ground(pose: &Pose) -> SMatrix<f64, 2, 6> {
    let ground = |p: &Pose| {
        let axis = p.rotation * Vector3::new(0.0, 0.0, -1.0);
        let t = -p.translation.vector.z / axis.z;
        (p.translation.vector + axis * t).xy()
    };
    let mut jacobian = SMatrix::<f64, 2, 6>::zeros();
    for k in 0..6 {
        let moved = |h: f64| {
            let mut position = pose.translation.vector;
            let mut rotation = pose.rotation;
            if k < 3 {
                position[k] += h;
            } else {
                rotation *= UnitQuaternion::from_scaled_axis(Vector3::ith(k - 3, h));
            }
            ground(&Pose::from_parts(position.into(), rotation))
        };
        jacobian.set_column(k, &((moved(1e-6) - moved(-1e-6)) / 2e-6));
    }
    jacobian
}

fn anchor(pose: Pose) -> AnchorObservation {
    AnchorObservation {
        frame: crate::FrameKey {
            stream: StreamId(0),
            continuity: ContinuityEpoch(0),
            index: 0,
        },
        observation_sha256: "a".repeat(64),
        map: navigate_visual::MapRevision {
            release_id: "synthetic".into(),
            manifest_sha256: "b".repeat(64),
        },
        local_frame: LocalFrame::anchor_mercator(47.0, 11.0).unwrap(),
        pose,
        geometry_covariance: plane_view(&pose),
        backend: "synthetic".into(),
        reliability: MapReliability {
            horizontal_m: 3.0,
            vertical_m: 5.0,
            imagery_age_years: None,
            age_growth_m_per_year: 1.0,
            unknown_age_years: 5.0,
            surface: SurfaceModel::BareEarth {
                max_object_height_m: 20.0,
            },
            change: RegionChange::Unknown,
        },
    }
}

#[test]
fn parallax_moves_the_camera_about_the_matched_ground() {
    let pose = Pose::from_parts(
        Translation3::new(6.0, -4.0, 100.0),
        UnitQuaternion::from_euler_angles(0.04, -0.03, 0.8),
    );
    let anchor = anchor(pose);
    let geometry = anchor.geometry_covariance;
    let found = budget(&anchor, 0.67, MIN_ROTATION_RAD);
    let camera = ((found.independent[(0, 0)] + found.independent[(1, 1)]) * 0.5).sqrt();
    let ground = centre_ground(&pose);
    let footprint = ground * found.independent * ground.transpose();
    let footprint = (footprint.trace() * 0.5).sqrt();
    // The camera carries the declared parallax, about 17 m; the ground under
    // the image moves much less.
    assert!(camera > 15.0, "camera {camera:.2} m");
    assert!(
        footprint < 0.15 * camera,
        "footprint {footprint:.2} m, camera {camera:.2} m"
    );
    // Heading is the floored image-geometry heading; parallax adds nothing.
    let to_map = pose.rotation.to_rotation_matrix().into_inner();
    let floor = (MIN_ROTATION_RAD.powi(2)
        / (geometry[(3, 3)] + geometry[(4, 4)] + geometry[(5, 5)]))
        .max(1.0);
    let heading = (to_map * geometry.fixed_view::<3, 3>(3, 3) * to_map.transpose())[(2, 2)];
    assert!((found.heading_rad - (heading * floor).sqrt()).abs() < 1e-9);
    assert!(found.tilt_rad > 0.1, "tilt {:.3} rad", found.tilt_rad);
}

#[test]
fn the_camera_keeps_the_declared_parallax_in_nadir_and_oblique_views() {
    for tilt in [0.0, 0.2, 0.35] {
        let pose = Pose::from_parts(
            Translation3::new(0.0, 0.0, 100.0),
            UnitQuaternion::from_euler_angles(tilt, 0.5 * tilt, 0.3),
        );
        let observation = anchor(pose);
        let found = budget(&observation, 0.67, MIN_ROTATION_RAD);
        // The pivot term, not the uniform fallback, holds the parallax: the
        // heading keeps its floored image-geometry value.
        let geometry = observation.geometry_covariance;
        let floor = (MIN_ROTATION_RAD.powi(2)
            / (geometry[(3, 3)] + geometry[(4, 4)] + geometry[(5, 5)]))
            .max(1.0);
        let to_map = pose.rotation.to_rotation_matrix().into_inner();
        let heading = (to_map * geometry.fixed_view::<3, 3>(3, 3) * to_map.transpose())[(2, 2)];
        assert!((found.heading_rad - (heading * floor).sqrt()).abs() < 1e-9);
        let declared = 20.0 * (crate::pose::off_nadir(&pose) + 0.67).min(1.31).tan();
        let camera = ((found.independent[(0, 0)] + found.independent[(1, 1)]) * 0.5).sqrt();
        assert!(
            camera >= declared * 0.99,
            "tilt {tilt}: camera {camera:.2} m, declared {declared:.2} m"
        );
    }
}
