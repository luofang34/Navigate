//! Lunar south pole terrain-relative navigation through the public API.
//!
//! A camera 2 km above the Moon at 89.9 degrees south views terrain in a
//! `BodyTangentEnu` frame on `MOON_PA`. The pose verifier solves the camera pose
//! from pixel matches against a rendered reference view (PnP). The fusion bridge
//! converts the estimate into one six-degree-of-freedom `VisualPose`, and the
//! inertial filter on the Moon applies it as one update.

use image::GrayImage;
use nalgebra::{Matrix3, Rotation3, UnitQuaternion, Vector2, Vector3};
use navigate_contract::{
    AttitudeQuaternion, ClockDomainId, Ellipsoid, FrameId, GeodeticPosition, MonotonicNanos,
    NedVelocity, SensorClass, SourceComposition, SourceEpoch, SourceId, SymmetricCov3,
};
use navigate_fusion::{
    InertialConfig, InertialNavigationFilter, IngestOutcome, MeasurementKind, ObservationValue,
    PlanetaryBody, RejectionReason,
};
use navigate_geodesy::{BodyEllipsoid, LocalTangentPlane, NedOffset};
use navigate_visual::fusion_bridge::{
    EvidenceIndependence, VerticalDatum, VisualAttitudeBudget, VisualErrorBudget, VisualFixSource,
    VisualSourceIdentity,
};
use navigate_visual::{
    CameraModel, CameraPose, Estimate, Frame, FrameStamp, LocalFrame, LocalizerConfig, MapRevision,
    PixelMatch, PosePrior, PoseVerifier, ReferenceView,
};

const CLOCK: ClockDomainId = ClockDomainId::new(9);
const CAPTURE_NS: u64 = 5_000_000_000;
const CAMERA: CameraModel = CameraModel {
    width: 320,
    height: 240,
    fx: 275.0,
    fy: 275.0,
    cx: 159.5,
    cy: 119.5,
};

#[cfg(test)]
fn lunar_frame() -> LocalFrame {
    let origin = GeodeticPosition::new((-89.9_f64).to_radians(), 30_f64.to_radians(), 0.0);
    LocalFrame::body_tangent_enu(BodyEllipsoid::MOON_PA, origin).expect("polar origin")
}

/// The same matrix rotates east, north, up into north, east, down and back.
#[cfg(test)]
fn enu_ned() -> UnitQuaternion<f64> {
    UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(Matrix3::new(
        0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0,
    )))
}

/// A nadir camera: camera right is body right, camera up is body forward, and
/// camera back is body up.
#[cfg(test)]
fn eye_to_body() -> UnitQuaternion<f64> {
    enu_ned()
}

/// The true camera and the true body attitude from body to local north, east, down.
#[cfg(test)]
fn truth(frame: &LocalFrame) -> (CameraPose, UnitQuaternion<f64>) {
    let location = [-89.905, 45.0, 2_000.0];
    let body_to_ned = UnitQuaternion::from_euler_angles(-0.02, 0.03, 40_f64.to_radians());
    let local_to_frame = frame
        .local_rotation([location[0], location[1]])
        .expect("polar rotation");
    let pose = CameraPose {
        position: frame.local(location).expect("polar camera"),
        orientation: local_to_frame * enu_ned() * body_to_ned * eye_to_body(),
    };
    (pose, body_to_ned)
}

/// Optical-axis depth from `pose` to the lunar sphere with a small terrain relief.
#[cfg(test)]
fn lunar_depth(pose: &CameraPose, x: u32, y: u32) -> f32 {
    let radius =
        BodyEllipsoid::MOON_PA.semi_major_axis_m() + 40.0 * (f64::from(x + 2 * y) * 0.013).sin();
    let eye = Vector3::new(
        (f64::from(x) - CAMERA.cx) / CAMERA.fx,
        (CAMERA.cy - f64::from(y)) / CAMERA.fy,
        -1.0,
    );
    let ray = pose.orientation * eye;
    // The origin is on the sphere, so the centre is one radius below it.
    let offset =
        pose.position - Vector3::new(0.0, 0.0, -BodyEllipsoid::MOON_PA.semi_major_axis_m());
    let (a, b, c) = (
        ray.norm_squared(),
        2.0 * ray.dot(&offset),
        offset.norm_squared() - radius * radius,
    );
    ((-b - (b * b - 4.0 * a * c).sqrt()) / (2.0 * a)) as f32
}

/// A reference render at a perturbed pose, its pixel matches, and the query frame.
#[cfg(test)]
fn scene(frame: LocalFrame, truth: &CameraPose) -> (Frame, ReferenceView, Vec<PixelMatch>) {
    let rendered = CameraPose {
        position: truth.position + Vector3::new(25.0, -15.0, 10.0),
        orientation: truth.orientation * UnitQuaternion::from_euler_angles(0.01, -0.008, 0.012),
    };
    let depth_m: Vec<f32> = (0..CAMERA.height)
        .flat_map(|y| (0..CAMERA.width).map(move |x| lunar_depth(&rendered, x, y)))
        .collect();
    let mut matches = Vec::new();
    for y in (30..210).step_by(20) {
        for x in (30..290).step_by(20) {
            let pixel = Vector2::new(f64::from(x), f64::from(y));
            let depth = f64::from(depth_m[(y * CAMERA.width + x) as usize]);
            let world = CAMERA.unproject(&rendered, pixel, depth);
            matches.push(PixelMatch {
                reference: pixel,
                query: CAMERA.project(truth, world).expect("visible terrain"),
            });
        }
    }
    let query = Frame {
        stamp: FrameStamp {
            sequence: 1,
            capture_time_ns: CAPTURE_NS,
        },
        camera: CAMERA,
        image: GrayImage::from_fn(CAMERA.width, CAMERA.height, |x, y| {
            image::Luma([((x * 7 + y * 3) % 251) as u8])
        }),
    };
    let reference = ReferenceView {
        map: MapRevision {
            release_id: "lunar-south-pole".into(),
            manifest_sha256: "c".repeat(64),
        },
        frame,
        pose: rendered,
        image: GrayImage::new(CAMERA.width, CAMERA.height),
        depth_m,
    };
    (query, reference, matches)
}

#[cfg(test)]
fn solve(frame: LocalFrame, truth: &CameraPose) -> Estimate {
    let (query, reference, matches) = scene(frame, truth);
    let prior = PosePrior {
        pose: reference.pose,
        position_radius_m: 100.0,
        attitude_radius_rad: 0.1,
    };
    PoseVerifier::new(LocalizerConfig::default())
        .expect("default policy")
        .verify(&query, &reference, &prior, &matches, "lunar-synthetic")
        .expect("polar pose")
}

#[cfg(test)]
fn contract(q: UnitQuaternion<f64>) -> AttitudeQuaternion {
    AttitudeQuaternion::new(q.w, q.i, q.j, q.k)
}

#[cfg(test)]
fn unit(q: AttitudeQuaternion) -> UnitQuaternion<f64> {
    UnitQuaternion::new_normalize(nalgebra::Quaternion::new(q.w, q.x, q.y, q.z))
}

#[cfg(test)]
fn pose_source() -> VisualFixSource {
    let identity = VisualSourceIdentity {
        source: SourceId::new(77),
        epoch: SourceEpoch::new(0),
        clock: CLOCK,
    };
    let budget = VisualErrorBudget {
        map_horizontal_m: 2.0,
        map_vertical_m: 2.0,
        calibration_m: 0.5,
        geometry_model_m: 0.5,
        vertical_datum: VerticalDatum::Wgs84Ellipsoid,
        vertical_datum_m: 0.5,
    };
    VisualFixSource::new(identity, budget).expect("budget")
}

/// A lunar filter at a position 50 m and an attitude 40 mrad from the truth.
#[cfg(test)]
fn lunar_filter(
    truth_position: GeodeticPosition,
    truth_attitude: UnitQuaternion<f64>,
) -> InertialNavigationFilter {
    let mut filter = InertialNavigationFilter::new_surface(
        InertialConfig::default(),
        CLOCK,
        PlanetaryBody::MOON_PA,
    );
    let plane = LocalTangentPlane::new(truth_position, BodyEllipsoid::MOON_PA).expect("plane");
    filter
        .initialize_surface(
            plane.from_ned(&NedOffset::new(40.0, -30.0, 0.0)),
            NedVelocity::new(0.0, 0.0, 0.0),
            contract(truth_attitude * UnitQuaternion::from_euler_angles(0.02, -0.015, 0.03)),
            SymmetricCov3::from_diagonal(2_500.0, 2_500.0, 2_500.0),
            SymmetricCov3::from_diagonal(1.0, 1.0, 1.0),
            SymmetricCov3::from_diagonal(2.5e-3, 2.5e-3, 2.5e-3),
            MonotonicNanos::from_nanos(CAPTURE_NS),
            SourceComposition::of(SensorClass::Inertial),
        )
        .expect("lunar initialization");
    filter
}

#[test]
fn a_lunar_south_pole_pose_enters_the_inertial_filter_as_one_visual_pose() {
    let frame = lunar_frame();
    let (truth_pose, truth_attitude) = truth(&frame);
    let estimate = solve(frame, &truth_pose);
    assert!((estimate.pose.position - truth_pose.position).norm() < 0.05);
    assert!(estimate.pose.orientation.angle_to(&truth_pose.orientation) < 1e-4);

    let attitude_budget = VisualAttitudeBudget {
        eye_to_body: eye_to_body(),
        mounting_rad: 1e-3,
        map_rad: 1e-3,
    };
    let fix = pose_source()
        .convert_pose(
            &estimate,
            EvidenceIndependence::ValidatedIndependent,
            &attitude_budget,
        )
        .expect("visual pose");
    assert_eq!(fix.observation.frame, Some(FrameId::MOON_PA));
    let ObservationValue::VisualPose {
        position, attitude, ..
    } = fix.observation.value
    else {
        panic!("visual pose expected");
    };
    let [lat, lon, height] = frame.geodetic(truth_pose.position);
    let truth_position = GeodeticPosition::new(lat.to_radians(), lon.to_radians(), height);
    let plane = LocalTangentPlane::new(truth_position, BodyEllipsoid::MOON_PA).expect("plane");
    let offset = plane.to_ned(&position);
    assert!(Vector3::new(offset.north_m, offset.east_m, offset.down_m).norm() < 0.05);
    assert!(unit(attitude).angle_to(&truth_attitude) < 1e-4);

    let mut filter = lunar_filter(truth_position, truth_attitude);
    let now = MonotonicNanos::from_nanos(CAPTURE_NS);
    assert_eq!(
        filter.ingest(&fix.observation, now),
        IngestOutcome::Accepted
    );
    let stats = filter
        .innovation_stats(MeasurementKind::VisualPose)
        .expect("visual pose statistics");
    assert_eq!(stats.admitted_count, 1);
    let solution = filter.tick(now).expect("lunar solution");
    assert_eq!(solution.ellipsoid, Ellipsoid::MOON_PA);
    let error = plane.to_ned(&solution.position);
    let error = Vector3::new(error.north_m, error.east_m, error.down_m);
    // One update moves the state most of the way to the accurate visual pose.
    assert!(error.norm() < 0.25 * 50.0, "position error {error:?}");
    // The remaining error is consistent with the published covariance (99.9 %, 3 axes).
    let [nn, ne, nd, ee, ed, dd] = solution.position_cov.upper_triangle();
    let covariance = Matrix3::new(nn, ne, nd, ne, ee, ed, nd, ed, dd);
    let information = covariance.try_inverse().expect("positive definite");
    assert!(error.dot(&(information * error)) < 16.27);
    let attitude_error = unit(solution.attitude.expect("attitude")).angle_to(&truth_attitude);
    assert!(attitude_error < 5e-3, "attitude error {attitude_error} rad");
}

#[test]
fn a_lunar_pose_cannot_enter_an_earth_filter() {
    let frame = lunar_frame();
    let (truth_pose, _) = truth(&frame);
    let estimate = solve(frame, &truth_pose);
    let attitude_budget = VisualAttitudeBudget {
        eye_to_body: eye_to_body(),
        mounting_rad: 1e-3,
        map_rad: 1e-3,
    };
    let fix = pose_source()
        .convert_pose(
            &estimate,
            EvidenceIndependence::ValidatedIndependent,
            &attitude_budget,
        )
        .expect("visual pose");
    let mut earth = InertialNavigationFilter::new_surface(
        InertialConfig::default(),
        CLOCK,
        PlanetaryBody::EARTH_WGS84,
    );
    let now = MonotonicNanos::from_nanos(CAPTURE_NS);
    earth
        .initialize_surface(
            GeodeticPosition::new(0.8, 0.2, 2_000.0),
            NedVelocity::new(0.0, 0.0, 0.0),
            AttitudeQuaternion::new(1.0, 0.0, 0.0, 0.0),
            SymmetricCov3::from_diagonal(2_500.0, 2_500.0, 2_500.0),
            SymmetricCov3::from_diagonal(1.0, 1.0, 1.0),
            SymmetricCov3::from_diagonal(2.5e-3, 2.5e-3, 2.5e-3),
            now,
            SourceComposition::of(SensorClass::Inertial),
        )
        .expect("earth initialization");
    assert!(matches!(
        earth.ingest(&fix.observation, now),
        IngestOutcome::Rejected(RejectionReason::FrameMismatch {
            expected: FrameId::EARTH_ITRF93,
            got: FrameId::MOON_PA,
        })
    ));
}
