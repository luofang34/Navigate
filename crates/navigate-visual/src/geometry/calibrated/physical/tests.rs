//! Independent ellipsoidal observations expose map-model pose bias.
use crate::{
    CameraModel, CameraPose, Estimate, Frame, FrameStamp, LocalFrame, LocalizerConfig, MapRevision,
    PixelMatch, PosePrior, PoseVerifier, ReferenceView,
};
use image::GrayImage;
use nalgebra::{UnitQuaternion, Vector3};

#[test]
fn point_displacement_does_not_bound_a_geometrically_accepted_pose() {
    for (latitude, height, minimum_bias) in [(50.0, 6000.0, 10.0), (75.0, 30_000.0, 100.0)] {
        let sphere = fit(latitude, height, true);
        let control = fit(latitude, height, false);
        assert_eq!(sphere.quality.inliers, 99);
        assert!(sphere.quality.reprojection_rms_px < 0.5);
        assert!((sphere.pose.position.z - height).abs() > minimum_bias);
        assert!(sphere.frame.model_error_m(sphere.pose.position) < 0.01);
        assert!((control.pose.position - Vector3::new(0.0, 0.0, height)).norm() < 0.01);
    }
}
#[cfg(test)]
fn fit(latitude: f64, height: f64, sphere: bool) -> Estimate {
    let camera = CameraModel {
        width: 1280,
        height: 960,
        fx: 900.0,
        fy: 900.0,
        cx: 639.5,
        cy: 479.5,
    };
    let frame_model = LocalFrame::spherical_enu(latitude, 0.0).unwrap();
    let truth = CameraPose {
        position: Vector3::new(0.0, 0.0, height),
        orientation: UnitQuaternion::identity(),
    };
    let seed = CameraPose {
        position: truth.position + Vector3::new(37.0, -23.0, 41.0),
        orientation: UnitQuaternion::from_euler_angles(0.015, -0.012, 0.008),
    };
    let frame = Frame {
        camera,
        stamp: FrameStamp {
            sequence: 0,
            capture_time_ns: 1,
        },
        image: GrayImage::new(1280, 960),
    };
    // The control uses XYZ geometry only; its frame tag does not convert an oracle pose.
    let mut reference = ReferenceView {
        frame: frame_model,
        pose: seed,
        map: MapRevision {
            release_id: "synthetic".into(),
            manifest_sha256: "a".repeat(64),
        },
        image: frame.image.clone(),
        depth_m: vec![0.0; 1280 * 960],
    };
    let mut pairs = Vec::new();
    for geographic in locations(latitude, height) {
        let oracle = ellipsoid_enu(latitude, geographic);
        let world = if sphere {
            frame_model.local(geographic).unwrap()
        } else {
            oracle
        };
        let pixel = camera.project(&seed, world).unwrap();
        let index = pixel.y.round() as usize * 1280 + pixel.x.round() as usize;
        assert_eq!(reference.depth_m[index], 0.0);
        reference.depth_m[index] = -(seed.orientation.inverse() * (world - seed.position)).z as f32;
        pairs.push(PixelMatch {
            reference: pixel,
            query: camera.project(&truth, oracle).unwrap(),
        });
    }
    PoseVerifier::new(LocalizerConfig::default())
        .unwrap()
        .verify_calibrated_reference(
            &frame,
            &reference,
            &camera,
            &PosePrior {
                pose: seed,
                position_radius_m: 1000.0,
                attitude_radius_rad: 0.3,
            },
            &pairs,
            "physical-model-control",
        )
        .unwrap()
}
#[cfg(test)]
fn locations(latitude: f64, height: f64) -> Vec<[f64; 3]> {
    let a = 6_378_137.0;
    let f = 1.0 / 298.257_223_563;
    let e2 = f * (2.0 - f);
    let lat = latitude.to_radians();
    let d = 1.0 - e2 * lat.sin().powi(2);
    let n = a / d.sqrt();
    let m = a * (1.0 - e2) / d.powf(1.5);
    (0..9)
        .flat_map(|y| {
            (0..11).map(move |x| {
                let east = (-0.4 + 0.08 * f64::from(x)) * height;
                let north = (-0.3 + 0.075 * f64::from(y)) * height;
                [
                    latitude + (north / m).to_degrees(),
                    (east / (n * lat.cos())).to_degrees(),
                    0.0,
                ]
            })
        })
        .collect()
}
#[cfg(test)]
fn ellipsoid_enu(anchor_latitude: f64, geographic: [f64; 3]) -> Vector3<f64> {
    let ecef = |[latitude, longitude, h]: [f64; 3]| {
        let a = 6_378_137.0;
        let f = 1.0 / 298.257_223_563;
        let e2 = f * (2.0 - f);
        let (s, c) = latitude.to_radians().sin_cos();
        let (sl, cl) = longitude.to_radians().sin_cos();
        let n = a / (1.0 - e2 * s * s).sqrt();
        Vector3::new((n + h) * c * cl, (n + h) * c * sl, (n * (1.0 - e2) + h) * s)
    };
    let delta = ecef(geographic) - ecef([anchor_latitude, 0.0, 0.0]);
    let (s, c) = anchor_latitude.to_radians().sin_cos();
    Vector3::new(
        delta.y,
        -s * delta.x + c * delta.z,
        c * delta.x + s * delta.z,
    )
}
