use super::*;
use navigate_camera::{CameraMount, ImageView, Intrinsics, Pixels};

#[test]
fn intrinsics_convert_both_ways_without_loss() {
    let model = Intrinsics {
        width: 1600,
        height: 1200,
        fx: 1310.5,
        fy: 1308.25,
        cx: 801.75,
        cy: 597.125,
    };
    let mode = CalibrationMode::PriorFocal { sigma_rel: 0.15 };
    let live = CameraIntrinsics::from_intrinsics(model, mode);
    assert_eq!(live.calibration_confidence, mode);
    assert_eq!(Intrinsics::from(live), model);
    assert_eq!(live.to_matrix3(), model.matrix());
}

#[test]
fn scaling_keeps_pixel_centres_where_the_resampler_puts_them() {
    let model = Intrinsics::centred(1600, 1200, 1.0).unwrap();
    let live = CameraIntrinsics::from_intrinsics(model, CalibrationMode::Calibrated);
    let scaled = live.scaled_to([400, 300]);
    let expected = model.resized([400, 300]).unwrap();
    assert_eq!(Intrinsics::from(scaled), expected);
    // The optical axis of a centred camera stays on the image centre.
    assert!((scaled.cx - 199.5).abs() < 1e-12 && (scaled.cy - 149.5).abs() < 1e-12);
}

#[test]
fn mount_profile_keeps_the_frd_to_optical_rotation() {
    let mount = CameraMount::nadir().with_lever_arm(nalgebra::Vector3::new(1.0, 0.0, 0.5));
    let profile = CameraMountProfile::from_camera_mount(&mount, true, PlatformKind::FixedWing);
    assert_eq!(profile.cam_from_body, mount.cam_from_body);
    assert!(profile.has_airframe_occlusion);
    assert_eq!(profile.platform, PlatformKind::FixedWing);
    // A nadir camera looks along body down (+Z FRD).
    let boresight = profile.cam_from_body.inverse() * nalgebra::Vector3::z();
    assert!((boresight - nalgebra::Vector3::z()).norm() < 1e-12);
}

#[test]
fn gray8_and_rgb8_views_borrow_with_the_same_stride() {
    let gray = vec![7_u8; 40 * 3];
    let view = ImageView::new(
        32,
        3,
        Pixels::Gray8 {
            data: &gray,
            stride: 40,
        },
    )
    .unwrap();
    match FramePixels::try_from(view).unwrap() {
        FramePixels::Gray8 {
            data,
            width,
            height,
            stride,
        } => {
            assert_eq!((width, height, stride), (32, 3, 40));
            assert_eq!(data.as_ptr(), gray.as_ptr());
        }
        other => panic!("unexpected {other:?}"),
    }
    let rgb = vec![9_u8; 100 * 2];
    let view = ImageView::new(
        32,
        2,
        Pixels::Rgb8 {
            data: &rgb,
            stride: 100,
        },
    )
    .unwrap();
    let pixels = FramePixels::try_from(view).unwrap();
    assert!(matches!(pixels, FramePixels::Rgb8 { stride: 100, .. }));
    assert_eq!(pixels.dimensions(), [32, 2]);
}

#[test]
fn gray16_view_is_refused_with_the_agc_hint() {
    let raw = vec![1000_u16; 32 * 4];
    let view = ImageView::new(
        32,
        4,
        Pixels::Gray16 {
            data: &raw,
            stride: 32,
            significant_bits: 14,
        },
    )
    .unwrap();
    let err = FramePixels::try_from(view).unwrap_err();
    assert!(err.to_string().contains("to_gray8_percentile"), "{err}");
}

/// Asserts the angles of `prior` in degrees.
fn assert_angles(prior: &AttitudePrior, roll_deg: f64, pitch_deg: f64, yaw_deg: f64) {
    let close = |a: f64, b: f64| (a - b).abs() < 1e-9;
    assert!(close(prior.roll_rad.to_degrees(), roll_deg), "{prior:?}");
    assert!(close(prior.pitch_rad.to_degrees(), pitch_deg), "{prior:?}");
    let yaw = prior.yaw_rad.expect("yaw").to_degrees();
    assert!(close(yaw, yaw_deg), "{prior:?}");
}

/// `ned_from_body` from aerospace roll, pitch, and yaw in degrees.
fn body(roll_deg: f64, pitch_deg: f64, yaw_deg: f64) -> nalgebra::UnitQuaternion<f64> {
    nalgebra::UnitQuaternion::from_euler_angles(
        roll_deg.to_radians(),
        pitch_deg.to_radians(),
        yaw_deg.to_radians(),
    )
}

#[test]
fn the_camera_prior_applies_the_mount() {
    let prior = |ned_from_body, mount: &CameraMountProfile| {
        AttitudePrior::from_body_attitude(&ned_from_body, mount, 0.01, 0.02, None)
    };
    let oblique = CameraMountProfile::fixed_wing_oblique(20.0_f64.to_radians());
    assert_angles(&prior(body(0.0, 0.0, 30.0), &oblique), 0.0, 20.0, 30.0);
    // Nose up 5°: the camera looks 15° down.
    assert_angles(&prior(body(0.0, 5.0, 30.0), &oblique), 0.0, 15.0, 30.0);
    let forward = CameraMountProfile::fixed_wing_oblique(0.0);
    assert_angles(&prior(body(15.0, 0.0, 300.0), &forward), 15.0, 0.0, 300.0);
    // A vertical axis takes its heading from the image top, the nose.
    let nadir = CameraMountProfile::rotary_gimbal_nadir();
    assert_angles(&prior(body(0.0, 0.0, 70.0), &nadir), 0.0, 90.0, 70.0);
}

/// The engine's camera axes `(right, down, forward)` in NED for a heading,
/// a depression, and a roll in radians, from `Surface::camera_rotation` at
/// the origin of the surface, where its east, north, up axes are local.
fn engine_axes(h: f64, d: f64, r: f64) -> [nalgebra::Vector3<f64>; 3] {
    let surface = crate::Surface::synthetic(39.0, -75.5, 0.0, image::RgbImage::new(4, 4));
    let rotation =
        surface.camera_rotation(39.0, -75.5, h.to_degrees(), d.to_degrees(), r.to_degrees());
    let ned = |row: usize| {
        let enu = rotation.row(row).transpose();
        nalgebra::Vector3::new(enu.y, enu.x, -enu.z)
    };
    [ned(0), ned(1), ned(2)]
}

#[test]
fn the_camera_prior_inverts_the_engine_camera_rotation() {
    let mount = CameraMountProfile::fixed_wing_oblique(35.0_f64.to_radians());
    for (h, d, r) in [
        (10.0, 30.0, -20.0),
        (200.0, -10.0, 45.0),
        (355.0, 80.0, 5.0),
    ] {
        let [right, down, fwd] =
            engine_axes(f64::to_radians(h), f64::to_radians(d), f64::to_radians(r));
        let ned_from_cam = nalgebra::Rotation3::from_basis_unchecked(&[right, down, fwd]);
        let ned_from_body =
            nalgebra::UnitQuaternion::from_rotation_matrix(&ned_from_cam) * mount.cam_from_body;
        let prior = AttitudePrior::from_body_attitude(&ned_from_body, &mount, 0.01, 0.02, None);
        assert_angles(&prior, r, d, h);
    }
}
