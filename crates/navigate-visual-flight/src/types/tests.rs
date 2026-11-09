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
