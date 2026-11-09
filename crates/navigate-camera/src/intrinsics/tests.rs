use nalgebra::{Vector2, Vector3};

use super::*;
use crate::Distortion;

#[test]
fn centred_800_by_600_has_the_pixel_centre_principal_point() {
    let k = Intrinsics::centred(800, 600, 60_f64.to_radians()).unwrap();
    assert_eq!((k.cx, k.cy), (399.5, 299.5));
    let centre = k.project(&Vector3::new(0.0, 0.0, 10.0)).unwrap();
    assert_eq!(centre, Vector2::new(399.5, 299.5));
}

#[test]
fn project_and_ray_are_inverse() {
    let k = Intrinsics {
        width: 640,
        height: 512,
        fx: 700.0,
        fy: 710.0,
        cx: 300.0,
        cy: 260.0,
    };
    let pixel = Vector2::new(12.25, 480.5);
    let ray = k.ray(&pixel);
    let back = k.project(&(ray * 37.0)).unwrap();
    assert!((back - pixel).norm() < 1e-9);
}

#[test]
fn points_behind_the_camera_do_not_project() {
    let k = Intrinsics::centred(64, 64, 1.0).unwrap();
    assert!(k.project(&Vector3::new(0.0, 0.0, -1.0)).is_none());
    assert!(k.project(&Vector3::new(1.0, 0.0, 0.0)).is_none());
}

#[test]
fn resize_keeps_the_ray_of_each_pixel_corner() {
    let k = Intrinsics::centred(800, 600, 1.0).unwrap();
    let half = k.resized([400, 300]).unwrap();
    let corner_full = k.ray(&Vector2::new(-0.5, -0.5));
    let corner_half = half.ray(&Vector2::new(-0.5, -0.5));
    assert!((corner_full - corner_half).norm() < 1e-12);
}

#[test]
fn invalid_intrinsics_are_refused() {
    let mut k = Intrinsics::centred(64, 64, 1.0).unwrap();
    k.fx = f64::NAN;
    assert!(k.validate().is_err());
    assert!(Intrinsics::centred(64, 64, 4.0).is_err());
}

#[test]
fn brown_conrady_round_trips() {
    let d = Distortion::BrownConrady {
        k1: -0.21,
        k2: 0.05,
        p1: 1e-3,
        p2: -5e-4,
        k3: -0.004,
    };
    let ideal = Vector2::new(0.31, -0.22);
    let back = d.undistort(&d.distort(&ideal)).unwrap();
    assert!((back - ideal).norm() < 1e-10);
    assert_eq!(Distortion::None.distort(&ideal), ideal);
}
