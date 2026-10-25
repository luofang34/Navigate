//! Routing decisions use the real geometric acceptance path.
use super::*;
use nalgebra::UnitQuaternion;
use navigate_visual::{
    CameraModel, CameraPose, FrameStamp, MapRevision, PixelMatch, RasterGeometry,
};
#[cfg(test)]
fn scene() -> (Frame, OrthographicReference, PosePrior, Vec<PixelMatch>) {
    let camera = CameraModel {
        width: 640,
        height: 480,
        fx: 450.0,
        fy: 460.0,
        cx: 319.5,
        cy: 239.5,
    };
    let reference = OrthographicReference {
        map: MapRevision {
            release_id: "test".into(),
            manifest_sha256: "b".repeat(64),
        },
        image: image::GrayImage::new(640, 480),
        heights_m: (0..480)
            .flat_map(|y| (0..640).map(move |x| 5.0 * ((x + y) as f32 * 0.03).sin()))
            .collect(),
        geometry: RasterGeometry {
            origin: [-80.0, 60.0].into(),
            column: [0.25, 0.0].into(),
            row: [0.0, -0.25].into(),
        },
    };
    let pose = CameraPose {
        position: [2.0, -3.0, 130.0].into(),
        orientation: UnitQuaternion::from_euler_angles(0.03, -0.04, 0.2),
    };
    let prior = PosePrior {
        pose: CameraPose {
            position: [0.0, 0.0, 125.0].into(),
            orientation: UnitQuaternion::identity(),
        },
        position_radius_m: 50.0,
        attitude_radius_rad: 0.5,
    };
    let mut pairs = Vec::new();
    for y in (50..430).step_by(40) {
        for x in (50..590).step_by(40) {
            let r = [f64::from(x), f64::from(y)].into();
            let q = camera
                .project(&pose, reference.world_at(r).unwrap())
                .unwrap();
            if q.x > 0.0 && q.x < 639.0 && q.y > 0.0 && q.y < 479.0 {
                pairs.push(PixelMatch {
                    reference: r,
                    query: q,
                });
            }
        }
    }
    (
        Frame {
            camera,
            stamp: FrameStamp {
                sequence: 0,
                capture_time_ns: 1,
            },
            image: image::GrayImage::new(640, 480),
        },
        reference,
        prior,
        pairs,
    )
}
#[test]
fn rejected_prefix_expands_and_successful_prefix_still_checks_two_directions() {
    let (frame, reference, prior, pairs) = scene();
    let verifier = PoseVerifier::new(LocalizerConfig::default()).unwrap();
    for good in [Some(0), Some(2), None] {
        let mut evaluated = Vec::new();
        let rows = routed_blocking(&[0, 1, 2, 3], |rotation| {
            evaluated.push(rotation);
            let evidence = if Some(rotation) == good {
                pairs.clone()
            } else {
                vec![pairs[0]; 40]
            };
            Ok(report::verify(
                &verifier,
                &frame,
                &reference,
                &prior,
                super::super::matching::Matches {
                    rotation,
                    pairs: evidence,
                    matching_ms: 0.0,
                },
                "test",
                false,
            ))
        })
        .unwrap();
        assert_eq!(accepted(&rows), good.is_some());
        assert_eq!(
            evaluated,
            if good == Some(0) {
                vec![0, 1]
            } else {
                vec![0, 1, 2, 3]
            }
        );
    }
}

#[test]
fn fallback_budget_checks_both_images() {
    let (frame, mut reference, _, _) = scene();
    assert!(validate_pixels(&frame, &reference, 640 * 480).is_ok());
    assert!(validate_pixels(&frame, &reference, 640 * 480 - 1).is_err());
    reference.image = image::GrayImage::new(641, 480);
    assert!(validate_pixels(&frame, &reference, 640 * 480).is_err());
}
