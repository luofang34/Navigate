use super::*;
use nalgebra::{Vector2, Vector3};
use navigate_visual::{
    CameraPose, FrameStamp, LocalizerConfig, PixelMatch, PosePrior, PoseVerifier, ReferenceView,
};

#[test]
fn map_correction_does_not_enter_continuous_motion_or_create_extra_anchors() {
    let (camera, reference) = super::super::tests::fixture();
    let mut baseline = Navigation::new(camera, settings()).unwrap();
    let mut corrected = Navigation::new(camera, settings()).unwrap();
    let (first, initial) = fit(&reference, 0, 0.0, 0.0);
    let evidence = Evidence {
        tracked: None,
        matched: Some(initial.clone()),
    };
    baseline.step(&first, &evidence, Some(&initial)).unwrap();
    corrected.step(&first, &evidence, Some(&initial)).unwrap();
    let (second, tracked) = fit(&reference, 1, 20.0, 0.0);
    let (_, matched) = fit(&reference, 1, 20.0, 3.0);
    let a = baseline
        .step(
            &second,
            &Evidence {
                tracked: Some(tracked.clone()),
                matched: Some(tracked.clone()),
            },
            Some(&tracked),
        )
        .unwrap();
    let b = corrected
        .step(
            &second,
            &Evidence {
                tracked: Some(tracked),
                matched: Some(matched.clone()),
            },
            Some(&matched),
        )
        .unwrap();
    assert_eq!(a.record["odometry"], b.record["odometry"]);
    assert_eq!(b.record["anchor"]["status"], "accepted");
    assert!((a.position.unwrap()[1] - b.position.unwrap()[1]).abs() > 0.000001);
    let anchors = b.record["usage"]["anchors"].clone();
    let (third, unshifted) = fit(&reference, 2, 40.0, 0.0);
    let (_, reseeded) = fit(&reference, 2, 40.0, 3.0);
    let a = baseline
        .step(
            &third,
            &Evidence {
                tracked: Some(unshifted.clone()),
                matched: None,
            },
            Some(&unshifted),
        )
        .unwrap();
    let b = corrected
        .step(
            &third,
            &Evidence {
                tracked: Some(reseeded.clone()),
                matched: None,
            },
            Some(&reseeded),
        )
        .unwrap();
    let x = a.record["odometry"]["pose"]["position"][0]
        .as_f64()
        .unwrap();
    let y = b.record["odometry"]["pose"]["position"][0]
        .as_f64()
        .unwrap();
    assert!((x - y).abs() < 1e-6);
    assert!((x - 40.0).abs() < 1e-4);
    assert!(b.record["anchor"].is_null());
    assert_eq!(b.record["usage"]["anchors"], anchors);
    assert!(
        b.record["odometry"]["scale"]
            .as_str()
            .unwrap()
            .starts_with("SeededMapPoints")
    );
}

#[test]
fn a_lost_track_starts_a_new_segment_instead_of_inventing_motion() {
    let (camera, reference) = super::super::tests::fixture();
    let mut navigation = Navigation::new(camera, settings()).unwrap();
    let (first, initial) = fit(&reference, 0, 0.0, 0.0);
    let a = navigation
        .step(
            &first,
            &Evidence {
                tracked: None,
                matched: Some(initial.clone()),
            },
            Some(&initial),
        )
        .unwrap();
    let (next, far) = fit(&reference, 1, 0.0, 80.0);
    let b = navigation
        .step(
            &next,
            &Evidence {
                tracked: None,
                matched: Some(far.clone()),
            },
            Some(&far),
        )
        .unwrap();
    assert_ne!(
        a.record["odometry"]["segment"],
        b.record["odometry"]["segment"]
    );
    assert_eq!(b.record["continuity"], 1);
    assert_eq!(b.record["odometry"]["path_m"], 0.0);
    assert_eq!(b.record["map"]["confirmed"], false);
}

#[test]
fn error_declarations_must_be_finite_and_positive() {
    let (camera, _) = super::super::tests::fixture();
    let mut input = settings();
    input.map_horizontal_sigma_m = 0.0;
    assert!(Navigation::new(camera, input).is_err());
    input = settings();
    input.max_object_height_m = f64::NAN;
    assert!(Navigation::new(camera, input).is_err());
}

#[cfg(test)]
fn settings() -> Settings {
    Settings {
        map_horizontal_sigma_m: 2.0,
        map_vertical_sigma_m: 1.0,
        max_object_height_m: 0.0,
        capture_time_error_bound_ns: 1_000_000,
        max_located_bound_m: 1000.0,
    }
}

#[cfg(test)]
fn fit(
    reference: &ReferenceView,
    sequence: u64,
    travel: f64,
    map_offset: f64,
) -> (Frame, Estimate) {
    let (camera, _) = super::super::tests::fixture();
    let mut reference = ReferenceView {
        map: reference.map.clone(),
        frame: reference.frame,
        pose: reference.pose,
        image: reference.image.clone(),
        depth_m: reference.depth_m.clone(),
    };
    reference.pose.position.x += map_offset;
    let pose = CameraPose {
        position: Vector3::new(travel + map_offset, 0.0, 0.0),
        orientation: reference.pose.orientation,
    };
    let mut pairs = Vec::new();
    for y in (20..220).step_by(20) {
        for x in (90..300).step_by(20) {
            let p = Vector2::new(f64::from(x), f64::from(y));
            let world = camera.unproject(&reference.pose, p, 100.0);
            pairs.push(PixelMatch {
                reference: p,
                query: camera.project(&pose, world).unwrap(),
            });
        }
    }
    let frame = Frame {
        camera,
        image: image::GrayImage::from_fn(320, 240, |x, y| {
            *reference.image.get_pixel((x + sequence as u32).min(319), y)
        }),
        stamp: FrameStamp {
            sequence,
            capture_time_ns: sequence * 33_333_333,
        },
    };
    let prior = PosePrior {
        pose,
        position_radius_m: 100.0,
        attitude_radius_rad: 0.5,
    };
    let estimate = PoseVerifier::new(LocalizerConfig::default())
        .unwrap()
        .verify(&frame, &reference, &prior, &pairs, "session-test")
        .unwrap();
    (frame, estimate)
}
