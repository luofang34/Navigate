use super::*;
use crate::{PointTracker, PyramidalMatcher};

#[test]
fn cpu_local_refinement_uses_predictions_and_preserves_slots() {
    exercise(&mut PyramidalMatcher);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware compute adapter; run explicitly on the target device"]
fn gpu_local_refinement_uses_predictions_and_preserves_slots() {
    let mut gpu = pollster::block_on(crate::GpuPyramidalMatcher::new()).unwrap();
    exercise(&mut gpu);
}

fn exercise(tracker: &mut dyn PointTracker) {
    let reference = GrayImage::from_fn(320, 240, |x, y| {
        let h = (x / 3).wrapping_mul(374761393) ^ (y / 3).wrapping_mul(668265263);
        image::Luma([40 + ((h ^ (h >> 13)).wrapping_mul(1274126177) % 160) as u8])
    });
    let query = GrayImage::from_fn(320, 240, |x, y| {
        if x >= 120 && y >= 9 {
            image::Luma([reference.get_pixel(x - 120, y - 9)[0] + 10])
        } else {
            image::Luma([0])
        }
    });
    let mut points: Vec<_> = (20..160)
        .step_by(23)
        .flat_map(|x| {
            (30..200)
                .step_by(23)
                .map(move |y| Vector2::new(f64::from(x), f64::from(y)))
        })
        .collect();
    points.push(Vector2::new(1.0, 1.0));
    let predictions: Vec<_> = points
        .iter()
        .map(|p| p + Vector2::new(120.2, 8.7))
        .collect();
    let output = tracker
        .refine_points_blocking(&reference, &query, &points, &predictions)
        .unwrap();
    assert_eq!(output.len(), points.len());
    assert_eq!(output.last(), Some(&None));
    let correct = points
        .iter()
        .zip(&output)
        .filter(|(p, q)| q.is_some_and(|q| (q - **p - Vector2::new(120.0, 9.0)).norm() < 0.1))
        .count();
    assert!(
        correct > 20 && correct * 10 >= points.len() * 9,
        "{correct} accurate of {}",
        points.len()
    );
    assert!(
        tracker
            .refine_points_blocking(&reference, &query, &points, &predictions[..1])
            .is_err()
    );
    let mut invalid = predictions.clone();
    invalid[0].x = f64::NAN;
    assert!(
        tracker
            .refine_points_blocking(&reference, &query, &points, &invalid)
            .is_err()
    );
    assert!(
        tracker
            .refine_points_blocking(&reference, &query, &[], &[])
            .unwrap()
            .is_empty()
    );
    assert!(
        tracker
            .refine_points_blocking(&reference, &GrayImage::new(320, 240), &points, &predictions)
            .unwrap()
            .iter()
            .all(Option::is_none)
    );
}
