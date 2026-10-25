use super::*;
use navigate_visual::{CameraModel, FrameStamp, PyramidalMatcher};
fn frame(index: u64, shift: u32) -> Frame {
    Frame {
        camera: CameraModel {
            width: 320,
            height: 240,
            fx: 150.0,
            fy: 150.0,
            cx: 160.0,
            cy: 120.0,
        },
        stamp: FrameStamp {
            sequence: index,
            capture_time_ns: index * 33_333_333,
        },
        image: image::GrayImage::from_fn(320, 240, |x, y| {
            let h = (x.saturating_sub(shift) / 3).wrapping_mul(374761393)
                ^ (y / 3).wrapping_mul(668265263);
            image::Luma([40 + ((h ^ (h >> 13)).wrapping_mul(1274126177) % 160) as u8])
        }),
    }
}
#[test]
fn correction_keeps_map_identities_and_failed_slots_keep_predictions() {
    let first = frame(0, 0);
    let next = frame(1, 1);
    let mut tracker = PyramidalMatcher;
    let pairs: Vec<_> = tracker
        .features_blocking(&first.image)
        .unwrap()
        .into_iter()
        .filter(|p| p.x > 30.0 && p.x < 270.0 && p.y > 30.0 && p.y < 200.0)
        .map(|p| PixelMatch {
            reference: p,
            query: p,
        })
        .collect();
    let predictions: Vec<_> = pairs
        .iter()
        .map(|p| PixelMatch {
            reference: p.reference,
            query: p.query + Vector2::new(1.3, 0.2),
        })
        .collect();
    let mut anchor = Anchoring::new(Some(2));
    anchor.observe(&first, &pairs, true);
    let corrected = anchor
        .correct_blocking(&next, &predictions, &mut tracker)
        .unwrap()
        .unwrap();
    assert!(corrected.report["corrected_points"].as_u64().unwrap() > 20);
    for (a, b) in corrected.pairs.iter().zip(&pairs) {
        assert_eq!(a.reference, b.reference);
    }
    let blank = Frame {
        image: image::GrayImage::new(320, 240),
        ..frame(1, 1)
    };
    let failed = anchor
        .correct_blocking(&blank, &predictions, &mut tracker)
        .unwrap()
        .unwrap();
    assert_eq!(failed.report["corrected_points"], 0);
    for (a, b) in failed.pairs.iter().zip(&predictions) {
        assert_eq!(a.query, b.query);
        assert_eq!(a.reference, b.reference);
    }
    anchor.observe(&next, &corrected.pairs, false);
    assert_eq!(anchor.keyframe.as_ref().unwrap().frame.stamp.sequence, 0);
    anchor.observe(&frame(2, 2), &corrected.pairs, false);
    assert_eq!(anchor.keyframe.as_ref().unwrap().frame.stamp.sequence, 2);
    anchor.clear();
    assert!(
        anchor
            .correct_blocking(&next, &predictions, &mut tracker)
            .unwrap()
            .is_none()
    );
}
#[test]
fn ambiguous_map_identities_cannot_be_reassigned_by_correction() {
    let first = frame(0, 0);
    let next = frame(1, 1);
    let p = PixelMatch {
        reference: Vector2::new(80.0, 80.0),
        query: Vector2::new(80.0, 80.0),
    };
    let mut anchor = Anchoring::new(Some(5));
    anchor.observe(&first, &[p, p], true);
    let result = anchor
        .correct_blocking(&next, &[p], &mut PyramidalMatcher)
        .unwrap()
        .unwrap();
    assert_eq!(result.report["bound_points"], 0);
    assert_eq!(result.pairs[0].query, p.query);
    anchor.observe(&first, &[p], true);
    let result = anchor
        .correct_blocking(&next, &[p, p], &mut PyramidalMatcher)
        .unwrap()
        .unwrap();
    assert_eq!(result.report["unbound_or_ambiguous"], 2);
}
#[test]
fn disabled_policy_retains_no_keyframe_and_intervals_are_bounded() {
    let mut anchor = Anchoring::new(None);
    anchor.observe(&frame(0, 0), &[], true);
    assert!(anchor.keyframe.is_none());
    assert!(valid_interval(None) && valid_interval(Some(2)) && valid_interval(Some(30)));
    assert!(!valid_interval(Some(1)) && !valid_interval(Some(31)));
}
