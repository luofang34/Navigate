use super::*;
use navigate_visual::{FrameStamp, PyramidalMatcher};

#[test]
fn probe_frame_selection_is_bounded_unique_and_increasing() {
    assert!(valid_frames(&[], 1));
    assert!(valid_frames(&[0, 39, 40, 49, 50], 97));
    assert!(!valid_frames(&[40, 39], 97));
    assert!(!valid_frames(&[39, 39], 97));
    assert!(!valid_frames(&[97], 97));
    assert!(!valid_frames(&(0..17).collect::<Vec<_>>(), 97));
}

#[test]
fn real_tracking_retains_slots_and_fixed_map_coordinates() {
    let (camera, reference) = crate::evaluate::tests::fixture();
    let previous = Frame {
        camera,
        image: reference.image,
        stamp: FrameStamp {
            sequence: 0,
            capture_time_ns: 0,
        },
    };
    let frame = Frame {
        camera,
        image: image::GrayImage::from_fn(320, 240, |x, y| {
            *previous.image.get_pixel((x + 1).min(319), y)
        }),
        stamp: FrameStamp {
            sequence: 1,
            capture_time_ns: 33_333_333,
        },
    };
    let mut tracker = PyramidalMatcher;
    let mut pairs: Vec<_> = tracker
        .features_blocking(&previous.image)
        .unwrap()
        .into_iter()
        .map(|p| PixelMatch {
            reference: p,
            query: p,
        })
        .collect();
    pairs.push(PixelMatch {
        reference: [5.0, 5.0].into(),
        query: [1.0, 1.0].into(),
    });
    let (report, propagated) = propagate_blocking(&previous, &frame, &pairs, &mut tracker).unwrap();
    let slots = report["slots"].as_array().unwrap();
    assert_eq!(slots.len(), pairs.len());
    assert!(slots.last().unwrap()["query"].is_null());
    assert!(propagated.len() > 20);
    let mut retained = propagated.iter();
    for (slot, (row, pair)) in slots.iter().zip(&pairs).enumerate() {
        assert_eq!(row["slot"], slot);
        if let Some(q) = row["query"].as_array() {
            let result = retained.next().unwrap();
            assert_eq!(result.reference, pair.reference);
            assert_eq!(result.query.x, q[0].as_f64().unwrap());
            assert_eq!(result.query.y, q[1].as_f64().unwrap());
        }
    }
    assert!(retained.next().is_none());
}
