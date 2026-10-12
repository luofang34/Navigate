use super::*;
use navigate_visual::{CameraModel, FrameStamp, PyramidalMatcher};
fn frame(id: u64, time: u64) -> Frame {
    Frame {
        image: image::GrayImage::from_fn(128, 128, |x, y| {
            image::Luma([((x * 71 + y * 113 + x * y * 17) % 251) as u8])
        }),
        camera: CameraModel {
            width: 128,
            height: 128,
            fx: 100.0,
            fy: 100.0,
            cx: 64.0,
            cy: 64.0,
        },
        stamp: FrameStamp {
            sequence: id,
            capture_time_ns: time,
        },
    }
}
fn point(x: f64, origin: u64, age: usize) -> Evidence {
    Evidence {
        pair: PixelMatch {
            query: [x, 64.0].into(),
            reference: [x, 64.0].into(),
        },
        origin,
        source_pair_index: 0,
        age,
        rotation: 0,
    }
}
#[test]
fn repeated_map_or_query_points_do_not_inflate_support() {
    let fresh = point(64.0, 2, 0);
    let mut same_map = point(90.0, 1, 1);
    same_map.pair.reference = fresh.pair.reference;
    let mut same_query = point(90.0, 1, 1);
    same_query.pair.query = fresh.pair.query;
    let kept = merge(
        vec![fresh],
        vec![point(64.0, 1, 1), same_map, same_query, point(32.0, 1, 1)],
    );
    assert_eq!(kept.len(), 2);
    assert_eq!(kept[0].origin, 2);
    assert_eq!(kept[1].origin, 1);
}
#[test]
fn optical_flow_keeps_source_identity_and_expires_old_evidence() -> Result<(), Error> {
    let a = frame(1, 0);
    let b = frame(2, 33_333_333);
    let mut tracker = PyramidalMatcher;
    let found = advance_blocking(vec![point(64.0, 1, 0)], &a, &b, 2, &mut tracker)?;
    assert_eq!(found.pairs.len(), 1);
    assert_eq!(found.pairs[0].origin, 1);
    assert_eq!(found.pairs[0].age, 1);
    assert!((found.pairs[0].pair.query.x - 64.0).abs() < 0.01);
    assert!(
        advance_blocking(found.pairs, &b, &frame(3, 66_666_666), 2, &mut tracker)?
            .pairs
            .is_empty()
    );
    assert!(
        advance_blocking(
            vec![point(64.0, 1, 0)],
            &a,
            &frame(3, 200_000_001),
            5,
            &mut tracker
        )?
        .pairs
        .is_empty()
    );
    assert!(
        advance_blocking(vec![point(64.0, 1, 0)], &a, &b, 1, &mut tracker)?
            .pairs
            .is_empty()
    );
    Ok(())
}

#[test]
fn textureless_tracks_are_removed_and_single_frame_skips_flow() -> Result<(), Error> {
    let mut a = frame(1, 0);
    let mut b = frame(2, 33_333_333);
    a.image.fill(0);
    b.image.fill(0);
    let mut tracker = PyramidalMatcher;
    let failed = advance_blocking(vec![point(64.0, 1, 0)], &a, &b, 5, &mut tracker)?;
    assert!(failed.attempted);
    assert!(failed.pairs.is_empty());
    assert_eq!(failed.tracking_rejected, 1);
    let skipped = advance_blocking(vec![point(64.0, 1, 0)], &a, &b, 1, &mut tracker)?;
    assert!(!skipped.attempted);
    assert_eq!(skipped.expired_or_reset, 1);
    assert_eq!(skipped.tracking_rejected, 0);
    Ok(())
}
