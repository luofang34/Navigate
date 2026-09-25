//! Check image associations without requiring map support.
#![allow(clippy::expect_used, clippy::panic)]
use super::*;
fn pairs(step: usize) -> Vec<PixelMatch> {
    (0..12)
        .flat_map(|y| {
            (0..12).map(move |x| {
                let reference = Vector2::new(
                    30.0 + x as f64 * 12.0 + step as f64 * 2.0,
                    30.0 + y as f64 * 12.0,
                );
                PixelMatch {
                    reference,
                    query: reference + Vector2::new(2.0, 0.0),
                }
            })
        })
        .collect()
}
fn builder() -> DenseTrackBuilder {
    DenseTrackBuilder::new(CameraModel {
        width: 320,
        height: 240,
        fx: 250.0,
        fy: 250.0,
        cx: 159.5,
        cy: 119.5,
    })
    .expect("camera")
}
#[test]
fn identity_survives_motion_and_missing_pairs_break_links() {
    let mut b = builder();
    b.push(format!("{:064x}", 0), &[]).expect("first");
    for i in 0..5 {
        b.push(format!("{:064x}", i + 1), &pairs(i)).expect("frame");
    }
    let snapshot = b.snapshot();
    let stable: Vec<_> = snapshot
        .tracks
        .iter()
        .filter(|t| t.observations.len() == 6)
        .collect();
    assert!(stable.len() >= 40);
    for t in stable {
        for pair in t.observations.windows(2) {
            assert!((pair[1].pixel - pair[0].pixel - Vector2::new(2.0, 0.0)).norm() < 1e-8);
        }
    }
    let old_ids: BTreeSet<_> = b.graph.tracks.iter().map(|t| t.feature_id).collect();
    b.push(format!("{:064x}", 6), &[]).expect("gap");
    b.push(format!("{:064x}", 7), &pairs(6))
        .expect("new interval");
    assert!(!b.active.is_empty());
    assert!(
        b.active
            .iter()
            .all(|&i| !old_ids.contains(&b.graph.tracks[i].feature_id))
    );
    assert!(
        b.graph
            .tracks
            .iter()
            .filter(|t| old_ids.contains(&t.feature_id))
            .all(|t| t.observations.last().expect("support").camera_index <= 5)
    );
}
#[test]
fn rejected_append_preserves_graph_and_group_bounds() {
    let mut b = builder();
    b.push("a".repeat(64), &[]).expect("first");
    assert!(b.push("A".repeat(64), &pairs(0)).is_err());
    assert_eq!(b.graph.observation_sha256.len(), 1);
    assert!(b.graph.tracks.is_empty());
    let bad = PixelMatch {
        reference: Vector2::new(0.0, 0.0),
        query: Vector2::new(320.0, 0.0),
    };
    assert!(b.push("b".repeat(64), &[bad]).is_err());
    assert_eq!(b.graph.observation_sha256.len(), 1);
    for i in 1..129 {
        b.push(format!("{i:064x}"), &[]).expect("bounded group");
    }
    assert!(matches!(
        b.push(format!("{:064x}", 130), &[]),
        Err(ReconstructionError::Limits { .. })
    ));
    assert_eq!(b.graph.observation_sha256.len(), 129);
}

#[test]
fn abandoned_two_image_links_do_not_fill_the_group() {
    let mut b = builder();
    b.push(format!("{:064x}", 0), &[]).expect("first");
    for i in 1..120 {
        let links = if i % 2 == 0 { Vec::new() } else { pairs(0) };
        b.push(format!("{i:064x}"), &links)
            .expect("bounded failed associations");
        if i % 2 == 0 {
            assert!(b.graph.tracks.is_empty());
        }
    }
    assert!(b.graph.tracks.iter().all(|t| t.observations.len() == 2));
    assert!(b.graph.tracks.iter().all(|t| t.feature_id > 5000));
}
