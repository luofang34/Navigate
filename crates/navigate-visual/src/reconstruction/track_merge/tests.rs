//! Behavioral controls for joining overlapping image evidence.
#![allow(clippy::expect_used, clippy::panic)]
use super::*;
use crate::{ScenePointObservation, reconstruction::ImageTrack};
fn camera() -> CameraModel {
    CameraModel {
        width: 960,
        height: 544,
        fx: 700.0,
        fy: 690.0,
        cx: 479.5,
        cy: 271.5,
    }
}
fn digest(id: u64) -> String {
    format!("{id:064x}")
}
fn graph(ids: &[u64], features: &[(u64, f64)]) -> ImageTracks {
    ImageTracks {
        observation_sha256: ids.iter().map(|&id| digest(id)).collect(),
        tracks: features
            .iter()
            .map(|&(feature_id, x)| ImageTrack {
                feature_id,
                observations: ids
                    .iter()
                    .enumerate()
                    .map(|(camera_index, _)| ScenePointObservation {
                        camera_index,
                        pixel: Vector2::new(x, 150.0),
                    })
                    .collect(),
            })
            .collect(),
    }
}
fn select(merger: &ImageTrackMerger) -> MergedTrackSelection {
    merger
        .select(merger.observation_sha256(), 65536, 1)
        .expect("source selection")
}
#[test]
fn joins_two_shared_observations_and_keeps_exact_source_provenance() {
    let mut merger = ImageTrackMerger::new(camera()).expect("camera");
    merger
        .push(&digest(10), &graph(&[0, 1, 2], &[(u64::MAX, 100.0)]))
        .expect("first group");
    merger
        .push(&digest(11), &graph(&[1, 2, 3], &[(52, 100.5)]))
        .expect("overlap");
    let selected = select(&merger);
    assert_eq!(selected.graph.tracks.len(), 1);
    assert_eq!(selected.graph.tracks[0].feature_id, 0);
    assert_eq!(selected.graph.tracks[0].observations.len(), 4);
    assert_eq!(
        selected.graph.tracks[0].observations[1].pixel.x, 100.0,
        "shared evidence is not averaged"
    );
    assert_eq!(selected.graph.tracks[0].observations[3].pixel.x, 100.5);
    assert_eq!(
        selected.source_tracks[0],
        vec![
            TrackSource {
                group_sha256: digest(10),
                feature_id: u64::MAX
            },
            TrackSource {
                group_sha256: digest(11),
                feature_id: 52
            }
        ]
    );
    let repeated = select(&merger);
    assert_eq!(repeated.source_tracks, selected.source_tracks);
    assert_eq!(repeated.graph.tracks[0].observations.len(), 4);
}
#[test]
fn one_shared_image_cannot_join_features() {
    let mut merger = ImageTrackMerger::new(camera()).expect("camera");
    merger
        .push(&digest(10), &graph(&[0, 1], &[(0, 100.0)]))
        .expect("first");
    merger
        .push(&digest(11), &graph(&[1, 2], &[(0, 100.0)]))
        .expect("one shared image");
    let selected = select(&merger);
    assert_eq!(selected.graph.tracks.len(), 2);
    assert!(
        selected
            .source_tracks
            .iter()
            .all(|sources| sources.len() == 1)
    );
}
#[test]
fn disagreement_in_any_shared_image_prevents_joining() {
    let mut merger = ImageTrackMerger::new(camera()).expect("camera");
    merger
        .push(&digest(10), &graph(&[0, 1, 2], &[(0, 100.0)]))
        .expect("first");
    let mut next = graph(&[0, 1, 2, 3], &[(0, 100.0)]);
    next.tracks[0].observations[2].pixel.x = 103.0;
    merger.push(&digest(11), &next).expect("conflicting group");
    assert_eq!(merger.tracks.len(), 2);
    assert!(merger.tracks.iter().all(|t| t.sources.len() == 1));
}
#[test]
fn ambiguous_previous_features_stay_separate() {
    let mut merger = ImageTrackMerger::new(camera()).expect("camera");
    merger
        .push(&digest(10), &graph(&[0, 1], &[(0, 100.0), (1, 100.0)]))
        .expect("ambiguous first group");
    merger
        .push(&digest(11), &graph(&[0, 1, 2], &[(0, 100.0)]))
        .expect("second");
    assert_eq!(merger.tracks.len(), 3);
    assert!(merger.tracks.iter().all(|t| t.sources.len() == 1));
}
#[test]
fn one_previous_feature_cannot_receive_two_tracks_from_one_group() {
    let mut merger = ImageTrackMerger::new(camera()).expect("camera");
    merger
        .push(&digest(10), &graph(&[0, 1], &[(0, 100.0)]))
        .expect("first");
    merger
        .push(&digest(11), &graph(&[0, 1, 2], &[(0, 100.0), (1, 100.0)]))
        .expect("competing source links");
    assert_eq!(merger.tracks.len(), 2);
    assert_eq!(merger.tracks[0].sources.len(), 2);
    assert_eq!(merger.tracks[1].sources.len(), 1);
    assert_eq!(merger.tracks[0].pixels.len(), 3);
}
#[test]
fn invalid_or_repeated_groups_do_not_mutate_evidence() {
    let mut merger = ImageTrackMerger::new(camera()).expect("camera");
    let first = graph(&[0, 1], &[(0, 100.0)]);
    merger.push(&"ab".repeat(32), &first).expect("first");
    assert!(matches!(
        merger.push(&"AB".repeat(32), &first),
        Err(TrackMergeError::Group { .. })
    ));
    let mut invalid = graph(&[2, 3], &[(0, 100.0)]);
    invalid.tracks[0].observations[0].pixel.x = f64::NAN;
    let error = merger
        .push(&digest(11), &invalid)
        .expect_err("invalid source");
    assert!(std::error::Error::source(&error).is_some());
    assert_eq!(merger.observation_sha256(), first.observation_sha256);
    assert_eq!(merger.tracks.len(), 1);
    assert_eq!(merger.tracks[0].sources.len(), 1);
}
#[test]
fn selection_keeps_requested_order_and_reports_deferred_tracks() {
    let mut merger = ImageTrackMerger::new(camera()).expect("camera");
    merger
        .push(
            &digest(10),
            &graph(&[0, 1, 2], &[(0, 100.0), (1, 200.0), (2, 300.0)]),
        )
        .expect("first");
    let requested = vec![digest(2), digest(0)];
    let selected = merger
        .select(&requested, 2, 16)
        .expect("bounded work selection");
    assert_eq!(selected.graph.observation_sha256, requested);
    assert_eq!(selected.graph.tracks.len(), 2);
    assert_eq!(selected.deferred_tracks, 1);
    for track in &selected.graph.tracks {
        assert_eq!(
            track
                .observations
                .iter()
                .map(|o| o.camera_index)
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
    }
    assert!(merger.select(&[digest(0), digest(0)], 2, 16).is_err());
    assert!(merger.select(&[digest(0), digest(3)], 2, 16).is_err());
    assert_eq!(
        select(&merger).graph.tracks.len(),
        3,
        "selection does not discard source tracks"
    );
}
#[test]
fn exceeding_group_budget_leaves_all_original_sources_available() {
    let mut merger = ImageTrackMerger::new(camera()).expect("camera");
    let source = graph(&[0, 1], &[(0, 100.0)]);
    for i in 0..256 {
        merger.push(&digest(i), &source).expect("bounded group");
    }
    assert!(matches!(
        merger.push(&digest(256), &source),
        Err(TrackMergeError::Limits { groups: 257, .. })
    ));
    let selected = select(&merger);
    assert_eq!(selected.source_tracks[0].len(), 256);
    assert_eq!(
        selected.graph.tracks[0].observations.len(),
        2,
        "reused observations are not new evidence"
    );
}

#[test]
fn camera_links_omit_ambiguous_pixels_and_preserve_point_identity() {
    let mut merger = ImageTrackMerger::new(camera()).expect("camera");
    merger
        .push(
            &digest(10),
            &graph(&[0, 1], &[(0, 100.0), (1, 100.0), (2, 200.0)]),
        )
        .expect("source");
    let points: Vec<_> = (0..3)
        .map(|feature_id| crate::LocalScenePoint {
            feature_id,
            position: nalgebra::Vector3::new(0.0, 0.0, -4.0),
            observations: vec![],
        })
        .collect();
    let links = merger
        .scene_point_links(&digest(0), &points)
        .expect("source links");
    assert_eq!(links.ambiguous_feature_ids, vec![0, 1]);
    assert_eq!(links.matches.len(), 1);
    assert_eq!(links.matches[0].feature_id, 2);
    assert_eq!(links.matches[0].pixel.x, 200.0);
    assert!(merger.scene_point_links(&digest(8), &points).is_err());
    let mut invalid = points.clone();
    invalid[0].feature_id = 500;
    assert!(matches!(
        merger.scene_point_links(&digest(0), &invalid),
        Err(TrackMergeError::Feature {
            feature_id: 500,
            ..
        })
    ));
    invalid = points.clone();
    invalid[0].feature_id = 1;
    assert!(merger.scene_point_links(&digest(0), &invalid).is_err());
    invalid = points;
    invalid[0].position.x = f64::NAN;
    assert!(merger.scene_point_links(&digest(0), &invalid).is_err());
}
