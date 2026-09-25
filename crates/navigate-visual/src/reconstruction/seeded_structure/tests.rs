//! Seed estimates can initialize image links without establishing support.
#![allow(clippy::expect_used, clippy::panic)]
use super::*;
use crate::reconstruction::fixed_structure::tests::{fixture, graph};

#[test]
fn traced_estimates_initialize_inconsistent_links_without_changing_sources() {
    let (camera, cameras, points) = fixture();
    let mut graph = graph(&camera, &cameras, &points);
    for track in &mut graph.tracks {
        for (i, o) in track.observations.iter_mut().enumerate() {
            o.pixel.y += if i % 2 == 0 { 8.0 } else { -8.0 };
        }
    }
    let strict = fixed_structure::triangulate_scene_tracks(&camera, &graph, &cameras)
        .expect("valid inconsistent links");
    assert!(strict.scene.points.is_empty());
    let seeds: Vec<_> = graph
        .tracks
        .iter()
        .zip(&points)
        .map(|(t, &position)| ScenePointSeed {
            feature_id: t.feature_id,
            position,
        })
        .collect();
    let initialized =
        initialize_scene_tracks(&camera, &graph, &cameras, &seeds).expect("valid seeds");
    assert!(initialized.unresolved_feature_ids.is_empty());
    assert_eq!(initialized.scene.points.len(), points.len());
    for (p, seed) in initialized.scene.points.iter().zip(&seeds) {
        assert_eq!(p.feature_id, seed.feature_id);
        assert_eq!(p.position, seed.position);
        assert_eq!(p.observations.len(), cameras.len());
    }
    let repeated = initialize_scene_tracks(&camera, &graph, &cameras, &seeds).expect("repeat");
    for (a, b) in initialized.scene.points.iter().zip(repeated.scene.points) {
        assert_eq!(a.position, b.position);
        assert_eq!(a.observations.len(), b.observations.len());
    }
    let invalid_depth: Vec<_> = seeds
        .iter()
        .map(|s| ScenePointSeed {
            feature_id: s.feature_id,
            position: cameras[0].pose.position,
        })
        .collect();
    let unsupported =
        initialize_scene_tracks(&camera, &graph, &cameras, &invalid_depth).expect("weak estimates");
    assert!(unsupported.scene.points.is_empty());
    assert_eq!(unsupported.unresolved_feature_ids.len(), graph.tracks.len());
}

#[test]
fn strict_geometry_takes_priority_and_invalid_seed_identities_are_rejected() {
    let (camera, cameras, points) = fixture();
    let graph = graph(&camera, &cameras, &points);
    let seed = ScenePointSeed {
        feature_id: graph.tracks[0].feature_id,
        position: points[0] + Vector3::new(100.0, 0.0, 0.0),
    };
    let result = initialize_scene_tracks(&camera, &graph, &cameras, std::slice::from_ref(&seed))
        .expect("valid estimate");
    assert!((result.scene.points[0].position - points[0]).norm() < 1e-10);
    for seeds in [
        vec![seed.clone(), seed],
        vec![ScenePointSeed {
            feature_id: 7,
            position: points[0],
        }],
        vec![ScenePointSeed {
            feature_id: graph.tracks[0].feature_id,
            position: Vector3::new(f64::NAN, 0.0, 0.0),
        }],
    ] {
        assert!(matches!(
            initialize_scene_tracks(&camera, &graph, &cameras, &seeds),
            Err(ReconstructionError::Track { .. })
        ));
    }
}
