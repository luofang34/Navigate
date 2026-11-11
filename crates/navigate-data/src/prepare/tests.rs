use super::*;
use crate::resource::{GroundHeight, LayerLimits, PlannedCamera, ResourceRequest, plan_resources};
use crate::{MemoryStore, Package, SourceRef, TileRead};
use core::cell::Cell;
use pollster::block_on;

#[cfg(test)]
struct Flat;

impl GroundHeight for Flat {
    fn ground_m(&self, _: [f64; 2]) -> Option<f64> {
        Some(200.0)
    }
}

/// Synthetic tiles for every address, except imagery columns that are a
/// multiple of 5, which the source does not have.
#[cfg(test)]
struct Synthetic {
    revision: &'static str,
    calls: Cell<usize>,
}

impl FetchPort for Synthetic {
    fn source(&self, layer: LayerKind) -> Option<SourceRef> {
        Some(SourceRef {
            id: alloc::format!("synthetic-{layer:?}"),
            revision: self.revision.into(),
            attribution: "test".into(),
        })
    }

    async fn fetch(&self, key: &TileKey) -> Result<Fetched, String> {
        self.calls.set(self.calls.get() + 1);
        let a = key.address;
        if key.layer == LayerKind::Imagery && a.x.is_multiple_of(5) {
            return Ok(Fetched::Absent);
        }
        Ok(Fetched::Tile(alloc::format!("{key}").into_bytes()))
    }
}

#[cfg(test)]
fn legs() -> Vec<PlannedLeg> {
    alloc::vec![
        PlannedLeg {
            from: [41.70, -1.05, 1_200.0],
            to: [41.70, -1.00, 1_200.0]
        },
        PlannedLeg {
            from: [41.70, -1.00, 1_200.0],
            to: [41.73, -1.00, 1_600.0]
        },
    ]
}

#[cfg(test)]
fn planned() -> PlannedTiles {
    let request = ResourceRequest {
        legs: legs(),
        cameras: alloc::vec![PlannedCamera {
            focal_px: 400.0,
            width_px: 320,
            height_px: 240,
            depression_rad: core::f64::consts::FRAC_PI_2,
        }],
        search_radius_m: 50.0,
        deviation_m: 100.0,
        max_range_m: 6_000.0,
        imagery: LayerLimits::imagery(),
        elevation: LayerLimits::terrarium(),
        elevation_offset: 3,
    };
    let plan = plan_resources(&request, &Flat).unwrap();
    plan_tiles(&legs(), &plan).unwrap()
}

#[cfg(test)]
fn templates() -> Vec<LayerTemplate> {
    alloc::vec![
        LayerTemplate {
            id: "imagery".into(),
            kind: LayerKind::Imagery,
            encoding: Encoding::Jpeg,
            tile_pixels: 256,
            vertical_datum: None,
            accuracy: Accuracy::unknown(),
        },
        LayerTemplate {
            id: "elevation".into(),
            kind: LayerKind::Elevation,
            encoding: Encoding::TerrariumPng,
            tile_pixels: 256,
            vertical_datum: Some(VerticalDatum::Orthometric {
                geoid_model: "egm96".into()
            }),
            accuracy: Accuracy::unknown(),
        },
    ]
}

#[test]
fn a_prepared_package_holds_each_planned_tile_or_its_gap() {
    let tiles = planned();
    let source = Synthetic {
        revision: "r1",
        calls: Cell::new(0),
    };
    let mut store = MemoryStore::default();
    let summary = block_on(prepare(
        "trip",
        &tiles,
        &templates(),
        10_000,
        &source,
        &mut store,
    ))
    .unwrap();
    let total: usize = tiles.values().map(BTreeSet::len).sum();
    assert_eq!(summary.tiles + summary.gaps, total);
    assert_eq!(source.calls.get(), total);
    assert!(summary.gaps > 0 && summary.tiles > summary.gaps);
    let package = block_on(Package::open(&store, summary.identity)).unwrap();
    for (&kind, addresses) in &tiles {
        let id = if kind == LayerKind::Imagery {
            "imagery"
        } else {
            "elevation"
        };
        for &a in addresses {
            let read = block_on(package.tile(&store, id, a)).unwrap();
            if kind == LayerKind::Imagery && a.x.is_multiple_of(5) {
                assert_eq!(read, TileRead::Gap);
            } else {
                assert!(matches!(read, TileRead::Present(_)), "{a:?}");
            }
        }
    }
}

#[test]
fn the_identity_follows_the_content_and_the_source_revision() {
    let tiles = planned();
    let run = |revision: &'static str| {
        let source = Synthetic {
            revision,
            calls: Cell::new(0),
        };
        let mut store = MemoryStore::default();
        block_on(prepare(
            "trip",
            &tiles,
            &templates(),
            10_000,
            &source,
            &mut store,
        ))
        .unwrap()
        .identity
    };
    assert_eq!(run("r1"), run("r1"));
    assert_ne!(run("r1"), run("r2"));
}

#[test]
fn an_oversized_plan_is_refused_before_a_fetch() {
    let source = Synthetic {
        revision: "r1",
        calls: Cell::new(0),
    };
    let mut store = MemoryStore::default();
    let refused = block_on(prepare(
        "trip",
        &planned(),
        &templates(),
        3,
        &source,
        &mut store,
    ));
    assert!(matches!(
        refused,
        Err(DataError::Invalid {
            field: "max_tiles",
            ..
        })
    ));
    assert_eq!(source.calls.get(), 0);
    assert!(store.files.is_empty());
    let mut only_imagery = templates();
    only_imagery.truncate(1);
    assert!(
        block_on(prepare(
            "trip",
            &planned(),
            &only_imagery,
            10_000,
            &source,
            &mut store
        ))
        .is_err()
    );
    assert!(store.files.keys().all(|p| p.starts_with("chunks/")));
}

#[test]
fn the_plan_covers_the_strip_and_the_view_past_the_last_waypoint() {
    let tiles = planned();
    let imagery = &tiles[&LayerKind::Imagery];
    let zoom = imagery.iter().map(|a| a.zoom).max().unwrap();
    for point in [
        [41.70, -1.05],
        [41.70, -1.025],
        [41.715, -1.00],
        [41.73, -1.00],
    ] {
        let a = TileAddress::containing(point, zoom).unwrap();
        assert!(imagery.contains(&a), "{point:?}");
    }
    assert!(plan_tiles(&legs()[..1], &plan_resources_for_two()).is_err());
}

#[cfg(test)]
fn plan_resources_for_two() -> ResourcePlan {
    ResourcePlan {
        legs: alloc::vec![],
        bytes: 0.0,
    }
}
