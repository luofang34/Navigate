//! What the navigator loads and what it refuses, at load and on a frame.

use std::sync::Arc;

use navigate_camera::{CameraMount, SpectralBand};
use navigate_contract::Ellipsoid;
use navigate_data::index::IndexEntry;
use navigate_data::product::{KindName, ProductKind, ProductUse, write_list};
use navigate_data::{
    DataError, Encoding, GeoidModel, LayerKind, Sha256Digest, VerticalDatum, blob_path,
    manifest_path,
};
use navigate_geodesy::GeoidUndulation;

use super::fixes::{HEIGHT_M, belly, terrain};
use super::flight::{CAMERA, CLOCK, Flight, RATE, fly, imager, navigator, request, suite};
use super::world::{CENTRE, IMAGERY, World, WorldSpec, camera_model};
use crate::{
    AccuracySource, AssumedAccuracy, DataRequirement, HostGeoid, MapRefusal, Navigator, Need,
    OnboardError, Outcome, ReferenceNavigator, SensorData, SensorInput, TerrainMapRequest,
};

#[cfg(test)]
fn hover() -> Flight {
    Flight {
        start: [CENTRE[0], CENTRE[1], HEIGHT_M],
        speed_m_s: 0.0,
    }
}

/// The refusal of a load of `request` on the store of `world`.
#[cfg(test)]
fn refusal(world: &World, request: &TerrainMapRequest) -> MapRefusal {
    let mut navigator = navigator(&hover(), belly(), [0.0; 3], 20.0);
    match navigator.load_terrain_map(Arc::clone(&world.store), request, None) {
        Err(OnboardError::MapRefused { package, refusal }) => {
            assert_eq!(package, request.package);
            assert!(
                navigator.terrain_map().is_none(),
                "a refused load loads nothing"
            );
            refusal
        }
        other => panic!("the load is refused: {other:?}"),
    }
}

/// A geoid model that is 48 m above the ellipsoid everywhere.
#[cfg(test)]
struct FlatGeoid;

impl GeoidUndulation for FlatGeoid {
    fn ellipsoid(&self) -> Ellipsoid {
        Ellipsoid::WGS84
    }

    fn undulation_m(&self, _latitude_rad: f64, _longitude_rad: f64) -> Option<f64> {
        Some(48.0)
    }
}

#[test]
fn the_report_states_what_was_loaded() {
    let world = World::build(&WorldSpec::default());
    let keyframes = world.keyframes(camera_model(), HEIGHT_M);
    let mut navigator = navigator(&hover(), belly(), [0.0; 3], 20.0);
    assert_eq!(
        navigator
            .requirements()
            .iter()
            .map(|r| (r.layer, r.product, r.need))
            .collect::<Vec<_>>(),
        [
            (Some(LayerKind::Imagery), None, Need::Required),
            (Some(LayerKind::Elevation), None, Need::Required),
            (None, Some(ProductKind::Keyframes), Need::Optional),
        ]
    );
    assert!(navigator.load_report().is_empty() && navigator.terrain_map().is_none());
    let with_products = TerrainMapRequest {
        products: Some(keyframes.list),
        ..request(&world)
    };
    let report = navigator
        .load_terrain_map(Arc::clone(&world.store), &with_products, None)
        .unwrap()
        .clone();
    assert_eq!(report.package, world.package);
    assert_eq!(report.imagery.key, IMAGERY);
    assert_eq!(report.imagery.zooms, [15]);
    // 156 543 m · cos 41.7° / 2^15.
    assert!(
        (report.imagery.ground_sample_m - 3.567).abs() < 0.01,
        "{report:?}"
    );
    assert!(report.imagery.tiles > 30 && report.imagery.gaps == 0);
    assert_eq!(
        (report.imagery.sigma_m, report.imagery.sigma_from),
        (1.5, AccuracySource::Package)
    );
    assert_eq!(report.elevation.zooms, [12]);
    assert_eq!(report.elevation.sigma_m, 2.0);
    assert_eq!(report.undulation_bound_m, 0.0);
    assert_eq!(report.keyframes, 4);
    assert_eq!(report.products.len(), 1);
    assert_eq!(
        report.products[0].kind,
        KindName::Known(ProductKind::Keyframes)
    );
    assert_eq!(report.products[0].decision, ProductUse::Use);
    assert_eq!(navigator.load_report(), report.products.as_slice());
    assert_eq!(navigator.terrain_map(), Some(&report));

    // Keyframes for a camera of another calibration are not used: the
    // frames of this camera take the rendered reference.
    let mut other_camera = imager();
    other_camera.intrinsics.fx = 151.0;
    let mut other = ReferenceNavigator::new(suite(other_camera, belly()), CLOCK, None).unwrap();
    let report = other
        .load_terrain_map(Arc::clone(&world.store), &with_products, None)
        .unwrap();
    assert_eq!(report.keyframes, 0);
    assert!(
        matches!(&report.products[0].decision, ProductUse::Fallback(reason)
        if reason.contains("calibration"))
    );

    // A navigator without a terrain camera asks for no data.
    let bare = ReferenceNavigator::new(crate::reference::tests::suite(&[]), CLOCK, None).unwrap();
    assert_eq!(bare.requirements(), Vec::<DataRequirement>::new());
}

#[test]
fn a_package_that_is_not_the_expected_one_is_refused() {
    let world = World::build(&WorldSpec::default());
    let absent = TerrainMapRequest {
        package: Sha256Digest::of(b"a package that the host did not install"),
        ..request(&world)
    };
    assert!(matches!(
        refusal(&world, &absent),
        MapRefusal::Package(DataError::Missing { .. })
    ));
    // The manifest on the store is not the manifest of the identity.
    let manifest = world.dir.path().join(manifest_path(world.package));
    let mut bytes = std::fs::read(&manifest).unwrap();
    bytes.push(b' ');
    std::fs::write(&manifest, bytes).unwrap();
    assert!(matches!(
        refusal(&world, &request(&world)),
        MapRefusal::Package(DataError::Integrity { .. })
    ));
    // A product list of another package.
    let world = World::build(&WorldSpec::default());
    let mut store = (*world.store).clone();
    let other = Sha256Digest::of(b"another package");
    let list = pollster::block_on(write_list(&mut store, other, &[])).unwrap();
    let foreign = TerrainMapRequest {
        products: Some(list),
        ..request(&world)
    };
    assert!(matches!(
        refusal(&world, &foreign),
        MapRefusal::ProductList(DataError::Invalid {
            field: "product list",
            ..
        })
    ));
    // A suite without a terrain camera has no means that reads a map.
    let mut bare =
        ReferenceNavigator::new(crate::reference::tests::suite(&[]), CLOCK, None).unwrap();
    assert!(matches!(
        bare.load_terrain_map(Arc::clone(&world.store), &request(&world), None),
        Err(OnboardError::MapRefused {
            refusal: MapRefusal::NoTerrainCamera,
            ..
        })
    ));
}

#[test]
fn a_package_that_the_means_cannot_use_is_refused_with_the_reason() {
    let small = WorldSpec {
        radius_m: 400.0,
        ..WorldSpec::default()
    };
    let no_elevation = World::build(&WorldSpec {
        elevation: false,
        ..small
    });
    assert_eq!(
        refusal(&no_elevation, &request(&no_elevation)),
        MapRefusal::MissingLayer(LayerKind::Elevation)
    );
    let webp = World::build(&WorldSpec {
        imagery_encoding: Encoding::Webp,
        ..small
    });
    let large = World::build(&WorldSpec {
        imagery_tile_pixels: 512,
        ..small
    });
    for world in [&webp, &large] {
        assert!(matches!(
            refusal(world, &request(world)),
            MapRefusal::UnsupportedLayer {
                layer: LayerKind::Imagery,
                ..
            }
        ));
    }
}

#[test]
fn unknown_accuracy_and_an_unknown_geoid_are_stated_and_never_taken_as_zero() {
    let orthometric = VerticalDatum::Orthometric {
        geoid: GeoidModel::Egm96,
    };
    let world = World::build(&WorldSpec {
        accuracy_mm: None,
        vertical_datum: Some(orthometric),
        radius_m: 400.0,
        ..WorldSpec::default()
    });
    assert_eq!(
        refusal(&world, &request(&world)),
        MapRefusal::UnknownAccuracy(LayerKind::Imagery)
    );
    let assumed = TerrainMapRequest {
        assumed_accuracy: Some(AssumedAccuracy {
            horizontal_sigma_m: 4.0,
            vertical_sigma_m: 9.0,
        }),
        ..request(&world)
    };
    let mut navigator = navigator(&hover(), belly(), [0.0; 3], 20.0);
    let report = navigator
        .load_terrain_map(Arc::clone(&world.store), &assumed, None)
        .unwrap();
    assert_eq!(
        (report.imagery.sigma_m, report.imagery.sigma_from),
        (4.0, AccuracySource::Assumed)
    );
    assert_eq!(
        (report.elevation.sigma_m, report.elevation.sigma_from),
        (9.0, AccuracySource::Assumed)
    );
    // Without the geoid model of the datum, the height error carries the
    // bound of the undulation.
    assert_eq!(report.undulation_bound_m, 107.0);
    let geoid: HostGeoid = (GeoidModel::Egm96, Arc::new(FlatGeoid));
    let report = navigator
        .load_terrain_map(Arc::clone(&world.store), &assumed, Some(geoid))
        .unwrap();
    assert_eq!(report.undulation_bound_m, 0.0);
    let another: HostGeoid = (GeoidModel::Egm2008, Arc::new(FlatGeoid));
    let report = navigator
        .load_terrain_map(Arc::clone(&world.store), &assumed, Some(another))
        .unwrap();
    assert_eq!(report.undulation_bound_m, 107.0);
}

#[test]
fn keyframes_with_other_terrain_heights_than_the_navigator_are_not_used() {
    let orthometric = VerticalDatum::Orthometric {
        geoid: GeoidModel::Egm96,
    };
    let world = World::build(&WorldSpec {
        vertical_datum: Some(orthometric),
        radius_m: 1_200.0,
        ..WorldSpec::default()
    });
    // The builder of the keyframes had no geoid model.
    let keyframes = world.keyframes(camera_model(), HEIGHT_M);
    let with_products = TerrainMapRequest {
        products: Some(keyframes.list),
        ..request(&world)
    };
    let mut navigator = navigator(&hover(), belly(), [0.0; 3], 20.0);
    let mut load = |geoid: Option<HostGeoid>| {
        let report = navigator
            .load_terrain_map(Arc::clone(&world.store), &with_products, geoid)
            .unwrap();
        (
            report.keyframes,
            report.undulation_bound_m,
            report.products[0].decision.clone(),
        )
    };
    // No model on each side: the same heights, and the bound is in the
    // covariance of each fix.
    assert_eq!(load(None), (4, 107.0, ProductUse::Use));
    // A model of another geoid converts nothing, as the builder did.
    let other: HostGeoid = (GeoidModel::Egm2008, Arc::new(FlatGeoid));
    assert_eq!(load(Some(other)), (4, 107.0, ProductUse::Use));
    // With the model of the datum, the navigator has heights 48 m above
    // those of the keyframes and states no bound: it renders its references.
    let model: HostGeoid = (GeoidModel::Egm96, Arc::new(FlatGeoid));
    let (count, bound_m, decision) = load(Some(model));
    assert_eq!((count, bound_m), (0, 0.0));
    assert!(matches!(decision, ProductUse::Fallback(reason) if reason.contains("geoid")));
}

#[test]
fn a_camera_and_a_frame_that_the_means_cannot_take_are_refused() {
    let lwir = imager().with_band(SpectralBand::LongWaveInfrared);
    assert_eq!(
        ReferenceNavigator::new(suite(lwir, CameraMount::nadir()), CLOCK, None).err(),
        Some(OnboardError::UnsupportedBand {
            sensor: CAMERA,
            band: SpectralBand::LongWaveInfrared
        })
    );
    let world = World::build(&WorldSpec::default());
    let flight = hover();
    let mut navigator = navigator(&flight, belly(), [0.0; 3], 20.0);
    let mut sink = Vec::new();
    let frame = flight.frame(&world, &belly(), 1, 0);
    navigator.ingest(&flight.increment(1), &mut sink).unwrap();
    assert_eq!(
        navigator.ingest(&frame, &mut sink),
        Err(OnboardError::MapNotLoaded { sensor: CAMERA })
    );
    navigator
        .load_terrain_map(Arc::clone(&world.store), &request(&world), None)
        .unwrap();
    let half = SensorInput {
        data: SensorData::TerrainFrame {
            width: 80,
            height: 120,
            luma: vec![0; 80 * 120],
        },
        ..frame.clone()
    };
    assert_eq!(
        navigator.ingest(&half, &mut sink),
        Err(OnboardError::FrameSize {
            sensor: CAMERA,
            declared: (160, 120),
            frame: (80, 120),
            bytes: 9_600
        })
    );
    assert!(sink.is_empty(), "a refused input gives no record");
}

#[test]
fn a_frame_outside_the_package_is_not_matched_and_a_changed_tile_is_an_error() {
    let world = World::build(&WorldSpec::default());
    // 17 km east of the package.
    let outside = Flight {
        start: [CENTRE[0], CENTRE[1] + 0.2, HEIGHT_M],
        speed_m_s: 0.0,
    };
    let frame = hover().frame(&world, &belly(), RATE, 0);
    let mut navigator_outside = navigator(&outside, belly(), [0.0; 3], 20.0);
    navigator_outside
        .load_terrain_map(Arc::clone(&world.store), &request(&world), None)
        .unwrap();
    let mut sink = Vec::new();
    fly(
        &mut navigator_outside,
        &outside,
        &world,
        &belly(),
        1..=RATE,
        |_| false,
        &mut sink,
    )
    .unwrap();
    navigator_outside.ingest(&frame, &mut sink).unwrap();
    assert_eq!(
        terrain(&sink),
        [(Outcome::Unused("outside_coverage"), None)]
    );

    // Each imagery tile on the store changes after the load.
    let mut navigator_inside = navigator(&hover(), belly(), [0.0; 3], 20.0);
    navigator_inside
        .load_terrain_map(Arc::clone(&world.store), &request(&world), None)
        .unwrap();
    for (_, entry) in pollster::block_on(world.open().layer_entries(IMAGERY)).unwrap() {
        if let IndexEntry::Tile { blob, .. } = entry {
            std::fs::write(world.dir.path().join(blob_path(blob)), b"changed").unwrap();
        }
    }
    let mut sink = Vec::new();
    fly(
        &mut navigator_inside,
        &hover(),
        &world,
        &belly(),
        1..=RATE,
        |_| false,
        &mut sink,
    )
    .unwrap();
    assert!(matches!(
        navigator_inside.ingest(&frame, &mut sink),
        Err(OnboardError::MapRead { sensor: CAMERA, detail }) if detail.contains("integrity")
    ));
    assert!(sink.is_empty());
}
