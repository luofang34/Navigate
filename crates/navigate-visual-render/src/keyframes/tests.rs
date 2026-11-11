use super::*;
use crate::test_package::{CENTRE, ELEVATION, IMAGERY, synthetic_package};
use navigate_data::product::{KindName, open_chunk, read_recipe};

#[cfg(test)]
fn spec() -> KeyframeSpec {
    KeyframeSpec {
        imagery_layer: IMAGERY,
        imagery_zoom: 15,
        elevation_layer: ELEVATION,
        elevation_zoom: 12,
        camera: CameraModel {
            width: 160,
            height: 120,
            fx: 150.0,
            fy: 150.0,
            cx: 79.5,
            cy: 59.5,
        },
        depression_rad: std::f64::consts::FRAC_PI_2,
        spacing_m: 400.0,
        envelope: ValidityEnvelope {
            lateral_mm: 120_000,
            vertical_mm: 80_000,
            heading_urad: 200_000,
        },
        far_m: 1_500.0,
        steps: 32,
        refinements: 8,
        stride: 8,
        luma_edge: 8,
        depth_edge: 0.02,
        altitude_band: 3,
    }
}

#[cfg(test)]
fn leg() -> PlannedLeg {
    let [lat, lon] = CENTRE;
    PlannedLeg {
        from: [lat, lon - 0.006, 1_050.0],
        to: [lat, lon + 0.006, 1_050.0],
    }
}

#[test]
fn keyframes_are_a_product_of_the_core_with_their_recipe_and_envelope() {
    let (mut store, package) = synthetic_package(&[]);
    let (identity, count) = pollster::block_on(build_keyframes(
        &package,
        &mut store,
        &[leg()],
        &spec(),
        None,
    ))
    .unwrap();
    assert_eq!(count, 3, "1 km leg at 400 m spacing");
    let chunk = pollster::block_on(open_chunk(&store, identity)).unwrap();
    assert_eq!(chunk.kind, KindName::Known(ProductKind::Keyframes));
    assert_eq!(chunk.binding.envelope, Some(spec().envelope));
    assert_eq!(chunk.binding.calibration.map(|c| c.width), Some(160));
    assert!(chunk.binding.route.is_some());
    assert!(chunk.applies_to(&pollster::block_on(package.blobs()).unwrap()));
    assert!(!chunk.binding.inputs.is_empty());
    let recipe = pollster::block_on(read_recipe(&store, chunk.recipe)).unwrap();
    assert_eq!(recipe.parameters.get("spacing_mm"), Some(&400_000));
    let first = chunk.entries[0];
    let pose = first.pose.unwrap();
    let (_, _, heading) = pose.to_si();
    assert!(
        (heading - std::f64::consts::FRAC_PI_2).abs() < 1e-3,
        "{heading}"
    );
    let (w, h, luma, depth) =
        decode_view(&pollster::block_on(chunk.entry_bytes(&store, &first)).unwrap()).unwrap();
    assert_eq!((w, h), (160, 120));
    let fresh = pollster::block_on(render_at(&package, &store, &spec(), &pose, None)).unwrap();
    assert_eq!(luma, fresh.image.as_raw().clone());
    assert_eq!(depth, fresh.depth_m);
    let hits = depth.iter().filter(|d| **d > 0.0).count();
    assert!(hits > depth.len() * 9 / 10, "{hits}");
    let centre_depth = depth[60 * 160 + 80];
    assert!(
        (f64::from(centre_depth) - 800.0).abs() < 5.0,
        "{centre_depth}"
    );
}

#[test]
fn the_product_identity_follows_the_recipe() {
    let (mut store, package) = synthetic_package(&[]);
    let a = pollster::block_on(build_keyframes(
        &package,
        &mut store,
        &[leg()],
        &spec(),
        None,
    ))
    .unwrap()
    .0;
    let again = pollster::block_on(build_keyframes(
        &package,
        &mut store,
        &[leg()],
        &spec(),
        None,
    ))
    .unwrap()
    .0;
    assert_eq!(a, again);
    let wider = KeyframeSpec {
        spacing_m: 500.0,
        ..spec()
    };
    assert_ne!(
        a,
        pollster::block_on(build_keyframes(
            &package,
            &mut store,
            &[leg()],
            &wider,
            None
        ))
        .unwrap()
        .0
    );
}

#[test]
fn a_forward_camera_looks_along_the_heading() {
    let q = camera_orientation(0.0, 0.0);
    let back = q * Vector3::z();
    assert!(
        (back - Vector3::new(0.0, -1.0, 0.0)).norm() < 1e-12,
        "{back}"
    );
    let q = camera_orientation(std::f64::consts::FRAC_PI_2, std::f64::consts::FRAC_PI_2);
    assert!((q * Vector3::z() - Vector3::z()).norm() < 1e-12);
    assert!((q * Vector3::y() - Vector3::new(1.0, 0.0, 0.0)).norm() < 1e-12);
    assert!(decode_view(b"NVKEYV01").is_err());
}
