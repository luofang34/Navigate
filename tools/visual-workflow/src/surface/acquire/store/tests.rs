use super::*;
#[test]
fn declared_coverage_includes_clamped_edges_and_all_fine_children() {
    let size = (6275, 6384);
    let parents = windows(size, false).unwrap();
    assert_eq!(parents.len(), 50);
    assert!(parents.contains(&Some([3203, 3312, 2048, 2048])));
    let fine = windows(size, true).unwrap();
    assert_eq!(fine.len(), 306);
    for p in parents.into_iter().flatten() {
        assert!(fine.contains(&Some(p)));
        for child in super::super::selection::children(p) {
            assert!(fine.contains(&Some(child)));
        }
    }
    assert_eq!(fine.len() * 4, 1224);
}
#[test]
fn every_reachable_seed_has_its_neighbors_in_the_catalog() {
    for size in [(2048, 2048), (2080, 3000), (6275, 6384), (8192, 8192)] {
        let coverage = windows(size, false).unwrap();
        for x in (0..size.0).step_by(29) {
            for y in (0..size.1).step_by(31) {
                let pairs = [navigate_visual::PixelMatch {
                    reference: nalgebra::Vector2::new(f64::from(x), f64::from(y)),
                    query: nalgebra::Vector2::zeros(),
                }];
                let seed = super::super::selection::seed(&pairs, size)
                    .unwrap()
                    .unwrap();
                for area in super::super::selection::neighbors(seed, size) {
                    assert!(coverage.contains(&Some(area)));
                }
            }
        }
    }
    assert!(windows((2047, 4096), false).is_err());
}
#[test]
fn source_identity_binds_geometry_and_rejects_query_fields() {
    let value = json!({"image":{"path":"map.png","sha256":"a".repeat(64)},"heights":{"path":"height.bin","sha256":"b".repeat(64)},"origin":[0.,0.],"column":[0.2,0.],"row":[0.,-0.2],"map_release":"map","map_sha256":"c".repeat(64)});
    let source: Surface = serde_json::from_value(value.clone()).unwrap();
    let mut changed = value.clone();
    changed["origin"] = json!([1., 0.]);
    assert_ne!(
        source_namespace(&source),
        source_namespace(&serde_json::from_value(changed).unwrap())
    );
    let mut changed = value;
    changed["query"] = json!({"path":"query.png"});
    assert!(serde_json::from_value::<Surface>(changed).is_err());
}
