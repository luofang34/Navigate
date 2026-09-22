#![allow(clippy::expect_used)]
use super::*;

#[test]
fn area_preserves_selection_and_covers_corners() {
    let request: CoverageRequest =
        serde_json::from_str(r#"{"bounds":[-74.480,40.530,-74.425,40.565]}"#).expect("request");
    let p = plan(request).expect("plan");
    assert_eq!(p.imagery_tiles.len(), 110);
    assert_eq!(p.requested.bounds, Some([-74.480, 40.530, -74.425, 40.565]));
    assert!(
        p.bounds[0] <= -74.480
            && p.bounds[1] <= 40.530
            && p.bounds[2] >= -74.425
            && p.bounds[3] >= 40.565
    );
}

#[test]
fn route_excludes_distant_bounding_box_tiles() {
    let r: CoverageRequest =
        serde_json::from_str(r#"{"route":[[-74.47,40.54],[-74.44,40.57]],"buffer_m":100}"#)
            .expect("route");
    let p = plan(r).expect("plan");
    assert!(p.imagery_tiles.len() < 30);
    for point in [[-74.47, 40.54], [-74.44, 40.57]] {
        let [x, y] = tile_position(point[0], point[1], 16);
        assert!(
            p.imagery_tiles
                .contains(&Tile(16, x.floor() as u32, y.floor() as u32))
        );
    }
}

#[test]
fn malformed_and_oversized_selections_fail_before_provider_access() {
    for json in [
        r#"{"bounds":[-180,-80,180,80]}"#,
        r#"{"bounds":[10,20,0,30]}"#,
        r#"{"route":[[179,10],[-179,10]]}"#,
        r#"{"bounds":[0,0,1,1],"route":[[0,0],[1,1]]}"#,
    ] {
        let request = serde_json::from_str(json).expect("valid JSON");
        assert!(plan(request).is_err());
    }
}
