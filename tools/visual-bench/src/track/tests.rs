#![allow(clippy::expect_used)]
use super::*;

#[test]
fn rejections_split_tracks_without_inventing_positions() {
    let point = |lon| json!({"accepted":true,"longitude_deg":lon,"latitude_deg":47.0,"map_manifest_sha256":"a"});
    let report = collection(&[
        point(11.0),
        point(11.1),
        json!({"accepted":false}),
        point(11.2),
        point(11.3),
    ])
    .expect("valid records");
    let features = report["features"].as_array().expect("features");
    let lines: Vec<_> = features
        .iter()
        .filter(|f| f["geometry"]["type"] == "LineString")
        .collect();
    assert_eq!(lines.len(), 2);
    assert_eq!(
        lines[0]["geometry"]["coordinates"],
        json!([[11.0, 47.0], [11.1, 47.0]])
    );
    assert_eq!(features.len(), 6);
}
