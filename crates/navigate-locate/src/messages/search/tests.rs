use super::*;

#[test]
fn a_request_and_a_response_round_trip_as_json() {
    let request = SearchRequest {
        version: SEARCH_MESSAGES_VERSION,
        region: SearchRegion::Circle {
            latitude_rad: 0.714,
            longitude_rad: -1.311,
            radius_m: 10_000.0,
        },
        focal_px: 1202.0,
        max_width_px: 1600,
        priors: SearchPriors {
            height_m: Some(Prior {
                value: 800.0,
                sigma: 300.0,
            }),
            ..SearchPriors::default()
        },
        index_url: "indexes/delaware/".into(),
    };
    let text = serde_json::to_string(&request).unwrap();
    assert!(text.contains(r#""kind":"circle""#), "{text}");
    assert_eq!(
        serde_json::from_str::<SearchRequest>(&text).unwrap(),
        request
    );
    let response = SearchResponse::Uncovered {
        dense_search: true,
        reason: "no index".into(),
    };
    let text = serde_json::to_string(&response).unwrap();
    assert!(text.contains(r#""outcome":"uncovered""#), "{text}");
}
