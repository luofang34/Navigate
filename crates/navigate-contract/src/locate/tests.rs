use super::*;

#[test]
fn a_found_response_has_a_stable_json_form() {
    let response = LocateResponse::Found {
        camera: LocatedCamera {
            latitude_rad: 0.9247,
            longitude_rad: 0.6294,
            height_m: 291.4,
            heading_rad: 4.807,
            pitch_rad: 0.686,
            roll_rad: -0.12,
            orientation_xyzw: [0.29, 0.30, 0.57, 0.71],
            width_px: 1024,
            height_px: 586,
            focal_px: 710.0,
            inliers: 148,
            reprojection_rms_px: 1.53,
            confirmed: true,
            map_sha256: "ab".repeat(32),
        },
    };
    let json = serde_json::to_value(&response).unwrap();
    assert_eq!(json["outcome"], "found");
    assert_eq!(json["camera"]["inliers"], 148);
    let back: LocateResponse = serde_json::from_value(json).unwrap();
    assert_eq!(back, response);
}

#[test]
fn stages_are_snake_case_names() {
    let progress = LocateProgress {
        stage: LocateStage::Rendering,
        done: 3,
        total: 27,
    };
    let json = serde_json::to_string(&progress).unwrap();
    assert_eq!(json, r#"{"stage":"rendering","done":3,"total":27}"#);
}

#[test]
fn a_request_round_trips() {
    let request = LocateRequest {
        version: LOCATE_MESSAGES_VERSION,
        target_latitude_rad: 0.9247,
        target_longitude_rad: 0.6293,
        heading_rad: 4.8,
        focal_px: 1420.0,
        max_width_px: 1024,
        area_radius_m: 800.0,
    };
    let json = serde_json::to_string(&request).unwrap();
    assert_eq!(
        serde_json::from_str::<LocateRequest>(&json).unwrap(),
        request
    );
}
