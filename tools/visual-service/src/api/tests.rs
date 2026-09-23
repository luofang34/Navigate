#![allow(clippy::expect_used)]
use super::*;
use axum::{body::Body, http::Request};
use http_body_util::BodyExt;
use tower::ServiceExt;

async fn request(
    app: Router,
    method: &str,
    path: &str,
    body: &str,
    origin: Option<&str>,
) -> (StatusCode, Value) {
    let mut builder = Request::builder()
        .method(method)
        .uri(path)
        .header("content-type", "application/json");
    if let Some(origin) = origin {
        builder = builder.header("origin", origin);
    }
    let response = app
        .oneshot(builder.body(Body::from(body.to_owned())).expect("request"))
        .await
        .expect("response");
    let status = response.status();
    let bytes = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    (
        status,
        serde_json::from_slice(&bytes).unwrap_or(Value::Null),
    )
}

#[tokio::test]
async fn browser_service_plans_without_inference_or_upload_routes() {
    let temp = tempfile::tempdir().expect("directory");
    let config = Config {
        state: temp.path().into(),
        webapp: temp.path().into(),
        origin: "http://127.0.0.1:8080".into(),
        catalog: None,
        bind: "127.0.0.1:0".parse().expect("address"),
    };
    config.validate().expect("config");
    let (jobs, handle) = crate::jobs::start(config.state.clone(), Default::default());
    let app = router(config, jobs.clone());
    let (status, health) = request(app.clone(), "GET", "/api/health", "", None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(health["package_runtime"], "rust-gdal");
    assert_eq!(health["native_inference"], false);
    let (status, plan) = request(
        app.clone(),
        "POST",
        "/api/coverage-plan",
        r#"{"bounds":[-74.480,40.530,-74.425,40.565]}"#,
        None,
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(plan["imagery_tiles"].as_array().expect("tiles").len(), 110);
    for path in ["/api/uploads", "/api/localize"] {
        assert_eq!(
            request(app.clone(), "POST", path, "{}", None).await.0,
            StatusCode::NOT_FOUND
        );
    }
    assert_eq!(
        request(
            app.clone(),
            "POST",
            "/api/coverage-download",
            "{}",
            Some("https://foreign.test")
        )
        .await
        .0,
        StatusCode::FORBIDDEN
    );
    assert_eq!(
        request(
            app.clone(),
            "POST",
            "/api/coverage-plan",
            r#"{"bounds":[-180,-80,180,80]}"#,
            None
        )
        .await
        .0,
        StatusCode::BAD_REQUEST
    );
    assert!(jobs.snapshot.read().await.jobs.is_empty());
    jobs.stop().await;
    handle.await.expect("shutdown");
}

#[tokio::test]
async fn static_assets_support_range_reads_and_cannot_escape_roots() {
    let temp = tempfile::tempdir().expect("directory");
    let state = temp.path().join("data");
    std::fs::create_dir_all(state.join("chunks")).expect("directory");
    std::fs::write(state.join("chunks/test.bin"), b"0123456789").expect("chunk");
    let config = Config {
        state: state.clone(),
        webapp: temp.path().join("web"),
        origin: "http://127.0.0.1:8080".into(),
        catalog: None,
        bind: "127.0.0.1:0".parse().expect("address"),
    };
    let (jobs, handle) = crate::jobs::start(state, Default::default());
    let app = router(config, jobs.clone());
    let response = app
        .clone()
        .oneshot(
            Request::builder()
                .uri("/chunks/test.bin")
                .header("range", "bytes=2-5")
                .body(Body::empty())
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status(), StatusCode::PARTIAL_CONTENT);
    assert_eq!(
        response
            .into_body()
            .collect()
            .await
            .expect("body")
            .to_bytes(),
        "2345"
    );
    assert_eq!(
        request(app, "GET", "/chunks/%2e%2e/%2e%2e/secret", "", None)
            .await
            .0,
        StatusCode::NOT_FOUND
    );
    jobs.stop().await;
    handle.await.expect("shutdown");
}
