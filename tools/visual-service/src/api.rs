use crate::{config::Config, jobs::Jobs};
use axum::{
    Json, Router,
    extract::{DefaultBodyLimit, Path, State},
    http::{HeaderMap, HeaderValue, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use navigate_imagery::{CoverageRequest, plan};
use serde_json::{Value, json};
use tower_http::services::ServeDir;

#[derive(Clone)]
struct App {
    config: Config,
    jobs: Jobs,
}
type ApiResult = Result<Json<Value>, (StatusCode, Json<Value>)>;
fn failure(code: StatusCode, message: impl ToString) -> (StatusCode, Json<Value>) {
    (code, Json(json!({"error":message.to_string()})))
}

pub(crate) fn router(config: Config, jobs: Jobs) -> Router {
    let state = App {
        config: config.clone(),
        jobs,
    };
    Router::new()
        .route("/api/health",get(||async {Json(json!({"status":"ready","processing":"browser-worker","native_inference":false,"package_runtime":"rust-gdal"}))}))
        .route("/api/catalog",get(catalog))
        .route("/api/offline-plan",post(offline))
        .route("/api/coverage-plan",post(coverage))
        .route("/api/coverage-download",post(download))
        .route("/api/downloads/{id}",get(download_status))
        .route("/api/{*path}",get(||async {StatusCode::NOT_FOUND}).post(||async {StatusCode::NOT_FOUND}))
        .nest_service("/chunks",ServeDir::new(config.state.join("chunks")))
        .nest_service("/packs",ServeDir::new(config.state.join("packs")))
        .nest_service("/catalog",ServeDir::new(config.state.join("catalog")))
        .fallback_service(ServeDir::new(config.webapp.clone()).append_index_html_on_directories(true))
        .layer(DefaultBodyLimit::max(65_536))
        .layer(middleware::from_fn_with_state(state.clone(),headers))
        .with_state(state)
}

async fn headers(
    State(state): State<App>,
    request: axum::extract::Request,
    next: Next,
) -> Response {
    if request.method() == axum::http::Method::POST
        && !allowed(request.headers(), &state.config.origin)
    {
        return failure(StatusCode::FORBIDDEN, "Origin rejected").into_response();
    }
    let mut response = next.run(request).await;
    response.headers_mut().insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
    response
}
fn allowed(headers: &HeaderMap, origin: &str) -> bool {
    headers
        .get("sec-fetch-site")
        .is_none_or(|v| v != "cross-site")
        && headers.get(header::ORIGIN).is_none_or(|v| v == origin)
}

async fn catalog(State(state): State<App>) -> Json<Value> {
    let values: Vec<_> = state
        .jobs
        .snapshot
        .read()
        .await
        .regions
        .values()
        .map(|r| {
            let mut value = json!(r);
            if let Some(object) = value.as_object_mut() {
                object.remove("manifest");
            }
            value
        })
        .collect();
    Json(json!(values))
}
async fn offline(State(state): State<App>, Json(value): Json<Value>) -> ApiResult {
    let id = value["region_id"]
        .as_str()
        .ok_or_else(|| failure(StatusCode::BAD_REQUEST, "region_id is required"))?;
    let snapshot = state.jobs.snapshot.read().await;
    let region = snapshot
        .regions
        .get(id)
        .ok_or_else(|| failure(StatusCode::NOT_FOUND, "region is unavailable"))?;
    Ok(Json(json!(region.manifest)))
}
async fn coverage(Json(request): Json<CoverageRequest>) -> ApiResult {
    plan(request)
        .map(|p| Json(json!(p)))
        .map_err(|e| failure(StatusCode::BAD_REQUEST, e))
}
async fn download(
    State(state): State<App>,
    Json(request): Json<CoverageRequest>,
) -> Result<(StatusCode, Json<Value>), (StatusCode, Json<Value>)> {
    let plan = plan(request).map_err(|e| failure(StatusCode::BAD_REQUEST, e))?;
    let job = state
        .jobs
        .submit(plan)
        .await
        .map_err(|e| failure(StatusCode::TOO_MANY_REQUESTS, e))?;
    Ok((StatusCode::ACCEPTED, Json(job)))
}
async fn download_status(State(state): State<App>, Path(id): Path<String>) -> ApiResult {
    state
        .jobs
        .snapshot
        .read()
        .await
        .jobs
        .get(&id)
        .cloned()
        .map(Json)
        .ok_or_else(|| failure(StatusCode::NOT_FOUND, "download is unavailable"))
}

#[cfg(test)]
mod tests;
