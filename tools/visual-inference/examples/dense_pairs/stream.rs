use image::GrayImage;
use navigate_visual::{ImageMatcher, VisualError};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    io::{BufRead, Write},
    path::{Path, PathBuf},
    time::Instant,
};

#[derive(Debug, thiserror::Error)]
pub(super) enum StreamError {
    #[error("{operation} matcher stream: {source}")]
    Io {
        operation: &'static str,
        #[source]
        source: std::io::Error,
    },
    #[error("decode matcher request: {0}")]
    Request(#[source] serde_json::Error),
    #[error("encode matcher response: {0}")]
    Response(#[source] serde_json::Error),
    #[error("request {request_id}: read image {path}: {source}")]
    Image {
        request_id: u64,
        path: PathBuf,
        #[source]
        source: image::ImageError,
    },
    #[error("request {request_id}: {backend}: {source}")]
    Match {
        request_id: u64,
        backend: String,
        #[source]
        source: VisualError,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    request_id: u64,
    reference: PathBuf,
    query: PathBuf,
}

pub(super) fn serve_blocking(
    matcher: &mut dyn ImageMatcher,
    input: impl BufRead,
    mut output: impl Write,
    load_ms: f64,
) -> Result<(), StreamError> {
    write_response_blocking(
        &mut output,
        &json!({
            "ready": true, "backend": matcher.identity(), "load_ms": load_ms,
            "scope": "pixel correspondences; no geometric acceptance or accuracy claim"
        }),
    )?;
    for line in input.lines() {
        let line = line.map_err(|source| StreamError::Io {
            operation: "read",
            source,
        })?;
        let request: Request = serde_json::from_str(&line).map_err(StreamError::Request)?;
        let response = match_request_blocking(matcher, request)?;
        write_response_blocking(&mut output, &response)?;
    }
    Ok(())
}

fn match_request_blocking(
    matcher: &mut dyn ImageMatcher,
    request: Request,
) -> Result<Value, StreamError> {
    let reference = read_image_blocking(request.request_id, &request.reference)?;
    let query = read_image_blocking(request.request_id, &request.query)?;
    let start = Instant::now();
    let pairs = matcher
        .match_images_blocking(&reference, &query)
        .map_err(|source| StreamError::Match {
            request_id: request.request_id,
            backend: matcher.identity().to_owned(),
            source,
        })?;
    let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;
    Ok(json!({
        "request_id": request.request_id, "backend": matcher.identity(), "elapsed_ms": elapsed_ms,
        "reference_image_sha256": format!("{:x}", Sha256::digest(reference.as_raw())),
        "query_image_sha256": format!("{:x}", Sha256::digest(query.as_raw())),
        "pairs": pairs.iter().map(|p| json!({
            "reference": [p.reference.x, p.reference.y], "query": [p.query.x, p.query.y]
        })).collect::<Vec<_>>()
    }))
}

fn read_image_blocking(request_id: u64, path: &Path) -> Result<GrayImage, StreamError> {
    image::open(path)
        .map(|image| image.to_luma8())
        .map_err(|source| StreamError::Image {
            request_id,
            path: path.to_owned(),
            source,
        })
}

fn write_response_blocking(output: &mut impl Write, response: &Value) -> Result<(), StreamError> {
    serde_json::to_writer(&mut *output, response).map_err(StreamError::Response)?;
    output
        .write_all(b"\n")
        .and_then(|()| output.flush())
        .map_err(|source| StreamError::Io {
            operation: "write",
            source,
        })
}

#[cfg(test)]
#[path = "stream/tests.rs"]
mod tests;
