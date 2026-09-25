//! Decode the same source links that the upload worker stores.
use super::*;
use navigate_visual::ScenePointObservation;
use navigate_visual::reconstruction::ImageTrack;
#[derive(Deserialize)]
struct Graph {
    observation_sha256: Vec<String>,
    tracks: Vec<Track>,
}
#[derive(Deserialize)]
struct Track {
    feature_id: String,
    observations: Vec<Observation>,
}
#[derive(Deserialize)]
struct Observation {
    camera_index: usize,
    pixel: [f64; 2],
}
pub(super) fn decode(json: &str) -> Result<ImageTracks, PreviewError> {
    let graph: Graph = serde_json::from_str(json)?;
    let tracks = graph
        .tracks
        .into_iter()
        .map(|t| {
            let feature_id = t
                .feature_id
                .parse()
                .map_err(|source| PreviewError::ScenePointId {
                    id: t.feature_id,
                    source,
                })?;
            Ok(ImageTrack {
                feature_id,
                observations: t
                    .observations
                    .into_iter()
                    .map(|o| ScenePointObservation {
                        camera_index: o.camera_index,
                        pixel: o.pixel.into(),
                    })
                    .collect(),
            })
        })
        .collect::<Result<Vec<_>, PreviewError>>()?;
    Ok(ImageTracks {
        observation_sha256: graph.observation_sha256,
        tracks,
    })
}
pub(super) fn camera(json: &str) -> Result<CameraModel, PreviewError> {
    let camera: Camera = serde_json::from_str(json)?;
    camera.validate()?;
    Ok(camera.model())
}
