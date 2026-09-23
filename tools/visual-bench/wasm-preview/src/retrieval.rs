use crate::{error::PreviewError, model::Camera};
use navigate_visual::{GroundCorrespondence, planar_proposal};
use serde::Deserialize;
use serde_json::json;
use wasm_bindgen::prelude::*;

#[derive(Deserialize)]
struct GroundMatch {
    world: [f64; 3],
    query: [f64; 2],
}

/// Propose a pose through the shared Rust retriever. Surface acceptance is separate.
#[wasm_bindgen]
pub fn propose(camera_json: String, matches_json: String) -> Result<String, JsValue> {
    let camera: Camera = serde_json::from_str(&camera_json).map_err(PreviewError::from)?;
    camera.validate()?;
    let pairs: Vec<GroundMatch> =
        serde_json::from_str(&matches_json).map_err(PreviewError::from)?;
    let pairs: Vec<_> = pairs
        .into_iter()
        .map(|p| GroundCorrespondence {
            world: p.world.into(),
            query: p.query.into(),
        })
        .collect();
    let proposal = planar_proposal(&camera.model(), &pairs).map_err(PreviewError::from)?;
    let value = match proposal {
        None => json!({"retrieved":false,"reason":"no consistent planar retrieval"}),
        Some(proposal) => {
            let p = proposal.pose.position;
            let q = proposal.pose.orientation.quaternion();
            json!({"retrieved":true,"retrieval_inliers":proposal.inliers,"position_enu_m":[p.x,p.y,p.z],"eye_to_enu_xyzw":[q.i,q.j,q.k,q.w],"acceptance_stage":"retrieval_only"})
        }
    };
    Ok(value.to_string())
}
