//! Decode the LighterGlue assignment without exposing model scores to geometry.
use crate::{InferenceError, features::Features, native_runtime::runtime};
use navigate_visual::PixelMatch;
use ort::{session::Session, value::Tensor};

pub(crate) fn match_features(
    session: &mut Session,
    a: &Features,
    b: &Features,
) -> Result<Vec<PixelMatch>, InferenceError> {
    if a.pixels.is_empty() || b.pixels.is_empty() {
        return Ok(vec![]);
    }
    let mut inputs = Vec::new();
    for (i, f) in [a, b].into_iter().enumerate() {
        // The sparse detector export uses an 800 by 600 pixel coordinate system.
        let pixels: Vec<f32> = f
            .model_pixels
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|p| [(p[0] - 400.0) / 400.0, (p[1] - 300.0) / 400.0])
            .collect();
        for (name, shape, data) in [
            ("keypoints", vec![1, f.pixels.len(), 2], pixels),
            (
                "descriptors",
                vec![1, f.pixels.len(), 64],
                f.descriptors.clone(),
            ),
        ] {
            inputs.push((
                format!("{name}{i}"),
                Tensor::from_array((shape, data))
                    .map_err(|e| runtime("build LighterGlue input", e))?
                    .into_dyn(),
            ));
        }
    }
    let output = session
        .run(inputs)
        .map_err(|e| runtime("run LighterGlue", e))?;
    let (shape, scores) = output
        .get("log_assignment")
        .ok_or_else(|| InferenceError::Invalid("missing LighterGlue log_assignment".into()))?
        .try_extract_tensor::<f32>()
        .map_err(|e| runtime("decode LighterGlue scores", e))?;
    if shape.as_ref() != [1, a.pixels.len() as i64, b.pixels.len() as i64] {
        return Err(InferenceError::Invalid(
            "invalid LighterGlue assignment shape".into(),
        ));
    }
    assignment_pairs(scores, &a.pixels, &b.pixels)
}

fn assignment_pairs(
    scores: &[f32],
    a: &[[f64; 2]],
    b: &[[f64; 2]],
) -> Result<Vec<PixelMatch>, InferenceError> {
    if scores.len() != a.len().saturating_mul(b.len()) || scores.iter().any(|s| !s.is_finite()) {
        return Err(InferenceError::Invalid(format!(
            "invalid LighterGlue assignment values: {} scores for {} by {} features; {} NaN, {} negative infinity, {} positive infinity",
            scores.len(),
            a.len(),
            b.len(),
            scores.iter().filter(|v| v.is_nan()).count(),
            scores.iter().filter(|v| **v == f32::NEG_INFINITY).count(),
            scores.iter().filter(|v| **v == f32::INFINITY).count()
        )));
    }
    let mut rows = vec![(usize::MAX, f32::NEG_INFINITY); a.len()];
    let mut columns = vec![(usize::MAX, f32::NEG_INFINITY); b.len()];
    for (i, row) in rows.iter_mut().enumerate() {
        for (j, column) in columns.iter_mut().enumerate() {
            let score = scores[i * b.len() + j];
            if score > row.1 {
                *row = (j, score);
            }
            if score > column.1 {
                *column = (i, score);
            }
        }
    }
    Ok(rows
        .into_iter()
        .enumerate()
        .filter(|(i, (j, score))| {
            *score > 0.1_f32.ln() && columns.get(*j).is_some_and(|c| c.0 == *i)
        })
        .map(|(i, (j, _))| PixelMatch {
            reference: a[i].into(),
            query: b[j].into(),
        })
        .collect())
}

#[cfg(test)]
mod tests;
