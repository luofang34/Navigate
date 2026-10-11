//! Decode the LighterGlue assignment without exposing model scores to geometry.
use crate::{InferenceError, features::Features, native_runtime::runtime};
use navigate_visual::PixelMatch;
use ort::{session::Session, value::TensorRef};
mod compact;

pub(crate) fn match_features(
    session: &mut Session,
    a: &Features,
    b: &Features,
) -> Result<Vec<PixelMatch>, InferenceError> {
    if a.pixels.is_empty() || b.pixels.is_empty() {
        return Ok(vec![]);
    }
    let normalized = [a, b].map(|f| {
        f.model_pixels
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|p| [(p[0] - 400.0) / 400.0, (p[1] - 300.0) / 400.0])
            .collect::<Vec<_>>()
    });
    match_prepared(session, a, b, &normalized[0], &normalized[1])
}

pub(crate) fn match_prepared(
    session: &mut Session,
    a: &Features,
    b: &Features,
    pa: &[f32],
    pb: &[f32],
) -> Result<Vec<PixelMatch>, InferenceError> {
    if a.pixels.is_empty() || b.pixels.is_empty() {
        return Ok(vec![]);
    }
    let mut inputs = Vec::new();
    for (i, (f, pixels)) in [(a, pa), (b, pb)].into_iter().enumerate() {
        for (name, shape, data) in [
            ("keypoints", [1, f.pixels.len(), 2], pixels),
            (
                "descriptors",
                [1, f.pixels.len(), 64],
                f.descriptors.as_slice(),
            ),
        ] {
            inputs.push((
                format!("{name}{i}"),
                TensorRef::from_array_view((shape, data))
                    .map_err(|e| runtime("borrow LighterGlue input", e))?
                    .into_dyn(),
            ));
        }
    }
    let output = session
        .run(inputs)
        .map_err(|e| runtime("run LighterGlue", e))?;
    if output.get("row_indices").is_some() {
        return compact::decode(&output, &a.pixels, &b.pixels);
    }
    let probability = output.get("assignment").is_some();
    if probability && output.get("log_assignment").is_some() {
        return Err(InferenceError::Invalid(
            "ambiguous LighterGlue score contract".into(),
        ));
    }
    let name = if probability {
        "assignment"
    } else {
        "log_assignment"
    };
    let (shape, scores) = output
        .get(name)
        .ok_or_else(|| InferenceError::Invalid("missing LighterGlue assignment output".into()))?
        .try_extract_tensor::<f32>()
        .map_err(|e| runtime("decode LighterGlue scores", e))?;
    if shape.as_ref() != [1, a.pixels.len() as i64, b.pixels.len() as i64] {
        return Err(InferenceError::Invalid(
            "invalid LighterGlue assignment shape".into(),
        ));
    }
    assignment_pairs(scores, &a.pixels, &b.pixels, probability)
}

fn assignment_pairs(
    scores: &[f32],
    a: &[[f64; 2]],
    b: &[[f64; 2]],
    probability: bool,
) -> Result<Vec<PixelMatch>, InferenceError> {
    if scores.len() != a.len().saturating_mul(b.len())
        || scores
            .iter()
            .any(|s| !s.is_finite() || (probability && !(0.0..=1.0).contains(s)))
    {
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
    let threshold = if probability { 0.1 } else { 0.1_f32.ln() };
    Ok(rows
        .into_iter()
        .enumerate()
        .filter(|(i, (j, score))| *score > threshold && columns.get(*j).is_some_and(|c| c.0 == *i))
        .map(|(i, (j, _))| PixelMatch {
            reference: a[i].into(),
            query: b[j].into(),
        })
        .collect())
}

#[cfg(all(test, target_os = "macos"))]
mod device_tests;
#[cfg(test)]
mod tests;
