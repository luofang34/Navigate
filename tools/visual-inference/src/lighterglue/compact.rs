use super::*;
use ort::session::SessionOutputs;

pub(super) fn decode(
    output: &SessionOutputs<'_>,
    a: &[[f64; 2]],
    b: &[[f64; 2]],
) -> Result<Vec<PixelMatch>, InferenceError> {
    decode_bounded(output, a, b, [a.len(), b.len()])
}

pub(super) fn decode_bounded(
    output: &SessionOutputs<'_>,
    a: &[[f64; 2]],
    b: &[[f64; 2]],
    capacity: [usize; 2],
) -> Result<Vec<PixelMatch>, InferenceError> {
    decode_with(output, a, b, capacity, |pair, _| pair)
}
pub(super) fn decode_scored_bounded(
    output: &SessionOutputs<'_>,
    a: &[[f64; 2]],
    b: &[[f64; 2]],
    capacity: [usize; 2],
) -> Result<Vec<crate::ScoredPixelMatch>, InferenceError> {
    decode_with(output, a, b, capacity, |pair, assignment_score| {
        crate::ScoredPixelMatch {
            pair,
            assignment_score,
        }
    })
}
fn decode_with<T>(
    output: &SessionOutputs<'_>,
    a: &[[f64; 2]],
    b: &[[f64; 2]],
    capacity: [usize; 2],
    make: impl Fn(PixelMatch, f32) -> T,
) -> Result<Vec<T>, InferenceError> {
    if output.len() != 3 {
        return Err(InferenceError::Invalid(
            "compact matcher requires exactly three outputs".into(),
        ));
    }
    let indices = |name, count| {
        let (shape, data) = output
            .get(name)
            .ok_or_else(|| InferenceError::Invalid(format!("missing {name}")))?
            .try_extract_tensor::<i64>()
            .map_err(|e| runtime(format!("decode {name}"), e))?;
        if shape.as_ref() != [1, count as i64] {
            return Err(InferenceError::Invalid(format!("invalid {name} shape")));
        }
        Ok(data)
    };
    let rows = indices("row_indices", capacity[0])?;
    let columns = indices("column_indices", capacity[1])?;
    let (shape, scores) = output
        .get("row_scores")
        .ok_or_else(|| InferenceError::Invalid("missing row_scores".into()))?
        .try_extract_tensor::<f32>()
        .map_err(|e| runtime("decode row scores", e))?;
    if shape.as_ref() != [1, capacity[0] as i64] {
        return Err(InferenceError::Invalid("invalid row_scores shape".into()));
    }
    map_pairs(rows, columns, scores, a, b, make)
}

#[cfg(test)]
fn pairs(
    rows: &[i64],
    columns: &[i64],
    scores: &[f32],
    a: &[[f64; 2]],
    b: &[[f64; 2]],
) -> Result<Vec<PixelMatch>, InferenceError> {
    map_pairs(rows, columns, scores, a, b, |pair, _| pair)
}
fn map_pairs<T>(
    rows: &[i64],
    columns: &[i64],
    scores: &[f32],
    a: &[[f64; 2]],
    b: &[[f64; 2]],
    make: impl Fn(PixelMatch, f32) -> T,
) -> Result<Vec<T>, InferenceError> {
    if rows.len() < a.len()
        || scores.len() != rows.len()
        || columns.len() < b.len()
        || rows
            .iter()
            .any(|i| *i < 0 || *i as u64 >= columns.len() as u64)
        || columns
            .iter()
            .any(|i| *i < 0 || *i as u64 >= rows.len() as u64)
        || scores
            .iter()
            .any(|s| !s.is_finite() || !(0.0..=1.0).contains(s))
    {
        return Err(InferenceError::Invalid(
            "compact matcher indices, scores or lengths".into(),
        ));
    }
    Ok(rows
        .iter()
        .take(a.len())
        .zip(scores)
        .enumerate()
        .filter(|(i, (j, s))| {
            **s > 0.1 && (**j as usize) < b.len() && columns[**j as usize] == *i as i64
        })
        .map(|(i, (j, score))| {
            make(
                PixelMatch {
                    reference: a[i].into(),
                    query: b[*j as usize].into(),
                },
                *score,
            )
        })
        .collect())
}

#[cfg(test)]
mod tests;
