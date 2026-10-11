use super::*;
use ort::session::SessionOutputs;

pub(super) fn decode(
    output: &SessionOutputs<'_>,
    a: &[[f64; 2]],
    b: &[[f64; 2]],
) -> Result<Vec<PixelMatch>, InferenceError> {
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
    let rows = indices("row_indices", a.len())?;
    let columns = indices("column_indices", b.len())?;
    let (shape, scores) = output
        .get("row_scores")
        .ok_or_else(|| InferenceError::Invalid("missing row_scores".into()))?
        .try_extract_tensor::<f32>()
        .map_err(|e| runtime("decode row scores", e))?;
    if shape.as_ref() != [1, a.len() as i64] {
        return Err(InferenceError::Invalid("invalid row_scores shape".into()));
    }
    pairs(rows, columns, scores, a, b)
}

fn pairs(
    rows: &[i64],
    columns: &[i64],
    scores: &[f32],
    a: &[[f64; 2]],
    b: &[[f64; 2]],
) -> Result<Vec<PixelMatch>, InferenceError> {
    if rows.len() != a.len()
        || scores.len() != a.len()
        || columns.len() != b.len()
        || rows.iter().any(|i| *i < 0 || *i as u64 >= b.len() as u64)
        || columns
            .iter()
            .any(|i| *i < 0 || *i as u64 >= a.len() as u64)
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
        .zip(scores)
        .enumerate()
        .filter(|(i, (j, s))| **s > 0.1 && columns[**j as usize] == *i as i64)
        .map(|(i, (j, _))| PixelMatch {
            reference: a[i].into(),
            query: b[*j as usize].into(),
        })
        .collect())
}

#[cfg(test)]
mod tests;
