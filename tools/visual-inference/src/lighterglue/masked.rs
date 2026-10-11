use super::*;
use ort::value::ValueType;
use std::borrow::Cow;

pub(super) fn capacity(session: &Session) -> Result<Option<usize>, InferenceError> {
    let masks: Vec<_> = session
        .inputs()
        .iter()
        .filter(|i| i.name().starts_with("key_bias"))
        .collect();
    if masks.is_empty() {
        return Ok(None);
    }
    if masks.len() != 2 || session.inputs().len() != 6 {
        return Err(InferenceError::Invalid(
            "masked matcher input contract".into(),
        ));
    }
    let mut count = None;
    for (i, input) in masks.iter().enumerate() {
        let ValueType::Tensor { shape, ty, .. } = input.dtype() else {
            return Err(InferenceError::Invalid("masked matcher bias type".into()));
        };
        if input.name() != format!("key_bias{i}")
            || shape.len() != 3
            || shape[0] != 1
            || shape[1] != 1
            || !(32..=4096).contains(&shape[2])
            || *ty != ort::value::TensorElementType::Float32
            || count.is_some_and(|c| c != shape[2] as usize)
        {
            return Err(InferenceError::Invalid("masked matcher bias shape".into()));
        }
        count = Some(shape[2] as usize);
    }
    Ok(count)
}

pub(super) fn run(
    session: &mut Session,
    features: [&Features; 2],
    pixels: [&[f32]; 2],
    capacity: usize,
) -> Result<Vec<PixelMatch>, InferenceError> {
    run_with(session, features, pixels, capacity, |output| {
        if output.get("assignment").is_some() {
            return probabilities(output, features, capacity);
        }
        compact::decode_bounded(
            output,
            &features[0].pixels,
            &features[1].pixels,
            [capacity; 2],
        )
    })
}
pub(super) fn run_scored(
    session: &mut Session,
    features: [&Features; 2],
    pixels: [&[f32]; 2],
    capacity: usize,
) -> Result<Vec<crate::ScoredPixelMatch>, InferenceError> {
    run_with(session, features, pixels, capacity, |output| {
        compact::decode_scored_bounded(
            output,
            &features[0].pixels,
            &features[1].pixels,
            [capacity; 2],
        )
    })
}
fn run_with<T>(
    session: &mut Session,
    features: [&Features; 2],
    pixels: [&[f32]; 2],
    capacity: usize,
    decode: impl Fn(&ort::session::SessionOutputs<'_>) -> Result<Vec<T>, InferenceError>,
) -> Result<Vec<T>, InferenceError> {
    let a = Padded::new(features[0], pixels[0], capacity)?;
    let b = Padded::new(features[1], pixels[1], capacity)?;
    let mut inputs = Vec::new();
    for (i, data) in [&a, &b].into_iter().enumerate() {
        for (name, shape, values) in [
            ("keypoints", [1, capacity, 2], data.pixels.as_ref()),
            ("descriptors", [1, capacity, 64], data.descriptors.as_ref()),
            ("key_bias", [1, 1, capacity], data.bias.as_slice()),
        ] {
            inputs.push((
                format!("{name}{i}"),
                TensorRef::from_array_view((shape, values))
                    .map_err(|e| runtime("borrow masked matcher input", e))?
                    .into_dyn(),
            ));
        }
    }
    let output = session
        .run(inputs)
        .map_err(|e| runtime("run masked matcher", e))?;
    decode(&output)
}

struct Padded<'a> {
    pixels: Cow<'a, [f32]>,
    descriptors: Cow<'a, [f32]>,
    bias: Vec<f32>,
}
impl<'a> Padded<'a> {
    fn new(
        features: &'a Features,
        pixels: &'a [f32],
        capacity: usize,
    ) -> Result<Self, InferenceError> {
        let count = features.pixels.len();
        if count == 0
            || count > capacity
            || pixels.len() != count * 2
            || features.descriptors.len() != count * 64
        {
            return Err(InferenceError::Invalid(
                "masked matcher feature dimensions".into(),
            ));
        }
        let pad = |data: &'a [f32], width: usize| {
            if count == capacity {
                Cow::Borrowed(data)
            } else {
                let mut buffer = vec![0.0; capacity * width];
                buffer[..data.len()].copy_from_slice(data);
                Cow::Owned(buffer)
            }
        };
        // Finite suppression avoids NaNs in unused rows. Empty sets skip inference.
        let mut bias = vec![-10000.0; capacity];
        bias[..count].fill(0.0);
        Ok(Self {
            pixels: pad(pixels, 2),
            descriptors: pad(&features.descriptors, 64),
            bias,
        })
    }
}

fn probabilities(
    output: &ort::session::SessionOutputs<'_>,
    features: [&Features; 2],
    capacity: usize,
) -> Result<Vec<PixelMatch>, InferenceError> {
    let (shape, scores) = output
        .get("assignment")
        .ok_or_else(|| InferenceError::Invalid("missing masked assignment".into()))?
        .try_extract_tensor::<f32>()
        .map_err(|e| runtime("read masked assignment", e))?;
    if shape.as_ref() != [1, capacity as i64, capacity as i64] {
        return Err(InferenceError::Invalid("masked assignment shape".into()));
    }
    let valid: Vec<_> = scores
        .chunks_exact(capacity)
        .take(features[0].pixels.len())
        .flat_map(|row| row.iter().take(features[1].pixels.len()).copied())
        .collect();
    assignment_pairs(&valid, &features[0].pixels, &features[1].pixels, true)
}

#[cfg(test)]
mod tests;
