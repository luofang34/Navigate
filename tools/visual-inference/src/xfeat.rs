//! Sparse XFeat output decoding, bounded to the supplied image.
use crate::{InferenceError, features::Features, native_runtime::runtime, preprocessing};
use image::GrayImage;
use ort::{session::Session, value::Tensor};

pub(crate) fn extract(
    session: &mut Session,
    image: &GrayImage,
    limit: usize,
) -> Result<Features, InferenceError> {
    let input = preprocessing::prepare(image, 800, 600, 3)?;
    let tensor = Tensor::from_array(([1, 3, input.height, input.width], input.data.clone()))
        .map_err(|e| runtime("build XFeat tensor", e))?;
    let output = match session.run(ort::inputs!["images"=>tensor]) {
        Ok(output) => output,
        Err(error)
            if error.to_string().contains("Name:'/Where'")
                && error
                    .to_string()
                    .contains("Condition Shape: {1,0,2}, X Shape: {1,0}, Y Shape: {}") =>
        {
            return Ok(Features::empty(64));
        }
        Err(error) => return Err(runtime("run XFeat", error)),
    };
    let get = |name| {
        output
            .get(name)
            .ok_or_else(|| InferenceError::Invalid(format!("missing XFeat {name}")))
    };
    let (shape, points) = get("keypoints")?
        .try_extract_tensor::<f32>()
        .map_err(|e| runtime("read XFeat pixels", e))?;
    let (_, scores) = get("scores")?
        .try_extract_tensor::<f32>()
        .map_err(|e| runtime("read XFeat scores", e))?;
    let (desc_shape, descriptors) = get("descriptors")?
        .try_extract_tensor::<f32>()
        .map_err(|e| runtime("read XFeat descriptors", e))?;
    if shape.len() != 2
        || shape[1] != 2
        || desc_shape.len() != 2
        || desc_shape[1] != 64
        || scores.len() * 2 != points.len()
        || scores.len() * 64 != descriptors.len()
    {
        return Err(InferenceError::Invalid(
            "incompatible XFeat output shapes".into(),
        ));
    }
    let mut selected: Vec<_> = scores
        .iter()
        .enumerate()
        .filter_map(|(i, score)| {
            let p = input.pixel([points[i * 2] as f64, points[i * 2 + 1] as f64]);
            (score.is_finite() && *score > 0.0 && preprocessing::inside(p, image))
                .then_some((i, p, *score))
        })
        .collect();
    selected.sort_by(|a, b| b.2.total_cmp(&a.2));
    let mut result = Features::empty(64);
    for (i, p, score) in selected.into_iter().take(limit) {
        result.push(
            p,
            [points[i * 2], points[i * 2 + 1]],
            score,
            &descriptors[i * 64..(i + 1) * 64],
        )?;
    }
    Ok(result)
}
