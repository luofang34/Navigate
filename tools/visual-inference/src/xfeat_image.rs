//! Dense image graph with preprocessing inside the export.
use crate::{
    InferenceError, features::Features, native_runtime::runtime, preprocessing, xfeat_dense,
};
use image::GrayImage;
use ort::{session::Session, value::TensorRef};

pub(crate) fn extract(
    session: &mut Session,
    image: &GrayImage,
    limit: usize,
) -> Result<Features, InferenceError> {
    let channels = input_channels(session)?;
    let input = preprocessing::prepare(image, 800, 600, channels)?;
    let tensor = TensorRef::from_array_view(([1, channels, 600, 800], input.data.as_slice()))
        .map_err(|e| runtime("build dense image input", e))?;
    let output = session
        .run(ort::inputs!["images" => tensor])
        .map_err(|e| runtime("run dense image graph", e))?;
    let read = |name: &str, channels: i64| -> Result<&[f32], InferenceError> {
        let (shape, values) = output
            .get(name)
            .ok_or_else(|| InferenceError::Invalid(format!("missing dense output {name}")))?
            .try_extract_tensor::<f32>()
            .map_err(|e| runtime("read dense image output", e))?;
        if shape.as_ref() != [1, channels, 72, 100] || values.iter().any(|v| !v.is_finite()) {
            return Err(InferenceError::Invalid(format!(
                "invalid dense output {name}"
            )));
        }
        Ok(values)
    };
    xfeat_dense::decode(
        read("dense_descriptors", 64)?,
        read("keypoint_logits", 65)?,
        read("reliability", 1)?,
        &input,
        image,
        limit,
    )
}

fn input_channels(session: &Session) -> Result<usize, InferenceError> {
    let [input] = session.inputs() else {
        return Err(InferenceError::Invalid(
            "dense graph requires one image input".into(),
        ));
    };
    let ort::value::ValueType::Tensor { ty, shape, .. } = input.dtype() else {
        return Err(InferenceError::Invalid(
            "dense input is not a tensor".into(),
        ));
    };
    if input.name() != "images" || *ty != ort::value::TensorElementType::Float32 {
        return Err(InferenceError::Invalid("dense image name or type".into()));
    }
    match shape.as_ref() {
        [1, channels @ (1 | 3), 600, 800] => Ok(*channels as usize),
        _ => Err(InferenceError::Invalid(
            "dense image shape must be 1x{1,3}x600x800".into(),
        )),
    }
}
