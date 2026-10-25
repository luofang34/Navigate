use super::decode;
use crate::{ExecutionConfig, InferenceError, native_runtime, preprocessing::InputImage};
use image::GrayImage;
use navigate_visual::PixelMatch;
use ort::{
    session::{Session, SessionOutputs},
    value::{TensorElementType, TensorRef, ValueType},
};
use sha2::{Digest, Sha256};
use std::path::Path;
const PATCH: usize = 25 * 128;
pub(super) struct FineStage {
    session: Session,
    batch: usize,
    buffers: [Vec<f32>; 2],
    pub identity: String,
    pub source: String,
}
impl FineStage {
    pub fn load_blocking(path: &Path, execution: &ExecutionConfig) -> Result<Self, InferenceError> {
        let bytes = std::fs::read(path).map_err(|source| InferenceError::Io {
            path: path.into(),
            source,
        })?;
        let session = native_runtime::session_blocking(path, execution)?;
        let mut batch = None;
        if session.inputs().len() != 2 {
            return Err(InferenceError::Invalid(
                "fine stage requires two patch inputs".into(),
            ));
        }
        for (input, name) in session.inputs().iter().zip(["patches0", "patches1"]) {
            let ValueType::Tensor { ty, shape, .. } = input.dtype() else {
                return Err(InferenceError::Invalid("fine patch tensor".into()));
            };
            let [n, 25, 128] = shape.as_ref() else {
                return Err(InferenceError::Invalid("fine patch dimensions".into()));
            };
            if input.name() != name
                || *ty != TensorElementType::Float32
                || !(1..=1024).contains(n)
                || batch.is_some_and(|b| b != *n as usize)
            {
                return Err(InferenceError::Invalid("fine patch contract".into()));
            }
            batch = Some(*n as usize);
        }
        let batch = batch.ok_or_else(|| InferenceError::Invalid("missing fine batch".into()))?;
        let source = source(&session, &format!("fine-offsets-v1/batch{batch}"))?;
        Ok(Self {
            session,
            source,
            batch,
            buffers: [vec![0.0; batch * PATCH], vec![0.0; batch * PATCH]],
            identity: format!(
                "fine/{}/batch{batch}/ort-requested-{:?}",
                hex::encode(Sha256::digest(bytes)),
                execution.provider
            ),
        })
    }
    pub fn decode(
        &mut self,
        output: &SessionOutputs<'_>,
        transforms: [&InputImage; 2],
        images: [&GrayImage; 2],
    ) -> Result<Vec<PixelMatch>, InferenceError> {
        let (shape, confidence) = output
            .get("confidence")
            .ok_or_else(|| InferenceError::Invalid("missing coarse confidence".into()))?
            .try_extract_tensor::<f32>()
            .map_err(|e| native_runtime::runtime("read coarse confidence", e))?;
        let n = confidence.len();
        if shape.as_ref() != [n as i64] || n > 4800 || confidence.iter().any(|v| !v.is_finite()) {
            return Err(InferenceError::Invalid(
                "coarse match count or confidence".into(),
            ));
        }
        let reference = read(output, "keypoints0", &[n as i64, 2])?;
        let mut query = read(output, "query_coarse", &[n as i64, 2])?.to_vec();
        let patches = [
            read(output, "patches0", &[n as i64, 25, 128])?,
            read(output, "patches1", &[n as i64, 25, 128])?,
        ];
        for start in (0..n).step_by(self.batch) {
            let count = (n - start).min(self.batch);
            for (buffer, source) in self.buffers.iter_mut().zip(patches) {
                buffer[..count * PATCH]
                    .copy_from_slice(&source[start * PATCH..(start + count) * PATCH]);
                buffer[count * PATCH..].fill(0.0);
            }
            let tensor = |i: usize| {
                TensorRef::from_array_view(([self.batch, 25, 128], self.buffers[i].as_slice()))
                    .map_err(|e| native_runtime::runtime("borrow fine patch batch", e))
            };
            let fine = self
                .session
                .run(ort::inputs!["patches0"=>tensor(0)?,"patches1"=>tensor(1)?])
                .map_err(|e| native_runtime::runtime("run fine patch batch", e))?;
            let offsets = read(&fine, "offsets", &[self.batch as i64, 2])?;
            for (pixel, offset) in query[start * 2..(start + count) * 2]
                .iter_mut()
                .zip(offsets)
            {
                *pixel += offset;
            }
        }
        decode(reference, &query, confidence, transforms, images)
    }
}
fn read<'a>(
    output: &'a SessionOutputs<'_>,
    name: &str,
    dimensions: &[i64],
) -> Result<&'a [f32], InferenceError> {
    let (shape, data) = output
        .get(name)
        .ok_or_else(|| InferenceError::Invalid(format!("missing split LoFTR {name}")))?
        .try_extract_tensor::<f32>()
        .map_err(|e| native_runtime::runtime(format!("read split LoFTR {name}"), e))?;
    if shape.as_ref() != dimensions || data.iter().any(|v| !v.is_finite()) {
        return Err(InferenceError::Invalid(format!(
            "invalid split LoFTR {name}"
        )));
    }
    Ok(data)
}

pub(super) fn source(session: &Session, stage: &str) -> Result<String, InferenceError> {
    let metadata = session
        .metadata()
        .map_err(|e| native_runtime::runtime("read LoFTR stage metadata", e))?;
    let source = metadata
        .custom("navigate.loftr.source_sha256")
        .ok_or_else(|| InferenceError::Invalid("missing LoFTR source digest".into()))?;
    if source.len() != 64
        || !source.bytes().all(|b| b.is_ascii_hexdigit())
        || metadata.custom("navigate.loftr.stage").as_deref() != Some(stage)
    {
        return Err(InferenceError::Invalid(
            "invalid LoFTR stage metadata".into(),
        ));
    }
    Ok(source)
}

#[cfg(test)]
mod tests;
