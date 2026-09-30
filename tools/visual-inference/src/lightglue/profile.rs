//! Exact feature-count specialization of the same dynamic matcher model.
use crate::{ExecutionConfig, InferenceError, native_runtime};
use ort::{
    session::Session,
    value::{TensorElementType, ValueType},
};
use std::{collections::BTreeSet, path::Path};

pub(super) struct KeypointProfile {
    session: Session,
    count: usize,
}
impl KeypointProfile {
    pub fn load_blocking(
        path: &Path,
        dynamic: &Session,
        count: usize,
        execution: &ExecutionConfig,
    ) -> Result<Self, InferenceError> {
        if !(32..=4096).contains(&count) {
            return Err(invalid("feature count must be 32 through 4096"));
        }
        let inputs: Vec<_> = dynamic
            .inputs()
            .iter()
            .map(|input| (input.name(), input.dtype()))
            .collect();
        let overrides = dimensions(&inputs)?
            .into_iter()
            .map(|name| (name, count as i64))
            .collect::<Vec<_>>();
        let session =
            native_runtime::session_with_dimensions_blocking(path, execution, &overrides)?;
        for input in session.inputs() {
            let ValueType::Tensor { shape, .. } = input.dtype() else {
                return Err(invalid("profile input is not a tensor"));
            };
            if shape.len() != 3 || shape[1] != count as i64 {
                return Err(invalid("runtime did not apply the feature-count profile"));
            }
        }
        Ok(Self { session, count })
    }
    pub fn session_for(&mut self, reference: usize, query: usize) -> Option<&mut Session> {
        (reference == self.count && query == self.count).then_some(&mut self.session)
    }
}
fn dimensions(inputs: &[(&str, &ValueType)]) -> Result<Vec<String>, InferenceError> {
    if inputs.len() != 4 {
        return Err(invalid("expected four LightGlue inputs"));
    }
    let mut names = BTreeSet::new();
    for (name, width) in [
        ("keypoints0", 2),
        ("keypoints1", 2),
        ("descriptors0", 256),
        ("descriptors1", 256),
    ] {
        let dtype = inputs
            .iter()
            .find(|(key, _)| *key == name)
            .map(|(_, dtype)| *dtype)
            .ok_or_else(|| invalid("LightGlue input is absent"))?;
        let ValueType::Tensor {
            ty,
            shape,
            dimension_symbols,
        } = dtype
        else {
            return Err(invalid("LightGlue input is not a tensor"));
        };
        if *ty != TensorElementType::Float32
            || shape.len() != 3
            || shape[0] != 1
            || shape[1] != -1
            || shape[2] != width
        {
            return Err(invalid(
                "profile requires dynamic float32 LightGlue features",
            ));
        }
        let symbol = dimension_symbols
            .get(1)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| invalid("feature dimension has no name"))?;
        names.insert(symbol.clone());
    }
    Ok(names.into_iter().collect())
}
fn invalid(reason: &str) -> InferenceError {
    InferenceError::Invalid(format!("LightGlue feature profile: {reason}"))
}

#[cfg(test)]
mod tests;
