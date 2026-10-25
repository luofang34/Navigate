//! Named weight tensors read from and written to safetensors files.
use crate::MatchError;
use half::f16;
use safetensors::{Dtype, SafeTensors, View, tensor::TensorView};
use std::{borrow::Cow, collections::BTreeMap};

/// One weight tensor in host memory.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightTensor {
    /// Dimensions, outermost first.
    pub shape: Vec<usize>,
    /// Values in row-major order.
    pub values: Vec<f32>,
}

/// Named weight tensors of one model.
///
/// The model takes each tensor by name and checks its shape. After loading, no
/// tensor may remain: an unused tensor shows a file for a different model.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Weights {
    tensors: BTreeMap<String, WeightTensor>,
}

impl Weights {
    /// Read a safetensors file with F32 or F16 tensors.
    ///
    /// # Errors
    ///
    /// Returns an error for a malformed file or an unsupported data type.
    pub fn from_safetensors(bytes: &[u8]) -> Result<Self, MatchError> {
        let file = SafeTensors::deserialize(bytes).map_err(MatchError::WeightFile)?;
        let mut tensors = BTreeMap::new();
        for (name, view) in file.tensors() {
            let values = decode(&name, &view)?;
            tensors.insert(
                name,
                WeightTensor {
                    shape: view.shape().to_vec(),
                    values,
                },
            );
        }
        Ok(Self { tensors })
    }

    /// Write the tensors as a safetensors file, in F16 when `half` is set.
    ///
    /// # Errors
    ///
    /// Returns an error when the serializer fails.
    pub fn to_safetensors(&self, half: bool) -> Result<Vec<u8>, MatchError> {
        let views: Vec<(&str, Encoded)> = self
            .tensors
            .iter()
            .map(|(name, tensor)| (name.as_str(), Encoded::new(tensor, half)))
            .collect();
        safetensors::serialize(views, None).map_err(MatchError::Serialize)
    }

    /// Add or replace one tensor.
    pub fn insert(&mut self, name: impl Into<String>, tensor: WeightTensor) {
        self.tensors.insert(name.into(), tensor);
    }

    /// Number of tensors.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    /// True when there are no tensors.
    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// Number of scalar values in all tensors.
    pub fn parameter_count(&self) -> usize {
        self.tensors.values().map(|t| t.values.len()).sum()
    }

    /// Remove one tensor without a shape check.
    pub(crate) fn remove(&mut self, name: &str) -> Option<WeightTensor> {
        self.tensors.remove(name)
    }

    /// Remove and return all tensors in name order.
    pub(crate) fn drain(&mut self) -> impl Iterator<Item = (String, WeightTensor)> {
        std::mem::take(&mut self.tensors).into_iter()
    }

    /// Remove one tensor and check its shape.
    pub(crate) fn take(&mut self, name: &str, shape: &[usize]) -> Result<Vec<f32>, MatchError> {
        let tensor = self
            .tensors
            .remove(name)
            .ok_or_else(|| MatchError::MissingWeight { name: name.into() })?;
        if tensor.shape != shape {
            return Err(MatchError::WeightShape {
                name: name.into(),
                actual: tensor.shape,
                expected: shape.to_vec(),
            });
        }
        Ok(tensor.values)
    }

    /// Fail when tensors remain after the model took all that it needs.
    pub(crate) fn finish(self) -> Result<(), MatchError> {
        match self.tensors.keys().next() {
            None => Ok(()),
            Some(example) => Err(MatchError::UnusedWeights {
                count: self.tensors.len(),
                example: example.clone(),
            }),
        }
    }
}

fn decode(name: &str, view: &TensorView<'_>) -> Result<Vec<f32>, MatchError> {
    let data = view.data();
    match view.dtype() {
        Dtype::F32 => Ok(data
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect()),
        Dtype::F16 => Ok(data
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| f16::from_le_bytes(*b).to_f32())
            .collect()),
        other => Err(MatchError::WeightType {
            name: name.into(),
            dtype: format!("{other:?}"),
        }),
    }
}

struct Encoded {
    shape: Vec<usize>,
    dtype: Dtype,
    bytes: Vec<u8>,
}

impl Encoded {
    fn new(tensor: &WeightTensor, half: bool) -> Self {
        let bytes = if half {
            tensor
                .values
                .iter()
                .flat_map(|v| f16::from_f32(*v).to_le_bytes())
                .collect()
        } else {
            tensor.values.iter().flat_map(|v| v.to_le_bytes()).collect()
        };
        Self {
            shape: tensor.shape.clone(),
            dtype: if half { Dtype::F16 } else { Dtype::F32 },
            bytes,
        }
    }
}

impl View for Encoded {
    fn dtype(&self) -> Dtype {
        self.dtype
    }
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn data(&self) -> Cow<'_, [u8]> {
        Cow::Borrowed(&self.bytes)
    }
    fn data_len(&self) -> usize {
        self.bytes.len()
    }
}

#[cfg(test)]
mod tests;
