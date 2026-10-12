//! Bound supported dense encoder shapes and their output grids.
use crate::InferenceError;

pub(crate) struct DenseLayout {
    pub channels: usize,
    pub width: usize,
    pub height: usize,
}
impl DenseLayout {
    pub fn from_shape(shape: &[i64]) -> Result<Self, InferenceError> {
        match shape {
            [1, channels @ (1 | 3), height @ 600, width @ 800]
            | [1, channels @ (1 | 3), height @ 1200, width @ 1600] => Ok(Self {
                channels: *channels as usize,
                width: *width as usize,
                height: *height as usize,
            }),
            _ => Err(InferenceError::Invalid(
                "dense image shape must be 1x{1,3}x600x800 or 1x{1,3}x1200x1600".into(),
            )),
        }
    }
    pub fn grid(&self) -> [usize; 2] {
        [self.height / 32 * 4, self.width / 32 * 4]
    }
}

#[cfg(test)]
mod tests;
