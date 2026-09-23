//! Adapter-private feature data and descriptor matching.
use crate::InferenceError;
use navigate_visual::PixelMatch;

pub(crate) struct Features {
    pub pixels: Vec<[f64; 2]>,
    pub model_pixels: Vec<f32>,
    pub scores: Vec<f32>,
    pub descriptors: Vec<f32>,
    pub dimensions: usize,
}
impl Features {
    pub fn empty(dimensions: usize) -> Self {
        Self {
            pixels: vec![],
            model_pixels: vec![],
            scores: vec![],
            descriptors: vec![],
            dimensions,
        }
    }
    pub fn push(
        &mut self,
        pixel: [f64; 2],
        model_pixel: [f32; 2],
        score: f32,
        descriptor: &[f32],
    ) -> Result<(), InferenceError> {
        let norm = descriptor.iter().map(|x| x * x).sum::<f32>().sqrt();
        if descriptor.len() != self.dimensions || !norm.is_finite() || norm <= 0.0 {
            return Err(InferenceError::Invalid("invalid descriptor".into()));
        }
        self.pixels.push(pixel);
        self.model_pixels.extend(model_pixel);
        self.scores.push(score);
        self.descriptors.extend(descriptor.iter().map(|v| v / norm));
        Ok(())
    }
    pub fn channel_major(&self) -> Vec<f32> {
        let n = self.pixels.len();
        let mut result = vec![0.0; n * self.dimensions];
        for i in 0..n {
            for c in 0..self.dimensions {
                result[c * n + i] = self.descriptors[i * self.dimensions + c];
            }
        }
        result
    }
}
#[derive(Clone, Copy)]
struct Nearest {
    index: usize,
    first: f32,
    second: f32,
}
impl Nearest {
    fn update(&mut self, index: usize, score: f32) {
        if score > self.first {
            self.second = self.first;
            self.first = score;
            self.index = index;
        } else if score > self.second {
            self.second = score;
        }
    }
    fn distinct(self) -> bool {
        self.first > 0.82 && 1.0 - self.first < 0.9 * (1.0 - self.second)
    }
}
pub(crate) fn mutual(first: &Features, second: &Features) -> Vec<PixelMatch> {
    let empty = Nearest {
        index: 0,
        first: -2.0,
        second: -2.0,
    };
    let mut a = vec![empty; first.pixels.len()];
    let mut b = vec![empty; second.pixels.len()];
    for (i, da) in first.descriptors.chunks_exact(first.dimensions).enumerate() {
        for (j, db) in second
            .descriptors
            .chunks_exact(second.dimensions)
            .enumerate()
        {
            let score = da.iter().zip(db).map(|(a, b)| a * b).sum();
            a[i].update(j, score);
            b[j].update(i, score);
        }
    }
    a.iter()
        .enumerate()
        .filter_map(|(i, n)| {
            let reverse = b.get(n.index)?;
            (n.distinct() && reverse.distinct() && reverse.index == i).then(|| PixelMatch {
                reference: first.pixels[i].into(),
                query: second.pixels[n.index].into(),
            })
        })
        .collect()
}
#[cfg(test)]
mod tests;
