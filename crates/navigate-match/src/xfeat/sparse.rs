//! Sparse 64-D keypoint feature sets and mutual cosine matching for [`super::XFeat`].

use navigate_visual::PixelMatch;

/// Sparse keypoints and L2-normalized 64-D descriptors extracted from one image.
#[derive(Clone, Debug, PartialEq)]
pub struct SparseFeatures {
    /// Keypoint positions `[x, y]` in source-image pixel coordinates.
    pub pixels: Vec<[f64; 2]>,
    /// Keypoint positions `[x, y]` in `800 × 600` model-frame coordinates.
    pub model_pixels: Vec<[f32; 2]>,
    /// Detection confidence score of each keypoint.
    pub scores: Vec<f32>,
    /// Row-major L2-normalized descriptors (`64` values per keypoint).
    pub descriptors: Vec<f32>,
}

impl SparseFeatures {
    /// Construct an empty feature set.
    pub fn empty() -> Self {
        Self {
            pixels: Vec::new(),
            model_pixels: Vec::new(),
            scores: Vec::new(),
            descriptors: Vec::new(),
        }
    }

    /// Return the number of keypoints.
    pub fn len(&self) -> usize {
        self.pixels.len()
    }

    /// Return `true` when there are no keypoints.
    pub fn is_empty(&self) -> bool {
        self.pixels.is_empty()
    }

    /// Return normalized LighterGlue coordinates `[(x - 400) / 400, (y - 300) / 400]`.
    pub fn normalized_keypoints(&self) -> Vec<f32> {
        self.model_pixels
            .iter()
            .flat_map(|p| [(p[0] - 400.0) / 400.0, (p[1] - 300.0) / 400.0])
            .collect()
    }

    /// Return up to `max_keypoints` keypoints with the highest confidence scores.
    pub fn truncate_top(&self, max_keypoints: usize) -> Self {
        let count = self
            .pixels
            .len()
            .min(self.model_pixels.len())
            .min(self.scores.len())
            .min(self.descriptors.len() / 64);
        if max_keypoints == 0 || count == 0 {
            return Self::empty();
        }
        let keep = max_keypoints.min(count);
        if self.scores[..count].windows(2).all(|w| w[0] >= w[1]) {
            return Self {
                pixels: self.pixels[..keep].to_vec(),
                model_pixels: self.model_pixels[..keep].to_vec(),
                scores: self.scores[..keep].to_vec(),
                descriptors: self.descriptors[..keep * 64].to_vec(),
            };
        }
        let mut indices: Vec<usize> = (0..count).collect();
        indices.sort_by(|&a, &b| {
            self.scores[b]
                .total_cmp(&self.scores[a])
                .then_with(|| a.cmp(&b))
        });
        indices.truncate(keep);
        let mut out = Self {
            pixels: Vec::with_capacity(keep),
            model_pixels: Vec::with_capacity(keep),
            scores: Vec::with_capacity(keep),
            descriptors: Vec::with_capacity(keep * 64),
        };
        for idx in indices {
            out.pixels.push(self.pixels[idx]);
            out.model_pixels.push(self.model_pixels[idx]);
            out.scores.push(self.scores[idx]);
            out.descriptors
                .extend_from_slice(&self.descriptors[idx * 64..(idx + 1) * 64]);
        }
        out
    }

    /// Match two feature sets with distinct mutual cosine nearest neighbours.
    pub fn mutual_matches(&self, other: &Self) -> Vec<PixelMatch> {
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
        if self.is_empty() || other.is_empty() {
            return Vec::new();
        }
        let empty = Nearest {
            index: 0,
            first: -2.0,
            second: -2.0,
        };
        let mut a = vec![empty; self.len()];
        let mut b = vec![empty; other.len()];
        for (i, da) in self.descriptors.as_chunks::<64>().0.iter().enumerate() {
            for (j, db) in other.descriptors.as_chunks::<64>().0.iter().enumerate() {
                let score: f32 = da.iter().zip(db).map(|(x, y)| x * y).sum();
                a[i].update(j, score);
                b[j].update(i, score);
            }
        }
        a.iter()
            .enumerate()
            .filter_map(|(i, n)| {
                let reverse = b.get(n.index)?;
                (n.distinct() && reverse.distinct() && reverse.index == i).then(|| PixelMatch {
                    reference: self.pixels[i].into(),
                    query: other.pixels[n.index].into(),
                })
            })
            .collect()
    }
}
