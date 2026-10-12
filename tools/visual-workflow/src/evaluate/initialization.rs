//! Perturb only the first observation to measure conditional sequence sensitivity.
use navigate_visual::PixelMatch;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum MatchOrder {
    #[default]
    Original,
    Reverse,
    QueryX,
    Shuffle0,
    Shuffle1,
    Shuffle2,
    Shuffle3,
}
impl MatchOrder {
    pub fn apply(self, sequence: u64, pairs: &mut [PixelMatch]) {
        if sequence != 0 {
            return;
        }
        match self {
            Self::Original => {}
            Self::Reverse => pairs.reverse(),
            Self::QueryX => pairs.sort_by(|a, b| {
                a.query
                    .x
                    .total_cmp(&b.query.x)
                    .then(a.query.y.total_cmp(&b.query.y))
            }),
            Self::Shuffle0 => shuffle(pairs, 0),
            Self::Shuffle1 => shuffle(pairs, 1),
            Self::Shuffle2 => shuffle(pairs, 2),
            Self::Shuffle3 => shuffle(pairs, 3),
        }
    }
}
fn shuffle(pairs: &mut [PixelMatch], mut seed: u64) {
    for i in (1..pairs.len()).rev() {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        pairs.swap(i, (seed >> 32) as usize % (i + 1));
    }
}

#[cfg(test)]
mod tests;
