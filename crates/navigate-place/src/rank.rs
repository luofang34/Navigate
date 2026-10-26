//! Ranking of index records against the query descriptors.
use crate::{Descriptor, Record, region::distance_m};

/// One ranked place.
#[derive(Clone, Debug, PartialEq)]
pub struct Ranked {
    /// Latitude and longitude of the crop centre, degrees.
    pub centre: [f64; 2],
    /// Best cosine over the query descriptors.
    pub score: f32,
    /// Index of the query descriptor that gave the best score.
    pub query: usize,
}

/// The `limit` best records among those that `eligible` accepts, scored by their best
/// cosine over `queries`. A place within `spacing_m` of a better place is dropped, so
/// that the list holds distinct places for verification.
pub fn rank<'a>(
    queries: &[Descriptor],
    records: impl IntoIterator<Item = &'a Record>,
    eligible: impl Fn([f64; 2]) -> bool,
    limit: usize,
    spacing_m: f64,
) -> Vec<Ranked> {
    let mut scored: Vec<Ranked> = records
        .into_iter()
        .filter(|r| eligible(r.centre))
        .filter_map(|r| {
            let (query, score) = queries
                .iter()
                .enumerate()
                .map(|(i, q)| {
                    (
                        i,
                        q.values
                            .iter()
                            .zip(&r.descriptor)
                            .map(|(a, b)| a * b)
                            .sum::<f32>(),
                    )
                })
                .max_by(|a, b| a.1.total_cmp(&b.1))?;
            Some(Ranked {
                centre: r.centre,
                score,
                query,
            })
        })
        .collect();
    scored.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Ranked> = Vec::with_capacity(limit);
    for candidate in scored {
        if kept.len() >= limit {
            break;
        }
        if kept
            .iter()
            .all(|k| distance_m(k.centre, candidate.centre) >= spacing_m)
        {
            kept.push(candidate);
        }
    }
    kept
}

#[cfg(test)]
mod tests;
