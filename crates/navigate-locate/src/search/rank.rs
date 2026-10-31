//! Descriptors of the photo projections and the ranked places of the index.
use super::{Hypothesis, IndexSource, PlaceEncoder, SearchProgress};
use crate::LocateError;
use navigate_place::{
    Canvas, Descriptor, IndexManifest, Ranked, ScaleEntry, ScalePlan, SearchRequest, cell_weights,
    decode_shard, describe, rank, refine,
};

/// Score difference that lowers a neighbor's weight by a factor of e, in [`refine`].
const REFINE_TEMPERATURE: f32 = 0.02;

/// The query head bytes of a scale, if it has one.
async fn head<I: IndexSource>(
    index: &I,
    scale: &ScaleEntry,
) -> Result<Option<(Vec<u8>, usize, usize)>, LocateError> {
    match &scale.query_head {
        Some(h) => Ok(Some((
            index.read(&h.path).await.map_err(LocateError::Index)?,
            h.rows,
            h.columns,
        ))),
        None => Ok(None),
    }
}

fn scale_entry(manifest: &IndexManifest, side_m: f64) -> Result<&ScaleEntry, LocateError> {
    manifest
        .scales
        .iter()
        .find(|s| s.side_m == side_m)
        .ok_or(LocateError::Request("planned scale not in the index"))
}

pub(super) async fn describe_all<I: IndexSource, E: PlaceEncoder>(
    index: &I,
    manifest: &IndexManifest,
    scales: &[ScalePlan],
    encoder: &mut E,
    hypotheses: &[Hypothesis],
    canvases: &[Canvas],
    progress: &mut impl FnMut(SearchProgress),
) -> Result<Vec<Descriptor>, LocateError> {
    let mut heads = Vec::new();
    for part in scales {
        heads.push(head(index, scale_entry(manifest, part.side_m)?).await?);
    }
    let mut out = Vec::with_capacity(canvases.len());
    for (chunk, hyp) in canvases.chunks(8).zip(hypotheses.chunks(8)) {
        let tokens = encoder.tokens(chunk).await.map_err(LocateError::Encoder)?;
        for ((canvas, h), t) in chunk.iter().zip(hyp).zip(&tokens) {
            let head = heads[h.scale]
                .as_ref()
                .map(|(b, r, c)| (b.as_slice(), *r, *c));
            out.push(describe(t, &cell_weights(canvas), head)?);
        }
        progress(SearchProgress::Describing {
            done: out.len(),
            total: canvases.len(),
        });
    }
    Ok(out)
}

/// The ranked places of every scale, best score first, each with its hypothesis.
pub(super) async fn rank_all<I: IndexSource>(
    index: &I,
    manifest: &IndexManifest,
    scales: &[ScalePlan],
    request: &SearchRequest,
    hypotheses: &[Hypothesis],
    descriptors: &[Descriptor],
    progress: &mut impl FnMut(SearchProgress),
) -> Result<Vec<(Ranked, usize)>, LocateError> {
    let shard_total: usize = scales.iter().map(|s| s.cells.len()).sum();
    let mut shard_done = 0;
    let mut out = Vec::new();
    for (s, part) in scales.iter().enumerate() {
        let entry = scale_entry(manifest, part.side_m)?;
        let mut records = Vec::new();
        for cell in &part.cells {
            if let Some(shard) = entry.shards.iter().find(|e| e.cell == *cell) {
                let bytes = index.read(&shard.path).await.map_err(LocateError::Index)?;
                records.extend(decode_shard(&shard.path, &bytes)?.2);
            }
            shard_done += 1;
            progress(SearchProgress::Index {
                done: shard_done,
                total: shard_total,
            });
        }
        let members: Vec<usize> = (0..hypotheses.len())
            .filter(|&h| hypotheses[h].scale == s)
            .collect();
        let queries: Vec<Descriptor> = members.iter().map(|&h| descriptors[h].clone()).collect();
        let reach = members
            .iter()
            .map(|&h| hypotheses[h].footprint[0].hypot(hypotheses[h].footprint[1]))
            .fold(0.0, f64::max);
        let area = request.region.grown(reach + part.side_m / 2.0);
        let ranked = rank(
            &queries,
            &records,
            |c| area.contains(c),
            part.verify,
            part.side_m / 2.0,
        );
        out.extend(ranked.into_iter().map(|mut r| {
            r.centre = refine(
                &r,
                &queries[r.query],
                &records,
                part.side_m,
                REFINE_TEMPERATURE,
            );
            let h = members[r.query];
            (r, h)
        }));
    }
    out.sort_by(|a, b| b.0.score.total_cmp(&a.0.score));
    Ok(out)
}
