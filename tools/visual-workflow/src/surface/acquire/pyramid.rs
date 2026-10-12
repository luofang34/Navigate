//! Bounded coarse-to-fine search retains multiple candidate regions.
use super::*;
use std::collections::BTreeSet;

#[derive(Clone, Copy)]
pub struct Options {
    pub side: u32,
    pub levels: usize,
    pub beam: usize,
}
impl Options {
    pub(super) fn plan(
        &self,
        size: (u32, u32),
        limit: usize,
    ) -> Result<(Vec<[u32; 4]>, usize), Error> {
        if !(1..=5).contains(&self.levels) || !(1..=16).contains(&self.beam) {
            return Err(Error::Input(
                "pyramid needs 1..=5 levels and 1..=16 parent windows".into(),
            ));
        }
        let divisor = 1 << (self.levels - 1);
        if !self.side.is_multiple_of(divisor * 2) || self.side / divisor < 32 {
            return Err(Error::Input(
                "each pyramid side must be even and at least 32 pixels".into(),
            ));
        }
        let windows = grid::windows(size, self.side, limit)?;
        let bound = windows.len() * 4 + (self.levels - 1) * self.beam * 9 * 4;
        if bound > limit || bound > 4096 {
            return Err(Error::Input(format!(
                "pyramid needs a candidate budget of {bound}; limit is {}",
                limit.min(4096)
            )));
        }
        Ok((windows, bound))
    }
}

pub fn run_blocking(
    config: &Config,
    root: &Path,
    input: &Path,
    output: &Path,
    options: Options,
    rectification_path: Option<&Path>,
) -> Result<(), Error> {
    grid::run_search_blocking(
        config,
        root,
        input,
        output,
        grid::Search::Pyramid(options),
        rectification_path,
    )
}

pub(super) fn search_blocking(
    trial: &Trial<'_>,
    engine: &mut Engine,
    mut windows: Vec<[u32; 4]>,
    options: Options,
) -> Result<Value, Error> {
    let started = Instant::now();
    let mut stages = Vec::new();
    let mut first = None;
    let mut candidates = 0_usize;
    for level in 0..options.levels {
        let mut rows = Vec::new();
        let mut scores = Vec::new();
        for area in &windows {
            let mut result = trial.windows_blocking(engine, &[*area], started, &mut Vec::new())?;
            if let Some((mut decision, time)) = window_pass(&result, "pyramid") {
                decision["level"] = json!(level);
                decision["window_index"] = json!(rows.len());
                first = Some((decision, time));
            }
            for row in &result {
                let score = row["candidates"]
                    .as_array()
                    .into_iter()
                    .flatten()
                    .filter_map(|c| c["matches"].as_u64())
                    .max()
                    .unwrap_or(0);
                scores.push((*area, score));
            }
            candidates = candidates.wrapping_add(4);
            rows.append(&mut result);
            if first.is_some() {
                break;
            }
        }
        let parents = parents(&scores, options.beam);
        stages.push(json!({"level":level,"windows":rows,"selected_parents":parents}));
        if first.is_some() || level + 1 == options.levels {
            break;
        }
        windows = children(&parents);
    }
    Ok(json!({"stages":stages,"accepted":first.is_some(),
        "first_pass":first.as_ref().map(|(decision,_)|decision),
        "search_to_first_pass_ms":first.map(|(_,time)|time),
        "search_total_ms":started.elapsed().as_secs_f64()*1000.,"candidates":candidates,
        "pyramid":{"side":options.side,"levels":options.levels,"beam":options.beam},
        "scope":"Bounded conditional search. First level is a full half-overlap grid. Rank windows by maximum raw match count across four rotations, with source order for ties. Retain at most beam parents, suppress intersection-over-union above 0.5, then examine nine half-size children per parent, centre first. Deduplicate exact child windows. A candidate score is not confidence or independent geographic evidence. Keep original geometry and optional raw pixel checks. Stop after all four rotations of the first passing window. No automatic full-grid fallback; budget exhaustion or pruning can miss a valid fix. Query encoded once. Full raster stays resident; memory accounting excludes models, runtime and result records."}))
}

fn parents(scores: &[([u32; 4], u64)], beam: usize) -> Vec<[u32; 4]> {
    let mut ranked: Vec<_> = scores.iter().enumerate().collect();
    ranked.sort_by_key(|(i, (_, score))| (std::cmp::Reverse(*score), *i));
    let mut chosen = Vec::new();
    for (_, (area, score)) in ranked {
        if *score == 0 || chosen.iter().any(|p| redundant(*p, *area)) {
            continue;
        }
        chosen.push(*area);
        if chosen.len() == beam {
            break;
        }
    }
    chosen
}
fn redundant(a: [u32; 4], b: [u32; 4]) -> bool {
    let w = (a[0] + a[2])
        .min(b[0] + b[2])
        .saturating_sub(a[0].max(b[0]));
    let h = (a[1] + a[3])
        .min(b[1] + b[3])
        .saturating_sub(a[1].max(b[1]));
    u64::from(w) * u64::from(h) * 3
        > u64::from(a[2]) * u64::from(a[3]) + u64::from(b[2]) * u64::from(b[3])
}
fn children(parents: &[[u32; 4]]) -> Vec<[u32; 4]> {
    let mut seen = BTreeSet::new();
    let mut result = Vec::new();
    for &[x, y, width, _] in parents {
        let side = width / 2;
        let step = side / 2;
        let mut windows = Vec::new();
        for dy in 0..3 {
            for dx in 0..3 {
                windows.push([x + dx * step, y + dy * step, side, side]);
            }
        }
        windows.sort_by_key(|w| {
            let dx = i64::from(w[0]) - i64::from(x + step);
            let dy = i64::from(w[1]) - i64::from(y + step);
            (dx * dx + dy * dy, w[1], w[0])
        });
        result.extend(windows.into_iter().filter(|w| seen.insert(*w)));
    }
    result
}

#[cfg(test)]
mod tests;
