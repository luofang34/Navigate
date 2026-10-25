use crate::Error;
use navigate_visual::{Frame, LocalizerConfig, PixelMatch, PointTracker};

#[derive(Clone, Copy)]
pub(super) struct Evidence {
    pub pair: PixelMatch,
    pub rotation: u8,
    pub origin: u64,
    pub source_pair_index: usize,
    pub age: usize,
}

#[derive(Default)]
pub(super) struct Propagation {
    pub pairs: Vec<Evidence>,
    pub expired_or_reset: usize,
    pub tracking_rejected: usize,
    pub attempted: bool,
}

pub(super) fn advance_blocking(
    evidence: Vec<Evidence>,
    previous: &Frame,
    current: &Frame,
    history: usize,
    tracker: &mut dyn PointTracker,
) -> Result<Propagation, Error> {
    if history == 1
        || current.camera != previous.camera
        || current.stamp.capture_time_ns <= previous.stamp.capture_time_ns
        || current.stamp.capture_time_ns - previous.stamp.capture_time_ns > 200_000_000
    {
        return Ok(Propagation {
            expired_or_reset: evidence.len(),
            ..Default::default()
        });
    }
    let previous_count = evidence.len();
    let retained: Vec<_> = evidence
        .into_iter()
        .filter(|p| p.age < history - 1)
        .collect();
    let expired_or_reset = previous_count - retained.len();
    if retained.is_empty() {
        return Ok(Propagation {
            expired_or_reset,
            ..Default::default()
        });
    }
    let input_count = retained.len();
    let points: Vec<_> = retained.iter().map(|p| p.pair.query).collect();
    let found = tracker.track_points_blocking(&previous.image, &current.image, &points)?;
    if found.len() != retained.len() {
        return Err(Error::Input("accumulation tracker feature count".into()));
    }
    let pairs: Vec<_> = retained
        .into_iter()
        .zip(found)
        .filter_map(|(mut p, q)| {
            q.filter(|q| {
                q.iter().all(|v| v.is_finite())
                    && q.x >= 0.0
                    && q.y >= 0.0
                    && q.x < f64::from(current.camera.width)
                    && q.y < f64::from(current.camera.height)
            })
            .map(|q| {
                p.pair.query = q;
                p.age = p.age.wrapping_add(1);
                p
            })
        })
        .collect();
    Ok(Propagation {
        tracking_rejected: input_count - pairs.len(),
        pairs,
        expired_or_reset,
        attempted: true,
    })
}

pub(super) fn merge(current: Vec<Evidence>, mut older: Vec<Evidence>) -> Vec<Evidence> {
    // New detections supersede correlated tracks; repeated points must not add support.
    older.sort_by_key(|p| p.age);
    let separation = LocalizerConfig::default().support_separation_px;
    let mut kept: Vec<Evidence> = Vec::new();
    let mut counts = [0usize; 4];
    for p in current.into_iter().chain(older) {
        let rotation = usize::from(p.rotation);
        if rotation >= counts.len() || counts[rotation] >= 4096 {
            continue;
        }
        if kept.iter().filter(|q| q.rotation == p.rotation).all(|q| {
            (q.pair.query - p.pair.query).norm() >= separation
                && (q.pair.reference - p.pair.reference).norm() >= separation
        }) {
            counts[rotation] = counts[rotation].wrapping_add(1);
            kept.push(p);
        }
    }
    kept
}

#[cfg(test)]
mod tests;
