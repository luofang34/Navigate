//! Bounded appearance-keyframe correction of existing map feature identities.
use crate::Error;
use nalgebra::Vector2;
use navigate_visual::{Frame, PixelMatch, PointTracker};
use serde_json::{Value, json};
use std::{collections::BTreeMap, time::Instant};

type Identity = (u64, u64);
struct Keyframe {
    frame: Frame,
    observation: String,
    locations: BTreeMap<Identity, Option<Vector2<f64>>>,
    age: usize,
}
pub(crate) struct Anchoring {
    interval: Option<usize>,
    keyframe: Option<Keyframe>,
}
pub(crate) struct Correction {
    pub pairs: Vec<PixelMatch>,
    pub report: Value,
}
pub(crate) fn valid_interval(interval: Option<usize>) -> bool {
    interval.is_none_or(|n| (2..=30).contains(&n))
}
fn identity(p: &PixelMatch) -> Identity {
    (p.reference.x.to_bits(), p.reference.y.to_bits())
}
impl Anchoring {
    pub fn new(interval: Option<usize>) -> Self {
        Self {
            interval,
            keyframe: None,
        }
    }
    pub fn clear(&mut self) {
        self.keyframe = None;
    }
    pub fn observe(&mut self, frame: &Frame, pairs: &[PixelMatch], map_refresh: bool) {
        let Some(interval) = self.interval else {
            return;
        };
        if let Some(keyframe) = &mut self.keyframe {
            keyframe.age = keyframe.age.wrapping_add(1);
            if !map_refresh && keyframe.age < interval {
                return;
            }
        }
        let mut locations = BTreeMap::new();
        for pair in pairs {
            locations
                .entry(identity(pair))
                .and_modify(|v| *v = None)
                .or_insert(Some(pair.query));
        }
        self.keyframe = Some(Keyframe {
            frame: Frame {
                stamp: frame.stamp,
                camera: frame.camera,
                image: frame.image.clone(),
            },
            observation: frame.evidence_sha256(),
            locations,
            age: 0,
        });
    }
    pub fn correct_blocking(
        &self,
        frame: &Frame,
        predicted: &[PixelMatch],
        tracker: &mut dyn PointTracker,
    ) -> Result<Option<Correction>, Error> {
        let Some(keyframe) = &self.keyframe else {
            return Ok(None);
        };
        let started = Instant::now();
        let mut counts = BTreeMap::<Identity, usize>::new();
        for p in predicted {
            counts
                .entry(identity(p))
                .and_modify(|n| *n = n.wrapping_add(1))
                .or_insert(1);
        }
        let bound: Vec<_> = predicted
            .iter()
            .enumerate()
            .filter_map(|(slot, p)| {
                (counts.get(&identity(p)) == Some(&1)).then_some(())?;
                let point = (*keyframe.locations.get(&identity(p))?)?;
                Some((slot, point, p.query))
            })
            .collect();
        let points: Vec<_> = bound.iter().map(|p| p.1).collect();
        let initial: Vec<_> = bound.iter().map(|p| p.2).collect();
        let refined = tracker.refine_points_blocking(
            &keyframe.frame.image,
            &frame.image,
            &points,
            &initial,
        )?;
        if refined.len() != bound.len() {
            return Err(Error::Input(
                "keyframe correction changed slot count".into(),
            ));
        }
        let mut pairs = predicted.to_vec();
        let mut corrected = 0_usize;
        let mut outside = 0_usize;
        let mut failed = 0_usize;
        for ((slot, _, prediction), result) in bound.iter().zip(refined) {
            match result {
                Some(point) if (point - prediction).norm() <= 2.0 => {
                    pairs[*slot].query = point;
                    corrected = corrected.wrapping_add(1);
                }
                Some(_) => outside = outside.wrapping_add(1),
                None => failed = failed.wrapping_add(1),
            }
        }
        Ok(Some(Correction {
            pairs,
            report: json!({
                "keyframe_index":keyframe.frame.stamp.sequence,"keyframe_capture_time_ns":keyframe.frame.stamp.capture_time_ns,
                "keyframe_observation_sha256":keyframe.observation,
                "age_frames":keyframe.age.wrapping_add(1),"input_points":predicted.len(),"bound_points":bound.len(),
                "corrected_points":corrected,"failed_alignment":failed,"outside_two_pixels":outside,
                "unbound_or_ambiguous":predicted.len()-bound.len(),"retained_prediction_points":predicted.len()-corrected,
                "keyframe_image_bytes":keyframe.frame.image.as_raw().len(),"patch_ms":started.elapsed().as_secs_f64()*1000.0,
                "scope":"Full-resolution local alignment initialized by adjacent-frame predictions; forward/reverse checks and a 2px correction bound. Failed, unbound or ambiguous slots retain their prediction. Same map identities, no new independent map evidence. Corrected geometry must pass unchanged gates; caller records whether it was selected."
            }),
        }))
    }
}
#[cfg(test)]
mod tests;
