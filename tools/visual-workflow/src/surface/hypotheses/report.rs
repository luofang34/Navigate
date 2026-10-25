//! Keep conditional uncertainty distinct from disagreement between accepted poses.
use super::{Runner, tracking::Hypothesis};
use navigate_visual::Frame;
use serde_json::{Value, json};

pub(super) fn hypothesis(h: &Hypothesis, mut row: Value) -> Value {
    if row.is_null() {
        row = json!({});
    }
    row["id"] = json!(h.id);
    row["reference_window"] = json!(h.window);
    row["reference_quarter_turns"] = json!(h.rotation);
    row["accepted"] = json!(h.estimate.is_some());
    row["retained_tracks"] = json!(h.pairs.len());
    if let Some(e) = &h.estimate {
        row["pose"] = json!(navigate_recording::Pose::from(e.pose));
        row["observation_sha256"] = json!(e.observation_sha256);
        row["quality"] = json!({"inliers":e.quality.inliers,"spatial_support":e.quality.spatial_support,
            "occupied_cells":e.quality.occupied_cells,"rms_px":e.quality.reprojection_rms_px,
            "condition_number":e.quality.condition_number,"depth_matches":e.quality.depth_matches});
        row["geometry_covariance"] = json!(
            (0..6)
                .map(|r| (0..6)
                    .map(|c| e.geometry_covariance[(r, c)])
                    .collect::<Vec<_>>())
                .collect::<Vec<_>>()
        );
    }
    row
}
impl Runner {
    pub(super) fn pair_bytes(&self) -> usize {
        self.hypotheses
            .iter()
            .map(|h| h.pairs.capacity() * std::mem::size_of::<navigate_visual::PixelMatch>())
            .sum()
    }
    pub(super) fn record(&self, frame: &Frame, mut row: Value) -> Value {
        row["frame_id"] = json!(frame.stamp.sequence);
        row["capture_time_ns"] = json!(frame.stamp.capture_time_ns);
        row["observation_sha256"] = json!(frame.evidence_sha256());
        let source_bytes = self.reference.image.as_raw().len() * 5;
        let max_crop = self
            .hypotheses
            .iter()
            .filter_map(|h| h.window)
            .map(|w| w[2] as usize * w[3] as usize * 5)
            .max()
            .unwrap_or(0);
        row["memory"] = json!({"resident_source_bytes":source_bytes,"retained_pair_capacity_bytes":self.pair_bytes(),
            "temporary_crop_upper_bound_bytes":max_crop,"peak_source_and_crop_bytes":source_bytes+max_crop,
            "resident_previous_image_bytes":self.previous.image.as_raw().len(),"current_image_bytes":frame.image.as_raw().len(),
            "image_memory_scope":"The initial record borrows the previous image; subsequent records hold previous and current images."});
        let mut disagreements = Vec::new();
        for (index, a) in self.hypotheses.iter().enumerate() {
            for b in self.hypotheses.iter().skip(index + 1) {
                if let (Some(ea), Some(eb)) = (&a.estimate, &b.estimate) {
                    disagreements.push(json!({"a":a.id,"b":b.id,"position_m":(ea.pose.position-eb.pose.position).norm(),
                        "attitude_rad":ea.pose.orientation.angle_to(&eb.pose.orientation)}));
                }
            }
        }
        row["disagreements"] = json!(disagreements);
        row
    }
}
