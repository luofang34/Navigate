//! External position comparisons are diagnostics, separate from pose fitting.
use navigate_visual::Estimate;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct PositionCheck {
    pub source: String,
    pub latitude_longitude: [f64; 2],
    pub reported_horizontal_error_m: Option<f64>,
    pub associated_capture_time_ns: u64,
}

impl PositionCheck {
    pub fn valid(&self) -> bool {
        !self.source.trim().is_empty()
            && self.source.len() <= 1024
            && self.latitude_longitude.iter().all(|v| v.is_finite())
            && (-90.0..=90.0).contains(&self.latitude_longitude[0])
            && (-180.0..=180.0).contains(&self.latitude_longitude[1])
            && self
                .reported_horizontal_error_m
                .is_none_or(|v| v.is_finite() && v > 0.0)
    }

    pub fn compare(&self, capture_time_ns: u64, estimate: Option<&Estimate>) -> Value {
        self.compare_position(
            capture_time_ns,
            estimate.map(|e| e.frame.geodetic(e.pose.position)),
        )
    }

    pub fn compare_position(&self, capture_time_ns: u64, position: Option<[f64; 3]>) -> Value {
        let distance = position.map(|p| horizontal_distance(self.latitude_longitude, [p[0], p[1]]));
        json!({
            "source":self.source,"latitude_longitude":self.latitude_longitude,
            "reported_horizontal_error_m":self.reported_horizontal_error_m,
            "associated_capture_time_ns":self.associated_capture_time_ns,
            "capture_time_difference_ns":capture_time_ns.abs_diff(self.associated_capture_time_ns),
            "horizontal_disagreement_m":distance,
            "scope":"Great-circle separation on a sphere of radius 6371008.8 m. This check does not enter fitting, selection, or acceptance. Source accuracy, time association, and independence from map preparation or priors are not verified. No altitude comparison."
        })
    }
}

fn horizontal_distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    let [a_lat, a_lon] = a.map(f64::to_radians);
    let [b_lat, b_lon] = b.map(f64::to_radians);
    let h = ((a_lat - b_lat) / 2.0).sin().powi(2)
        + a_lat.cos() * b_lat.cos() * ((a_lon - b_lon) / 2.0).sin().powi(2);
    2.0 * 6_371_008.8 * h.clamp(0.0, 1.0).sqrt().asin()
}

#[cfg(test)]
mod tests;
