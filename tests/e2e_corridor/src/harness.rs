//! Test harness helpers, error definitions, and synthetic data generators.

use serde::{Deserialize, Serialize};
use std::error::Error;

/// Standard test result type returning a boxed error on failure.
pub type TestResult<T = ()> = Result<T, Box<dyn Error>>;

/// Verifies that a boolean condition is true, returning an error if false.
pub fn check(condition: bool, message: &str) -> TestResult {
    if !condition {
        return Err(message.into());
    }
    Ok(())
}

/// Checks whether two floating point values match within an epsilon.
pub fn approx_eq(a: f64, b: f64, eps: f64) -> bool {
    (a - b).abs() <= eps
}

/// Synthetic photo test case mirroring the upgraded `PhotoCase` contract.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SyntheticPhotoCase {
    /// Unique photo case identifier.
    pub id: String,
    /// Image dimensions [width, height].
    pub size: [u32; 2],
    /// Monotonic frame sequence number.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sequence: Option<u64>,
    /// Capture timestamp in seconds since epoch or flight start.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub time_s: Option<f64>,
    /// Valid ground fraction in range [0.0, 1.0].
    #[serde(default = "default_valid_fraction")]
    pub valid_fraction: f64,
}

fn default_valid_fraction() -> f64 {
    1.0
}

/// Generates a synthetic photo case for testing.
pub fn synthetic_photo_case(
    id: &str,
    time_s: Option<f64>,
    sequence: Option<u64>,
) -> SyntheticPhotoCase {
    SyntheticPhotoCase {
        id: id.to_owned(),
        size: [1920, 1080],
        sequence,
        time_s,
        valid_fraction: 1.0,
    }
}

/// Synthetic digital elevation model tile.
#[derive(Debug, Clone)]
pub struct SyntheticDem {
    /// Base surface elevation in meters above mean sea level.
    pub base_elevation_m: f64,
    /// North-South and East-West extent in meters.
    pub extent_m: f64,
}

/// Generates a synthetic DEM tile at the given elevation.
pub fn synthetic_dem_tile(base_elevation_m: f64) -> SyntheticDem {
    SyntheticDem {
        base_elevation_m,
        extent_m: 10_000.0,
    }
}
