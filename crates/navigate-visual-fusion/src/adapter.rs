//! Conversion of one visual estimate into one fusion observation.

use nalgebra::Matrix3;
use navigate_contract::{
    ClockDomainId, GeodeticPosition, MonotonicNanos, ObservationStamp, SensorClass,
    SourceComposition, SourceEpoch, SourceId, SymmetricCov3, WrappingSequence,
};
use navigate_fusion::{Observation, ObservationValue};
use navigate_visual::{Estimate, MapRevision};

use crate::{VisualErrorBudget, VisualFusionError};

/// Host-assigned identity of one camera capture stream.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VisualSourceIdentity {
    /// Source instance of the camera stream.
    pub source: SourceId,
    /// Restart epoch of the camera stream.
    pub epoch: SourceEpoch,
    /// Clock domain of the frame capture times.
    pub clock: ClockDomainId,
}

/// One position fix with the map release and the frame evidence it came from.
#[derive(Clone, Debug, PartialEq)]
pub struct VisualFix {
    /// Observation for [`navigate_fusion::NavigationFilter::ingest`].
    pub observation: Observation,
    /// Map release of the reference that located the frame.
    pub map: MapRevision,
    /// Evidence digest of the frame.
    pub observation_sha256: String,
}

/// Converts the visual estimates of one capture stream into position fixes.
///
/// One frame gives at most one fix. A second estimate from the same frame
/// evidence is refused, because re-evaluation is not a new measurement.
/// Use a new source for a new stream epoch.
#[derive(Debug)]
pub struct VisualFixSource {
    identity: VisualSourceIdentity,
    budget: VisualErrorBudget,
    last: Option<(u64, String)>,
}

impl VisualFixSource {
    /// Declare a capture stream and the error budget of its map and camera.
    ///
    /// # Errors
    /// Refuses a budget with a term that is not finite and positive.
    pub fn new(
        identity: VisualSourceIdentity,
        budget: VisualErrorBudget,
    ) -> Result<Self, VisualFusionError> {
        budget.validate()?;
        Ok(Self {
            identity,
            budget,
            last: None,
        })
    }

    /// Convert one accepted visual estimate into a fusion position fix.
    ///
    /// # Errors
    /// Refuses repeated frame evidence, frames out of capture order,
    /// positions outside geodetic range, and unusable covariances. A refusal
    /// does not change the state of the source.
    pub fn convert(&mut self, estimate: &Estimate) -> Result<VisualFix, VisualFusionError> {
        self.check_order(estimate)?;
        let position = estimate.pose.position;
        let [lat, lon, alt] = estimate.frame.geodetic(position);
        let geodetic = GeodeticPosition::new(
            lat.to_radians(),
            lon.to_radians(),
            self.budget.ellipsoid_height_m(alt),
        );
        if !geodetic.is_plausible() {
            return Err(VisualFusionError::ImplausiblePosition);
        }
        let covariance = ned_covariance(estimate, &self.budget)?;
        let stamp = ObservationStamp::new(
            self.identity.source,
            self.identity.epoch,
            WrappingSequence::new(low_bits(estimate.stamp.sequence)),
            MonotonicNanos::from_nanos(estimate.stamp.capture_time_ns),
            self.identity.clock,
        );
        let observation = Observation::new(
            stamp,
            ObservationValue::PositionFix {
                position: geodetic,
                covariance,
            },
            SourceComposition::of(SensorClass::VisualLandmark),
        );
        self.last = Some((
            estimate.stamp.capture_time_ns,
            estimate.observation_sha256.clone(),
        ));
        Ok(VisualFix {
            observation,
            map: estimate.map.clone(),
            observation_sha256: estimate.observation_sha256.clone(),
        })
    }

    fn check_order(&self, estimate: &Estimate) -> Result<(), VisualFusionError> {
        let Some((previous_ns, digest)) = &self.last else {
            return Ok(());
        };
        if *digest == estimate.observation_sha256 {
            return Err(VisualFusionError::RepeatedEvidence {
                observation_sha256: digest.clone(),
            });
        }
        if estimate.stamp.capture_time_ns <= *previous_ns {
            return Err(VisualFusionError::FrameOrder {
                previous_ns: *previous_ns,
                received_ns: estimate.stamp.capture_time_ns,
            });
        }
        Ok(())
    }
}

/// The low 32 bits keep the wrap order of a wrapping 64-bit frame sequence.
fn low_bits(sequence: u64) -> u32 {
    u32::try_from(sequence & u64::from(u32::MAX)).unwrap_or(u32::MAX)
}

/// Rotate the image-geometry position covariance from frame axes (east,
/// north, up) into north, east, down, and add the declared budget.
fn ned_covariance(
    estimate: &Estimate,
    budget: &VisualErrorBudget,
) -> Result<SymmetricCov3, VisualFusionError> {
    let enu: Matrix3<f64> = estimate
        .geometry_covariance
        .fixed_view::<3, 3>(0, 0)
        .into_owned();
    let axes = Matrix3::new(0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0);
    let mut ned = axes * enu * axes.transpose();
    let model_error_m = estimate.frame.model_error_m(estimate.pose.position);
    let (horizontal, vertical) = budget.variances_m2(model_error_m);
    ned[(0, 0)] += horizontal;
    ned[(1, 1)] += horizontal;
    ned[(2, 2)] += vertical;
    let usable = ned.iter().all(|v| v.is_finite())
        && ned.symmetric_eigen().eigenvalues.iter().all(|v| *v >= 0.0);
    if !usable {
        return Err(VisualFusionError::InvalidCovariance);
    }
    Ok(SymmetricCov3::from_upper_triangle([
        ned[(0, 0)],
        ned[(0, 1)],
        ned[(0, 2)],
        ned[(1, 1)],
        ned[(1, 2)],
        ned[(2, 2)],
    ]))
}

#[cfg(test)]
mod tests;
