//! The reference navigator: Navigate's inertial filter on WGS84 with GNSS,
//! barometric, star-camera, terrain-camera, and pass-through aiding. The
//! filter has 22 states, and a correlated-error block when a sensor
//! declares a time-correlated error.

use std::collections::BTreeMap;

use navigate_contract::{
    ClockDomainId, MonotonicNanos, NavigationSolution, ObservationStamp, SensorClass,
    SourceComposition, SourceEpoch, SourceId, WrappingSequence,
};
use navigate_fusion::{
    FogmBias, ImuIncrement, InertialConfig, IngestOutcome, Observation, ObservationValue,
};
use navigate_startrack::line_of_sight_observations;

use crate::{
    AcceptanceRecord, AcceptanceSink, Alignment, AlignmentConfig, AlignmentRecord, AlignmentStatus,
    ErrorBudget, GnssIntegrityConfig, GnssIntegrityStatus, MagneticReference, Navigator,
    OnboardError, Outcome, SensorData, SensorInput, SensorKind, SensorSuite,
};

mod aligner;
mod filter;
mod gnss;
mod monitor;
mod stars;
#[cfg(feature = "terrain-camera")]
pub(crate) mod terrain;
pub use filter::NavigatorFilter;
pub use stars::{StarCatalogueConfig, own_prior};

use aligner::Aligner;
use monitor::GnssMonitor;
use stars::{Sighting, StarChannel};
#[cfg(feature = "terrain-camera")]
use terrain::TerrainMeans;

/// The reference composition. Each sensor of the suite feeds the one
/// filter; a sensor whose inputs do not come gives nothing.
#[derive(Debug, Clone)]
pub struct ReferenceNavigator {
    suite: SensorSuite,
    clock: ClockDomainId,
    filter: NavigatorFilter,
    /// Keyed by `SourceId::get`.
    sequences: BTreeMap<u32, WrappingSequence>,
    /// Keyed by `SourceId::get`.
    stars: BTreeMap<u32, StarChannel>,
    aligner: Aligner,
    /// What the alignment of the present state used.
    alignment: Option<AlignmentRecord>,
    /// GNSS positions in a row that the filter refused at its gate.
    gate_refusals: u32,
    /// The GNSS check, when the host started one.
    monitor: Option<Box<GnssMonitor>>,
    #[cfg(feature = "terrain-camera")]
    terrain: TerrainMeans,
}

/// The filter configuration of a suite: the IMU budget and the bias of the
/// first barometer.
fn inertial_config(suite: &SensorSuite) -> Result<InertialConfig, OnboardError> {
    let mut config = InertialConfig::default();
    let imu = suite.of_kind(SensorKind::Imu).next().map(|s| s.budget);
    let Some(ErrorBudget::Imu {
        accel_noise_mps2_sqrt_hz,
        gyro_noise_rad_s_sqrt_hz,
        accel_bias,
        gyro_bias,
    }) = imu
    else {
        return Err(OnboardError::Config {
            field: "sensor suite",
            detail: "no IMU budget".into(),
        });
    };
    config.accel_noise_sigma_mps2_sqrt_hz = accel_noise_mps2_sqrt_hz;
    config.gyro_noise_sigma_rad_s_sqrt_hz = gyro_noise_rad_s_sqrt_hz;
    config.accel_bias = accel_bias;
    config.gyro_bias = gyro_bias;
    if let Some(ErrorBudget::Barometer { bias, .. }) = suite
        .of_kind(SensorKind::Barometer)
        .next()
        .map(|s| s.budget)
    {
        config.baro_bias = bias;
    } else {
        config.baro_bias = FogmBias::new(0.0, 0.0);
    }
    Ok(config)
}

impl ReferenceNavigator {
    /// A navigator for `suite` on the clock `clock`. `stars` gives the star
    /// catalogue and the absolute time; it is required when the suite has a
    /// star camera.
    ///
    /// # Errors
    ///
    /// Refuses a suite without an IMU budget, a configuration whose model
    /// has no block in the layout, a star camera without a
    /// catalogue configuration, and a terrain camera that is not a visible
    /// camera with usable intrinsics. Without the feature `terrain-camera`
    /// it refuses each terrain camera
    /// ([`OnboardError::FeatureDisabled`]).
    pub fn new(
        suite: SensorSuite,
        clock: ClockDomainId,
        stars: Option<StarCatalogueConfig>,
    ) -> Result<Self, OnboardError> {
        let config = inertial_config(&suite)?;
        let mut channels = BTreeMap::new();
        for spec in suite.of_kind(SensorKind::StarCamera) {
            let catalogue = stars.ok_or_else(|| OnboardError::Config {
                field: "star catalogue",
                detail: format!(
                    "star camera {:?} needs the catalogue configuration",
                    spec.id
                ),
            })?;
            channels.insert(spec.id.get(), StarChannel::new(spec, catalogue)?);
        }
        #[cfg(not(feature = "terrain-camera"))]
        if let Some(spec) = suite.of_kind(SensorKind::TerrainCamera).next() {
            return Err(OnboardError::FeatureDisabled {
                sensor: spec.id,
                kind: spec.kind.name(),
                feature: "terrain-camera",
            });
        }
        Ok(Self {
            aligner: Aligner::new(&suite),
            alignment: None,
            gate_refusals: 0,
            monitor: None,
            #[cfg(feature = "terrain-camera")]
            terrain: TerrainMeans::new(&suite, clock)?,
            filter: NavigatorFilter::new(config, clock, suite.has_correlated_error()).map_err(
                |error| OnboardError::Config {
                    field: "filter layout",
                    detail: error.to_string(),
                },
            )?,
            suite,
            clock,
            sequences: BTreeMap::new(),
            stars: channels,
        })
    }

    /// The filter, for its configuration and rejection counters.
    #[must_use]
    pub fn filter(&self) -> &NavigatorFilter {
        &self.filter
    }

    fn stamp(&mut self, sensor: SourceId, acquired_at: MonotonicNanos) -> ObservationStamp {
        let sequence = self
            .sequences
            .entry(sensor.get())
            .or_insert_with(|| WrappingSequence::new(0));
        let stamp = ObservationStamp::new(
            sensor,
            SourceEpoch::new(0),
            *sequence,
            acquired_at,
            self.clock,
        );
        *sequence = sequence.next();
        stamp
    }

    fn skip_sequence(&mut self, sensor: SourceId, count: usize) {
        if let Some(sequence) = self.sequences.get_mut(&sensor.get()) {
            for _ in 0..count {
                *sequence = sequence.next();
            }
        }
    }

    fn offer(
        &mut self,
        input: &SensorInput,
        measurement: &'static str,
        observation: &Observation,
        context: Option<&'static str>,
        sink: &mut dyn AcceptanceSink,
    ) -> bool {
        let outcome = match self.filter.ingest(observation, input.delivered_at) {
            IngestOutcome::Accepted => Outcome::Accepted,
            IngestOutcome::Rejected(reason) => Outcome::Rejected(reason),
        };
        if let Some(monitor) = self.monitor.as_deref_mut() {
            monitor.mirror(observation, input.delivered_at);
        }
        let gated = matches!(
            outcome,
            Outcome::Rejected(navigate_fusion::RejectionReason::InnovationGate { .. })
        );
        sink.record(AcceptanceRecord {
            sensor: input.sensor,
            acquired_at: input.acquired_at,
            measurement,
            outcome,
            context,
        });
        gated
    }

    fn offer_value(
        &mut self,
        input: &SensorInput,
        measurement: &'static str,
        value: ObservationValue,
        class: SensorClass,
        sink: &mut dyn AcceptanceSink,
    ) -> bool {
        let stamp = self.stamp(input.sensor, input.acquired_at);
        let observation = Observation::new(stamp, value, SourceComposition::of(class));
        self.offer(input, measurement, &observation, None, sink)
    }

    fn imu(
        &mut self,
        input: &SensorInput,
        dt_s: f64,
        theta: [f64; 3],
        dv: [f64; 3],
    ) -> Result<(), OnboardError> {
        let stamp = self.stamp(input.sensor, input.acquired_at);
        let sample = ImuIncrement::new(
            stamp,
            SourceComposition::of(SensorClass::Inertial),
            dt_s,
            theta,
            dv,
        );
        match self
            .filter
            .ingest_imu_increment(&sample, input.delivered_at)
        {
            IngestOutcome::Accepted => {
                if let Some(monitor) = self.monitor.as_deref_mut() {
                    monitor.mirror_imu(&sample, input.delivered_at);
                }
                Ok(())
            }
            IngestOutcome::Rejected(reason) => Err(OnboardError::Refused {
                stage: "IMU increment",
                reason,
            }),
        }
    }

    fn star_frame(
        &mut self,
        input: &SensorInput,
        sink: &mut dyn AcceptanceSink,
    ) -> Result<(), OnboardError> {
        let SensorData::StarCentroids { centroids } = &input.data else {
            return Ok(());
        };
        let Some(channel) = self.stars.get_mut(&input.sensor.get()) else {
            return Err(OnboardError::UnknownSensor(input.sensor));
        };
        let sighting = channel.sight(centroids, input.acquired_at)?;
        let variance = channel.mount_variance_rad2;
        let (solution, mode) = match sighting {
            Sighting::Solved(solution, mode) => (solution, mode),
            Sighting::Unsolved(reason, mode) => {
                sink.record(AcceptanceRecord {
                    sensor: input.sensor,
                    acquired_at: input.acquired_at,
                    measurement: "star_line_of_sight",
                    outcome: Outcome::Unused(reason),
                    context: Some(mode),
                });
                return Ok(());
            }
        };
        let stamp = self.stamp(input.sensor, input.acquired_at);
        let observations =
            line_of_sight_observations(&solution, stamp, self.filter.frame(), variance)
                .map_err(|e| OnboardError::StarTrack(e.to_string()))?;
        // The observations of one frame take consecutive sequence numbers
        // from the stamp; the next frame continues after the last one.
        self.skip_sequence(input.sensor, observations.len().saturating_sub(1));
        for observation in &observations {
            self.offer(input, "star_line_of_sight", observation, Some(mode), sink);
        }
        Ok(())
    }
}

/// Whether the reference navigator takes `data` from a sensor of `kind`,
/// before its first state and after it.
const fn fits(data: &SensorData, kind: SensorKind) -> bool {
    let terrain = matches!(
        (data, kind),
        (SensorData::TerrainFrame { .. }, SensorKind::TerrainCamera)
    );
    (terrain && cfg!(feature = "terrain-camera"))
        || matches!(
            (data, kind),
            (SensorData::ImuIncrement { .. }, SensorKind::Imu)
                | (SensorData::GnssFix { .. }, SensorKind::Gnss)
                | (SensorData::BaroAltitude { .. }, SensorKind::Barometer)
                | (
                    SensorData::StarExposureMid | SensorData::StarCentroids { .. },
                    SensorKind::StarCamera,
                )
                | (SensorData::MagneticField { .. }, SensorKind::Magnetometer)
                | (
                    SensorData::SurveyedPosition { .. } | SensorData::KnownPosition { .. },
                    SensorKind::SurveyedPoint,
                )
                | (SensorData::DeclaredStart(_), SensorKind::Operator)
                | (SensorData::Observation(_), _)
        )
}

impl Navigator for ReferenceNavigator {
    fn suite(&self) -> &SensorSuite {
        &self.suite
    }

    fn align(&mut self, a: &Alignment) -> Result<(), OnboardError> {
        self.set_state(a, aligner::transfer_record(a))
    }

    fn alignment_status(&self) -> Option<AlignmentStatus> {
        Some(self.alignment_now())
    }

    fn set_magnetic_reference(&mut self, reference: MagneticReference) -> Result<(), OnboardError> {
        if !reference.is_usable() {
            return Err(OnboardError::Config {
                field: "magnetic reference",
                detail: format!("{reference:?}"),
            });
        }
        self.aligner.reference = Some(reference);
        Ok(())
    }

    fn set_alignment_config(&mut self, config: AlignmentConfig) -> Result<(), OnboardError> {
        self.configure_alignment(config)
    }

    fn realign(&mut self) -> Result<(), OnboardError> {
        self.forget_state();
        Ok(())
    }

    fn ingest(
        &mut self,
        input: &SensorInput,
        sink: &mut dyn AcceptanceSink,
    ) -> Result<(), OnboardError> {
        let spec = self
            .suite
            .get(input.sensor)
            .ok_or(OnboardError::UnknownSensor(input.sensor))?;
        let (kind, budget) = (spec.kind, spec.budget);
        let wrong = || OnboardError::WrongData {
            sensor: input.sensor,
            kind: kind.name(),
            data: input.data.name(),
        };
        if !fits(&input.data, kind) {
            return Err(wrong());
        }
        if !self.filter.is_initialized() {
            return self.aligning(input, budget, sink);
        }
        self.follow_filter(input.delivered_at);
        match (&input.data, kind) {
            (SensorData::MagneticField { .. }, SensorKind::Magnetometer)
            | (
                SensorData::SurveyedPosition { .. } | SensorData::KnownPosition { .. },
                SensorKind::SurveyedPoint,
            )
            | (SensorData::DeclaredStart(_), SensorKind::Operator) => {
                Self::alignment_only(input, sink);
                Ok(())
            }
            (
                SensorData::ImuIncrement {
                    dt_s,
                    delta_theta_rad,
                    delta_velocity_m_s,
                },
                SensorKind::Imu,
            ) => self.imu(input, *dt_s, *delta_theta_rad, *delta_velocity_m_s),
            (SensorData::GnssFix { .. }, SensorKind::Gnss) => {
                self.gnss(input, budget, sink);
                Ok(())
            }
            (SensorData::BaroAltitude { altitude_m }, SensorKind::Barometer) => {
                let ErrorBudget::Barometer { white_sigma_m, .. } = budget else {
                    return Err(wrong());
                };
                let value = ObservationValue::Altitude {
                    altitude_m: *altitude_m,
                    variance_m2: white_sigma_m * white_sigma_m,
                };
                self.offer_value(input, "baro_altitude", value, SensorClass::Barometric, sink);
                Ok(())
            }
            (SensorData::StarExposureMid, SensorKind::StarCamera) => {
                let solution = self.filter.tick(input.delivered_at);
                if let Some(channel) = self.stars.get_mut(&input.sensor.get()) {
                    channel.hold_prior(solution.as_ref());
                }
                Ok(())
            }
            (SensorData::StarCentroids { .. }, SensorKind::StarCamera) => {
                self.star_frame(input, sink)
            }
            #[cfg(feature = "terrain-camera")]
            (SensorData::TerrainFrame { .. }, SensorKind::TerrainCamera) => {
                self.terrain_frame(input, sink)
            }
            (SensorData::Observation(observation), _) => {
                self.offer(input, "observation", observation, None, sink);
                Ok(())
            }
            _ => Err(wrong()),
        }
    }

    fn solution(&mut self, now: MonotonicNanos) -> Option<NavigationSolution> {
        self.checked_solution(now)
    }

    fn set_gnss_integrity(
        &mut self,
        config: Option<GnssIntegrityConfig>,
    ) -> Result<(), OnboardError> {
        self.configure_gnss_integrity(config)
    }

    fn gnss_integrity(&self) -> Option<GnssIntegrityStatus> {
        self.gnss_integrity_status()
    }

    #[cfg(feature = "terrain-camera")]
    fn requirements(&self) -> Vec<crate::DataRequirement> {
        self.terrain.requirements()
    }

    #[cfg(feature = "terrain-camera")]
    fn load_report(&self) -> &[crate::ProductReport] {
        self.terrain.products()
    }

    fn snapshot(&self) -> Box<dyn Navigator> {
        Box::new(self.clone())
    }

    fn restore(&mut self, snapshot: &dyn Navigator) -> Result<(), OnboardError> {
        let other =
            snapshot
                .as_any()
                .downcast_ref::<Self>()
                .ok_or_else(|| OnboardError::Config {
                    field: "snapshot",
                    detail: "a snapshot of another navigator type".into(),
                })?;
        self.clone_from(other);
        Ok(())
    }

    fn as_any(&self) -> &dyn core::any::Any {
        self
    }
}

#[cfg(test)]
pub(crate) mod tests;
