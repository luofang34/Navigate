//! The reference navigator: Navigate's inertial filter on WGS84 with GNSS,
//! barometric, star-camera, terrain-camera, and pass-through aiding. The
//! filter has 22 states, and a correlated-error block when a sensor
//! declares a time-correlated error.

use std::collections::BTreeMap;

use navigate_contract::{
    ClockDomainId, MonotonicNanos, NavigationSolution, ObservationStamp, SensorClass,
    SourceComposition, SourceEpoch, SourceId, SymmetricCov3, WrappingSequence,
};
use navigate_fusion::{
    CorrelatedError, FogmBias, ImuIncrement, InertialConfig, IngestOutcome, Observation,
    ObservationValue,
};
use navigate_startrack::line_of_sight_observations;

use crate::{
    AcceptanceRecord, AcceptanceSink, Alignment, ErrorBudget, Navigator, OnboardError, Outcome,
    SensorData, SensorInput, SensorKind, SensorSuite,
};

mod filter;
mod stars;
#[cfg(feature = "terrain-camera")]
mod terrain;
pub use filter::NavigatorFilter;
pub use stars::{StarCatalogueConfig, own_prior};
#[cfg(feature = "terrain-camera")]
pub use terrain::{
    AccuracySource, AssumedAccuracy, BeliefSource, HostGeoid, LayerReport, MapReport,
    TerrainConfig, TerrainMapRequest, WorkingCamera, WorkingLimit,
};

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
    ) {
        let outcome = match self.filter.ingest(observation, input.delivered_at) {
            IngestOutcome::Accepted => Outcome::Accepted,
            IngestOutcome::Rejected(reason) => Outcome::Rejected(reason),
        };
        sink.record(AcceptanceRecord {
            sensor: input.sensor,
            acquired_at: input.acquired_at,
            measurement,
            outcome,
            context,
        });
    }

    fn offer_value(
        &mut self,
        input: &SensorInput,
        measurement: &'static str,
        value: ObservationValue,
        class: SensorClass,
        sink: &mut dyn AcceptanceSink,
    ) {
        let stamp = self.stamp(input.sensor, input.acquired_at);
        let observation = Observation::new(stamp, value, SourceComposition::of(class));
        self.offer(input, measurement, &observation, None, sink);
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
            IngestOutcome::Accepted => Ok(()),
            IngestOutcome::Rejected(reason) => Err(OnboardError::Refused {
                stage: "IMU increment",
                reason,
            }),
        }
    }

    fn gnss(&mut self, input: &SensorInput, budget: ErrorBudget, sink: &mut dyn AcceptanceSink) {
        let SensorData::GnssFix {
            position,
            velocity,
            position_cov,
            velocity_cov,
        } = &input.data
        else {
            return;
        };
        let (h, v, s, correlated) = match budget {
            ErrorBudget::Gnss {
                horizontal_sigma_m,
                vertical_sigma_m,
                velocity_sigma_m_s,
                correlated,
            } => (
                horizontal_sigma_m,
                vertical_sigma_m,
                velocity_sigma_m_s,
                correlated,
            ),
            _ => (f64::NAN, f64::NAN, f64::NAN, None),
        };
        let position_cov =
            position_cov.unwrap_or(SymmetricCov3::from_diagonal(h * h, h * h, v * v));
        let value = ObservationValue::PositionFix {
            position: *position,
            covariance: position_cov,
        };
        let stamp = self.stamp(input.sensor, input.acquired_at);
        let observation = Observation::new(stamp, value, SourceComposition::of(SensorClass::Gnss));
        // The velocity is from Doppler and stays white.
        let observation = match correlated {
            Some(c) => observation.with_correlated_error(CorrelatedError::new(
                c.tau_s,
                [
                    c.horizontal_sigma_m,
                    c.horizontal_sigma_m,
                    c.vertical_sigma_m,
                ],
            )),
            None => observation,
        };
        self.offer(input, "gnss_position", &observation, None, sink);
        if let Some(velocity) = velocity {
            let covariance =
                velocity_cov.unwrap_or(SymmetricCov3::from_diagonal(s * s, s * s, s * s));
            let value = ObservationValue::VelocityFix {
                velocity: *velocity,
                covariance,
            };
            self.offer_value(input, "gnss_velocity", value, SensorClass::Gnss, sink);
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

impl Navigator for ReferenceNavigator {
    fn suite(&self) -> &SensorSuite {
        &self.suite
    }

    fn align(&mut self, a: &Alignment) -> Result<(), OnboardError> {
        self.filter
            .initialize_surface(
                a.position,
                a.velocity,
                a.attitude,
                a.position_cov,
                a.velocity_cov,
                a.attitude_cov,
                a.at,
                SourceComposition::of(SensorClass::Inertial),
            )
            .map_err(|reason| OnboardError::Refused {
                stage: "alignment",
                reason,
            })
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
        match (&input.data, kind) {
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
        self.filter.tick(now)
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
