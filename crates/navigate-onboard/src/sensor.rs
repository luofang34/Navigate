//! The declared sensor suite and the inputs of a navigator.

use navigate_camera::{CameraMount, ImagerModel};
use navigate_contract::{GeodeticPosition, MonotonicNanos, NedVelocity, SourceId, SymmetricCov3};
use navigate_fusion::{FogmBias, Observation};
use navigate_startrack::StarCentroid;

use crate::OnboardError;

/// What a sensor measures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SensorKind {
    /// An inertial measurement unit: angle and velocity increments.
    Imu,
    /// A GNSS receiver: position and velocity fixes.
    Gnss,
    /// A barometric altimeter.
    Barometer,
    /// A star camera.
    StarCamera,
    /// A camera that sees the terrain.
    TerrainCamera,
    /// A sensor that gives fusion observations itself.
    ObservationSource,
}

impl SensorKind {
    /// The kind as text.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Imu => "IMU",
            Self::Gnss => "GNSS receiver",
            Self::Barometer => "barometer",
            Self::StarCamera => "star camera",
            Self::TerrainCamera => "terrain camera",
            Self::ObservationSource => "observation source",
        }
    }
}

/// The declared error budget of a sensor.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum ErrorBudget {
    /// IMU noise densities and bias models.
    Imu {
        /// Accelerometer white noise density, (m/s²)/√Hz.
        accel_noise_mps2_sqrt_hz: f64,
        /// Gyroscope white noise density, (rad/s)/√Hz.
        gyro_noise_rad_s_sqrt_hz: f64,
        /// Accelerometer bias, m/s².
        accel_bias: FogmBias,
        /// Gyroscope bias, rad/s.
        gyro_bias: FogmBias,
    },
    /// GNSS one-sigma errors, used when a fix carries no covariance.
    Gnss {
        /// Horizontal position, each axis, metres.
        horizontal_sigma_m: f64,
        /// Vertical position, metres.
        vertical_sigma_m: f64,
        /// Velocity, each axis, m/s.
        velocity_sigma_m_s: f64,
    },
    /// Barometric altitude errors.
    Barometer {
        /// White noise, metres.
        white_sigma_m: f64,
        /// Bias, metres.
        bias: FogmBias,
    },
    /// Star camera mount knowledge.
    StarCamera {
        /// One-sigma alignment error of the mount, radians.
        mount_alignment_sigma_rad: f64,
    },
    /// The inputs carry their own covariances.
    Declared,
}

/// One declared sensor.
#[derive(Debug, Clone, PartialEq)]
pub struct SensorSpec {
    /// Source identifier; unique in the suite.
    pub id: SourceId,
    /// What the sensor measures.
    pub kind: SensorKind,
    /// The imager model of a camera, with its spectral band.
    pub imager: Option<ImagerModel>,
    /// The mount of a camera, with its lever arm.
    pub mount: Option<CameraMount>,
    /// Nominal rate, Hz.
    pub rate_hz: f64,
    /// The error budget.
    pub budget: ErrorBudget,
}

/// The sensors of one vehicle, declared once.
#[derive(Debug, Clone, PartialEq)]
pub struct SensorSuite {
    sensors: Vec<SensorSpec>,
}

impl SensorSuite {
    /// A suite of `sensors`.
    ///
    /// # Errors
    ///
    /// Refuses a repeated identifier, a suite without exactly one IMU, a
    /// budget that does not match its kind, a camera without an imager and
    /// a mount, and a rate that is not finite and positive.
    pub fn new(sensors: Vec<SensorSpec>) -> Result<Self, OnboardError> {
        let invalid = |detail: String| OnboardError::Config {
            field: "sensor suite",
            detail,
        };
        for (i, a) in sensors.iter().enumerate() {
            if sensors.iter().skip(i + 1).any(|b| b.id == a.id) {
                return Err(invalid(format!("sensor {:?} is declared twice", a.id)));
            }
            if !(a.rate_hz.is_finite() && a.rate_hz > 0.0) {
                return Err(invalid(format!("sensor {:?} rate {}", a.id, a.rate_hz)));
            }
            let budget_ok = matches!(
                (a.kind, a.budget),
                (SensorKind::Imu, ErrorBudget::Imu { .. })
                    | (
                        SensorKind::Gnss,
                        ErrorBudget::Gnss { .. } | ErrorBudget::Declared
                    )
                    | (SensorKind::Barometer, ErrorBudget::Barometer { .. })
                    | (SensorKind::StarCamera, ErrorBudget::StarCamera { .. })
                    | (SensorKind::TerrainCamera | SensorKind::ObservationSource, _)
            );
            let camera = matches!(a.kind, SensorKind::StarCamera | SensorKind::TerrainCamera);
            if !budget_ok || (camera && (a.imager.is_none() || a.mount.is_none())) {
                return Err(invalid(format!(
                    "sensor {:?} ({}) budget or camera model",
                    a.id,
                    a.kind.name()
                )));
            }
        }
        if sensors.iter().filter(|s| s.kind == SensorKind::Imu).count() != 1 {
            return Err(invalid("a suite has exactly one IMU".into()));
        }
        Ok(Self { sensors })
    }

    /// The declared sensors.
    #[must_use]
    pub fn sensors(&self) -> &[SensorSpec] {
        &self.sensors
    }

    /// The sensor with identifier `id`.
    #[must_use]
    pub fn get(&self, id: SourceId) -> Option<&SensorSpec> {
        self.sensors.iter().find(|s| s.id == id)
    }

    /// The sensors of one kind.
    pub fn of_kind(&self, kind: SensorKind) -> impl Iterator<Item = &SensorSpec> {
        self.sensors.iter().filter(move |s| s.kind == kind)
    }
}

/// One input of a navigator.
#[derive(Debug, Clone, PartialEq)]
pub struct SensorInput {
    /// The declared sensor.
    pub sensor: SourceId,
    /// When the sensor acquired the data, on the navigator clock.
    pub acquired_at: MonotonicNanos,
    /// When the data reached the navigator, on the navigator clock.
    pub delivered_at: MonotonicNanos,
    /// The data.
    pub data: SensorData,
}

/// The data of one input.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum SensorData {
    /// IMU angle and velocity increments over `dt_s` that end at
    /// `acquired_at`, in the body axes.
    ImuIncrement {
        /// Interval, seconds.
        dt_s: f64,
        /// Angle increment, radians.
        delta_theta_rad: [f64; 3],
        /// Velocity increment, m/s.
        delta_velocity_m_s: [f64; 3],
    },
    /// A GNSS fix. A missing covariance takes the declared budget.
    GnssFix {
        /// Position.
        position: GeodeticPosition,
        /// Velocity, if the receiver gives one.
        velocity: Option<NedVelocity>,
        /// Position covariance, NED, m².
        position_cov: Option<SymmetricCov3>,
        /// Velocity covariance, NED, (m/s)².
        velocity_cov: Option<SymmetricCov3>,
    },
    /// Barometric altitude, metres.
    BaroAltitude {
        /// The altitude.
        altitude_m: f64,
    },
    /// The middle of a star camera exposure: the navigator takes its
    /// attitude prior for the frame here.
    StarExposureMid,
    /// The star centroids of one frame, acquired at mid-exposure.
    StarCentroids {
        /// The centroids.
        centroids: Vec<StarCentroid>,
    },
    /// An observation that the sensor made itself.
    Observation(Observation),
}

impl SensorData {
    /// The data kind as text.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::ImuIncrement { .. } => "IMU increment",
            Self::GnssFix { .. } => "GNSS fix",
            Self::BaroAltitude { .. } => "barometric altitude",
            Self::StarExposureMid => "star exposure",
            Self::StarCentroids { .. } => "star centroids",
            Self::Observation(_) => "observation",
        }
    }
}
