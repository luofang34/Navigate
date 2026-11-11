//! The declared sensor suite and the inputs of a navigator.

use navigate_camera::{CameraMount, ImagerModel};
use navigate_contract::{GeodeticPosition, MonotonicNanos, NedVelocity, SourceId, SymmetricCov3};
use navigate_fusion::{CorrelatedError, FogmBias, Observation};
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
    /// A camera that sees the terrain. The navigator matches its frames
    /// against the terrain map that the host loads.
    TerrainCamera,
    /// A three-axis magnetometer. It gives the heading of an alignment at
    /// rest.
    Magnetometer,
    /// A point that the host measured before the flight, for example the
    /// surveyed launch point. It is a measurement with a declared error,
    /// and its input says that the vehicle is at rest on the point.
    SurveyedPoint,
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
            Self::Magnetometer => "magnetometer",
            Self::SurveyedPoint => "surveyed point",
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
    /// With a correlated part, the position sigmas and the position
    /// covariance of each fix are the white part of the error only.
    Gnss {
        /// Horizontal position, each axis, metres.
        horizontal_sigma_m: f64,
        /// Vertical position, metres.
        vertical_sigma_m: f64,
        /// Velocity, each axis, m/s.
        velocity_sigma_m_s: f64,
        /// The time-correlated part of the position error, or `None` for
        /// a white error.
        correlated: Option<GnssCorrelatedError>,
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
    /// Terrain camera knowledge.
    TerrainCamera {
        /// One-sigma residual of the geometric calibration (focal length,
        /// principal point, lens model) as an angle, radians. Its position
        /// error is this angle times the range to the ground.
        calibration_sigma_rad: f64,
    },
    /// Magnetometer errors, each as an error of the heading that the
    /// sensor gives at rest.
    Magnetometer {
        /// White noise of one sample, radians.
        noise_sigma_rad: f64,
        /// Residual of the body field after calibration: hard and soft
        /// iron and the field of motor currents, radians. It is a bias,
        /// so more samples do not make it smaller.
        disturbance_sigma_rad: f64,
        /// Declination of the place, radians, east positive, from a
        /// field model. Navigate holds no field model.
        declination_rad: f64,
        /// One-sigma of that declination, radians.
        declination_sigma_rad: f64,
    },
    /// The inputs carry their own covariances.
    Declared,
}

/// The time-correlated part of a GNSS position error: ionosphere, orbit,
/// and clock residuals that successive fixes share. A first-order
/// Gauss–Markov process with the correlation time `tau_s` and the
/// stationary one-sigma errors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GnssCorrelatedError {
    /// Correlation time, seconds.
    pub tau_s: f64,
    /// Horizontal position, each axis, metres.
    pub horizontal_sigma_m: f64,
    /// Vertical position, metres.
    pub vertical_sigma_m: f64,
}

impl GnssCorrelatedError {
    /// Whether the correlation time is finite and positive and each sigma
    /// is finite and not negative.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        self.tau_s.is_finite()
            && self.tau_s > 0.0
            && [self.horizontal_sigma_m, self.vertical_sigma_m]
                .iter()
                .all(|sigma| sigma.is_finite() && *sigma >= 0.0)
    }

    /// The error as the filter takes it, on the north, east and down axes
    /// of a position fix.
    pub(crate) fn filter_error(&self) -> CorrelatedError {
        let (h, v) = (self.horizontal_sigma_m, self.vertical_sigma_m);
        CorrelatedError::new(self.tau_s, [h, h, v])
    }
}

impl ErrorBudget {
    /// The correlated part of a GNSS budget, or `None`.
    #[must_use]
    pub const fn gnss_correlated(&self) -> Option<GnssCorrelatedError> {
        match self {
            Self::Gnss { correlated, .. } => *correlated,
            _ => None,
        }
    }
}

/// One declared sensor.
#[derive(Debug, Clone, PartialEq)]
pub struct SensorSpec {
    /// Source identifier; unique in the suite.
    pub id: SourceId,
    /// What the sensor measures.
    pub kind: SensorKind,
    /// The imager model of a camera, with its spectral band: pinhole
    /// intrinsics `fx`, `fy`, `cx`, `cy` in pixels, where pixel centres have
    /// integer coordinates.
    pub imager: Option<ImagerModel>,
    /// The mount of a camera. The lever arm is the optical centre from the
    /// body reference point, in metres, in the body axes: forward, right,
    /// down. The rotation takes the body axes into the camera axes: `+X`
    /// right in the image, `+Y` down in the image, `+Z` along the optical
    /// axis. `CameraMount::rotated_forward` gives it from roll, pitch, and
    /// yaw of a forward camera. The body reference point is the point
    /// whose position the navigator gives.
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
    /// budget that does not match its kind, an implausible correlated GNSS
    /// error, a camera without an imager and a mount, and a rate that is
    /// not finite and positive.
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
                    | (SensorKind::TerrainCamera, ErrorBudget::TerrainCamera { .. })
                    | (SensorKind::Magnetometer, ErrorBudget::Magnetometer { .. })
                    | (SensorKind::SurveyedPoint, ErrorBudget::Declared)
                    | (SensorKind::ObservationSource, _)
            );
            let camera = matches!(a.kind, SensorKind::StarCamera | SensorKind::TerrainCamera);
            let correlated_ok = a
                .budget
                .gnss_correlated()
                .is_none_or(|correlated| correlated.is_plausible());
            if !budget_ok || !correlated_ok || (camera && (a.imager.is_none() || a.mount.is_none()))
            {
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

    /// Whether a sensor declares a time-correlated error.
    #[must_use]
    pub fn has_correlated_error(&self) -> bool {
        self.sensors
            .iter()
            .any(|s| s.budget.gnss_correlated().is_some())
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
    /// One frame of a terrain camera, as the sensor gave it. The
    /// acquisition time of the input is the middle of the exposure. The
    /// intrinsics, the mount, and the spectral band are those of the
    /// declared [`SensorSpec`]: the image is undistorted, its size is the
    /// size of the declared intrinsics, and its axes are those of
    /// `navigate_camera::Intrinsics` (columns to the right, rows down).
    /// The navigator takes each row at the acquisition time, so it does
    /// not model a rolling shutter.
    TerrainFrame {
        /// Image width, pixels.
        width: u32,
        /// Image height, pixels.
        height: u32,
        /// Luminance, eight bits for each pixel, row-major, top row first.
        luma: Vec<u8>,
    },
    /// The magnetic field in the body axes, tesla. The navigator uses its
    /// direction for the heading of an alignment at rest.
    MagneticField {
        /// The field, forward, right, down.
        field_body_t: [f64; 3],
    },
    /// The vehicle is at rest on a surveyed point.
    SurveyedPosition {
        /// The point.
        position: GeodeticPosition,
        /// Horizontal one-sigma of the survey, each axis, metres.
        horizontal_sigma_m: f64,
        /// Vertical one-sigma of the survey, metres.
        vertical_sigma_m: f64,
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
            Self::TerrainFrame { .. } => "terrain frame",
            Self::MagneticField { .. } => "magnetic field",
            Self::SurveyedPosition { .. } => "surveyed position",
            Self::Observation(_) => "observation",
        }
    }
}
