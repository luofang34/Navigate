//! The flight of the GNSS tests over terrain: the suite, the receiver
//! error, and the inputs of each step.

use navigate_contract::{GeodeticPosition, NedVelocity, SourceId};
use navigate_fusion::FogmBias;

use super::super::fixes::{HEIGHT_M, belly};
use super::super::flight::{CAMERA, CLOCK, Flight, IMU, RATE, at, imager};
use super::super::world::{CENTRE, World, WorldSpec};
use crate::reference::terrain::MEASUREMENT;
use crate::{
    AcceptanceRecord, Alignment, ErrorBudget, Navigator, Outcome, ReferenceNavigator, SensorData,
    SensorInput, SensorKind, SensorSpec, SensorSuite,
};

pub(super) const GNSS: SourceId = SourceId::new(2);
pub(super) const BARO: SourceId = SourceId::new(3);
/// GNSS fixes at 5 Hz.
pub(super) const GNSS_EVERY: u64 = 10;
pub(super) const GNSS_SIGMA_M: [f64; 3] = [1.0, 1.0, 2.0];
pub(super) const GNSS_VELOCITY_SIGMA: f64 = 0.1;
pub(super) const SPEED_M_S: f64 = 40.0;

/// The receiver error that leaves the truth: from `from_step` the
/// position moves east at `rate_m_s` until `to_step`, and holds or drops
/// its offset after it.
#[derive(Debug, Clone, Copy)]
pub(super) struct Ramp {
    pub from_step: u64,
    pub to_step: u64,
    pub rate_m_s: f64,
    /// Whether the Doppler velocity moves with the position.
    pub velocity_too: bool,
    /// Whether the offset goes back to zero at `to_step`.
    pub drops: bool,
}

impl Ramp {
    /// The east offset of the position and of the velocity at `step`.
    fn offset(&self, step: u64) -> (f64, f64) {
        let seconds = |steps: u64| steps as f64 / RATE as f64;
        if step < self.from_step {
            (0.0, 0.0)
        } else if step < self.to_step {
            let rate = if self.velocity_too {
                self.rate_m_s
            } else {
                0.0
            };
            (self.rate_m_s * seconds(step - self.from_step), rate)
        } else if self.drops {
            (0.0, 0.0)
        } else {
            (self.rate_m_s * seconds(self.to_step - self.from_step), 0.0)
        }
    }
}

/// SplitMix64 finalizer.
pub(super) fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A standard normal draw for `(seed, index)`.
pub(super) fn normal(seed: u64, index: u64) -> f64 {
    let uniform =
        |k: u64| ((mix((seed << 40) ^ (index * 2 + k)) >> 11) as f64 + 0.5) / (1_u64 << 53) as f64;
    (-2.0 * uniform(0).ln()).sqrt() * (core::f64::consts::TAU * uniform(1)).cos()
}

/// The flight of the tests: north through the world at 40 m/s.
pub(super) fn flight(world: &WorldSpec) -> Flight {
    let south = CENTRE[0] - ((world.radius_m - 400.0) / 6_364_000.0_f64).to_degrees();
    Flight {
        start: [south, CENTRE[1], HEIGHT_M],
        speed_m_s: SPEED_M_S,
    }
}

/// A world with map cells of the size of the flight packages: zoom 16,
/// about 450 m on a side.
pub(super) fn world_spec() -> WorldSpec {
    WorldSpec {
        radius_m: 4_000.0,
        correlation_zoom: 16,
        ..WorldSpec::default()
    }
}

/// An IMU, a GNSS receiver, a barometer, and the belly camera.
pub(super) fn suite() -> SensorSuite {
    let spec = |id, kind, rate_hz, budget| SensorSpec {
        id,
        kind,
        imager: None,
        mount: None,
        rate_hz,
        budget,
    };
    SensorSuite::new(vec![
        spec(
            IMU,
            SensorKind::Imu,
            RATE as f64,
            ErrorBudget::Imu {
                accel_noise_mps2_sqrt_hz: 6e-4,
                gyro_noise_rad_s_sqrt_hz: 3e-6,
                accel_bias: FogmBias::new(3_600.0, 5e-4),
                gyro_bias: FogmBias::new(3_600.0, 5e-6),
            },
        ),
        spec(
            GNSS,
            SensorKind::Gnss,
            5.0,
            ErrorBudget::Gnss {
                horizontal_sigma_m: GNSS_SIGMA_M[0],
                vertical_sigma_m: GNSS_SIGMA_M[2],
                velocity_sigma_m_s: GNSS_VELOCITY_SIGMA,
                correlated: None,
            },
        ),
        spec(
            BARO,
            SensorKind::Barometer,
            1.0,
            ErrorBudget::Barometer {
                white_sigma_m: 0.5,
                bias: FogmBias::new(3_600.0, 3.0),
            },
        ),
        SensorSpec {
            imager: Some(imager()),
            mount: Some(belly()),
            ..spec(
                CAMERA,
                SensorKind::TerrainCamera,
                1.0,
                ErrorBudget::TerrainCamera {
                    calibration_sigma_rad: 2e-4,
                },
            )
        },
    ])
    .unwrap()
}

/// A navigator of [`suite`] aligned at the truth of `flight`.
pub(super) fn navigator(flight: &Flight) -> ReferenceNavigator {
    let mut navigator = ReferenceNavigator::new(suite(), CLOCK, None).unwrap();
    let [lat, lon, h] = flight.start;
    navigator
        .align(&Alignment {
            position: GeodeticPosition::new(lat.to_radians(), lon.to_radians(), h),
            velocity: NedVelocity::new(flight.speed_m_s, 0.0, 0.0),
            attitude: navigate_contract::AttitudeQuaternion::new(1.0, 0.0, 0.0, 0.0),
            position_cov: navigate_contract::SymmetricCov3::from_diagonal(4.0, 4.0, 4.0),
            velocity_cov: navigate_contract::SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
            attitude_cov: navigate_contract::SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-5),
            at: at(0),
            imu_bias_prior: None,
        })
        .unwrap();
    navigator
}

/// The GNSS fix of `step` with the error of `ramp` and seeded noise.
pub(super) fn gnss_fix(flight: &Flight, step: u64, ramp: Option<&Ramp>, seed: u64) -> SensorInput {
    let [lat, lon, h] = flight.body(step);
    let (east_m, east_m_s) = ramp.map_or((0.0, 0.0), |r| r.offset(step));
    let noise = |axis: u64| normal(seed, step * 8 + axis);
    let north = noise(0) * GNSS_SIGMA_M[0];
    let east = east_m + noise(1) * GNSS_SIGMA_M[1];
    let up = noise(2) * GNSS_SIGMA_M[2];
    let east_radius = 6_378_137.0 * lat.to_radians().cos();
    SensorInput {
        sensor: GNSS,
        acquired_at: at(step),
        delivered_at: at(step),
        data: SensorData::GnssFix {
            position: GeodeticPosition::new(
                lat.to_radians() + north / 6_364_000.0,
                lon.to_radians() + east / east_radius,
                h + up,
            ),
            velocity: Some(NedVelocity::new(
                flight.speed_m_s + noise(3) * GNSS_VELOCITY_SIGMA,
                east_m_s + noise(4) * GNSS_VELOCITY_SIGMA,
                noise(5) * GNSS_VELOCITY_SIGMA,
            )),
            position_cov: None,
            velocity_cov: None,
        },
    }
}

/// The inputs of one step: the increment, then the GNSS fix, the
/// barometric altitude, and the camera frame at their rates.
pub(super) fn step_inputs(
    flight: &Flight,
    world: Option<&World>,
    step: u64,
    ramp: Option<&Ramp>,
    seed: u64,
) -> Vec<SensorInput> {
    let mut inputs = vec![flight.increment(step)];
    if step.is_multiple_of(GNSS_EVERY) {
        inputs.push(gnss_fix(flight, step, ramp, seed));
    }
    if step.is_multiple_of(RATE) {
        inputs.push(SensorInput {
            sensor: BARO,
            acquired_at: at(step),
            delivered_at: at(step),
            data: SensorData::BaroAltitude {
                altitude_m: flight.body(step)[2] + normal(seed, step * 8 + 6) * 0.5,
            },
        });
        if let Some(world) = world {
            inputs.push(flight.frame(world, &belly(), step, 0));
        }
    }
    inputs
}

/// The count of terrain frames with each outcome: accepted, refused at
/// the gate, and not offered.
pub(super) fn terrain_counts(records: &[AcceptanceRecord]) -> (usize, usize, usize) {
    let of = |f: &dyn Fn(&Outcome) -> bool| {
        records
            .iter()
            .filter(|r| r.sensor == CAMERA && r.measurement == MEASUREMENT && f(&r.outcome))
            .count()
    };
    (
        of(&|o| *o == Outcome::Accepted),
        of(&|o| matches!(o, Outcome::Rejected(_))),
        of(&|o| matches!(o, Outcome::Unused(_))),
    )
}
