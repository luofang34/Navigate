//! A GNSS receiver whose position leaves the truth at a constant rate,
//! on a flight over the test world with the real matcher: what the
//! navigator does with it.

use nalgebra::Vector3;
use navigate_contract::{GeodeticPosition, NedVelocity, SourceId};
use navigate_fusion::FogmBias;

use super::fixes::{FIX_BOUND_M, HEIGHT_M, belly};
use super::flight::{CAMERA, CLOCK, Flight, IMU, RATE, at, imager, request, show_events};
use super::world::{CENTRE, World, WorldSpec};
use crate::reference::monitor::GNSS_EXCLUDED;
use crate::reference::terrain::MEASUREMENT;
use crate::{
    AcceptanceRecord, Alignment, ErrorBudget, GnssIntegrityConfig, GnssIntegrityState, Navigator,
    Outcome, ReferenceNavigator, SensorData, SensorInput, SensorKind, SensorSpec, SensorSuite,
};
use navigate_contract::SolutionQuality;

const GNSS: SourceId = SourceId::new(2);
const BARO: SourceId = SourceId::new(3);
/// GNSS fixes at 5 Hz.
const GNSS_EVERY: u64 = 10;
const GNSS_SIGMA_M: [f64; 3] = [1.0, 1.0, 2.0];
const GNSS_VELOCITY_SIGMA: f64 = 0.1;
const SPEED_M_S: f64 = 40.0;

/// The receiver error that leaves the truth: from `from_step` the
/// position moves east at `rate_m_s` until `to_step`, and holds or drops
/// its offset after it.
#[derive(Debug, Clone, Copy)]
struct Ramp {
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
    fn at(&self, step: u64) -> (f64, f64) {
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
fn mix(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A standard normal draw for `(seed, index)`.
fn normal(seed: u64, index: u64) -> f64 {
    let uniform =
        |k: u64| ((mix((seed << 40) ^ (index * 2 + k)) >> 11) as f64 + 0.5) / (1_u64 << 53) as f64;
    (-2.0 * uniform(0).ln()).sqrt() * (core::f64::consts::TAU * uniform(1)).cos()
}

/// The flight of the tests: north through the world at 40 m/s.
fn flight(world: &WorldSpec) -> Flight {
    let south = CENTRE[0] - ((world.radius_m - 400.0) / 6_364_000.0_f64).to_degrees();
    Flight {
        start: [south, CENTRE[1], HEIGHT_M],
        speed_m_s: SPEED_M_S,
    }
}

/// A world with map cells of the size of the flight packages: zoom 16,
/// about 450 m on a side.
fn world_spec() -> WorldSpec {
    WorldSpec {
        radius_m: 4_000.0,
        correlation_zoom: 16,
        ..WorldSpec::default()
    }
}

/// An IMU, a GNSS receiver, a barometer, and the belly camera.
fn suite() -> SensorSuite {
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
fn navigator(flight: &Flight) -> ReferenceNavigator {
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
fn gnss_fix(flight: &Flight, step: u64, ramp: Option<&Ramp>, seed: u64) -> SensorInput {
    let [lat, lon, h] = flight.body(step);
    let (east_m, east_m_s) = ramp.map_or((0.0, 0.0), |r| r.at(step));
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
fn step_inputs(
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
fn terrain_counts(records: &[AcceptanceRecord]) -> (usize, usize, usize) {
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

/// What one flight with the check saw.
#[derive(Debug, Default)]
struct Seen {
    /// Seconds of flight at the first doubt, at the exclusion, and when
    /// the receiver was taken again.
    doubted_s: Option<f64>,
    excluded_s: Option<f64>,
    taken_again_s: Option<f64>,
    /// Largest horizontal error of the solution up to the exclusion, and
    /// the largest exclusion bound that the status stated up to it.
    worst_before_m: f64,
    largest_bound_m: f64,
    /// Largest horizontal error of the solution after the exclusion.
    worst_after_m: f64,
    /// Whether each doubted solution stated a quality below `Good` and a
    /// one-sigma that holds its error within three sigma.
    doubt_is_stated: bool,
    end_error_m: f64,
    end_state: Option<GnssIntegrityState>,
    doubted_fixes: u32,
    worst_chi2: f64,
    records: Vec<AcceptanceRecord>,
}

/// Flies `seconds` over `world` with the check, with the receiver error
/// of `ramp`.
fn watch(world: &World, seconds: u64, ramp: Option<&Ramp>, seed: u64) -> Seen {
    let flight = flight(&world.spec);
    let mut navigator = navigator(&flight);
    navigator
        .set_gnss_integrity(Some(GnssIntegrityConfig::default()))
        .unwrap();
    navigator
        .load_terrain_map(std::sync::Arc::clone(&world.store), &request(world), None)
        .unwrap();
    let mut seen = Seen {
        doubt_is_stated: true,
        ..Seen::default()
    };
    for step in 1..=seconds * RATE {
        for input in step_inputs(&flight, Some(world), step, ramp, seed) {
            navigator.ingest(&input, &mut seen.records).unwrap();
        }
        if !step.is_multiple_of(GNSS_EVERY) {
            continue;
        }
        let status = navigator.gnss_integrity().unwrap();
        let solution = navigator.solution(at(step)).unwrap();
        let error: Vector3<f64> = flight.error_m(&solution, step);
        let (error_m, t_s) = (error.xy().norm(), step as f64 / RATE as f64);
        let last = status.last.unwrap();
        seen.worst_chi2 = seen.worst_chi2.max(last.chi2);
        match status.state {
            GnssIntegrityState::Excluded => {
                seen.excluded_s.get_or_insert(t_s);
            }
            GnssIntegrityState::Disagreeing => {
                seen.doubted_s.get_or_insert(t_s);
                seen.doubted_fixes += 1;
                let integrity = solution.integrity;
                seen.doubt_is_stated &= integrity.quality != SolutionQuality::Good
                    && error_m <= 3.0 * integrity.horizontal_1sigma_m;
            }
            GnssIntegrityState::Consistent if seen.excluded_s.is_some() => {
                seen.taken_again_s.get_or_insert(t_s);
            }
            GnssIntegrityState::Consistent => {}
        }
        if seen.excluded_s.is_none() {
            seen.worst_before_m = seen.worst_before_m.max(error_m);
            seen.largest_bound_m = seen.largest_bound_m.max(last.exclusion_bound_m);
        } else {
            seen.worst_after_m = seen.worst_after_m.max(error_m);
        }
        seen.end_error_m = error_m;
        seen.end_state = Some(status.state);
    }
    seen
}

/// The drift starts 30 s into the flight.
const FROM_S: u64 = 30;
/// The receiver is at the truth again 110 s into the flight.
const TO_S: u64 = 110;

fn ramp(rate_m_s: f64, drops: bool) -> Ramp {
    Ramp {
        from_step: FROM_S * RATE,
        to_step: TO_S * RATE,
        rate_m_s,
        velocity_too: true,
        drops,
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore = "needs an optimized build")]
fn a_drifting_receiver_is_doubted_then_excluded_and_taken_again_when_it_agrees() {
    show_events();
    let world = World::build(&world_spec());
    // Measured, from the start of the drift: 0.5 m/s is doubted after
    // 31.2 s and excluded after 42.6 s with at most 21.5 m of error;
    // 1 m/s after 20.2 s and 31.6 s with 7.4 m; 2 m/s after 8.2 s and
    // 17.2 s with 1.6 m. The filter refuses the two faster receivers at
    // its innovation gate before the exclusion, so they move it less.
    for (rate_m_s, within_s, error_m) in [(0.5, 60.0, 30.0), (1.0, 45.0, 15.0), (2.0, 30.0, 10.0)] {
        let seen = watch(&world, 170, Some(&ramp(rate_m_s, true)), 1);
        let (doubted_s, excluded_s, worst_m) = (
            seen.doubted_s.unwrap() - FROM_S as f64,
            seen.excluded_s.unwrap() - FROM_S as f64,
            seen.worst_before_m,
        );
        tracing::info!(
            rate_m_s,
            doubted_s,
            excluded_s,
            worst_m,
            "a drifting receiver"
        );
        assert!(0.0 < doubted_s && doubted_s < excluded_s, "{seen:?}");
        assert!(excluded_s < within_s, "{rate_m_s}: {excluded_s}");
        // The error stays under the bound that the status stated.
        assert!(worst_m < error_m && worst_m < seen.largest_bound_m);
        assert!(seen.doubt_is_stated);
        // After the exclusion the solution is that of the terrain fixes.
        assert!(seen.worst_after_m < FIX_BOUND_M, "{}", seen.worst_after_m);
        // The receiver agrees again from `TO_S`, and the navigator takes
        // it after the configured time, with a terrain fix in that time.
        let again_s = seen.taken_again_s.unwrap() - TO_S as f64;
        let wait_s = GnssIntegrityConfig::default().readmission_s;
        assert!((wait_s..wait_s + 15.0).contains(&again_s), "{again_s}");
        assert!(seen.end_error_m < 3.0, "{}", seen.end_error_m);
        // The values of the excluded receiver have their records.
        let unused = |measurement| {
            seen.records
                .iter()
                .filter(|r| r.sensor == GNSS && r.measurement == measurement)
                .filter(|r| r.outcome == Outcome::Unused(GNSS_EXCLUDED))
                .count()
        };
        assert!(unused("gnss_position") > 100 && unused("gnss_velocity") > 100);
    }
}

#[test]
#[cfg_attr(debug_assertions, ignore = "needs an optimized build")]
fn a_receiver_that_keeps_its_offset_over_terrain_is_not_taken_again() {
    let world = World::build(&world_spec());
    // The receiver stops 80 m from the truth.
    let seen = watch(&world, 170, Some(&ramp(1.0, false)), 1);
    assert!(seen.excluded_s.is_some() && seen.taken_again_s.is_none());
    assert_eq!(seen.end_state, Some(GnssIntegrityState::Excluded));
    assert!(seen.worst_after_m < FIX_BOUND_M, "{}", seen.worst_after_m);
}

#[test]
#[cfg_attr(debug_assertions, ignore = "needs an optimized build")]
fn an_honest_receiver_over_terrain_is_not_excluded() {
    show_events();
    // The imagery is 1.4 m from where it says, inside its stated
    // accuracy of 1.5 m.
    let world = World::build(&WorldSpec {
        imagery_shift_m: [1.0, -1.0],
        ..world_spec()
    });
    let (mut fixes, mut doubted, mut worst_chi2) = (0_u32, 0_u32, 0.0_f64);
    for seed in 1..=4 {
        let seen = watch(&world, 170, None, seed);
        assert!(seen.excluded_s.is_none(), "seed {seed}: {seen:?}");
        // The terrain fixes go on with the belief of the reference: one
        // for each map cell of about 450 m on 6.8 km.
        let (accepted, refused, _) = terrain_counts(&seen.records);
        assert!(accepted >= 12 && refused <= 1, "{accepted} {refused}");
        assert!(seen.end_error_m < 3.0);
        fixes += 170 * RATE as u32 / GNSS_EVERY as u32;
        doubted += seen.doubted_fixes;
        worst_chi2 = worst_chi2.max(seen.worst_chi2);
    }
    tracing::info!(fixes, doubted, worst_chi2, "honest receivers over terrain");
    assert!(doubted * 200 <= fixes, "{doubted} of {fixes}, {worst_chi2}");
}
