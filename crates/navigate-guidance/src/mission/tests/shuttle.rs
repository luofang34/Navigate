//! A Space Shuttle class reentry in closed loop: a deorbit burn from a
//! 400 km orbit, a coast, the entry corridor with bank reversals, and
//! the handoff to the first leg of a runway route at the TAEM energy.
//!
//! The test planet is the WGS84 sphere of radius `a` with point-mass
//! gravity, and it does not turn. Thus the inertial axes and the
//! body-fixed axes agree, and the navigation output changes only the
//! name of its frame at the entry. The atmosphere is exponential.

use navigate_contract::{
    Ellipsoid, FaultDetection, FlightPlan, IntegrityAssessment, NavigationSolution, NedVelocity,
    PlanRole, Redundancy, SensorClass, SolutionQuality, SolutionStamp, SourceComposition,
    SourceEpoch, SymmetricCov3, Waypoint, WrappingSequence,
};
use navigate_geodesy::{BodyEllipsoid, BodyFixedPosition, body_fixed_to_geodetic};
use navigate_route::{ExecutionConfig, PlanExecution};

use super::*;
use crate::scenario::id;

const SCALE_HEIGHT_M: f64 = 7_200.0;
const SEA_LEVEL_DENSITY_KG_PER_M3: f64 = 1.225;
const BALLISTIC_COEFFICIENT_KG_PER_M2: f64 = 400.0;
const LIFT_TO_DRAG: f64 = 1.0;
const INCLINATION_RAD: f64 = 51.6 * core::f64::consts::PI / 180.0;
const TAEM_ENERGY_J_PER_KG: f64 = -6.196e7;
const STEP_MS: u64 = 500;

/// The in-plane unit vector at the orbit angle `theta` and the orbit
/// normal of the initial orbit.
fn orbit_axes(theta: f64) -> (Vector3<f64>, Vector3<f64>) {
    let (s, c) = theta.sin_cos();
    let (si, ci) = INCLINATION_RAD.sin_cos();
    (Vector3::new(c, s * ci, s * si), Vector3::new(0.0, -si, ci))
}

/// The landing site: 207° along the initial orbit and 1.5° to its left.
fn landing_site_m() -> Vector3<f64> {
    let (along, normal) = orbit_axes(207f64.to_radians());
    let cross = 1.5f64.to_radians();
    (along * cross.cos() + normal * cross.sin()) * EARTH_RADIUS_M
}

/// The drag in m/s² of the test vehicle at `state`.
fn drag_mps2(state: &StateVector) -> f64 {
    let altitude_m = state.position_m.norm() - EARTH_RADIUS_M;
    let density = SEA_LEVEL_DENSITY_KG_PER_M3 * (-altitude_m / SCALE_HEIGHT_M).exp();
    0.5 * density * state.velocity_mps.norm_squared() / BALLISTIC_COEFFICIENT_KG_PER_M2
}

/// Drag plus lift at the bank `bank_rad`. A positive bank tilts the lift
/// to the right of the velocity.
fn aerodynamic(p: &Vector3<f64>, v: &Vector3<f64>, bank_rad: f64) -> Vector3<f64> {
    let drag = drag_mps2(&StateVector::new(*p, *v));
    let along = v.normalize();
    let right = along.cross(&p.normalize()).normalize();
    let up = right.cross(&along);
    let lift = (up * bank_rad.cos() + right * bank_rad.sin()) * (drag * LIFT_TO_DRAG);
    lift - along * drag
}

/// Reference drag knots from the equilibrium glide at a bank of about
/// 53°, at a nominal altitude of 60 km, between 1 and 12 m/s².
fn reference_nodes() -> [EntryNode; 16] {
    let gm = EARTH.gm_m3_per_s2;
    let r = EARTH_RADIUS_M + 60e3;
    let drag_at = |energy: f64| {
        let v_sq = (2.0 * (energy + gm / r)).max(0.0);
        ((gm / (r * r) - v_sq / r) / (LIFT_TO_DRAG * 0.6)).clamp(1.0, 12.0)
    };
    let (top, bottom) = (-2.9e7, -6.3e7);
    let mut nodes = [EntryNode::new(0.0, 1.0, 0.0); 16];
    let step = (top - bottom) / 15.0;
    for (i, node) in nodes.iter_mut().enumerate() {
        let energy = top - step * i as f64;
        let drag = drag_at(energy);
        let slope = (drag_at(energy + 1e4) - drag_at(energy - 1e4)) / 2e4;
        let speed = (2.0 * (energy + gm / r)).max(0.0).sqrt();
        *node = EntryNode::new(energy, drag, -slope * drag * speed);
    }
    nodes
}

/// What the closed loop records.
#[derive(Default)]
struct Record {
    burn_delta_v_mps: f64,
    burn_end: Option<StateVector>,
    max_drag_mps2: f64,
    min_altitude_m: f64,
    last_course_rad: f64,
    drag_error_sum_sq: f64,
    drag_error_count: u32,
}

/// One closed-loop step: the command of the phase, then a propagation.
fn fly_step(
    mission: &mut (MissionState, StateVector, f64),
    profile: &MissionProfile<'_>,
    millis: u64,
    record: &mut Record,
) {
    let (state, vehicle, bank) = mission;
    let frame = profile.expected_frame(state.phase).unwrap();
    let solution = space_solution(frame, vehicle, at_millis(millis), 5.0);
    let drag = drag_mps2(vehicle);
    let input = MissionInput::Space {
        solution: &solution,
        sensed_drag_mps2: drag,
    };
    let step = step_mission(*state, profile, input, at_millis(millis), CLOCK, &config()).unwrap();
    let mut thrust = Vector3::zeros();
    match step.command.map(|c| c.setpoint) {
        Some(GuidanceSetpoint::ThrustVector {
            direction_unit,
            specific_force_mps2,
            cutoff,
            ..
        }) => {
            thrust = Vector3::from(direction_unit) * specific_force_mps2;
            if cutoff {
                record.burn_end = Some(*vehicle);
            }
        }
        Some(GuidanceSetpoint::EntryBank {
            bank_rad,
            drag_error_mps2,
            heading_course_rad,
            ..
        }) => {
            *bank = bank_rad;
            record.last_course_rad = heading_course_rad;
            if drag > 3.0 {
                record.drag_error_sum_sq += drag_error_mps2 * drag_error_mps2;
                record.drag_error_count = record.drag_error_count.wrapping_add(1);
            }
        }
        _ => {}
    }
    record.burn_delta_v_mps += thrust.norm() * (STEP_MS as f64 / 1e3);
    record.max_drag_mps2 = record.max_drag_mps2.max(drag);
    *state = step.state;
    let bank_now = *bank;
    *vehicle = rk4(vehicle, STEP_MS as f64 / 1e3, |p, v| {
        point_mass(EARTH.gm_m3_per_s2, p) + thrust + aerodynamic(p, v, bank_now)
    });
    let altitude = vehicle.position_m.norm() - EARTH_RADIUS_M;
    record.min_altitude_m = record.min_altitude_m.min(altitude);
}

fn config() -> MissionConfig {
    MissionConfig::default()
}

/// The surface solution of the vehicle on the WGS84 ellipsoid.
fn surface_solution(vehicle: &StateVector, millis: u64) -> NavigationSolution {
    let p = vehicle.position_m;
    let body = BodyFixedPosition::new(p.x, p.y, p.z);
    let position = body_fixed_to_geodetic(&body, BodyEllipsoid::WGS84);
    let (sin_lat, cos_lat) = position.latitude_rad.sin_cos();
    let (sin_lon, cos_lon) = position.longitude_rad.sin_cos();
    let north = Vector3::new(-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat);
    let east = Vector3::new(-sin_lon, cos_lon, 0.0);
    let down = -north.cross(&east);
    let v = vehicle.velocity_mps;
    let composition = SourceComposition::of(SensorClass::Inertial);
    NavigationSolution::new(
        SolutionStamp::new(
            SourceEpoch::new(1),
            WrappingSequence::new(2),
            at_millis(millis),
            CLOCK,
        ),
        position,
        NedVelocity::new(v.dot(&north), v.dot(&east), v.dot(&down)),
        SymmetricCov3::from_diagonal(25.0, 25.0, 25.0),
        SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
        IntegrityAssessment::new(
            SolutionQuality::Good,
            composition,
            Redundancy::None,
            5.0,
            5.0,
            FaultDetection::Unavailable,
        ),
        composition,
        Ellipsoid::WGS84,
    )
}

/// The azimuth of the horizontal part of `d` at the position `p`, in
/// radians clockwise from north, in `[0, 2π)`.
fn azimuth_rad(p: &Vector3<f64>, d: &Vector3<f64>) -> f64 {
    let up = p.normalize();
    let east = Vector3::z().cross(&up).normalize();
    let north = up.cross(&east);
    d.dot(&east)
        .atan2(d.dot(&north))
        .rem_euclid(core::f64::consts::TAU)
}

/// The periapsis radius of the osculating conic of `state` in meters.
fn periapsis_radius_m(state: &StateVector) -> f64 {
    let gm = EARTH.gm_m3_per_s2;
    let energy = 0.5 * state.velocity_mps.norm_squared() - gm / state.position_m.norm();
    let a = -gm / (2.0 * energy);
    let h = state.position_m.cross(&state.velocity_mps).norm();
    a * (1.0 - (1.0 - h * h / (gm * a)).max(0.0).sqrt())
}

/// The geodetic position on WGS84 of the body-fixed point `p`.
fn geodetic(p: &Vector3<f64>) -> navigate_contract::GeodeticPosition {
    body_fixed_to_geodetic(&BodyFixedPosition::new(p.x, p.y, p.z), BodyEllipsoid::WGS84)
}

/// Flies the runway route from the handoff at constant groundspeed and
/// level, on the course of the route guidance. The result is the number
/// of legs flown, or `None` when the route does not complete.
fn fly_route(
    mut state: MissionState,
    profile: &MissionProfile<'_>,
    mut vehicle: StateVector,
    mut millis: u64,
    waypoints: &[Waypoint],
) -> Option<usize> {
    let plan = FlightPlan::new(id("RWY15"), PlanRole::Mission, Ellipsoid::WGS84, waypoints);
    let mut route = PlanExecution::new(plan, ExecutionConfig::new(500.0)).ok()?;
    let speed = 200.0;
    for _ in 0..3_000 {
        let solution = surface_solution(&vehicle, millis);
        let leg = route.active_leg()?;
        let input = MissionInput::Surface {
            solution: &solution,
            leg,
        };
        let step = step_mission(state, profile, input, at_millis(millis), CLOCK, &config()).ok()?;
        state = step.state;
        let GuidanceSetpoint::DeviationTracking {
            lateral_m,
            course_rad,
            ..
        } = step.command?.setpoint
        else {
            return None;
        };
        let heading = course_rad - (lateral_m / 1_000.0).atan();
        let up = vehicle.position_m.normalize();
        let east = Vector3::z().cross(&up).normalize();
        let north = up.cross(&east);
        vehicle.velocity_mps = (north * heading.cos() + east * heading.sin()) * speed;
        vehicle.position_m += vehicle.velocity_mps;
        vehicle.position_m *= (EARTH_RADIUS_M + 3_000.0) / vehicle.position_m.norm();
        millis += 1_000;
        let event = route.advance(&geodetic(&vehicle.position_m), speed);
        if matches!(event, navigate_route::SequenceEvent::PlanComplete { .. }) {
            return Some(waypoints.len());
        }
    }
    None
}

#[test]
fn a_deorbit_burn_and_an_entry_corridor_hand_off_to_the_runway_route() {
    let nodes = reference_nodes();
    let mut profile = reentry(&nodes);
    if let MissionProfile::Reentry { entry, .. } = &mut profile {
        entry.landing_site_m = landing_site_m();
        entry.taem_energy_j_per_kg = TAEM_ENERGY_J_PER_KG;
    }
    let (along, normal) = orbit_axes(0.0);
    let speed = (EARTH.gm_m3_per_s2 / ORBIT_RADIUS_M).sqrt();
    let start = StateVector::new(along * ORBIT_RADIUS_M, normal.cross(&along) * speed);
    let mut mission = (MissionState::START, start, 0.0);
    let mut record = Record {
        min_altitude_m: f64::INFINITY,
        ..Record::default()
    };
    let mut millis = 0;
    while mission.0.phase != MissionPhase::AtmosphericRoute && millis < 6_000_000 {
        fly_step(&mut mission, &profile, millis, &mut record);
        millis += STEP_MS;
    }
    let (state, vehicle, _) = mission;
    assert_eq!(state.phase, MissionPhase::AtmosphericRoute);

    // The burn reaches the target periapsis for about the impulsive cost.
    let burn_end = record.burn_end.expect("the burn must reach cutoff");
    let periapsis_error_m = periapsis_radius_m(&burn_end) - (EARTH_RADIUS_M + 30e3);
    assert!(periapsis_error_m.abs() < 3_000.0, "{periapsis_error_m} m");
    assert!((105.0..113.0).contains(&record.burn_delta_v_mps));

    // The entry tracks the drag profile within the load limit and turns
    // toward the site with bank reversals.
    let rms = (record.drag_error_sum_sq / f64::from(record.drag_error_count)).sqrt();
    assert!(rms < 0.5, "RMS drag error {rms} m/s²");
    assert!(record.max_drag_mps2 < 15.0, "{} m/s²", record.max_drag_mps2);
    assert!(
        state.entry.reversals >= 2,
        "{} reversals",
        state.entry.reversals
    );
    assert!(record.min_altitude_m > 25e3, "{} m", record.min_altitude_m);

    // The handoff is at the TAEM energy, ahead of the site, with the
    // velocity inside the heading deadband.
    let site = landing_site_m();
    let range_m = vehicle.position_m.normalize().dot(&site.normalize()).acos() * EARTH_RADIUS_M;
    assert!((50e3..400e3).contains(&range_m), "{range_m} m");
    assert!(vehicle.velocity_mps.norm() < 1_000.0);
    let heading = azimuth_rad(&vehicle.position_m, &vehicle.velocity_mps);
    let heading_error = (heading - record.last_course_rad + core::f64::consts::PI)
        .rem_euclid(core::f64::consts::TAU)
        - core::f64::consts::PI;
    assert!(
        heading_error.abs() < 12f64.to_radians(),
        "{heading_error} rad"
    );

    // The route guidance flies the approach fix and the runway threshold.
    let approach = (site + (vehicle.position_m.normalize() * EARTH_RADIUS_M - site) * 0.1)
        .normalize()
        * EARTH_RADIUS_M;
    let waypoints = [
        Waypoint::new(id("HAC"), geodetic(&approach)),
        Waypoint::new(id("RW15"), geodetic(&site)),
    ];
    assert_eq!(
        fly_route(state, &profile, vehicle, millis, &waypoints),
        Some(2)
    );
}
