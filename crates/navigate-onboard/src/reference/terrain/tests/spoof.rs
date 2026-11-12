//! A GNSS receiver whose position leaves the truth at a constant rate,
//! on a flight over the test world with the real matcher: what the
//! navigator does with it.

use nalgebra::Vector3;
use navigate_contract::{SensorClass, SolutionQuality};

use super::fixes::FIX_BOUND_M;
use super::flight::{RATE, at, request, show_events};
use super::world::{DemError, Texture, World, WorldSpec};
use crate::reference::monitor::GNSS_EXCLUDED;
use crate::{AcceptanceRecord, GnssIntegrityConfig, GnssIntegrityState, Navigator, Outcome};

mod scenario;

use scenario::{
    GNSS, GNSS_EVERY, Ramp, flight, navigator, step_inputs, terrain_counts, world_spec,
};

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
    /// Solutions after the first terrain fix that do not name the terrain
    /// camera as a contributing sensor.
    visual_gaps: u32,
    records: Vec<AcceptanceRecord>,
}

/// Flies `seconds` over `world` with the check, with the receiver error
/// of `ramp`.
fn watch(world: &World, seconds: u64, ramp: Option<&Ramp>, seed: u64) -> Seen {
    let check = Some(GnssIntegrityConfig::default());
    watch_over(world, world, seconds, ramp, seed, check)
}

/// Flies `seconds` with the package of `map` over the ground of `ground`,
/// with the receiver error of `ramp`, with or without the check.
fn watch_over(
    map: &World,
    ground: &World,
    seconds: u64,
    ramp: Option<&Ramp>,
    seed: u64,
    check: Option<GnssIntegrityConfig>,
) -> Seen {
    let world = ground;
    let flight = flight(&world.spec);
    let mut navigator = navigator(&flight);
    navigator.set_gnss_integrity(check).unwrap();
    navigator
        .load_terrain_map(std::sync::Arc::clone(&map.store), &request(map), None)
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
        let solution = navigator.solution(at(step)).unwrap();
        let error: Vector3<f64> = flight.error_m(&solution, step);
        let (error_m, t_s) = (error.xy().norm(), step as f64 / RATE as f64);
        seen.end_error_m = error_m;
        let fixed = terrain_counts(&seen.records).0 > 0;
        let named = solution
            .integrity
            .contributing
            .contains(SensorClass::VisualLandmark);
        seen.visual_gaps += u32::from(fixed && !named);
        let Some(status) = navigator.gnss_integrity() else {
            continue;
        };
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
    // 32.0 s and excluded after 42.6 s with at most 21.4 m of error;
    // 1 m/s after 20.6 s and 31.6 s with 7.3 m; 2 m/s after 8.6 s and
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
        // After the exclusion the solution is that of the terrain fixes,
        // and the filter takes each of them.
        assert!(seen.worst_after_m < FIX_BOUND_M, "{}", seen.worst_after_m);
        let excluded_at = at((seen.excluded_s.unwrap() * RATE as f64).round() as u64);
        let later: Vec<_> = seen
            .records
            .iter()
            .filter(|r| r.acquired_at > excluded_at)
            .cloned()
            .collect();
        let (accepted, refused, _) = terrain_counts(&later);
        assert!(accepted >= 6 && refused == 0, "{accepted} {refused}");
        // Between two fixes, 11 s apart, the solution still names the
        // terrain camera: with the receiver, and alone after the
        // exclusion.
        assert_eq!(seen.visual_gaps, 0);
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

/// A world of the flights with one property changed.
fn variant(change: impl FnOnce(&mut WorldSpec)) -> World {
    let mut spec = world_spec();
    change(&mut spec);
    World::build(&spec)
}

#[test]
#[cfg_attr(debug_assertions, ignore = "needs an optimized build")]
fn an_honest_receiver_over_terrain_is_not_excluded() {
    show_events();
    // A reference that the terrain fixes move off the truth would make an
    // honest receiver look wrong. The cases give the fixes the errors
    // that a map has inside its stated accuracy, and other ground: an
    // imagery 1.4 m from where it says (stated 1.5 m), an elevation
    // error of 2 m one-sigma in features of 300 m (stated 2 m), a town,
    // relief of 250 m, and fields in another season than the map.
    let plain = World::build(&world_spec());
    let shifted = variant(|spec| spec.imagery_shift_m = [1.0, -1.0]);
    let rough = variant(|spec| {
        spec.dem_error = DemError::Field {
            sigma_m: 2.0,
            length_m: 300.0,
            seed: 103,
        };
    });
    let town = variant(|spec| spec.texture = Texture::Town);
    let hills = variant(|spec| spec.relief_m = 250.0);
    let season = variant(|spec| spec.texture = Texture::FarmlandSeason);
    let cases = [
        ("shifted imagery", &shifted, &shifted, 1),
        ("shifted imagery", &shifted, &shifted, 2),
        ("elevation error", &rough, &rough, 3),
        ("town", &town, &town, 4),
        ("hills", &hills, &hills, 5),
        ("another season", &plain, &season, 6),
    ];
    let check = Some(GnssIntegrityConfig::default());
    let (mut fixes, mut doubted, mut worst_chi2) = (0_u32, 0_u32, 0.0_f64);
    for (name, map, ground, seed) in cases {
        let seen = watch_over(map, ground, 170, None, seed, check);
        let (accepted, refused, _) = terrain_counts(&seen.records);
        let chi2 = seen.worst_chi2;
        tracing::info!(name, accepted, refused, chi2, "an honest receiver");
        assert!(seen.excluded_s.is_none(), "{name}: {seen:?}");
        assert!(seen.end_error_m < 3.0, "{name}");
        fixes += 170 * RATE as u32 / GNSS_EVERY as u32;
        doubted += seen.doubted_fixes;
        worst_chi2 = worst_chi2.max(chi2);
    }
    tracing::info!(fixes, doubted, worst_chi2, "honest receivers over terrain");
    assert!(doubted * 200 <= fixes, "{doubted} of {fixes}, {worst_chi2}");
}

#[test]
#[cfg_attr(debug_assertions, ignore = "needs an optimized build")]
fn the_check_keeps_the_terrain_fixes_of_an_honest_flight() {
    show_events();
    // With the check the belief of a frame is that of the reference,
    // which is wider than that of the filter with the receiver.
    let world = World::build(&world_spec());
    let timed = |check| {
        let started = std::time::Instant::now();
        let seen = watch_over(&world, &world, 170, None, 1, check);
        (seen, started.elapsed().as_secs_f64())
    };
    let (without, without_s) = timed(None);
    let (with, with_s) = timed(Some(GnssIntegrityConfig::default()));
    let (fixes_without, fixes_with) = (
        terrain_counts(&without.records),
        terrain_counts(&with.records),
    );
    tracing::info!(
        ?fixes_without,
        ?fixes_with,
        without_s,
        with_s,
        "terrain fixes"
    );
    // One fix for each map cell in both flights; the filter with the
    // receiver refuses none of them.
    assert_eq!(fixes_with.1, 0);
    assert!(fixes_with.0.abs_diff(fixes_without.0) <= 1);
    assert!(with.end_error_m < 3.0 && without.end_error_m < 3.0);
    assert_eq!((with.visual_gaps, without.visual_gaps), (0, 0));
}
