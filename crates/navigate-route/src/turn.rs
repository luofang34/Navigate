//! Turn-performance geometry for fly-by sequencing (NAV-TT-002).
//!
//! The model is vehicle-class-free and body-explicit: radius follows from
//! the commanded groundspeed, a configured bank-angle limit, and the
//! surface gravity of the body (see `docs/procedure-requirements.md` and
//! ADR-0015).

use crate::float_math::{rem_euclid, tan};

/// Standard Earth surface gravity in meters per second squared.
pub const EARTH_STANDARD_GRAVITY_MPS2: f64 = 9.80665;

/// Mean surface gravity of Mars in meters per second squared.
pub const MARS_SURFACE_GRAVITY_MPS2: f64 = 3.72076;

/// Mean surface gravity of Titan in meters per second squared.
pub const TITAN_SURFACE_GRAVITY_MPS2: f64 = 1.352;

/// Level-turn radius at `groundspeed_mps` under `bank_limit_rad` and
/// `surface_gravity_mps2`: `r = v² / (g · tan φ)`. A non-finite or
/// non-positive speed, bank tangent, or surface gravity yields a zero
/// radius (no anticipation).
#[must_use]
pub fn turn_radius_m(groundspeed_mps: f64, bank_limit_rad: f64, surface_gravity_mps2: f64) -> f64 {
    if !groundspeed_mps.is_finite()
        || groundspeed_mps <= 0.0
        || !surface_gravity_mps2.is_finite()
        || surface_gravity_mps2 <= 0.0
    {
        return 0.0;
    }
    let tan_bank = tan(bank_limit_rad);
    if !tan_bank.is_finite() || tan_bank <= 0.0 {
        return 0.0;
    }
    groundspeed_mps * groundspeed_mps / (surface_gravity_mps2 * tan_bank)
}

/// Distance of turn anticipation before a fly-by fix:
/// `DTA = r · tan(Δ/2)` with the track change `Δ` folded into `[0, π]`.
/// A straight-ahead fix (`Δ ≈ 0`) anticipates nothing. The tangent
/// blows up toward a reversal; bounding is the sequencer's policy — it
/// refuses track changes beyond its anticipation limit and caps the
/// result at half the shorter adjoining leg.
#[must_use]
pub fn turn_anticipation_m(radius_m: f64, track_change_rad: f64) -> f64 {
    if !radius_m.is_finite() || radius_m <= 0.0 || !track_change_rad.is_finite() {
        return 0.0;
    }
    let folded = fold_to_half_turn(track_change_rad);
    let dta = radius_m * tan(folded / 2.0);
    if dta.is_finite() {
        dta.max(0.0)
    } else {
        f64::MAX
    }
}

/// Folds any angle into `[0, π]` — the magnitude of a track change.
pub(crate) fn fold_to_half_turn(angle_rad: f64) -> f64 {
    let wrapped = rem_euclid(angle_rad, core::f64::consts::TAU);
    if wrapped > core::f64::consts::PI {
        core::f64::consts::TAU - wrapped
    } else {
        wrapped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nav_tt_002_fixed_wing_radius_at_the_rnp_fly_by_bank() {
        // 60 m/s at the 18° standard on Earth: r = 3600 / (9.80665 · tan 18°).
        let radius = turn_radius_m(60.0, 18.0_f64.to_radians(), EARTH_STANDARD_GRAVITY_MPS2);
        assert!((radius - 1129.81).abs() < 1.0, "radius {radius}");
        // A 90° track change anticipates one radius: tan 45° = 1.
        let dta = turn_anticipation_m(radius, core::f64::consts::FRAC_PI_2);
        assert!((dta - radius).abs() < 1e-9, "dta {dta}");
    }

    #[test]
    fn lower_planetary_gravity_widens_the_turn_radius() {
        let bank = 18.0_f64.to_radians();
        let earth = turn_radius_m(60.0, bank, EARTH_STANDARD_GRAVITY_MPS2);
        let mars = turn_radius_m(60.0, bank, MARS_SURFACE_GRAVITY_MPS2);
        let titan = turn_radius_m(60.0, bank, TITAN_SURFACE_GRAVITY_MPS2);
        assert!(mars > 2.6 * earth, "mars {mars} vs earth {earth}");
        // The radius is inversely proportional to the surface gravity.
        let expected = EARTH_STANDARD_GRAVITY_MPS2 / TITAN_SURFACE_GRAVITY_MPS2;
        assert!((titan / earth - expected).abs() < 1e-12, "titan {titan}");
    }

    #[test]
    fn nav_tt_002_multirotor_radius_degenerates_gracefully() {
        let radius = turn_radius_m(2.0, 18.0_f64.to_radians(), EARTH_STANDARD_GRAVITY_MPS2);
        assert!((radius - 1.2553).abs() < 0.01, "radius {radius}");
    }

    #[test]
    fn a_straight_ahead_fix_anticipates_nothing() {
        assert_eq!(turn_anticipation_m(1000.0, 0.0), 0.0);
        assert!(turn_anticipation_m(1000.0, 1e-12) < 1e-6);
    }

    #[test]
    fn unflyable_inputs_yield_zero_anticipation_never_a_panic() {
        assert_eq!(
            turn_radius_m(f64::NAN, 0.3, EARTH_STANDARD_GRAVITY_MPS2),
            0.0
        );
        assert_eq!(turn_radius_m(-5.0, 0.3, EARTH_STANDARD_GRAVITY_MPS2), 0.0);
        assert_eq!(turn_radius_m(60.0, 0.3, 0.0), 0.0);
        assert_eq!(turn_radius_m(60.0, 0.3, f64::NAN), 0.0);
        assert_eq!(turn_anticipation_m(f64::NAN, 1.0), 0.0);
        assert_eq!(turn_anticipation_m(-1.0, 1.0), 0.0);
    }

    #[test]
    fn a_track_change_beyond_a_half_turn_folds_back() {
        let radius = 100.0;
        let quarter = turn_anticipation_m(radius, core::f64::consts::FRAC_PI_2);
        let mirrored = turn_anticipation_m(radius, -core::f64::consts::FRAC_PI_2);
        assert!((quarter - mirrored).abs() < 1e-9);
        let three_quarters = turn_anticipation_m(radius, 3.0 * core::f64::consts::FRAC_PI_2);
        assert!((three_quarters - quarter).abs() < 1e-9, "270° folds to 90°");
    }
}
