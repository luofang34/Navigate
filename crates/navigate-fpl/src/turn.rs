//! Turn-performance geometry for fly-by sequencing (NAV-TT-002).
//!
//! The model is deliberately vehicle-class-free: radius follows from the
//! commanded groundspeed and a configured bank-angle limit, so a
//! multirotor at 2 m/s and a fixed-wing at 60 m/s run the same code with
//! different numbers (see `docs/procedure-requirements.md`).

/// Standard gravity, meters per second squared.
const STANDARD_GRAVITY_MPS2: f64 = 9.80665;

/// Level-turn radius at `groundspeed_mps` under `bank_limit_rad`:
/// `r = v² / (g · tan φ)`. The bank limit is validated positive and
/// finite by the execution config; a zero or negative groundspeed yields
/// a zero radius (no anticipation).
#[must_use]
pub fn turn_radius_m(groundspeed_mps: f64, bank_limit_rad: f64) -> f64 {
    if !groundspeed_mps.is_finite() || groundspeed_mps <= 0.0 {
        return 0.0;
    }
    let tan_bank = bank_limit_rad.tan();
    if !tan_bank.is_finite() || tan_bank <= 0.0 {
        return 0.0;
    }
    groundspeed_mps * groundspeed_mps / (STANDARD_GRAVITY_MPS2 * tan_bank)
}

/// Distance of turn anticipation before a fly-by fix:
/// `DTA = r · tan(Δ/2)` with the track change `Δ` folded into `[0, π]`.
/// A straight-ahead fix (`Δ ≈ 0`) anticipates nothing; a near-reversal
/// approaches the tangent's blowup and is capped by the caller's
/// inbound-leg length, not here.
#[must_use]
pub fn turn_anticipation_m(radius_m: f64, track_change_rad: f64) -> f64 {
    if !radius_m.is_finite() || radius_m <= 0.0 || !track_change_rad.is_finite() {
        return 0.0;
    }
    let folded = fold_to_half_turn(track_change_rad);
    let dta = radius_m * (folded / 2.0).tan();
    if dta.is_finite() {
        dta.max(0.0)
    } else {
        f64::MAX
    }
}

/// Folds any angle into `[0, π]` — the magnitude of a track change.
fn fold_to_half_turn(angle_rad: f64) -> f64 {
    let wrapped = angle_rad.rem_euclid(core::f64::consts::TAU);
    if wrapped > core::f64::consts::PI {
        core::f64::consts::TAU - wrapped
    } else {
        wrapped
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::expect_used, clippy::panic)]

    use super::*;

    #[test]
    fn nav_tt_002_fixed_wing_radius_at_the_rnp_fly_by_bank() {
        // 60 m/s at the 18° standard: r = 3600 / (9.80665 · tan 18°).
        let radius = turn_radius_m(60.0, 18.0_f64.to_radians());
        assert!((radius - 1129.81).abs() < 1.0, "radius {radius}");
        // A 90° track change anticipates exactly one radius: tan 45° = 1.
        let dta = turn_anticipation_m(radius, core::f64::consts::FRAC_PI_2);
        assert!((dta - radius).abs() < 1e-9, "dta {dta}");
    }

    #[test]
    fn nav_tt_002_multirotor_radius_degenerates_gracefully() {
        let radius = turn_radius_m(2.0, 18.0_f64.to_radians());
        assert!((radius - 1.2553).abs() < 0.01, "radius {radius}");
    }

    #[test]
    fn a_straight_ahead_fix_anticipates_nothing() {
        assert_eq!(turn_anticipation_m(1000.0, 0.0), 0.0);
        assert!(turn_anticipation_m(1000.0, 1e-12) < 1e-6);
    }

    #[test]
    fn unflyable_inputs_yield_zero_anticipation_never_a_panic() {
        assert_eq!(turn_radius_m(f64::NAN, 0.3), 0.0);
        assert_eq!(turn_radius_m(-5.0, 0.3), 0.0);
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
