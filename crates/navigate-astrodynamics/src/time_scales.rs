//! Conversions of a [`PhysicalInstant`] between coordinate time scales
//! (ADR-0015, ADR-0016).
//!
//! Each function reads an instant on one scale and gives the same event
//! on another scale. A function refuses an instant on a different scale
//! with [`TimeScaleError::WrongScale`]. The conversions are of three
//! types:
//!
//! - Fixed offsets: TT − TAI is exactly 32.184 s, and GPST − TAI is
//!   exactly −19 s.
//! - Secular rates: TCG, TCB, and TCL are linear in TT or TDB. The
//!   functions apply the defining rates in exact integer arithmetic on
//!   attoseconds, from the epoch `T0` = 1977-01-01T00:00:32.184 TT.
//! - Periodic terms: TDB − TT is the Fairhead–Bretagnon series to its two
//!   largest terms, `0.001657 s·sin g + 0.000014 s·sin 2g`. The largest
//!   term that the two terms omit has an amplitude of 22 µs. UT1 − TAI
//!   comes from the caller in [`EarthOrientation`].
//!
//! ## Lunar Coordinate Time
//!
//! A clock on the lunar surface is higher in the potential of the
//! Earth–Moon system than a clock on the geoid. Its rate relative to TT
//! is `1 + ΔL_L`, with `ΔL_L = 6.484·10^-10` (IAU 2024 Resolution II),
//! which is 56.021 µs per day. TCL is equal to TT at `T0`:
//! `TCL − TT = ΔL_L·(TT − T0)`. The model omits the periodic terms of
//! TCL − TT.

use navigate_contract::{PhysicalInstant, SignedDuration, TimeScale};

use crate::float_math::sin;

#[cfg(test)]
mod tests;

/// `T0` = 1977-01-01T00:00:32.184 TT (Julian date 2443144.5003725), as a
/// span from J2000.0: −725 803 167.816 s. TCG and TCL are equal to TT at
/// `T0`, and TCB is equal to TDB − TDB0 at `T0`.
const T0_FROM_J2000: Option<SignedDuration> =
    SignedDuration::new(-725_803_168, 184_000_000_000_000_000);

/// `L_G` = 6.969290134·10^-10, a defining constant (IAU 2000 Resolution
/// B1.9): `dTT/dTCG = 1 − L_G`.
const L_G: Ratio = Ratio::new(6_969_290_134, 10_000_000_000_000_000_000);

/// `L_B` = 1.550519768·10^-8, a defining constant (IAU 2006 Resolution
/// B3): `dTDB/dTCB = 1 − L_B`.
const L_B: Ratio = Ratio::new(1_550_519_768, 100_000_000_000_000_000);

/// `TDB0` = −6.55·10^-5 s, a defining constant (IAU 2006 Resolution B3).
const TDB0: Option<SignedDuration> = SignedDuration::new(0, -65_500_000_000_000);

/// `ΔL_L` = 6.484·10^-10: the secular rate of TCL relative to TT.
const TCL_RATE: Ratio = Ratio::new(6_484, 10_000_000_000_000);

/// Seconds in one day.
const SECONDS_PER_DAY: f64 = 86_400.0;

/// The mean anomaly `g` of the Earth at J2000.0 TT, in degrees.
const MEAN_ANOMALY_J2000_DEG: f64 = 357.53;

/// The rate of the mean anomaly `g`, in degrees per day.
const MEAN_ANOMALY_RATE_DEG_PER_DAY: f64 = 0.985_600_28;

/// The amplitude of the `sin g` term of TDB − TT, in seconds.
const TDB_ANNUAL_S: f64 = 0.001_657;

/// The amplitude of the `sin 2g` term of TDB − TT, in seconds.
const TDB_SEMIANNUAL_S: f64 = 0.000_014;

/// A rate as an exact ratio of integers.
#[derive(Clone, Copy)]
struct Ratio {
    numerator: i128,
    denominator: i128,
}

impl Ratio {
    const fn new(numerator: i128, denominator: i128) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    /// `r / (1 − r)`: the rate of the faster scale from the rate of the
    /// slower one.
    const fn over_one_minus(self) -> Self {
        Self::new(self.numerator, self.denominator - self.numerator)
    }

    /// `r / (1 + r)`.
    const fn over_one_plus(self) -> Self {
        Self::new(self.numerator, self.denominator + self.numerator)
    }

    fn of(self, span: SignedDuration) -> Option<SignedDuration> {
        span.checked_mul_ratio(self.numerator, self.denominator)
    }
}

/// Why a time-scale conversion failed.
#[derive(Debug, Clone, Copy, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum TimeScaleError {
    /// The instant is not on the scale that the function reads.
    #[error("the function reads a {expected:?} instant, not a {got:?} instant")]
    WrongScale {
        /// The scale that the function reads.
        expected: TimeScale,
        /// The scale of the given instant.
        got: TimeScale,
    },
    /// A value of the Earth orientation is not finite or not plausible.
    #[error("the Earth orientation parameters are not finite or not plausible")]
    ImplausibleEarthOrientation,
    /// The instant is outside the range of the conversion.
    #[error("the instant is outside the range of the conversion")]
    OutOfRange,
}

/// The Earth orientation parameters at one time, from an IERS bulletin
/// or a prediction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EarthOrientation {
    /// UT1 − TAI in seconds.
    pub ut1_minus_tai_s: f64,
    /// The polar motion `x_p` of the pole in radians.
    pub polar_motion_xp_rad: f64,
    /// The polar motion `y_p` of the pole in radians.
    pub polar_motion_yp_rad: f64,
}

impl EarthOrientation {
    /// Largest magnitude of UT1 − TAI in seconds that
    /// [`Self::is_plausible`] accepts. It is far beyond the value for any
    /// date of the modern era.
    const MAX_UT1_MINUS_TAI_S: f64 = 86_400.0;

    /// Largest polar motion in radians that [`Self::is_plausible`]
    /// accepts: 0.01 rad, which is more than 2000 arcseconds.
    const MAX_POLAR_MOTION_RAD: f64 = 0.01;

    /// Whether every value is finite and in its plausible range.
    #[must_use]
    pub fn is_plausible(&self) -> bool {
        self.ut1_minus_tai_s.is_finite()
            && self.ut1_minus_tai_s.abs() <= Self::MAX_UT1_MINUS_TAI_S
            && [self.polar_motion_xp_rad, self.polar_motion_yp_rad]
                .iter()
                .all(|p| p.is_finite() && p.abs() <= Self::MAX_POLAR_MOTION_RAD)
    }
}

/// TAI to TT: `TT = TAI + 32.184 s`.
///
/// # Errors
///
/// [`TimeScaleError::WrongScale`] for an instant that is not on TAI, and
/// [`TimeScaleError::OutOfRange`] on overflow.
pub fn tai_to_tt(tai: PhysicalInstant) -> Result<PhysicalInstant, TimeScaleError> {
    from_tai(tai, TimeScale::Tt)
}

/// TT to TAI: `TAI = TT − 32.184 s`.
///
/// # Errors
///
/// The errors of [`tai_to_tt`], for an instant that is not on TT.
pub fn tt_to_tai(tt: PhysicalInstant) -> Result<PhysicalInstant, TimeScaleError> {
    to_tai(tt, TimeScale::Tt)
}

/// TAI to GPST: `GPST = TAI − 19 s`.
///
/// # Errors
///
/// The errors of [`tai_to_tt`].
pub fn tai_to_gpst(tai: PhysicalInstant) -> Result<PhysicalInstant, TimeScaleError> {
    from_tai(tai, TimeScale::Gpst)
}

/// GPST to TAI: `TAI = GPST + 19 s`.
///
/// # Errors
///
/// The errors of [`tai_to_tt`], for an instant that is not on GPST.
pub fn gpst_to_tai(gpst: PhysicalInstant) -> Result<PhysicalInstant, TimeScaleError> {
    to_tai(gpst, TimeScale::Gpst)
}

/// TT to TCG: `TCG = TT + (TT − T0)·L_G / (1 − L_G)`.
///
/// # Errors
///
/// The errors of [`tai_to_tt`], for an instant that is not on TT.
pub fn tt_to_tcg(tt: PhysicalInstant) -> Result<PhysicalInstant, TimeScaleError> {
    expect_scale(tt, TimeScale::Tt)?;
    let tcg = faster(
        tt.offset_from_j2000(),
        SignedDuration::ZERO,
        L_G.over_one_minus(),
    )?;
    Ok(PhysicalInstant::new(TimeScale::Tcg, tcg))
}

/// TCG to TT: `TT = TCG − (TCG − T0)·L_G`.
///
/// # Errors
///
/// The errors of [`tai_to_tt`], for an instant that is not on TCG.
pub fn tcg_to_tt(tcg: PhysicalInstant) -> Result<PhysicalInstant, TimeScaleError> {
    expect_scale(tcg, TimeScale::Tcg)?;
    let tt = slower(tcg.offset_from_j2000(), SignedDuration::ZERO, L_G)?;
    Ok(PhysicalInstant::new(TimeScale::Tt, tt))
}

/// TDB to TCB: `TCB = y + (y − T0)·L_B / (1 − L_B)` with `y = TDB − TDB0`.
///
/// # Errors
///
/// The errors of [`tai_to_tt`], for an instant that is not on TDB.
pub fn tdb_to_tcb(tdb: PhysicalInstant) -> Result<PhysicalInstant, TimeScaleError> {
    expect_scale(tdb, TimeScale::Tdb)?;
    let tcb = faster(
        tdb.offset_from_j2000(),
        in_range(TDB0)?,
        L_B.over_one_minus(),
    )?;
    Ok(PhysicalInstant::new(TimeScale::Tcb, tcb))
}

/// TCB to TDB: `TDB = TCB − (TCB − T0)·L_B + TDB0`.
///
/// # Errors
///
/// The errors of [`tai_to_tt`], for an instant that is not on TCB.
pub fn tcb_to_tdb(tcb: PhysicalInstant) -> Result<PhysicalInstant, TimeScaleError> {
    expect_scale(tcb, TimeScale::Tcb)?;
    let tdb = slower(tcb.offset_from_j2000(), in_range(TDB0)?, L_B)?;
    Ok(PhysicalInstant::new(TimeScale::Tdb, tdb))
}

/// TT to TCL: `TCL = TT + ΔL_L·(TT − T0)` (see the module documentation).
///
/// # Errors
///
/// The errors of [`tai_to_tt`], for an instant that is not on TT.
pub fn tt_to_tcl(tt: PhysicalInstant) -> Result<PhysicalInstant, TimeScaleError> {
    expect_scale(tt, TimeScale::Tt)?;
    let tcl = faster(tt.offset_from_j2000(), SignedDuration::ZERO, TCL_RATE)?;
    Ok(PhysicalInstant::new(TimeScale::Tcl, tcl))
}

/// TCL to TT: `TT = TCL − (TCL − T0)·ΔL_L / (1 + ΔL_L)`, the inverse of
/// [`tt_to_tcl`] to within two attoseconds of rounding.
///
/// # Errors
///
/// The errors of [`tai_to_tt`], for an instant that is not on TCL.
pub fn tcl_to_tt(tcl: PhysicalInstant) -> Result<PhysicalInstant, TimeScaleError> {
    expect_scale(tcl, TimeScale::Tcl)?;
    let tt = slower(
        tcl.offset_from_j2000(),
        SignedDuration::ZERO,
        TCL_RATE.over_one_plus(),
    )?;
    Ok(PhysicalInstant::new(TimeScale::Tt, tt))
}

/// TT to TDB: `TDB = TT + 0.001657 s·sin g + 0.000014 s·sin 2g` with
/// `g = 357.53° + 0.98560028°·d` and `d` the days of TT from J2000.0.
///
/// # Errors
///
/// The errors of [`tai_to_tt`], for an instant that is not on TT.
pub fn tt_to_tdb(tt: PhysicalInstant) -> Result<PhysicalInstant, TimeScaleError> {
    expect_scale(tt, TimeScale::Tt)?;
    let offset = tt.offset_from_j2000();
    let tdb = in_range(offset.checked_add(periodic_tdb_minus_tt(offset)?))?;
    Ok(PhysicalInstant::new(TimeScale::Tdb, tdb))
}

/// TDB to TT: the inverse of [`tt_to_tdb`]. Two fixed-point steps
/// evaluate the periodic term at the TT instant. The residual of the
/// inverse is less than one picosecond.
///
/// # Errors
///
/// The errors of [`tai_to_tt`], for an instant that is not on TDB.
pub fn tdb_to_tt(tdb: PhysicalInstant) -> Result<PhysicalInstant, TimeScaleError> {
    expect_scale(tdb, TimeScale::Tdb)?;
    let offset = tdb.offset_from_j2000();
    let mut tt = offset;
    for _ in 0..2 {
        tt = in_range(offset.checked_sub(periodic_tdb_minus_tt(tt)?))?;
    }
    Ok(PhysicalInstant::new(TimeScale::Tt, tt))
}

/// TAI to UT1: `UT1 = TAI + (UT1 − TAI)`.
///
/// # Errors
///
/// The errors of [`tai_to_tt`], and
/// [`TimeScaleError::ImplausibleEarthOrientation`].
pub fn tai_to_ut1(
    tai: PhysicalInstant,
    orientation: &EarthOrientation,
) -> Result<PhysicalInstant, TimeScaleError> {
    expect_scale(tai, TimeScale::Tai)?;
    let shift = ut1_minus_tai(orientation)?;
    let ut1 = in_range(tai.offset_from_j2000().checked_add(shift))?;
    Ok(PhysicalInstant::new(TimeScale::Ut1, ut1))
}

/// UT1 to TAI: `TAI = UT1 − (UT1 − TAI)`.
///
/// # Errors
///
/// The errors of [`tai_to_ut1`], for an instant that is not on UT1.
pub fn ut1_to_tai(
    ut1: PhysicalInstant,
    orientation: &EarthOrientation,
) -> Result<PhysicalInstant, TimeScaleError> {
    expect_scale(ut1, TimeScale::Ut1)?;
    let shift = ut1_minus_tai(orientation)?;
    let tai = in_range(ut1.offset_from_j2000().checked_sub(shift))?;
    Ok(PhysicalInstant::new(TimeScale::Tai, tai))
}

/// The periodic TDB − TT at the TT span `tt` from J2000.0.
fn periodic_tdb_minus_tt(tt: SignedDuration) -> Result<SignedDuration, TimeScaleError> {
    let days = tt.to_seconds_f64() / SECONDS_PER_DAY;
    let g = (MEAN_ANOMALY_J2000_DEG + MEAN_ANOMALY_RATE_DEG_PER_DAY * days).to_radians();
    let seconds = TDB_ANNUAL_S * sin(g) + TDB_SEMIANNUAL_S * sin(2.0 * g);
    in_range(SignedDuration::from_seconds_f64(seconds))
}

/// UT1 − TAI of a plausible `orientation` as a span.
fn ut1_minus_tai(orientation: &EarthOrientation) -> Result<SignedDuration, TimeScaleError> {
    if !orientation.is_plausible() {
        return Err(TimeScaleError::ImplausibleEarthOrientation);
    }
    in_range(SignedDuration::from_seconds_f64(
        orientation.ut1_minus_tai_s,
    ))
}

/// `instant` on the fixed-offset scale `target`.
fn from_tai(
    instant: PhysicalInstant,
    target: TimeScale,
) -> Result<PhysicalInstant, TimeScaleError> {
    expect_scale(instant, TimeScale::Tai)?;
    in_range(PhysicalInstant::from_tai_fixed(instant, target))
}

/// `instant` on the fixed-offset scale `source`, on TAI.
fn to_tai(instant: PhysicalInstant, source: TimeScale) -> Result<PhysicalInstant, TimeScaleError> {
    expect_scale(instant, source)?;
    in_range(instant.to_tai_fixed())
}

/// The faster coordinate scale from the slower one: `y + (y − T0)·rate`
/// with `y = slow − shift`.
fn faster(
    slow: SignedDuration,
    shift: SignedDuration,
    rate: Ratio,
) -> Result<SignedDuration, TimeScaleError> {
    let y = in_range(slow.checked_sub(shift))?;
    let delta = in_range(rate.of(in_range(y.checked_sub(in_range(T0_FROM_J2000)?))?))?;
    in_range(y.checked_add(delta))
}

/// The slower scale from the faster coordinate scale:
/// `fast − (fast − T0)·rate + shift`.
fn slower(
    fast: SignedDuration,
    shift: SignedDuration,
    rate: Ratio,
) -> Result<SignedDuration, TimeScaleError> {
    let delta = in_range(rate.of(in_range(fast.checked_sub(in_range(T0_FROM_J2000)?))?))?;
    in_range(
        fast.checked_sub(delta)
            .and_then(|value| value.checked_add(shift)),
    )
}

/// [`TimeScaleError::OutOfRange`] for `None`.
fn in_range<T>(value: Option<T>) -> Result<T, TimeScaleError> {
    value.ok_or(TimeScaleError::OutOfRange)
}

/// [`TimeScaleError::WrongScale`] for an instant that is not on
/// `expected`.
pub(crate) fn expect_scale(
    instant: PhysicalInstant,
    expected: TimeScale,
) -> Result<(), TimeScaleError> {
    if instant.scale() == expected {
        Ok(())
    } else {
        Err(TimeScaleError::WrongScale {
            expected,
            got: instant.scale(),
        })
    }
}
