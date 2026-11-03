//! Conversions between the physical time of `navigate-contract` and
//! `hifitime` epochs, and between the coordinate time scales (ADR-0015).
//!
//! A [`PhysicalInstant`] is a span from the J2000.0 epoch of its scale in
//! attoseconds. A `hifitime` [`Epoch`] has a resolution of one
//! nanosecond. Thus [`physical_from_epoch`] is exact, and
//! [`epoch_from_physical`] rounds to the nearest nanosecond. An epoch
//! that goes to a physical instant and back is the same instant, on the
//! `hifitime` scale that [`epoch_from_physical`] names.
//!
//! TAI, TT, and GPST differ by exact constants. UTC adds the leap seconds
//! of `hifitime`. TDB follows the TDB model of `hifitime`. UT1 needs the
//! caller value of UT1 − TAI, for example from [`ut1_minus_tai_s`] and a
//! [`crate::DeltaT`]. TCG, TCB, and TCL are linear in TT or TDB, and this
//! module applies their defining rates in exact integer arithmetic.
//!
//! ## Lunar Coordinate Time
//!
//! A clock on the lunar surface is higher in the potential of the
//! Earth–Moon system and moves differently from a clock on the geoid. Its
//! rate relative to TT is `1 + ΔL_L`, with `ΔL_L = 6.484·10^-10`, which is
//! 56.021 µs per day. This module uses that secular rate for TCL relative
//! to TT, from the epoch `T0` = 1977-01-01T00:00:32.184 TT, where TCL is
//! equal to TT:
//!
//! `TCL − TT = ΔL_L · (TT − T0)`.
//!
//! The model omits the periodic terms of TCL − TT. Use it where the
//! secular drift is the dominant term, for example to keep a lunar clock
//! and an Earth clock in one time base over days.

use hifitime::{Duration, Epoch, TimeScale as HifitimeScale};
use navigate_contract::{PhysicalInstant, SignedDuration, TimeScale};

use crate::DeltaT;

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

/// TT − TAI in seconds.
const TT_MINUS_TAI_S: f64 = 32.184;

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

/// Why a time conversion failed.
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
#[non_exhaustive]
pub enum TimeBridgeError {
    /// A UT1 conversion needs UT1 − TAI, and the caller gave none.
    #[error("a UT1 conversion needs UT1 - TAI, and none was given")]
    MissingUt1Offset,
    /// The UT1 − TAI value is not finite.
    #[error("UT1 - TAI of {0} s is not finite")]
    NonFiniteUt1Offset(f64),
    /// The source instant is not on the scale that the function reads.
    #[error("the function reads a {expected:?} instant, not a {got:?} instant")]
    WrongScale {
        /// The scale that the function reads.
        expected: TimeScale,
        /// The scale of the given instant.
        got: TimeScale,
    },
    /// The time scale has no conversion in this module.
    #[error("the {0:?} time scale has no conversion in this module")]
    UnsupportedScale(TimeScale),
    /// The instant is outside the range of the conversion.
    #[error("the instant is outside the range of the conversion")]
    OutOfRange,
}

/// UT1 − TAI in seconds from Delta T: `(TT − TAI) − (TT − UT1)`.
#[must_use]
pub fn ut1_minus_tai_s(delta_t: DeltaT) -> f64 {
    TT_MINUS_TAI_S - delta_t.seconds
}

/// The instant `epoch` on `target`. The conversion is exact to the
/// nanosecond resolution of `epoch`.
///
/// # Errors
///
/// [`TimeBridgeError::MissingUt1Offset`] or
/// [`TimeBridgeError::NonFiniteUt1Offset`] for UT1 without a usable
/// `ut1_minus_tai_s`, [`TimeBridgeError::UnsupportedScale`] for a scale
/// this module does not know, and [`TimeBridgeError::OutOfRange`] on
/// overflow.
pub fn physical_from_epoch(
    epoch: Epoch,
    target: TimeScale,
    ut1_minus_tai_s: Option<f64>,
) -> Result<PhysicalInstant, TimeBridgeError> {
    let on = |scale, hifitime_scale| {
        offset_on(epoch, hifitime_scale)
            .map(|offset| PhysicalInstant::new(scale, offset))
            .ok_or(TimeBridgeError::OutOfRange)
    };
    let tai = || on(TimeScale::Tai, HifitimeScale::TAI);
    let tt = || fixed_from_tai(tai()?, TimeScale::Tt);
    match target {
        TimeScale::Tai | TimeScale::Tt | TimeScale::Gpst => fixed_from_tai(tai()?, target),
        TimeScale::Utc => on(TimeScale::Utc, HifitimeScale::UTC),
        TimeScale::Tdb => on(TimeScale::Tdb, HifitimeScale::TDB),
        TimeScale::Tcg => tt_to_tcg(tt()?),
        TimeScale::Tcb => tdb_to_tcb(on(TimeScale::Tdb, HifitimeScale::TDB)?),
        TimeScale::Tcl => tt_to_tcl(tt()?),
        TimeScale::Ut1 => {
            let ut1_minus_tai = ut1_offset(ut1_minus_tai_s)?;
            let ut1 = tai()?.offset_from_j2000().checked_add(ut1_minus_tai);
            in_range(ut1).map(|offset| PhysicalInstant::new(TimeScale::Ut1, offset))
        }
        other => Err(TimeBridgeError::UnsupportedScale(other)),
    }
}

/// The epoch of `instant`, rounded to the nearest nanosecond. The epoch is on the matching `hifitime` scale: TAI, TT,
/// GPST, UTC, or TDB. A TCG or TCL instant gives a TT epoch, a TCB
/// instant gives a TDB epoch, and a UT1 instant gives a TAI epoch.
///
/// # Errors
///
/// The errors of [`physical_from_epoch`].
pub fn epoch_from_physical(
    instant: PhysicalInstant,
    ut1_minus_tai_s: Option<f64>,
) -> Result<Epoch, TimeBridgeError> {
    let offset = instant.offset_from_j2000();
    // TT, GPST, TCG, TCL, and UT1 go through TAI, so one exact path holds
    // their fixed offsets.
    let via_tai = |instant: PhysicalInstant, scale| {
        let tai = in_range(instant.to_tai_fixed())?;
        in_range(epoch_on(tai.offset_from_j2000(), HifitimeScale::TAI))
            .map(|epoch| epoch.to_time_scale(scale))
    };
    match instant.scale() {
        TimeScale::Tai => via_tai(instant, HifitimeScale::TAI),
        TimeScale::Tt => via_tai(instant, HifitimeScale::TT),
        TimeScale::Gpst => via_tai(instant, HifitimeScale::GPST),
        TimeScale::Utc => in_range(epoch_on(offset, HifitimeScale::UTC)),
        TimeScale::Tdb => in_range(epoch_on(offset, HifitimeScale::TDB)),
        TimeScale::Tcg => via_tai(tcg_to_tt(instant)?, HifitimeScale::TT),
        TimeScale::Tcb => {
            let tdb = tcb_to_tdb(instant)?.offset_from_j2000();
            in_range(epoch_on(tdb, HifitimeScale::TDB))
        }
        TimeScale::Tcl => via_tai(tcl_to_tt(instant)?, HifitimeScale::TT),
        TimeScale::Ut1 => {
            let tai = in_range(offset.checked_sub(ut1_offset(ut1_minus_tai_s)?))?;
            via_tai(
                PhysicalInstant::new(TimeScale::Tai, tai),
                HifitimeScale::TAI,
            )
        }
        other => Err(TimeBridgeError::UnsupportedScale(other)),
    }
}

/// TT to TCG: `TCG = TT + (TT − T0)·L_G / (1 − L_G)`.
///
/// # Errors
///
/// [`TimeBridgeError::WrongScale`] for an instant that is not on TT, and
/// [`TimeBridgeError::OutOfRange`] on overflow.
pub fn tt_to_tcg(tt: PhysicalInstant) -> Result<PhysicalInstant, TimeBridgeError> {
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
/// The errors of [`tt_to_tcg`], for an instant that is not on TCG.
pub fn tcg_to_tt(tcg: PhysicalInstant) -> Result<PhysicalInstant, TimeBridgeError> {
    expect_scale(tcg, TimeScale::Tcg)?;
    let tt = slower(tcg.offset_from_j2000(), SignedDuration::ZERO, L_G)?;
    Ok(PhysicalInstant::new(TimeScale::Tt, tt))
}

/// TDB to TCB: `TCB = y + (y − T0)·L_B / (1 − L_B)` with `y = TDB − TDB0`.
///
/// # Errors
///
/// The errors of [`tt_to_tcg`], for an instant that is not on TDB.
pub fn tdb_to_tcb(tdb: PhysicalInstant) -> Result<PhysicalInstant, TimeBridgeError> {
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
/// The errors of [`tt_to_tcg`], for an instant that is not on TCB.
pub fn tcb_to_tdb(tcb: PhysicalInstant) -> Result<PhysicalInstant, TimeBridgeError> {
    expect_scale(tcb, TimeScale::Tcb)?;
    let tdb = slower(tcb.offset_from_j2000(), in_range(TDB0)?, L_B)?;
    Ok(PhysicalInstant::new(TimeScale::Tdb, tdb))
}

/// TT to TCL: `TCL = TT + ΔL_L·(TT − T0)` (see the module documentation).
///
/// # Errors
///
/// The errors of [`tt_to_tcg`], for an instant that is not on TT.
pub fn tt_to_tcl(tt: PhysicalInstant) -> Result<PhysicalInstant, TimeBridgeError> {
    expect_scale(tt, TimeScale::Tt)?;
    let tcl = faster(tt.offset_from_j2000(), SignedDuration::ZERO, TCL_RATE)?;
    Ok(PhysicalInstant::new(TimeScale::Tcl, tcl))
}

/// TCL to TT: `TT = TCL − (TCL − T0)·ΔL_L / (1 + ΔL_L)`, the exact
/// inverse of [`tt_to_tcl`] to within two attoseconds of rounding.
///
/// # Errors
///
/// The errors of [`tt_to_tcg`], for an instant that is not on TCL.
pub fn tcl_to_tt(tcl: PhysicalInstant) -> Result<PhysicalInstant, TimeBridgeError> {
    expect_scale(tcl, TimeScale::Tcl)?;
    let tt = slower(
        tcl.offset_from_j2000(),
        SignedDuration::ZERO,
        TCL_RATE.over_one_plus(),
    )?;
    Ok(PhysicalInstant::new(TimeScale::Tt, tt))
}

/// The faster coordinate scale from the slower one: `y + (y − T0)·rate`
/// with `y = slow − shift`.
fn faster(
    slow: SignedDuration,
    shift: SignedDuration,
    rate: Ratio,
) -> Result<SignedDuration, TimeBridgeError> {
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
) -> Result<SignedDuration, TimeBridgeError> {
    let delta = in_range(rate.of(in_range(fast.checked_sub(in_range(T0_FROM_J2000)?))?))?;
    in_range(
        fast.checked_sub(delta)
            .and_then(|value| value.checked_add(shift)),
    )
}

/// [`TimeBridgeError::OutOfRange`] for `None`.
fn in_range<T>(value: Option<T>) -> Result<T, TimeBridgeError> {
    value.ok_or(TimeBridgeError::OutOfRange)
}

/// `tai` on the fixed-offset scale `target`.
fn fixed_from_tai(
    tai: PhysicalInstant,
    target: TimeScale,
) -> Result<PhysicalInstant, TimeBridgeError> {
    in_range(PhysicalInstant::from_tai_fixed(tai, target))
}

fn expect_scale(instant: PhysicalInstant, expected: TimeScale) -> Result<(), TimeBridgeError> {
    if instant.scale() == expected {
        Ok(())
    } else {
        Err(TimeBridgeError::WrongScale {
            expected,
            got: instant.scale(),
        })
    }
}

/// UT1 − TAI as a span, when the caller gave a finite value.
fn ut1_offset(ut1_minus_tai_s: Option<f64>) -> Result<SignedDuration, TimeBridgeError> {
    let seconds = ut1_minus_tai_s.ok_or(TimeBridgeError::MissingUt1Offset)?;
    SignedDuration::from_seconds_f64(seconds).ok_or(TimeBridgeError::NonFiniteUt1Offset(seconds))
}

/// 2000-01-01T12:00:00 on `scale`.
fn j2000_on(scale: HifitimeScale) -> Option<Epoch> {
    Epoch::maybe_from_gregorian(2000, 1, 1, 12, 0, 0, 0, scale).ok()
}

/// The span from the J2000.0 label of `scale` to `epoch` on `scale`.
fn offset_on(epoch: Epoch, scale: HifitimeScale) -> Option<SignedDuration> {
    let on_scale = epoch.to_time_scale(scale).duration;
    let nanos = (on_scale - j2000_on(scale)?.duration).total_nanoseconds();
    SignedDuration::from_nanos(nanos)
}

/// The epoch on `scale` at the span `offset` from the J2000.0 label of
/// `scale`, rounded to the nearest nanosecond (a half rounds up). The
/// rounding keeps a round trip through the linear scales on its
/// nanosecond: their inverse can return one attosecond below it. The
/// result is `None` beyond the range of a `hifitime` duration, which
/// would otherwise saturate.
fn epoch_on(offset: SignedDuration, scale: HifitimeScale) -> Option<Epoch> {
    const HALF_NANOSECOND_AS: i128 = 500_000_000;
    const ATTOSECONDS_PER_NANOSECOND: i128 = 1_000_000_000;
    let nanos = offset
        .as_attoseconds()
        .checked_add(HALF_NANOSECOND_AS)?
        .div_euclid(ATTOSECONDS_PER_NANOSECOND);
    let span = Duration::from_total_nanoseconds(nanos);
    if span.total_nanoseconds() != nanos {
        return None;
    }
    Some(Epoch::from_duration(
        j2000_on(scale)?.duration + span,
        scale,
    ))
}

#[cfg(test)]
mod tests;
