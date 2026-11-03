//! Coordinate time scales and exact instants on them.
//!
//! A [`PhysicalInstant`] is a signed span from the J2000.0 epoch of one
//! [`TimeScale`]. J2000.0 is 2000-01-01T12:00:00 in that scale, which is
//! Julian date 2451545.0. The span is a [`SignedDuration`]: whole seconds
//! and attoseconds in exact integer arithmetic. This module has no float
//! model of a time scale. The conversions that need an ephemeris, Earth
//! orientation data, or leap seconds are in `navigate-celestial`.

/// Attoseconds in one second (10^18).
///
/// The sum of two normalized attosecond parts is less than `2·10^18`,
/// which is less than `i64::MAX`. Thus the addition of two parts cannot
/// overflow.
pub const ATTOSECONDS_PER_SECOND: i64 = 1_000_000_000_000_000_000;

const ATTOSECONDS_I128: i128 = ATTOSECONDS_PER_SECOND as i128;
const NANOSECONDS_PER_SECOND: i128 = 1_000_000_000;
const ATTOSECONDS_PER_NANOSECOND: i128 = 1_000_000_000;

/// TT − TAI: exactly 32.184 s.
const TT_MINUS_TAI: SignedDuration = SignedDuration {
    seconds: 32,
    attoseconds: 184_000_000_000_000_000,
};

/// GPST − TAI: exactly −19 s.
const GPST_MINUS_TAI: SignedDuration = SignedDuration {
    seconds: -19,
    attoseconds: 0,
};

/// A time scale. Each scale has its own J2000.0 epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum TimeScale {
    /// International Atomic Time.
    Tai,
    /// Terrestrial Time: TAI + 32.184 s.
    Tt,
    /// Barycentric Dynamical Time.
    Tdb,
    /// Geocentric Coordinate Time.
    Tcg,
    /// Barycentric Coordinate Time.
    Tcb,
    /// Lunar Coordinate Time (IAU 2024).
    Tcl,
    /// Universal Time 1, the scale of the Earth rotation angle.
    Ut1,
    /// Coordinated Universal Time. The offset of an instant counts the
    /// UTC labels and does not count leap seconds.
    Utc,
    /// GPS Time: TAI − 19 s.
    Gpst,
}

impl TimeScale {
    /// The exact offset of this scale from TAI, when one constant gives
    /// it: zero for TAI, +32.184 s for TT, and −19 s for GPST.
    ///
    /// The result is `None` for a scale that needs a relativistic model,
    /// Earth orientation data, or leap seconds.
    #[must_use]
    pub const fn fixed_offset_from_tai(self) -> Option<SignedDuration> {
        match self {
            Self::Tai => Some(SignedDuration::ZERO),
            Self::Tt => Some(TT_MINUS_TAI),
            Self::Gpst => Some(GPST_MINUS_TAI),
            Self::Tdb | Self::Tcg | Self::Tcb | Self::Tcl | Self::Ut1 | Self::Utc => None,
        }
    }
}

/// A signed span of time in whole seconds and attoseconds.
///
/// The value is normalized: `0 <= attoseconds < 10^18`, and `seconds`
/// is the floor of the span. Thus −0.25 s is `seconds = −1` and
/// `attoseconds = 750_000_000_000_000_000`. The derived [`Ord`] compares
/// `(seconds, attoseconds)` in sequence, which is the correct order for
/// all values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SignedDuration {
    seconds: i64,
    attoseconds: i64,
}

impl SignedDuration {
    /// The zero span.
    pub const ZERO: Self = Self {
        seconds: 0,
        attoseconds: 0,
    };
    /// The most negative span.
    pub const MIN: Self = Self {
        seconds: i64::MIN,
        attoseconds: 0,
    };
    /// The most positive span.
    pub const MAX: Self = Self {
        seconds: i64::MAX,
        attoseconds: ATTOSECONDS_PER_SECOND - 1,
    };

    /// Builds a span from whole seconds and any attosecond count. The
    /// attoseconds carry into the seconds with floor division. The result
    /// is `None` when the seconds overflow `i64`.
    #[must_use]
    pub const fn new(seconds: i64, attoseconds: i64) -> Option<Self> {
        let carry = attoseconds.div_euclid(ATTOSECONDS_PER_SECOND);
        match seconds.checked_add(carry) {
            Some(seconds) => Some(Self {
                seconds,
                attoseconds: attoseconds.rem_euclid(ATTOSECONDS_PER_SECOND),
            }),
            None => None,
        }
    }

    /// Builds a span of whole seconds.
    #[must_use]
    pub const fn from_seconds(seconds: i64) -> Self {
        Self {
            seconds,
            attoseconds: 0,
        }
    }

    /// Builds a span from a total attosecond count. The result is `None`
    /// when the seconds overflow `i64`.
    #[must_use]
    pub const fn from_attoseconds(total: i128) -> Option<Self> {
        let seconds = total.div_euclid(ATTOSECONDS_I128);
        if seconds < i64::MIN as i128 || seconds > i64::MAX as i128 {
            return None;
        }
        Some(Self {
            seconds: seconds as i64,
            attoseconds: total.rem_euclid(ATTOSECONDS_I128) as i64,
        })
    }

    /// Builds a span from a total nanosecond count. The result is `None`
    /// when the seconds overflow `i64`.
    #[must_use]
    pub const fn from_nanos(nanos: i128) -> Option<Self> {
        let seconds = nanos.div_euclid(NANOSECONDS_PER_SECOND);
        if seconds < i64::MIN as i128 || seconds > i64::MAX as i128 {
            return None;
        }
        let attoseconds = nanos.rem_euclid(NANOSECONDS_PER_SECOND) * ATTOSECONDS_PER_NANOSECOND;
        Some(Self {
            seconds: seconds as i64,
            attoseconds: attoseconds as i64,
        })
    }

    /// Builds a span from seconds in `f64`. The conversion truncates
    /// below one attosecond. The result is `None` for a non-finite value
    /// or a value outside the `i64` second range.
    #[must_use]
    pub fn from_seconds_f64(seconds: f64) -> Option<Self> {
        // 2^63 s: the first magnitude that `i64` cannot hold.
        const LIMIT_S: f64 = 9_223_372_036_854_775_808.0;
        if !seconds.is_finite() || seconds.abs() >= LIMIT_S {
            return None;
        }
        // `as` truncates toward zero. The difference is exact because
        // `whole` is the integer part of an `f64`.
        let whole = seconds as i64;
        let fraction = seconds - whole as f64;
        let attoseconds = (fraction * ATTOSECONDS_PER_SECOND as f64) as i64;
        Self::new(whole, attoseconds)
    }

    /// The floor of the span in whole seconds.
    #[must_use]
    pub const fn seconds(self) -> i64 {
        self.seconds
    }

    /// The attoseconds after [`Self::seconds`], in `[0, 10^18)`.
    #[must_use]
    pub const fn subsec_attoseconds(self) -> i64 {
        self.attoseconds
    }

    /// The span as a total attosecond count. The count of every span
    /// fits in `i128`.
    #[must_use]
    pub const fn as_attoseconds(self) -> i128 {
        self.seconds as i128 * ATTOSECONDS_I128 + self.attoseconds as i128
    }

    /// The span in whole nanoseconds, rounded toward negative infinity.
    #[must_use]
    pub const fn as_nanos_floor(self) -> i128 {
        self.seconds as i128 * NANOSECONDS_PER_SECOND
            + self.attoseconds as i128 / ATTOSECONDS_PER_NANOSECOND
    }

    /// Whether the span is less than zero.
    #[must_use]
    pub const fn is_negative(self) -> bool {
        self.seconds < 0
    }

    /// The sum, or `None` on overflow.
    #[must_use]
    pub const fn checked_add(self, rhs: Self) -> Option<Self> {
        Self::from_attoseconds(self.as_attoseconds() + rhs.as_attoseconds())
    }

    /// The difference, or `None` on overflow.
    #[must_use]
    pub const fn checked_sub(self, rhs: Self) -> Option<Self> {
        Self::from_attoseconds(self.as_attoseconds() - rhs.as_attoseconds())
    }

    /// The negation, or `None` for [`Self::MIN`].
    #[must_use]
    pub const fn checked_neg(self) -> Option<Self> {
        Self::from_attoseconds(-self.as_attoseconds())
    }

    /// The magnitude. [`Self::MIN`] has no positive counterpart, so its
    /// magnitude saturates to [`Self::MAX`].
    #[must_use]
    pub const fn abs(self) -> Self {
        if !self.is_negative() {
            return self;
        }
        match self.checked_neg() {
            Some(magnitude) => magnitude,
            None => Self::MAX,
        }
    }

    /// The span times `numerator / denominator`, rounded toward negative
    /// infinity, in exact integer arithmetic. The result is `None` when
    /// `denominator` is not positive or on overflow.
    ///
    /// The calculation splits the span into seconds and attoseconds, so
    /// a large span times a ratio of large integers does not overflow
    /// `i128`. A relativistic rate such as `6_969_290_134 / 10^19` thus
    /// applies to any span of several thousand years.
    #[must_use]
    pub fn checked_mul_ratio(self, numerator: i128, denominator: i128) -> Option<Self> {
        if denominator <= 0 {
            return None;
        }
        // self = s·A + a, so self·N/D = q·A + (r·A + a·N)/D when
        // s·N = q·D + r with 0 <= r < D.
        let seconds_times_n = i128::from(self.seconds).checked_mul(numerator)?;
        let quotient = seconds_times_n.div_euclid(denominator);
        let remainder = seconds_times_n.rem_euclid(denominator);
        let fraction = remainder
            .checked_mul(ATTOSECONDS_I128)?
            .checked_add(i128::from(self.attoseconds).checked_mul(numerator)?)?;
        let total = quotient
            .checked_mul(ATTOSECONDS_I128)?
            .checked_add(fraction.div_euclid(denominator))?;
        Self::from_attoseconds(total)
    }

    /// The span in seconds as `f64`. Use it only for a short span. A span
    /// of one year keeps a resolution of approximately 4 ns, and the
    /// resolution decreases as the span increases.
    #[must_use]
    pub fn to_seconds_f64(self) -> f64 {
        self.seconds as f64 + self.attoseconds as f64 / ATTOSECONDS_PER_SECOND as f64
    }
}

/// One instant on one [`TimeScale`]: the signed span from the J2000.0
/// epoch of that scale.
///
/// Two instants on different scales have no difference without an
/// explicit conversion. [`Self::elapsed_since`] refuses such a pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PhysicalInstant {
    scale: TimeScale,
    offset_from_j2000: SignedDuration,
}

impl PhysicalInstant {
    /// Builds an instant from its scale and its span from J2000.0.
    #[must_use]
    pub const fn new(scale: TimeScale, offset_from_j2000: SignedDuration) -> Self {
        Self {
            scale,
            offset_from_j2000,
        }
    }

    /// Builds an instant from whole seconds and any attosecond count
    /// from J2000.0. The result is `None` when the seconds overflow.
    #[must_use]
    pub const fn from_j2000(scale: TimeScale, seconds: i64, attoseconds: i64) -> Option<Self> {
        match SignedDuration::new(seconds, attoseconds) {
            Some(offset_from_j2000) => Some(Self::new(scale, offset_from_j2000)),
            None => None,
        }
    }

    /// The J2000.0 epoch of `scale`: 2000-01-01T12:00:00 in that scale.
    #[must_use]
    pub const fn j2000(scale: TimeScale) -> Self {
        Self::new(scale, SignedDuration::ZERO)
    }

    /// The time scale of the instant.
    #[must_use]
    pub const fn scale(self) -> TimeScale {
        self.scale
    }

    /// The span from the J2000.0 epoch of [`Self::scale`].
    #[must_use]
    pub const fn offset_from_j2000(self) -> SignedDuration {
        self.offset_from_j2000
    }

    /// The floor of the span from J2000.0 in whole seconds.
    #[must_use]
    pub const fn seconds_since_j2000(self) -> i64 {
        self.offset_from_j2000.seconds()
    }

    /// The attoseconds after [`Self::seconds_since_j2000`].
    #[must_use]
    pub const fn subsec_attoseconds(self) -> i64 {
        self.offset_from_j2000.subsec_attoseconds()
    }

    /// The signed span from `earlier` to `self`. The result is `None`
    /// when the two scales are different or on overflow. A difference
    /// across two scales needs an explicit conversion first.
    #[must_use]
    pub fn elapsed_since(self, earlier: Self) -> Option<SignedDuration> {
        if self.scale != earlier.scale {
            return None;
        }
        self.offset_from_j2000
            .checked_sub(earlier.offset_from_j2000)
    }

    /// This instant moved by `delta` on the same scale, or `None` on
    /// overflow.
    #[must_use]
    pub fn checked_add(self, delta: SignedDuration) -> Option<Self> {
        let offset = self.offset_from_j2000.checked_add(delta)?;
        Some(Self::new(self.scale, offset))
    }

    /// This instant moved back by `delta` on the same scale, or `None` on
    /// overflow.
    #[must_use]
    pub fn checked_sub(self, delta: SignedDuration) -> Option<Self> {
        let offset = self.offset_from_j2000.checked_sub(delta)?;
        Some(Self::new(self.scale, offset))
    }

    /// The same instant on TAI, when a fixed offset relates the scale of
    /// the instant to TAI (see [`TimeScale::fixed_offset_from_tai`]). The
    /// result is `None` for any other scale or on overflow.
    #[must_use]
    pub fn to_tai_fixed(self) -> Option<Self> {
        let offset = self.scale.fixed_offset_from_tai()?;
        let tai = self.offset_from_j2000.checked_sub(offset)?;
        Some(Self::new(TimeScale::Tai, tai))
    }

    /// The instant `tai` on `target`, when a fixed offset relates
    /// `target` to TAI. The result is `None` when `tai` is not on TAI,
    /// when `target` needs a model, or on overflow.
    #[must_use]
    pub fn from_tai_fixed(tai: Self, target: TimeScale) -> Option<Self> {
        if tai.scale != TimeScale::Tai {
            return None;
        }
        let offset = target.fixed_offset_from_tai()?;
        let shifted = tai.offset_from_j2000.checked_add(offset)?;
        Some(Self::new(target, shifted))
    }
}
