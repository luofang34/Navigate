//! A deterministic random source: `SplitMix64` and the Box–Muller
//! transform. Identical seeds give identical particle clouds.

/// The `SplitMix64` generator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SplitMix64 {
    state: u64,
    spare: Option<u64>,
}

impl SplitMix64 {
    pub(crate) const fn new(seed: u64) -> Self {
        Self {
            state: seed,
            spare: None,
        }
    }

    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A uniform value in `[0, 1)` with 53 random bits.
    pub(crate) fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1_u64 << 53) as f64)
    }

    /// A standard normal value. The Box–Muller transform makes two values
    /// from two uniform values. The second value waits for the next call
    /// as raw bits, so the state stays `Eq`.
    pub(crate) fn gaussian(&mut self) -> f64 {
        if let Some(bits) = self.spare.take() {
            return f64::from_bits(bits);
        }
        // `1 − u` is in `(0, 1]`, so the logarithm is finite.
        let radius = (-2.0 * (1.0 - self.uniform()).ln()).sqrt();
        let angle = core::f64::consts::TAU * self.uniform();
        let (sin, cos) = angle.sin_cos();
        self.spare = Some((radius * sin).to_bits());
        radius * cos
    }
}
