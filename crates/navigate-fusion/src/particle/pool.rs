//! Particles and the caller-owned storage of the particle filter
//! (ADR-0016).
//!
//! The filter does not allocate. The caller gives two slices of the same
//! length: the active particles and a scratch area. An update writes the
//! new particles into the scratch area and checks them. Only then does
//! the filter swap the two slices, in constant time. Thus a refused
//! update changes nothing, and a resampling step copies each particle
//! one time.

use core::mem;

use nalgebra::{SVector, Vector3};

use super::{L, MpfError};

/// One particle: a position in the tangent plane, the conditional mean
/// of the linear states, and a log weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MpfParticle {
    pub(crate) position: Vector3<f64>,
    pub(crate) linear: SVector<f64, L>,
    pub(crate) log_weight: f64,
}

impl MpfParticle {
    /// A particle at the plane origin with zero linear states and a zero
    /// log weight. Use it to fill the storage of a [`ParticlePool`].
    pub const ZERO: Self = Self {
        position: Vector3::new(0.0, 0.0, 0.0),
        linear: SVector::<f64, L>::from_array_storage(nalgebra::ArrayStorage([[0.0; L]])),
        log_weight: 0.0,
    };

    /// The position in the NED axes of the plane origin, in meters.
    #[must_use]
    pub fn position_plane_m(&self) -> [f64; 3] {
        self.position.into()
    }

    /// The conditional mean velocity in the NED axes of the plane origin,
    /// in m/s.
    #[must_use]
    pub fn velocity_plane_mps(&self) -> [f64; 3] {
        [self.linear[0], self.linear[1], self.linear[2]]
    }

    /// The natural logarithm of the normalized weight.
    #[must_use]
    pub const fn log_weight(&self) -> f64 {
        self.log_weight
    }
}

/// The active particles and a scratch area of the same length, both
/// owned by the caller.
#[derive(Debug)]
pub struct ParticlePool<'a> {
    active: &'a mut [MpfParticle],
    scratch: &'a mut [MpfParticle],
}

impl<'a> ParticlePool<'a> {
    /// A pool over `active` and `scratch`. The filter overwrites both
    /// slices.
    ///
    /// # Errors
    ///
    /// [`MpfError::InvalidConfig`] when a slice is empty or the two
    /// slices have different lengths.
    pub fn new(
        active: &'a mut [MpfParticle],
        scratch: &'a mut [MpfParticle],
    ) -> Result<Self, MpfError> {
        if active.is_empty() || active.len() != scratch.len() {
            return Err(MpfError::InvalidConfig);
        }
        Ok(Self { active, scratch })
    }

    /// The number of particles.
    #[must_use]
    pub fn len(&self) -> usize {
        self.active.len()
    }

    /// Whether the pool has no particle. A pool from [`Self::new`] always
    /// has at least one.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.active.is_empty()
    }

    /// The active particles.
    #[must_use]
    pub fn particles(&self) -> &[MpfParticle] {
        self.active
    }

    pub(crate) fn active_mut(&mut self) -> &mut [MpfParticle] {
        self.active
    }

    /// The active particles and the scratch area.
    pub(crate) fn split(&mut self) -> (&[MpfParticle], &mut [MpfParticle]) {
        (self.active, self.scratch)
    }

    /// Makes the scratch area the active particles.
    pub(crate) fn swap(&mut self) {
        mem::swap(&mut self.active, &mut self.scratch);
    }
}
