//! Time-correlated source errors (ADR-0023).
//!
//! A source that declares a [`CorrelatedError`] has an error that a white
//! measurement model does not describe: successive fixes share most of
//! it, so a filter that takes each fix as independent becomes
//! over-confident. The filter handles the declared process in one of two
//! ways.
//!
//! - **Bound**: the source has a correlated-error block of the layout.
//!   The block is a first-order Gauss–Markov state with the declared
//!   correlation time and stationary covariance. The measurement reads
//!   it with the columns of its own position or attitude rows, so the
//!   filter estimates the error and keeps its correlation with the core
//!   states.
//! - **Inflated**: the layout has no free block. The white covariance
//!   grows by `Σ·(1+ρ)/(1−ρ)`, with `ρ = e^(−Δt/τ)` and `Δt` the time
//!   since the last accepted fix of the source. Over a correlation time
//!   the fixes then carry the information of the independent samples of
//!   the process, about one per `2τ`. The first fix of a source adds `Σ`
//!   alone. A fix whose factor is above [`MAX_INFLATION`] is refused, so
//!   a fast source decimates itself to the rate at which its fixes still
//!   carry information, and a constant error (`τ = ∞`) is used once.
//!
//! A block stays bound to its source until the source starts a new
//! epoch, which starts a new process, or until [`RELEASE_TAUS`]
//! correlation times pass without a fix, after which the old process
//! is independent of a new one. A block is not taken from a source that
//! still uses it: a source that returns to a reset block would have its
//! old error estimated as a new, independent one.

use nalgebra::Matrix3;
use navigate_contract::{MonotonicNanos, SourceEpoch, SourceId};

use super::layout::{MAX_CORRELATED_BLOCKS, StateLayout};
use super::nominal::{ATT, ErrorMat, InertialNominal, POS};
use crate::admission::seconds;
use crate::observation::{CorrelatedError, ObservationValue};
use crate::rejection::RejectionReason;

mod measurement;

/// The largest inflation factor `(1+ρ)/(1−ρ)` of an accepted fix.
pub(crate) const MAX_INFLATION: f64 = 1e4;

/// A block or a tracked source is free after this many correlation
/// times without a fix: the correlation is then below `e^(−5)`.
pub(crate) const RELEASE_TAUS: f64 = 5.0;

/// The number of sources without a block whose last fix the filter
/// tracks.
pub(crate) const MAX_UNBOUND_SOURCES: usize = 4;

/// The quantity that a correlated error adds to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ErrorKind {
    /// A position offset in the frame axes.
    Position,
    /// A rotation of the sensor reference in the body axes.
    Attitude,
}

impl ErrorKind {
    /// The kind of the correlated error of `value`, or none for a value
    /// without a correlated-error model.
    pub(crate) fn of(value: &ObservationValue) -> Option<Self> {
        match value {
            ObservationValue::PositionFix { .. }
            | ObservationValue::CartesianPositionFix { .. } => Some(Self::Position),
            ObservationValue::AttitudeFix { .. } | ObservationValue::LineOfSight { .. } => {
                Some(Self::Attitude)
            }
            _ => None,
        }
    }

    /// First index of the core states that the error adds to.
    const fn base(self) -> usize {
        match self {
            Self::Position => POS,
            Self::Attitude => ATT,
        }
    }
}

/// The process of one fix: the source, its epoch, the kind, and the
/// declaration. Without a key, the source, the epoch, and the kind
/// identify the process; with a key, the kind and the key do. A source
/// may change its declaration from fix to fix, for example with its
/// satellite geometry, and the process stays the same.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Process {
    source: SourceId,
    epoch: SourceEpoch,
    kind: ErrorKind,
    declared: CorrelatedError,
}

impl Process {
    /// Whether `other` has the owner of this process: the same kind and
    /// key, or without keys the same kind and source. A source keeps its
    /// block or its entry across its epochs.
    fn same_owner(&self, other: &Self) -> bool {
        self.kind == other.kind
            && match (self.declared.key, other.declared.key) {
                (Some(a), Some(b)) => a == b,
                (None, None) => self.source == other.source,
                _ => false,
            }
    }

    fn same_process(&self, other: &Self) -> bool {
        self.same_owner(other) && (self.declared.key.is_some() || self.epoch == other.epoch)
    }

    /// The kind and the key of a constant keyed process.
    fn constant_key(&self) -> Option<(ErrorKind, u64)> {
        self.declared
            .key
            .filter(|_| self.declared.tau_s.is_infinite())
            .map(|key| (self.kind, key))
    }
}

/// A source with its last accepted fix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Tracked {
    process: Process,
    /// The stationary covariance in the axes of the block.
    shape: Matrix3<f64>,
    last_at: MonotonicNanos,
}

impl Tracked {
    /// Whether the process had no fix for [`RELEASE_TAUS`] correlation
    /// times at `at`.
    fn released(&self, at: MonotonicNanos) -> bool {
        released(&self.process, self.last_at, at)
    }
}

/// A source without a block, with its last accepted fix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Seen {
    process: Process,
    last_at: MonotonicNanos,
}

impl Seen {
    fn released(&self, at: MonotonicNanos) -> bool {
        released(&self.process, self.last_at, at)
    }
}

/// Whether `process`, last fixed at `last_at`, had no fix for
/// [`RELEASE_TAUS`] correlation times at `at`.
fn released(process: &Process, last_at: MonotonicNanos, at: MonotonicNanos) -> bool {
    let tau = process.declared.tau_s;
    tau.is_finite()
        && at
            .elapsed_since(last_at)
            .is_some_and(|dt| seconds(dt) > RELEASE_TAUS * tau)
}

/// The bindings of the correlated-error blocks and the sources without
/// a block.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct CorrelatedSources {
    blocks: [Option<Tracked>; MAX_CORRELATED_BLOCKS],
    unbound: [Option<Seen>; MAX_UNBOUND_SOURCES],
    /// The constant keyed process that each block took over last. The
    /// filter refuses it, because its block no longer holds its error.
    retired: [Option<(ErrorKind, u64)>; MAX_CORRELATED_BLOCKS],
}

/// How one fix applies its correlated error.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum Plan {
    /// The fix reads the block `block` at the state index `offset`.
    Bound {
        block: usize,
        offset: usize,
        kind: ErrorKind,
        reset: bool,
    },
    /// The fix adds `factor·Σ` to its covariance; `entry` is the place
    /// of the source in the table of sources without a block.
    Inflated {
        entry: usize,
        kind: ErrorKind,
        shape: Matrix3<f64>,
        factor: f64,
    },
}

/// The process, its stationary covariance, and the plan of one fix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Pending {
    tracked: Tracked,
    pub(crate) plan: Plan,
    /// The constant keyed process whose block this fix takes over.
    retires: Option<(ErrorKind, u64)>,
}

impl CorrelatedSources {
    /// The plan of a fix of `process` at `at` with the stationary
    /// covariance `shape`.
    fn plan(
        &self,
        process: Process,
        shape: Matrix3<f64>,
        at: MonotonicNanos,
        layout: &StateLayout,
    ) -> Result<Pending, RejectionReason> {
        let tracked = Tracked {
            process,
            shape,
            last_at: at,
        };
        if let Some(key) = process.constant_key()
            && self.retired.contains(&Some(key))
        {
            return Err(RejectionReason::CorrelatedProcessRetired);
        }
        let offsets = layout.correlated();
        let bound =
            self.blocks
                .iter()
                .zip(offsets)
                .enumerate()
                .find_map(|(block, (slot, offset))| {
                    let current = (*slot)?;
                    (current.process.same_owner(&process) && !current.released(at))
                        .then_some((block, offset?, current))
                });
        if let Some((block, offset, current)) = bound {
            // The block of the same process keeps its estimate and takes
            // the latest declaration for its propagation.
            let reset = !current.process.same_process(&process);
            let plan = Plan::Bound {
                block,
                offset,
                kind: process.kind,
                reset,
            };
            return Ok(Pending {
                tracked,
                plan,
                retires: None,
            });
        }
        if let Some(pending) = self.inflated(&tracked, at, false)? {
            return Ok(pending);
        }
        let free =
            self.blocks
                .iter()
                .zip(offsets)
                .enumerate()
                .find_map(|(block, (slot, offset))| {
                    slot.is_none_or(|current| current.released(at))
                        .then_some((block, offset?))
                });
        if let Some((block, offset)) = free {
            let plan = Plan::Bound {
                block,
                offset,
                kind: process.kind,
                reset: true,
            };
            return Ok(Pending {
                tracked,
                plan,
                retires: None,
            });
        }
        if let Some(pending) = self.take_over(&tracked, offsets) {
            return Ok(pending);
        }
        self.inflated(&tracked, at, true)?
            .ok_or(RejectionReason::CorrelatedSourceUntracked)
    }

    /// The plan of a constant keyed process that takes over the block of
    /// the constant keyed process of the same kind with the oldest last
    /// fix, or none.
    fn take_over(
        &self,
        tracked: &Tracked,
        offsets: [Option<usize>; MAX_CORRELATED_BLOCKS],
    ) -> Option<Pending> {
        let (kind, _) = tracked.process.constant_key()?;
        let (block, offset, current) = self
            .blocks
            .iter()
            .zip(offsets)
            .enumerate()
            .filter_map(|(block, (slot, offset))| {
                let current = (*slot)?;
                let (held, _) = current.process.constant_key()?;
                (held == kind).then_some((block, offset?, current))
            })
            .min_by_key(|(_, _, current)| current.last_at)?;
        Some(Pending {
            tracked: *tracked,
            plan: Plan::Bound {
                block,
                offset,
                kind,
                reset: true,
            },
            retires: current.process.constant_key(),
        })
    }

    /// The inflated plan of `tracked` when its source has an entry in the
    /// table of sources without a block, or, with `admit_new`, when the
    /// table has room. A source in the table stays there until it is
    /// released, so its fixes keep one model.
    fn inflated(
        &self,
        tracked: &Tracked,
        at: MonotonicNanos,
        admit_new: bool,
    ) -> Result<Option<Pending>, RejectionReason> {
        let process = tracked.process;
        let found = self.unbound.iter().enumerate().find_map(|(entry, slot)| {
            let current = (*slot)?;
            (current.process.same_owner(&process) && !current.released(at))
                .then_some((entry, current))
        });
        let (entry, factor) = match found {
            Some((entry, current)) if current.process.same_process(&process) => {
                let dt_s = at.elapsed_since(current.last_at).map_or(0.0, seconds);
                let rho = process.declared.correlation(dt_s);
                (entry, (1.0 + rho) / (1.0 - rho))
            }
            Some((entry, _)) => (entry, 1.0),
            None if admit_new => {
                let Some(entry) = self
                    .unbound
                    .iter()
                    .position(|slot| slot.is_none_or(|current| current.released(at)))
                else {
                    return Ok(None);
                };
                (entry, 1.0)
            }
            None => return Ok(None),
        };
        if factor.is_nan() || factor > MAX_INFLATION {
            return Err(RejectionReason::CorrelatedErrorRepeated { factor });
        }
        let plan = Plan::Inflated {
            entry,
            kind: process.kind,
            shape: tracked.shape,
            factor,
        };
        Ok(Some(Pending {
            tracked: *tracked,
            plan,
            retires: None,
        }))
    }

    /// Records the accepted fix of `pending`.
    pub(crate) fn commit(&mut self, pending: &Pending) {
        match pending.plan {
            Plan::Bound { block, .. } => {
                if let Some(slot) = self.blocks.get_mut(block) {
                    *slot = Some(pending.tracked);
                }
                if let (Some(retired), Some(slot)) = (pending.retires, self.retired.get_mut(block))
                {
                    *slot = Some(retired);
                }
            }
            Plan::Inflated { entry, .. } => {
                if let Some(slot) = self.unbound.get_mut(entry) {
                    *slot = Some(Seen {
                        process: pending.tracked.process,
                        last_at: pending.tracked.last_at,
                    });
                }
            }
        }
    }

    /// Advances each bound block by `dt_s` seconds: the mean decays by
    /// `ρ`, the transition block is `ρ·I`, and the process noise block is
    /// `(1−ρ²)·Σ`. The blocks have no term in the continuous dynamics, so
    /// the transition and the noise of the core have identity and zero
    /// there, and this is the exact discretization.
    pub(crate) fn advance<const N: usize>(
        &self,
        layout: &StateLayout,
        dt_s: f64,
        nominal: &mut InertialNominal,
        phi: &mut ErrorMat<N>,
        q_d: &mut ErrorMat<N>,
    ) {
        let offsets = layout.correlated();
        for ((slot, offset), mean) in self.blocks.iter().zip(offsets).zip(&mut nominal.correlated) {
            let (Some(tracked), Some(i)) = (slot, offset) else {
                continue;
            };
            let rho = tracked.process.declared.correlation(dt_s);
            *mean *= rho;
            phi.fixed_view_mut::<3, 3>(i, i)
                .copy_from(&(Matrix3::identity() * rho));
            q_d.fixed_view_mut::<3, 3>(i, i)
                .copy_from(&(tracked.shape * (1.0 - rho * rho)));
        }
    }

    /// Whether a block holds a keyed process with `key`.
    pub(crate) fn binds(&self, key: u64) -> bool {
        self.blocks
            .iter()
            .flatten()
            .any(|tracked| tracked.process.declared.key == Some(key))
    }

    /// Rotates the position blocks and their stationary covariances by
    /// `c`, the rotation from the old frame axes to the new ones. The
    /// attitude blocks are in the body axes and do not change. A source
    /// without a block takes its covariance from each fix.
    pub(crate) fn rotate<const N: usize>(
        &mut self,
        c: &Matrix3<f64>,
        layout: &StateLayout,
        nominal: &mut InertialNominal,
        j: &mut ErrorMat<N>,
    ) {
        let offsets = layout.correlated();
        for ((slot, offset), mean) in self
            .blocks
            .iter_mut()
            .zip(offsets)
            .zip(&mut nominal.correlated)
        {
            if let (Some(tracked), Some(i)) = (slot, offset)
                && tracked.process.kind == ErrorKind::Position
            {
                *mean = c * *mean;
                tracked.shape = c * tracked.shape * c.transpose();
                j.fixed_view_mut::<3, 3>(i, i).copy_from(c);
            }
        }
    }
}
