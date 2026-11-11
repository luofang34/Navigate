# ADR-0023: Optional error-state blocks, and time-correlated source errors

- Status: Proposed
- Date: 2026-11-11

## Context

The inertial filter of `navigate-fusion` (ADR-0014, ADR-0017) has one
error state of 22 states: position, velocity, attitude, accelerometer and
gyroscope biases, a barometric altitude bias, a magnetic map bias, an
empirical acceleration, and a receiver clock. Each filter pays for each
state. Most vehicles need few of them: an aircraft with an altimeter
needs 16. The cost grows with the square and the cube of the dimension,
and it matters most in the browser and on a microcontroller.

Some sources have an error that is correlated in time:

- A single-frequency GNSS receiver has meter-level ionosphere, orbit, and
  clock residuals with a correlation time of minutes. A filter that takes
  each fix as white averages this error over the fixes as if it were
  noise. On a turning route with 1 Hz fixes (white 0.6 m horizontal and
  1.0 m vertical, correlated 1.2 m and 2.0 m, 300 s), the mean 3-D
  position NEES over 12 seeds is 16.5 to 29.4. The band is 1.28 to 5.67.
- A star camera has a mount error that is constant. Declared as white
  variance on each line of sight, it gives a mean attitude NEES of 17.8
  over 20 seeds. The band is 1.59 to 4.98.
- The anchors of one visual map cell share the registration error of the
  cell. `navigate-visual` labels them with `FusionEligibility`, and a
  `SharedMapError` anchor is not an independent fix.

## Decision

### 1. A core block and optional blocks

`StateLayout` gives the offsets of the error state. The core block
(position, velocity, attitude, and the two IMU biases, 15 states) has
fixed indices. Each optional block has an offset or none: the altitude
bias, the magnetic bias, the empirical acceleration, the clock, and up to
`MAX_CORRELATED_BLOCKS` correlated-error blocks of three states.
`InertialNavigationFilter<N>` stores its covariance with the dimension
`N` of its layout. The constructors `surface_with_layout` and
`space_with_layout` refuse a layout that does not fit `N`, and a
configuration with a model whose block the layout does not have.

The named layouts are `FULL` (22), `CORE` (15), and `AVIATION` (16).
`with_correlated_blocks(k)` adds `k` blocks to a layout. The layout is a
value, not a Cargo feature, so one build serves each vehicle.

A block that has no term in the core dynamics changes nothing when no
measurement reads it: `FULL`, `CORE`, and `AVIATION` give the same core
estimate bit for bit, and a layout with an unused correlated block gives
the same core estimate within `1e-9`.

### 2. The source declares a correlated error

An `Observation` can carry a `CorrelatedError`: a first-order
Gauss–Markov process with the correlation time `tau_s` and the stationary
1-sigma about three axes. The covariance of the observation then holds
the white part only. `tau_s = ∞` declares a constant error. No
declaration is a white error, and that path does not change.

The source declares the process because only the source knows its error
model. The filter keys the process by the source and the epoch of the
stamp, and by its kind: a position offset for position fixes, or a
rotation of the sensor reference for attitude fixes and lines of sight.
A source can change its declaration from fix to fix, for example with
its satellite geometry; the process stays the same, and the latest
declaration sets its propagation.
Other values with a declaration are refused with
`CorrelationNotModelled`, because the white model would be
over-confident. The kinematic filter has no correlated states and refuses
each declaration in the same way.

### 3. A bound block, or an inflated fix

When the layout has a free correlated block, the source binds it. The
block is a Gauss–Markov state with the declared parameters, and the fix
reads it with the columns of its own position or attitude rows. The
filter then estimates the error and its correlation with the core
states. The blocks have no term in the continuous dynamics, so their
transition `ρ·I` and noise `(1−ρ²)·Σ` are exact.

When no block is free, the fix covariance grows by `Σ·(1+ρ)/(1−ρ)`, with
`ρ = e^(−Δt/τ)` and `Δt` the time since the last accepted fix of the
source. Over a correlation time the fixes then carry the information of
the independent samples of the process. A fix whose factor is above
`MAX_INFLATION` is refused with `CorrelatedErrorRepeated`: a fast source
decimates itself, and a constant error is used once. The filter tracks
the last fix of up to `MAX_UNBOUND_SOURCES` sources, and refuses a new
source when the table is full (`CorrelatedSourceUntracked`).

A block stays bound until its source starts a new epoch, which starts a
new process, or until `RELEASE_TAUS` correlation times pass without a
fix. The filter does not take a block from a source that still uses it:
the old error of a returning source would then be estimated as a new,
independent error, which is the defect this decision removes.

The bindings, the last-fix times, and the block means are part of the
filter core. A replay of a delayed fix and a fork of the filter thus
reproduce them. A delayed fix can take a block before the fixes of
another source that arrived first. The replay then gives those fixes
the plan they would have had in order, and skips a fix that would have
been refused in order.

### 4. Keyed processes, and the map cells of terrain anchors

A declaration can carry a key (`CorrelatedError::with_key`). The key and
the kind alone then name the process, so the fixes of several sources
share it: two terrain cameras that see one map cell share the
registration error of that cell. When no block is free, a constant keyed
process takes over the block of the constant keyed process of the same
kind with the oldest fix. The filter refuses the older key until that
block is taken over again (`CorrelatedProcessRetired`), and a replay
skips such a fix as it would have been refused in order. The filter does
not remember earlier keys: the caller keeps that memory and does not
offer a key that lost its block. `binds_correlated_key` tells the caller
whether a block still holds a key.

The reference navigator declares the map error of each terrain fix, the
registration of the imagery and the height of the terrain, as a constant
correlated error keyed by the map cell of the matched ground. The fix
covariance holds the rest: the relief and calibration terms and the
image geometry. A suite with a terrain camera gets one block for the
map. The `MapCellLedger` of `navigate-visual` is the memory of the cells
that had a fix: a fix of a cell that had one is offered while the block
holds that cell, and withheld as `SharedMapError` once a newer cell took
the block. Every fix of a cell whose error a block holds reaches the
filter, not only the first fix of the cell.

### 5. An altitude update does not correct the horizontal states

The body-fixed error state reads the vertical along the local up of the
nominal position. Two points at the same height `d` apart differ along
that up by the chord `−d²/(2R)`. At kilometers of horizontal error, an
altitude update with full gain spreads this chord through the
vertical–horizontal cross-covariance. The altitude update therefore uses
a gain without its horizontal position and velocity rows, with the
Joseph covariance update (a Schmidt "consider" update of those states).

## Consequences

- A vehicle pays for the blocks it uses. On the development host, one
  IMU step of the release build takes 19.6 µs with 15 states, 24.5 µs
  with 16, and 79.7 µs with 22.
- The filter with its history is 233.6 KB with 19 states, 247.4 KB with
  22, 263.2 KB with 25, and 281.0 KB with 28. Each retained observation
  is 288 bytes with its declaration.
- A correlated GNSS error declared with a bound block gives a mean 3-D
  NEES of 2.2 to 4.3 on the route of the context. Declared with a
  correlation time three times too short or too long, it stays in the
  band (2.1 to 5.4). Without a block, the inflation gives 0.22 to 0.36:
  conservative, and never over-confident. The RMS 3-D position error
  after the first correlation time is 2.55 m with a block, 3.96 m with
  the inflation, and 2.53 m with the over-confident white declaration.
- A star mount error declared as a constant correlated error gives a
  mean attitude NEES of 2.86.
- With one fix for each map cell, a block gives the same core estimate as
  a white fix with the map variance added. In a synthetic loiter in one
  cell with 2 Hz anchors (constant registration and height error, a
  constant calibration angle, a relief term along the view, and white
  matcher noise), the filter takes all 1800 anchors with a mean 3-D NEES
  of 5.6 / 3.1 / 4.4 at 300 / 600 / 900 s; the first-fix rule took one.
  The relief term, declared white but constant in the cell, sets the
  margin: without it the NEES is 2.1 / 1.9 / 2.3.
- A filter with the map block and a correlated GNSS block has 28 states:
  281.0 KB with its history, and a Van Loan stack frame of 188.7 KB on
  thumbv8m.
- The declaration makes each observation 48 bytes larger.
- A source adapter must know its error model to declare it. A white
  declaration of a correlated error stays over-confident.

## Alternatives considered

- **Always 25 states.** Every filter would pay for a block that most
  vehicles do not use. Rejected.
- **Cargo features for the layout.** One build could then serve only one
  vehicle, and the features of a workspace merge. Rejected.
- **Evict the least recently used block.** A source that returns to a
  reset block has its old error estimated as new and independent, which
  makes the filter over-confident. Rejected.
- **A full-gain altitude update with a curvilinear injection.** The
  correction moves the nominal position along the geodetic coordinates
  instead of the body-fixed axes. On a 40 m/s route with 25° turns over
  12 seeds:

  | Variant | Horizontal NEES at 500 / 600 / 900 / 1300 s | Height NEES |
  |---|---|---|
  | Full gain, body-fixed injection | 5.1 / 27.5 / 49.9 / 134 | 0.84–0.95 |
  | Full gain, curvilinear injection | 9.2 / 57.6 / 95.4 / 296 | 0.84–0.95 |
  | Vertical-only gain | 2.45 / 2.52 / 2.55 / 2.46 | 0.84–0.95 |

  The height stays consistent in each variant, and the horizontal breaks
  with full gain in both injections. An altimeter that reads the
  first-order height keeps the full-gain filter consistent, so the chord
  is the cause, and the injection does not remove it. Rejected.
