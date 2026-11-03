# ADR-0019: Space trajectory laws and multiphase guidance

- Status: Proposed
- Date: 2026-11-03

## Context

ADR-0015 adds the `CartesianSolution` and the `SpaceIntegrityAssessment`.
ADR-0016 adds `navigate-astrodynamics`, which gives the body models and
the gravity fields. But `navigate-guidance` gives only two surface
setpoints: deviation tracking and an NED velocity. These problems
remain:

1. The contract has no setpoint for a thrust vector, a bank angle, or a
   velocity in a Cartesian frame. Thus an orbital burn, an atmospheric
   entry, a powered descent, and a rendezvous have no command.
2. No crate holds the trajectory laws of these flight phases.
3. No guidance function admits a Cartesian solution. A space solution
   in the wrong frame, or with too much position uncertainty, must give
   a refusal before a law reads it (ADR-0004).
4. No function changes the guidance from one flight phase to the next.
   A vehicle that flies from orbit to a runway, or from orbit to a
   touchdown, needs one sequence of phases.

## Decision

### Contract setpoints

`GuidanceSetpoint` gets three variants. Each variant is `Copy` and does
not allocate.

- `ThrustVector`: the frame, the thrust direction as a unit vector, the
  specific force in m/s², the magnitude of the velocity to go, and the
  `cutoff` flag.
- `EntryBank`: the bank angle in `[-π, π]`, the reference drag, the drag
  error, and the azimuth to the landing interface in `[0, 2π)`.
- `CartesianVelocity`: the frame and the velocity in m/s.

`GuidanceSetpoint` is `#[non_exhaustive]`, so the change is additive.
`CONTRACT_VERSION` stays 1.

### The `navigate-trajectory` crate

`navigate-trajectory` is a new crate. It is `#![no_std]`, it does not
allocate, and it has the features `std` (the default) and `libm`. Each
law reads a `StateVector` and a target, and gives one command or one
`TrajectoryError`. The crate reads no clock.

- `burn`: a `BurnArc` holds the frame, a `BurnTarget`, the ignition
  window, the cutoff threshold of the velocity to go, and the specific
  force. `evaluate_burn` calculates `v_go = v_req(r) − v` and applies the
  cross-product steering of Battin. The target `Apsides` gives the
  velocity on the conic with a fixed periapsis radius and a fixed
  apoapsis radius in the current orbit plane. A coast along that conic
  keeps the required velocity, so the steering stays correct over a long
  deorbit burn. The engine stops when `|v_go|` is the threshold or less.
- `rendezvous`: `LvlhFrame` gives the radial, along-track, and
  cross-track axes of a target. `cw_transfer_velocity` solves the
  two-impulse Clohessy–Wiltshire transfer. It refuses a time of flight
  near a whole number of half orbits, because the transfer matrix is
  singular there. `ApproachCorridor` holds a keep-out sphere and an
  approach cone. In the sphere, the chaser must be in the cone, and the
  scaled position uncertainty must fit in the margin to the cone surface.
  `evaluate_rendezvous` checks the corridor and then commands the
  inertial velocity of the transfer.
- `entry`: an `EntryCorridor` borrows its `EntryNode` knots from the
  caller. Each knot gives the reference drag and the reference drag rate
  at one specific energy `E = v²/2 − μ/r`. The law commands the vertical
  lift-to-drag ratio from the equilibrium glide plus a drag error term
  and a drag rate term. The gains give the drag error a set natural
  frequency and damping ratio. The bank sign changes when the heading
  error to the landing site goes out of the deadband on the side of the
  bank. `EntryState` holds the bank sign and the count of reversals. The
  count wraps. The law gives `taem_reached` at the TAEM energy.
- `powered_descent`: the E-guidance law fits a quadratic acceleration
  profile that meets the target position, velocity, and acceleration at
  the time to go `t`. The commanded acceleration is
  `a = 12·(p_f − p − v·t)/t² − 6·(v_f − v)/t + a_f`, and the thrust is
  `a − g(p)`. The form with `6/t²` and `−2/t` meets only the position and
  the velocity, so this crate does not use it. A floor on `t` keeps the
  gains bounded. The engine stops in the capture box. A glide-slope cone
  applies inside a set slant range, because the curvature of the body
  puts a far vehicle below the horizon of the target.

### Space guidance

- `SpaceGuidanceConfig` holds the minimum quality, the maximum age, and
  the maximum one-sigma position uncertainty.
- `admit_space_solution` enforces, in this order: the clock domain, the
  quality floor, the clock order, the age, the frame
  (`FrameMismatch`), and the uncertainty along the largest principal
  axis (`UncertaintyExceeded`). The surface admission and the space
  admission share the checks of the stamp.
- `guide_burn`, `guide_rendezvous`, `guide_entry`, and
  `guide_powered_descent` admit the solution and evaluate one law. A
  corridor or glide-slope refusal of a law becomes `CorridorViolated`.
  Every other error of a law becomes `TrajectoryInvalid`.
- `guide_rendezvous` refuses a frame that is not inertial. It admits the
  chaser and the target, and it checks the corridor with the root sum
  square of their uncertainties.
- `guide_powered_descent` reads the gravity from the `PlanetaryBody` of
  the plan, at the position of the solution.

### Mission phases

`step_mission` reads a `MissionState`, a `MissionProfile`, and one
`MissionInput`. It gives the next state and the command of the phase.
The state is `Copy` and holds the `EntryState` inline.

- `MissionProfile::Reentry`: `OrbitalCoast` → `Burn` when the ignition
  window opens → `EntryCorridor` at burn cutoff → `AtmosphericRoute` at
  the TAEM energy. A coast after the window closes gives
  `TrajectoryInvalid`. The route phase reads a `NavigationSolution` and
  the active `navigate_route::Leg`, and it calls `guide`.
- `MissionProfile::Landing`: `OrbitalCoast` → `PoweredDescent` at the
  ignition time → `Landed` at cutoff.
- `MissionProfile::expected_frame` gives the frame that each phase
  reads. The host publishes the solution in that frame.
- An input of the wrong kind gives `PhaseInputMismatch`. A phase that
  is not in the profile gives `PhaseNotInProfile`.

`navigate-guidance` depends on `navigate-route`, `navigate-trajectory`,
and `navigate-astrodynamics`.

## Consequences

- One guidance crate gives the commands of a winged vehicle from orbit to
  the first leg of a runway route, and of a lander from orbit to
  touchdown.
- A space solution in the wrong frame, or with too much uncertainty for
  the phase, gives a typed refusal. A landing sets a tight bound in
  `SpaceGuidanceConfig`.
- The entry law controls the drag profile and the heading. It does not
  control the range. The caller selects the knots for the range of the
  mission.
- The Clohessy–Wiltshire law is correct near a target on a circular
  orbit. A direct transfer can cross the keep-out sphere outside the
  cone. The corridor check refuses such a transfer, so the caller must
  plan a hold point at the entry of the corridor.
- The E-guidance law does not model the Coriolis acceleration. The
  closed loop corrects the error.
- A consumer that matches on `GuidanceSetpoint` or on `GuidanceRefusal`
  gets the new variants in its wildcard arm. It must handle them there
  or add arms for them.
