# Waypoint transition and procedure-constraint requirements

Requirements for turn types and instrument-procedure constraints in the
flight-plan and guidance crates (tracking issue: Navigate #1). Tests cite
these identifiers. The [ADRs](adr/README.md) own architecture; this
document owns behavioral requirements and their sources.

## Sources and scaling

The vocabulary and geometry follow the published concepts used in US
instrument procedure design and RNAV/RNP navigation:

- Fly-by waypoints connect segments by **turn anticipation** — the
  aircraft begins the turn before the fix and rolls out on the next
  course; airspace and obstacle protection assume it. Fly-over waypoints
  require crossing the fix before turning (missed-approach and
  missed-approach-holding waypoints are the canonical fly-over cases).
  (FAA AIM 1-2-1, Performance-Based Navigation and Area Navigation.)
- The **distance of turn anticipation (DTA)** is the along-track
  distance before a fly-by fix at which the turn begins. Turn radius
  derives from groundspeed and bank angle; RNP procedure design uses a
  standard fly-by bank of 18°, and the DO-236C low-altitude rule caps
  bank at half the track change, limited to 23°. (RTCA DO-236C concepts
  as adopted in FAA Order 8260.52 and the FAA PARC bank-angle
  recommendations; AC 90-101 for RNP AR.)
- Altitude constraints in procedure coding include at, at-or-above,
  at-or-below, and **window (between)** forms; vertical paths are flown
  against gradients (climb gradients in ft/NM, descent paths around 3°).

Scaling note: these sources govern crewed fixed-wing procedures. The
requirements below parameterize the same geometry (bank limit,
groundspeed, gradients) so a multirotor at 2 m/s and a fixed-wing at
60 m/s run the same code with different numbers; nothing hard-codes a
vehicle class.

## Turn types

- **NAV-TT-001** Every waypoint carries a turn type: `FlyBy` or
  `FlyOver`. `FlyBy` is the default (the RNAV default); the plan's final
  waypoint is terminal and its turn type has no effect. Turn types come
  from the plan source, never inferred by the executor.
- **NAV-TT-002** Turn geometry derives from a configured
  turn-performance model: radius `r = v² / (g · tan φ)` with `v` the
  commanded groundspeed toward the fix and `φ` a configured bank-angle
  limit (default 18°, the RNP fly-by standard; configurations MAY apply
  the DO-236C low-altitude cap of half the track change limited to 23°).
  The model is pure and unit-tested; no vehicle class is assumed.
- **NAV-TT-003** A fly-by waypoint sequences at
  `max(capture_radius, DTA)` before the fix, where
  `DTA = r · tan(Δtrack / 2)` and `Δtrack` is the course change between
  the inbound and outbound legs. A straight-ahead or terminal fix
  (`Δtrack ≈ 0`) degrades to the capture radius. DTA is capped at the
  leg length so short legs cannot sequence before they begin.
- **NAV-TT-004** A fly-over waypoint sequences only within the capture
  radius of the fix itself (no anticipation). After sequencing, guidance
  tracks the next leg from the overflown fix; the rejoin appears as
  honest lateral deviation that converges — no synthetic intercept path
  is fabricated.
- **NAV-TT-005** Sequencing events expose the turn type and the
  sequencing reason (anticipated vs overflown) so consumers (telemetry,
  tests, displays) can distinguish an early fly-by advance from a
  fly-over crossing.

## Vertical and speed constraints

- **NAV-VC-001** The altitude-constraint vocabulary gains a window form:
  between a lower and an upper altitude (`lower < upper`, validated at
  plan validation). Deviation semantics: inside the window is zero
  deviation; below reports the climb demand (negative, below profile);
  above reports the descent demand (positive, above profile) —
  consistent with the existing one-sided forms.
- **NAV-VC-002** Guidance vertical commands respect, in order: the
  vehicle's configured vertical-rate caps, and a per-leg **gradient
  limit** when the plan declares one (climb or descent, expressed as
  height per along-track distance; procedure sources use ft/NM, the
  vocabulary is dimensionless m/m in canonical units). The commanded
  vertical rate is `min(rate_cap, gradient · groundspeed)`; a leg with
  no declared gradient uses the caps alone.
- **NAV-VC-003** A waypoint MAY carry a maximum-speed constraint. On
  the leg toward that fix, commanded along-track speed is
  `min(cruise, constraint)`. Constraints below the vehicle's minimum
  controllable approach speed floor are refused at plan validation, not
  silently clamped in flight.
- **NAV-VC-004** Constraint provenance follows the plan: fixture and
  test plans set constraints explicitly; procedure-derived plans carry
  them from their source data. The executor never invents a constraint.

## Honesty requirements

- **NAV-HN-001** Turn anticipation never fakes on-course: during a
  fly-by transition the lateral deviation is reported against the leg
  guidance is actually tracking (inbound until sequencing, outbound
  after), so a consumer sees the real geometry of the corner being cut.
- **NAV-HN-002** All new vocabulary is additive: plans without turn
  types, windows, gradients, or speed constraints behave exactly as
  before (fly-by default with pure capture-radius sequencing when DTA is
  degenerate, caps-only vertical, cruise speed).

## Out of scope (recorded, deferred)

Radius-to-fix (RF) legs, holds, DME arcs, wind-compensated transition
areas (DO-236 theoretical transition boundaries), full ARINC-424 leg
types, and procedure databases (CIFP legs arrive through the
communication component's navdata, a separate integration).
