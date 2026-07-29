# Navigate

> ⚠️ **Work in progress — experimental. No guarantees of any kind.**
>
> Navigate is under active development and is **not** production-ready.
> Nothing here has been qualified or certified for any use. It is provided
> as-is, with no warranty or guarantee of any kind. Do not use it to
> navigate or otherwise operate a real vehicle, and do not rely on it in
> any safety-critical context.

Navigate is the global navigation component of a vehicle system: it fuses
heterogeneous navigation sources into a best estimate of ownship state with
explicit integrity, manages and executes flight plans, and produces guidance
setpoints for a flight controller. It is the navigation layer between a
motion-control kernel (such as Aviate) that owns stabilization, and a host
platform (such as Pilotage) that owns sessions, authority, and displays.

## Navigate does exactly four things

1. **Global navigation** — filter-based multi-sensor fusion (GNSS first;
   celestial and visual navigation join through the same typed seams) into a
   navigation-grade ownship solution. Operable with any subset of sources,
   including a single one.
2. **Integrity assessment** — every solution carries explicit quality,
   redundancy, and fault-detection honesty; confidence is never fabricated.
3. **Flight-plan management and execution** — plans, procedures (including
   loss-of-communication procedures), and leg sequencing.
4. **Guidance** — deviation-tracking setpoint generation toward the
   active plan, refusing to guide without the integrity the decision
   requires. Position and velocity setpoints are contract vocabulary
   mirroring the FC's command surface; generating them is deferred.

## Navigate never does

- ❌ Stabilization, inner-loop control, or actuation (the flight
  controller's job)
- ❌ Direct actuator or control-scope access — guidance commands enter the
  host platform's fenced authority path like any other commander
- ❌ Sessions, networking transports, UI, or displays (the host platform's
  job)
- ❌ Advisory aeronautical data sourcing — briefings, NOTAMs, weather come
  from the communication component; Navigate consumes, never fetches
- ❌ Hidden time or I/O in domain logic — cores are sans-IO and
  deterministic; `now` is always an argument

## Workspace layout

| Crate | Responsibility |
|---|---|
| `navigate-contract` | The typed boundary vocabulary external consumers pin: solutions, integrity, stamps, plans, guidance setpoints, draft aiding schema |
| `navigate-geodesy` | WGS84 geodesy: geodetic ↔ ECEF ↔ local NED, great-circle distance/bearing, cross-track geometry |
| `navigate-fusion` | The navigation filter: typed observations, admission gates, correlation discipline, integrity assessment |
| `navigate-fpl` | Flight-plan validation and leg-sequencing execution |
| `navigate-guidance` | Lateral/vertical guidance from solution + active leg to setpoints |
| `navigate-egpws` | Terrain-awareness seam: typed availability, alert vocabulary; honest `Unavailable` until a terrain database is bound |
| `tools/navrun` | Deterministic scripted scenario runner exercising fusion → plan → guidance end to end |

`docs/DESIGN.md` is the orientation map; `docs/adr/` holds the decision
records.

## Quality gates

`./ci.sh` runs the full local gate: `fmt --check`, `clippy --all-targets
-- -D warnings`, `test --all-targets`, `doc` with `-D missing_docs` and
broken-link denial, structure checks (file size, no `mod.rs`), and a
release build. CI runs the same gates.
