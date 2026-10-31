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
4. **Guidance** — deviation-tracking and NED velocity setpoint generation
   toward the active plan, refusing to guide without the integrity the
   decision requires. Position setpoints are contract vocabulary
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
| `navigate-celestial` | Celestial directions from a framed observer state and a reception event; supports non-Earth origins and moving observers |
| `navigate-geodesy` | WGS84 geodesy: geodetic ↔ ECEF ↔ local NED, great-circle distance/bearing, cross-track geometry |
| `navigate-fusion` | The navigation filter: typed observations, admission gates, correlation discipline, integrity assessment |
| `navigate-fpl` | Flight-plan validation and leg-sequencing execution |
| `navigate-guidance` | Lateral/vertical guidance from solution + active leg to setpoints |
| `navigate-egpws` | Terrain-awareness seam: typed availability, alert vocabulary; honest `Unavailable` until a terrain database is bound |
| `navigate-visual` | Camera pose observations, model-inference traits, and local scene refinement |
| `navigate-visual-render` | Terrain and orthoimagery reference renderer on CPU and `wgpu` |
| `navigate-match` | Rust `LoFTR`, `XFeat`, and `LighterGlue` matchers with native and `wasm32` dual-engine dispatch |
| `navigate-locate` | Bounded multi-hypothesis terrain pose search on rendered reference views |
| `navigate-locate-web` | Browser `wasm32` visual locator that uses WebNN with `burn` WebGPU and CPU fallback |
| `navigate-segment` | Rust SegFormer-B0 occlusion masks, propeller detection, and 4-class training |
| `navigate-visual-fusion` | Converts visual camera poses into fusion position fixes with a declared error budget (ADR-0007) |
| `navigate-visual-session` | Keeps continuous camera odometry for each video segment, anchors it to the map, closes revisits, revises past map poses, and gives per-frame heights |
| `navigate-place` | Hilbert-sharded visual place index and bounded region search plans |
| `navigate-recording` | Camera evidence, recording input and result types, and visual session processing |
| `tools/visual-model` | Deterministic ONNX-to-SafeTensors weight converter for `LoFTR`, `XFeat`, and `LighterGlue` |
| `tools/recording-worker` | JSON Lines worker for image matching and recording processing |
| `tools/visual-pipeline` | Rust photo and video preprocessing, pose search, segmentation training, and `sokoly-debrief` export |
| `tools/navrun` | Deterministic scripted scenario runner exercising fusion → plan → guidance end to end |

`docs/DESIGN.md` is the orientation map; `docs/adr/` holds the decision
records.

`navigate-celestial` takes position and velocity in metres and metres per TDB
second. Each state has a NAIF origin, a NAIF orientation, and a time scale.
The crate is an opt-in dependency. Other navigation crates do not depend on it.
The default build has no ANISE backend. Enable its `anise` feature to use
`AniseEphemeris`. The host supplies kernels and explicit NAIF-to-ANISE
orientation ID mappings. Other providers implement the `Ephemeris` trait.
Geometric queries and reception queries are separate modes. Reception queries
use Euclidean light time and special-relativistic aberration. They do not model
gravitational light deflection, gravitational delay, refraction, or clock drift.
The WGS84/NED `NavigationSolution` belongs to Earth navigation. Do not encode
states from other bodies as WGS84 positions to enter that filter.

Use the [visual recording pipeline](tools/visual-pipeline/README.md) to process
photo datasets and export camera views for Sokoly.

Build the visual API documentation with `cargo doc -p navigate-visual --open`.

## Quality gates

`./ci.sh` runs the full local gate: `fmt --check`, `clippy --all-targets
-- -D warnings`, `test --all-targets`, `doc` with `-D missing_docs` and
broken-link denial, structure checks (file size, no `mod.rs`), and a
release build. CI runs the same gates.

## Licence

Navigate is free software under the GNU Affero General Public License,
version 3 or later. Refer to [LICENSE](LICENSE). Some files keep their own
notices, for example `crates/navigate-celestial/LICENSE-ERFA`.
