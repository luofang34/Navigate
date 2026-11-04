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
| `navigate-geodesy` | Reference-ellipsoid geodesy on any body: geodetic ↔ body-fixed ↔ local NED, great-circle distance and bearing, cross-track geometry, and vertical datums |
| `navigate-astrodynamics` | Planetary bodies, gravity fields, orbital force models, coordinate time scales, and frame rotations, without heap allocation |
| `navigate-fusion` | The kinematic, inertial, and particle navigation filters: typed observations, admission gates, correlation discipline, and integrity assessment, without heap allocation |
| `navigate-magnav` | Magnetic anomaly navigation: Tolles–Lawson calibration, borrowed grid maps, and Cramér–Rao bounds |
| `navigate-route` | Route validation and leg-sequencing execution on any body |
| `navigate-trajectory` | Space trajectory laws: orbital burns, Clohessy–Wiltshire rendezvous, atmospheric entry corridors, and powered descent, without heap allocation |
| `navigate-guidance` | Surface and space guidance setpoints and mission-phase orchestration |
| `navigate-terrain-safety` | Terrain clearance with typed availability, alert vocabulary, and vertical-datum adapters on any body |
| `navigate-visual` | Camera pose observations, model-inference traits, holdout `L1-GNC` PnP solvers, local scene refinement, and exact body tangent frames at all latitudes on the Earth, the Moon, and Mars. The `session` module keeps continuous camera odometry, map anchors, revisit closures, and per-frame heights. The `fusion_bridge` module converts camera poses into fusion position fixes and visual poses with a declared error budget (ADR-0007, ADR-0020) |
| `navigate-visual-render` | Terrain and orthoimagery reference renderer on CPU and `wgpu` |
| `navigate-match` | Rust `LoFTR`, `XFeat`, `LighterGlue`, and `DINOv2` matchers across `burn` and optional `ort` runtimes. The `segment` module gives SegFormer-B0 occlusion masks, amortized homography warping, horizon estimation, propeller detection, and 4-class training on the same runtimes |
| `navigate-locate` | Sans-IO single-photo pose search, pair/feature localizers, and real-time corridor tracking. The `place` module holds the Hilbert-sharded place index and the bounded region search plans |
| `navigate-visual-flight` | Real-time 30 Hz flight visual engine in memory: borrowed pixel frames, bounded worker queues, shared GPU terrain views, multi-hypothesis anchor verification, and delayed-measurement covariance propagation on any body |
| `navigate-locate-web` | Browser `wasm32` visual locator that uses WebNN with `burn` WebGPU and CPU fallback |
| `navigate-recording` | Camera evidence, recording input and result types, and visual session processing |
| `tools/visual-model` | `navigate-visual-model`: deterministic ONNX and checkpoint converter for `LoFTR`, `XFeat`, and `LighterGlue` |
| `tools/visual-pipeline` | Rust photo and video preparation, corridor tracking, pose search, segmentation training, and `sokoly-debrief` export |
| `tools/navrun` | Deterministic scenario runner that tests fusion, flight plans, and guidance end to end |

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
