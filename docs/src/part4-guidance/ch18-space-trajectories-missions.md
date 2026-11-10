# Chapter 18: Space Trajectory Laws and Multi-Phase Mission Orchestration

A spacecraft mission from orbit to a planetary surface passes through distinct dynamical regimes: orbital coast, impulsive or finite-thrust deorbit burn, hypersonic atmospheric entry, terminal area energy management, and either runway approach or powered vertical landing. `navigate-trajectory` and `navigate-guidance` implement closed-loop space trajectory laws and stateless mission-phase orchestration in `#![no_std]` with zero heap allocation (ADR-0019).

## Technical names

- **Velocity-to-go ($\mathbf{v}_{\text{go}}$)** is the remaining velocity vector increment required to reach a target orbital state, accounting for gravity losses during a finite burn.
- **Local-Vertical Local-Horizontal (LVLH) frame** (`LvlhFrame`) is a target-centered rotating orbital frame with $+X$ radial outward ($\hat{\mathbf{r}}$), $+Y$ along-track ($\hat{\boldsymbol{\theta}}$), and $+Z$ cross-track along orbital angular momentum ($\hat{\mathbf{h}}$).
- **Specific mechanical energy** $E$ (`specific_energy_j_per_kg`) is the sum of kinetic and gravitational potential energy per unit mass, $E = \frac{1}{2}v^2 - \mu/r$.
- **Polynomial E-guidance** (`evaluate_powered_descent`) is a closed-loop boundary-value guidance law (Apollo / Chang'e / Falcon heritage) that drives both position and velocity to a soft touchdown target at time-to-go $t_{\text{go}} \to 0$.

## 1. The four space trajectory laws (`navigate-trajectory`)

Each law in `navigate-trajectory` maps a `StateVector` $(\mathbf{r}, \mathbf{v})$ and a typed target to a deterministic guidance command:

| Module | Function | Physical law | Output command |
|---|---|---|---|
| `burn` | `evaluate_burn` | Velocity-to-go $\mathbf{v}_{\text{go}} = \mathbf{v}_d(\mathbf{r}) - \mathbf{v}$ closed-loop steering with cross-product rate feedback, cutoff when $\|\mathbf{v}_{\text{go}}\| \le v_{\text{cutoff}}$ | `BurnCommand` (`ThrustVector` unit direction + throttle / cutoff flag) |
| `rendezvous` | `evaluate_rendezvous` | Clohessy–Wiltshire (Hill) linear relative-motion transfer matrix inversion in the target `LvlhFrame`, plus conical approach-corridor angle and closure-rate checks (`ApproachCorridor`) | `RendezvousCommand` (`CartesianVelocity` target velocity in inertial frame) |
| `entry` | `evaluate_entry` | Apollo/Shuttle drag-acceleration $D(E)$ versus specific energy $E$ tracking using lift-to-drag modulation $\cos\sigma$, with lateral heading-deadband bank reversals ($\pm\sigma$) | `EntryCommand` (`EntryBank` roll angle in radians) |
| `powered_descent` | `evaluate_powered_descent` | Quadratic acceleration profile $\mathbf{a}_{\text{cmd}}(t)$ solving the two-point boundary value problem $(\mathbf{r}, \mathbf{v}) \to (\mathbf{r}_f, \mathbf{v}_f)$ over $t_{\text{go}}$ with a conical terrain glide-slope guard (`GlideSlope`) | `DescentCommand` (`ThrustVector` acceleration vector in frame axes) |

### Key equations

1. **Clohessy–Wiltshire rendezvous transfer (`cw_transfer_velocity`, `[Clohessy1960]`):**
   With mean motion $n = \sqrt{\mu / r_{\text{tgt}}^3}$ and transfer time $T$, the relative position transition blocks $\boldsymbol{\Phi}_{rr}(T)$ and $\boldsymbol{\Phi}_{rv}(T)$ in the target LVLH frame yield the required initial relative velocity:

   $$\delta\mathbf{v}_0^+ = \boldsymbol{\Phi}_{rv}^{-1}(T)\bigl(\delta\mathbf{r}_f - \boldsymbol{\Phi}_{rr}(T)\delta\mathbf{r}_0\bigr)$$

2. **Atmospheric entry drag-versus-energy tracking (`evaluate_entry`, `[Harpold1979]`):**
   Using specific energy $E = \frac{1}{2}v_{\text{rel}}^2 - \mu/r$ as the monotonic independent variable instead of time, the law interpolates reference drag $D_{\text{ref}}(E)$ and drag rate $\dot{D}_{\text{ref}}(E)$ from caller-borrowed `EntryNode` knots, computes the required vertical lift fraction:

   $$\cos\sigma_{\text{cmd}} = \cos\sigma_{\text{ref}} + k_D(D - D_{\text{ref}}) + k_{\dot{D}}(\dot{D} - \dot{D}_{\text{ref}})$$

   and flips the sign of the bank angle $\sigma = \pm\arccos(\cos\sigma_{\text{cmd}})$ whenever the cross-range azimuth error exceeds the energy-scheduled heading deadband.

3. **Polynomial E-guidance powered descent (`evaluate_powered_descent`, `[Cherry1964]`):**
   Given current $(\mathbf{r}, \mathbf{v})$, target $(\mathbf{r}_f, \mathbf{v}_f)$, apparent gravity $\mathbf{g}$, and time-to-go $t_{\text{go}} > 0$:

   $$\mathbf{a}_{\text{thrust}} = -\frac{6}{t_{\text{go}}^2}\bigl(\mathbf{r} - \mathbf{r}_f + \mathbf{v}\,t_{\text{go}}\bigr) + \frac{2}{t_{\text{go}}}(\mathbf{v} - \mathbf{v}_f) - \mathbf{g}$$

   while verifying that the vehicle stays inside the upward conical `GlideSlope` above the landing site.

## 2. Space integrity admission and multi-phase missions (`navigate-guidance`)

Before evaluating any space law, `admit_space_solution` verifies the incoming `CartesianSolution`:

- Frame match (`solution.frame == expected_frame`), monotonic age, and `SpaceIntegrityAssessment::quality`.
- Radial, along-track, and cross-track $1\sigma$ position uncertainty (`SpaceUncertainty`) against `SpaceGuidanceConfig::max_position_1sigma_m`.

### Multi-phase orchestration (`step_mission`)

`step_mission` sequences complete orbit-to-surface profiles (`MissionProfile`) using a `Copy`, zero-heap `MissionState`:

```mermaid
stateDiagram-v2
    [*] --> OrbitalCoast
    OrbitalCoast --> DeorbitBurn : burn_ignition_time reached
    DeorbitBurn --> EntryCorridor : BurnCommand::cutoff (v_go <= threshold)
    EntryCorridor --> SurfaceRoute : Reentry profile & altitude <= handoff_altitude_m
    OrbitalCoast --> PoweredDescent : Landing profile descent ignition
    PoweredDescent --> Touchdown : t_go <= touchdown_threshold_s
    SurfaceRoute --> [*]
    Touchdown --> [*]
```

- **`MissionProfile::ReentryToRoute`**: Sequences `OrbitalCoast` $\to$ `DeorbitBurn` $\to$ `EntryCorridor` $\to$ atmospheric `SurfaceRoute` legs (`navigate-route`).
- **`MissionProfile::OrbitalLanding`**: Sequences `OrbitalCoast` $\to$ `PoweredDescent` $\to$ `Touchdown`.
