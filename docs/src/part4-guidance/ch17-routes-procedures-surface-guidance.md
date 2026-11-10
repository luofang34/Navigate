# Chapter 17: Flight Plans, Procedures, and Surface Guidance

A navigation solution exists to steer a vehicle safely along its mission trajectory. In atmospheric and surface flight on the Earth, Mars, or Titan, guidance must sequence flight-plan legs, anticipate banked turns using the local surface gravity of the body, enforce vertical altitude windows, and refuse to steer when the navigation solution loses integrity. `navigate-route` and `navigate-guidance` implement route validation, leg sequencing, and surface guidance setpoints (ADR-0006, ADR-0015, ADR-0016).

## Technical names

- **Path terminator** (`LegPath`) is an Aeronautical Radio Inc. (ARINC 424) / RNAV leg classification that specifies how a flight segment is guided and how it terminates (for example, `TrackToFix`, `DirectToFix`, `CourseToFix`, `HoldToAltitude`).
- **Turn anticipation** (`turn_anticipation_m`) is the lead distance before a fly-by waypoint at which the vehicle initiates a banked turn so that it rolls out tangent to the next leg without overshooting.
- **Cross-track error** ($e_{\text{xtk}}$) is the signed perpendicular distance from the vehicle position to the active great-circle track on the body ellipsoid.

## 1. Flight-plan validation and gravity-scaled leg sequencing (`navigate-route`)

A `FlightPlan` in `navigate-contract::plan` holds an ordered sequence of `Waypoint` items and a `PlanRole` (`Primary`, `Alternate`, `Contingency`, or `LostComm`). `PlanSet` manages loaded procedures so that a loss-of-communication event switches the active `PlanRole` using the same execution state machine.

### One leg vocabulary (ADR-0006)

Every leg carries a `LegPath` from one unified vocabulary (`TrackToFix`, `CourseToFix`, `DirectToFix`, `FixToAltitude`, `CourseToAltitude`, `ValueToAltitude`, `HoldToAltitude`, `HoldToFix`, `HoldToManual`). If a plan contains a leg variant that the current build does not execute, `PlanExecution` refuses the plan by name at activation (`PlanActivationError`) rather than flying it under another leg's geometry guard.

### Gravity-scaled turn anticipation (`navigate-route::turn`)

For a coordinated level turn at groundspeed $V_g$ and maximum bank angle $\phi_{\max}$ on a body with surface gravity $g$ (`EARTH_STANDARD_GRAVITY_MPS2 = 9.80665`, `MARS_SURFACE_GRAVITY_MPS2 = 3.72076`, `TITAN_SURFACE_GRAVITY_MPS2 = 1.352`), the steady turn radius $R_{\text{turn}}$ and fly-by turn anticipation lead distance $d_{\text{lead}}$ for a course change $\Delta\psi$ are:

$$R_{\text{turn}} = \frac{V_g^2}{g \tan\phi_{\max}}, \qquad d_{\text{lead}} = R_{\text{turn}} \tan\!\left(\frac{|\Delta\psi|}{2}\right)$$

Because Mars has $38\%$ of Earth's gravity, an aircraft at the same speed and bank angle on Mars requires $2.63\times$ the turn radius and anticipation distance. `ExecutionConfig` takes $g$ explicitly so a route on Mars or Titan never uses Earth gravity by mistake.

- **`TurnType::FlyBy`:** Sequences to the next leg early at distance $d_{\text{lead}}$ before the waypoint.
- **`TurnType::FlyOver` and terminal waypoints:** Sequence only inside the capture radius `capture_radius_m`.
- **Zero-wind vs. wind-triangle assumption:** The expression $R_{\text{turn}} = V_g^2 / (g \tan\phi_{\max})$ holds exactly when wind is zero ($\mathbf{v}_w = \mathbf{0}$, so groundspeed $V_g$ equals true airspeed $V_a$) or when the inner loop regulates ground-track curvature directly. Under a steady wind vector $\mathbf{v}_w$ ($\mathbf{v}_g = \mathbf{v}_a + \mathbf{v}_w$) at a fixed air-mass bank limit $\phi_{\max}$, the ground trajectory is a trochoid whose worst-case tailwind ground turn radius expands to $(V_a + \|\mathbf{v}_w\|)^2 / (g \tan\phi_{\max})$. When operating in strong winds relative to airspeed, callers should pass an upper-bounded groundspeed or apply a conservative bank-angle margin in `ExecutionConfig`.

## 2. Surface guidance laws and integrity gating (`navigate-guidance`)

`navigate-guidance` provides two pure surface guidance functions:

1. **`guide`**: Outputs a `GuidanceSetpoint::DeviationTracking` command containing cross-track error $e_{\text{xtk}}$, desired course $\psi_{\text{trk}}$, and target altitude $h_{\text{cmd}}$.
2. **`guide_velocity`**: Outputs a `GuidanceSetpoint::Velocity` command containing a 3D local NED velocity vector $\mathbf{v}_{\text{cmd}} = [v_N, v_E, v_D]^\top$.

### Lateral and vertical tracking laws

On the active great-circle segment (`navigate-geodesy::great_circle`), let $e_{\text{xtk}}$ be the cross-track deviation and $\psi_{\text{trk}}$ be the track bearing at the closest point on the leg.

- **Lateral intercept law:** Computes a bounded intercept angle $\Delta\psi_{\text{int}} = -\arctan(k_y e_{\text{xtk}})$ clamped to `max_intercept_angle_rad`, aligning the commanded horizontal velocity vector smoothly onto the great-circle track.
- **Vertical glideslope law:** Evaluates waypoint `AltitudeConstraint` rules (`At`, `AtOrAbove`, `AtOrBelow`, `Between`), clamps the vertical path gradient to `max_climb_gradient` / `max_descent_gradient`, and computes vertical rate $v_D$ toward the active constraint.

### Fail-closed integrity admission

Both `guide` and `guide_velocity` validate the incoming `NavigationSolution` through one shared admission helper before computing any geometry. Guidance refuses and returns `Err(GuidanceRefusal)` when:

- `solution.integrity.quality` is below `GuidanceConfig::min_quality` (`InsufficientQuality`).
- The solution timestamp is older than `max_solution_age` (`StaleSolution`) or lies in the future (`FutureSolution`).
- The clock domain (`ClockDomainMismatch`) or reference ellipsoid (`EllipsoidMismatch`) does not match the guidance configuration.

### Cross-layer fail-safe contract with `Pilotage` and `Aviate`

An aircraft or spacecraft in flight cannot halt in place when `navigate-guidance` returns `Err(GuidanceRefusal)`. In the three-layer architecture (`Navigate`, `Pilotage`, `Aviate`), `Err(GuidanceRefusal)` is an explicit mode-transition trigger rather than an unhandled exception:

1. **`Navigate` (`navigate-guidance`)** refuses to compute a cross-track or velocity setpoint from a stale or degraded position estimate, preventing corrupted navigation states from commanding hard bank angles into terrain.
2. **`Pilotage` (mission and procedure supervision)** catches `GuidanceRefusal`, records the cause (`StaleSolution` vs. `InsufficientQuality`), and selects the appropriate contingency response—such as switching `PlanSet` to `PlanRole::Contingency` or `PlanRole::LostComm`, or commanding a safe climb/loiter escalation.
3. **`Aviate` (inner-loop flight control)** drops back to an inertial/air-data stabilization mode that does not depend on global position (for example, wings-level attitude hold, constant barometric altitude hold, or constant-heading dead-reckoning) until `Navigate` restores an admissible `SolutionQuality`.
