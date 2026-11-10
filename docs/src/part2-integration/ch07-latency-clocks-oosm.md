# Chapter 7: Latency, Clocks, and Out-of-Sequence Measurements

In a real vehicle, measurements do not arrive at the filter in the order in which the physical events occurred. An IMU sample arrives within a millisecond, a GNSS receiver solution arrives $50\text{ ms}$ to $200\text{ ms}$ later, and a deep-learning visual terrain match (`navigate-visual-flight`) can take $100\text{ ms}$ to $500\text{ ms}$ of GPU compute after the camera shutter closes.

If a filter applies a delayed measurement against the current state $\hat{\mathbf{x}}(t_{\text{now}})$ instead of the state at the shutter capture time $t_{\text{acq}}$, the vehicle motion during $\Delta t = t_{\text{now}} - t_{\text{acq}}$ injects a systematic position bias:

$$\Delta\mathbf{p}_{\text{lag}} \approx \mathbf{v}\,\Delta t$$

At $70\text{ m/s}$, a $400\text{ ms}$ visual processing delay creates a $28\text{ m}$ along-track bias. `Navigate` eliminates this lag bias with a bounded state history ring, exact IMU interval splitting, and deterministic out-of-sequence measurement (OOSM) replay (ADR-0012, ADR-0014, ADR-0017).

## Technical names

- **Acquisition time** (`ObservationStamp::acquired_at`) is the monotonic instant at which the physical sensor sampled the environment (for example, the mid-exposure instant of a star or terrain camera).
- **Out-of-sequence measurement (OOSM)** is an observation whose acquisition time $t_{\text{acq}}$ is earlier than the current filter state time $t_{\text{state}}$.
- **History ring** (`history_ring`) is a fixed-capacity circular buffer on the stack that stores snapshots of the filter state and the applied event at each historical step.

## Bounded state history and deterministic replay

Both `NavigationFilter` and `InertialNavigationFilter` maintain a fixed-capacity history ring configured by `FusionConfig::history_capacity`. In the inertial filter, the history records four kinds of chronological events:

1. An IMU rate sample (`ImuSample`) or increment (`ImuIncrement`).
2. An admitted measurement (`Observation`).
3. A coordinate frame transition (`FrameChange`) or impulsive burn (`apply_impulsive_burn`).
4. A space environment update (`update_space_environment`) or surface ellipsoid attachment (`attach_surface_ellipsoid`).

When an observation arrives with $t_{\text{acq}} < t_{\text{state}}$, the filter executes a five-step transactional rewind and replay:

```mermaid
sequenceDiagram
    participant Caller
    participant Filter as Scratch Filter Core
    participant Ring as Bounded History Ring
    Caller->>Filter: ingest(delayed_obs at t_acq < t_state)
    Filter->>Ring: Find earliest entry after t_acq
    Ring-->>Filter: Restore state snapshot prior to t_acq
    Filter->>Filter: Split spanning IMU interval at t_acq & propagate to t_acq
    Filter->>Filter: Check frame gate, model admissibility & chi-square gate at t_acq
    alt Admission passes at t_acq
        Filter->>Filter: Apply Joseph update at t_acq
        Filter->>Ring: Replay all subsequent IMU, FrameChange & Observation events to t_state
        Filter-->>Caller: Commit scratch core & history; return Admitted
    else Admission fails at t_acq
        Filter-->>Caller: Discard scratch core; increment counter; return Rejected
    end
```

### Exact IMU interval splitting

Suppose an IMU sample spans the time interval $[t_0, t_1]$ with angular rate $\boldsymbol{\omega}$ and specific force $\mathbf{f}$, and a delayed camera observation was captured at $t_{\text{acq}} \in (t_0, t_1)$.

1. The filter rewinds to the snapshot at $t_0$.
2. It propagates the state from $t_0$ to $t_{\text{acq}}$ over $\Delta t_1 = t_{\text{acq}} - t_0$ using the rates of that spanning IMU sample.
3. It evaluates the measurement model, frame gate, and $\chi^2$ innovation gate at $t_{\text{acq}}$ and applies the Kalman update.
4. During replay, it propagates the remaining sub-interval $[t_{\text{acq}}, t_1]$ over $\Delta t_2 = t_1 - t_{\text{acq}}$ with the same IMU sample rates, and then replays all events after $t_1$.

### Fixed-decision replay vs. full chronological gate equivalence

When the filter replays historical observations after $t_{\text{acq}}$, it recomputes each observation's innovation and Joseph covariance update against the newly corrected trajectory, but it preserves the prior admission decision rather than re-evaluating the $\chi^2$ gate. This design preserves **admission monotonicity**: an observation that was already reported as `Admitted` to external callers is never retroactively revoked by a later arrival.

Consequently, out-of-sequence replay produces the exact same state and covariance as chronological in-order arrival when two conditions hold:

1. **Invariant gate decisions:** Inserting the delayed observation at $t_{\text{acq}}$ does not shift any subsequent observation's normalized innovation squared $\gamma_k = \mathbf{y}_k^\top \mathbf{S}_k^{-1} \mathbf{y}_k$ across its $\chi^2$ threshold $\tau$.
2. **Linear or small-relinearization regime:** The measurement and propagation models between $t_{\text{acq}}$ and $t_{\text{state}}$ are linear (as in `NavigationFilter`), or the trajectory correction is small enough that ESKF Jacobian relinearization differences are negligible.

### Frame-aware historical admission

A spacecraft may capture an optical navigation image in `EARTH_J2000` at $t_1$, execute `change_frame` to `MOON_J2000` at $t_2$, and finish processing the image at $t_3 > t_2$. Because `InertialNavigationFilter` checks the `FrameId` gate against the rewound state at $t_1$ (where the active frame was `EARTH_J2000`) and then replays `change_frame` at $t_2$, the delayed observation is admitted in its native frame and transformed into the lunar frame without error (ADR-0014, ADR-0017).

## Delayed covariance propagation in `navigate-visual-flight`

When the 30 Hz flight visual engine (`FlightVisualEngine`) runs a slow background map-anchor search on a worker thread while tracking optical flow at frame rate, the anchor result arrives several frames late. In addition to the filter's OOSM history ring, `FlightVisualEngine` can propagate the anchor pose forward along the accumulated relative odometry increments $(\Delta\mathbf{p}_{\text{odom}}, \mathbf{Q}_{\text{odom}})$ to emit a current-epoch `VisualFix` with inflated covariance when low-rate consumers cannot rewind.
