# Chapter 5: Strapdown Inertial Navigation and the 22-State Unified ESKF

A kinematic filter cannot integrate inertial sensors, cannot apply attitude-dependent measurements, and cannot transition from a launch pad on a rotating planet to an interplanetary orbit. `InertialNavigationFilter` in `navigate-fusion::inertial` solves these problems with a unified 22-state Error-State Kalman Filter (ESKF) formulated in one body-centered Cartesian frame (ADR-0014, ADR-0017, ADR-0018).

## Technical names

- **Nominal state** is the high-rate full nonlinear kinematic state integrated by the strapdown mechanization or orbital force model.
- **Error state** $\delta\mathbf{x}$ is the small perturbation between the true state and the nominal state, estimated by the linear Kalman recursion.
- **Right attitude perturbation** $\delta\boldsymbol{\theta} \in \mathbb{R}^3$ is an attitude error vector expressed in body FRD axes such that $\mathbf{C}_{\text{true}} = \mathbf{C}_{\text{nom}}\exp([\delta\boldsymbol{\theta}\times])$.
- **Coning and sculling** are high-rate non-commutative rotation and rotation-thrust coupling effects captured in pre-integrated `ImuIncrement` samples.
- **First-order Gauss–Markov (FOGM) bias** (`FogmBias`) is a colored noise process $\dot{b} = -b/\tau + w$ with time constant $\tau$ and stationary variance $\sigma^2$.

## Why one body-centered Cartesian frame?

Many inertial navigation systems maintain a surface filter in a local tangent plane or in geodetic coordinates $(\varphi, \lambda, h)$, and switch to a separate Cartesian filter in orbit. That split creates three defects:

1. Geodetic mechanization equations divide by $\cos\varphi$ and fail at the geographic poles.
2. Tangent-plane filters accumulate curvature errors away from the anchor and require periodic re-anchoring.
3. Switching filters during ascent or atmospheric entry requires ad-hoc covariance handoffs.

`InertialNavigationFilter` keeps its nominal position $\mathbf{p}$, frame-relative velocity $\mathbf{v}$, and body-to-frame quaternion $\mathbf{q}$ in a single body-centered Cartesian frame (`SpaceEnvironment`). On a planetary surface, the frame is the rotating body-fixed frame (`EARTH_ITRF`, `MOON_PA`, `MARS_IAU`) with angular velocity $\boldsymbol{\omega}_{\text{ie}} = [0, 0, \omega]^\top$ and an attached `BodyEllipsoid`. In space, the frame is an inertial or rotating orbital frame.

## The 22-dimensional error state vector

The error state $\delta\mathbf{x} \in \mathbb{R}^{22}$ comprises nine core kinematic states, eight sensor and map bias states, three dynamic model compensation (DMC) states, and two receiver clock states (ADR-0018):

| Indices | Symbol | Description | Frame / Units |
|---|---|---|---|
| `0..3` | $\delta\mathbf{p}$ | Position error | Active frame axes ($\text{m}$) |
| `3..6` | $\delta\mathbf{v}$ | Velocity error | Active frame axes ($\text{m/s}$) |
| `6..9` | $\delta\boldsymbol{\theta}$ | Right attitude error | Body FRD axes ($\text{rad}$) |
| `9..12` | $\delta\mathbf{b}_a$ | Accelerometer bias error | Body FRD axes ($\text{m/s}^2$) |
| `12..15` | $\delta\mathbf{b}_g$ | Gyroscope bias error | Body FRD axes ($\text{rad/s}$) |
| `15` | $\delta b_{\text{alt}}$ | Barometric altitude bias error | Scalar ($\text{m}$) |
| `16` | $\delta b_{\text{mag}}$ | Magnetic anomaly map bias error | Scalar ($\text{T}$) |
| `17..20` | $\delta\mathbf{a}_{\text{dmc}}$ | Empirical acceleration error (DMC) | Active frame axes ($\text{m/s}^2$) |
| `20` | $c\delta t$ | Receiver clock range bias error | Equivalent range ($\text{m}$) |
| `21` | $c\delta\dot{t}$ | Receiver clock range-rate drift error | Equivalent range rate ($\text{m/s}$) |

When `InertialConfig::dmc` or `InertialConfig::clock_model` is `None`, the corresponding rows and columns in the dynamics matrix $\mathbf{F}$, process noise $\mathbf{Q}_c$, and initial covariance $\mathbf{P}_0$ are identically zero. Disabled states never couple into active states.

## Two propagation modes: Sensed IMU and Force Model

### 1. Sensed-IMU mode (`PropagationMode::SensedImu`)

In sensed-IMU mode, the filter integrates bias-corrected specific force $\tilde{\mathbf{f}}_b = \mathbf{f}_b - \hat{\mathbf{b}}_a$ and angular rate $\tilde{\boldsymbol{\omega}}_b = \boldsymbol{\omega}_b - \hat{\mathbf{b}}_g$. The continuous-time nominal kinematics in a frame rotating at $\boldsymbol{\omega}_f$ are:

$$\begin{aligned} \dot{\mathbf{p}} &= \mathbf{v} \\ \dot{\mathbf{v}} &= \mathbf{C}_b^f \tilde{\mathbf{f}}_b + \mathbf{g}(\mathbf{p}) - 2\boldsymbol{\omega}_f \times \mathbf{v} \\ \dot{\mathbf{q}}_b^f &= \frac{1}{2}\mathbf{q}_b^f \otimes \begin{bmatrix} 0 \\ \tilde{\boldsymbol{\omega}}_b \end{bmatrix} - \frac{1}{2}\begin{bmatrix} 0 \\ \boldsymbol{\omega}_f \end{bmatrix} \otimes \mathbf{q}_b^f \end{aligned}$$

Here $\mathbf{g}(\mathbf{p})$ includes the gravitational acceleration and the centrifugal term $-\boldsymbol{\omega}_f \times (\boldsymbol{\omega}_f \times \mathbf{p})$. When the IMU supplies pre-integrated coning and sculling increments (`ImuIncrement`), `ingest_imu_increment` applies the rotational velocity correction $\Delta\mathbf{v}_{\text{rot}} = \Delta\tilde{\mathbf{v}} + \frac{1}{2}(\Delta\tilde{\boldsymbol{\theta}} \times \Delta\tilde{\mathbf{v}})$ and exact frame rotation over $\Delta t$.

Because $\delta\boldsymbol{\theta}$ is a right perturbation in body axes, the frame rotation cancels out of the attitude error block:

$$\begin{aligned} \delta\dot{\mathbf{p}} &= \delta\mathbf{v} \\ \delta\dot{\mathbf{v}} &= -\mathbf{C}_b^f [\tilde{\mathbf{f}}_b\times]\,\delta\boldsymbol{\theta} - \mathbf{C}_b^f \delta\mathbf{b}_a + \frac{\partial \mathbf{g}}{\partial \mathbf{p}}\delta\mathbf{p} - 2[\boldsymbol{\omega}_f\times]\delta\mathbf{v} \\ \delta\dot{\boldsymbol{\theta}} &= -[\tilde{\boldsymbol{\omega}}_b\times]\,\delta\boldsymbol{\theta} - \delta\mathbf{b}_g \end{aligned}$$

### 2. Force-model coast mode (`PropagationMode::ForceModel`)

During an orbital coast, accelerometer noise exceeds the micro-gravity environment. In `PropagationMode::ForceModel`, `OrbitalForceModel` integrates $\dot{\mathbf{v}} = \mathbf{a}(\mathbf{p}, \mathbf{v}) + \mathbf{a}_{\text{dmc}}$ with 4th-order Runge–Kutta steps, using the exact gravity gradient $\partial\mathbf{a}/\partial\mathbf{p}$, drag velocity Jacobian $\partial\mathbf{a}/\partial\mathbf{v}$, and DMC coupling $\mathbf{I}_3$.

## Van Loan discretization and closed-loop reset

To discretize the continuous $22 \times 22$ error dynamics matrix $\mathbf{F}$ and spectral density matrix $\mathbf{Q}_c$ over $\Delta t$, the filter forms the $44 \times 44$ Van Loan block matrix:

$$\mathbf{M} = \begin{bmatrix} -\mathbf{F} & \mathbf{Q}_c \\ \mathbf{0} & \mathbf{F}^\top \end{bmatrix}\Delta t, \qquad \exp(\mathbf{M}) = \begin{bmatrix} \mathbf{A}_{11} & \mathbf{A}_{12} \\ \mathbf{0} & \mathbf{A}_{22} \end{bmatrix}$$

A $[6/6]$ Padé approximant with scaling and squaring evaluates $\exp(\mathbf{M})$ on the stack. Then $\boldsymbol{\Phi} = \mathbf{A}_{22}^\top$ and $\mathbf{Q}_d = \boldsymbol{\Phi}\mathbf{A}_{12}$.

After each admitted measurement update computes the error estimate $\delta\hat{\mathbf{x}}^+$, the filter injects $\delta\hat{\mathbf{x}}^+$ into the nominal state ($\hat{\mathbf{q}}^+ = \hat{\mathbf{q}}^- \otimes \exp(\frac{1}{2}\delta\hat{\boldsymbol{\theta}}^+)$), zeros the error state, and applies the manifold covariance reset Jacobian:

$$\mathbf{P}^+ \leftarrow \mathbf{J}\mathbf{P}^+\mathbf{J}^\top, \qquad \mathbf{J}_{\theta\theta} = \mathbf{I}_3 - \frac{1}{2}[\delta\hat{\boldsymbol{\theta}}^+\times]$$

## Exact frame transitions (`change_frame`)

When a vehicle launches from a rotating surface into an inertial orbit or crosses into the lunar sphere of influence, `change_frame` applies an exact transformation with origin offset $\mathbf{o}$, origin velocity $\dot{\mathbf{o}}$, frame rotation $\mathbf{C}$, and relative angular rate $\boldsymbol{\omega}_{\text{rel}}$:

$$\begin{aligned} \mathbf{p}' &= \mathbf{C}(\mathbf{p} - \mathbf{o}) \\ \mathbf{v}' &= \mathbf{C}\bigl(\mathbf{v} - \dot{\mathbf{o}} - \boldsymbol{\omega}_{\text{rel}} \times (\mathbf{p} - \mathbf{o})\bigr) \\ \mathbf{q}' &= \mathbf{q}_{\mathbf{C}} \otimes \mathbf{q} \end{aligned}$$

The error covariance transforms via $\mathbf{P}' = \mathbf{J}_{\text{frame}}\mathbf{P}\mathbf{J}_{\text{frame}}^\top$ with coupling block $\mathbf{J}_{vp} = -\mathbf{C}[\boldsymbol{\omega}_{\text{rel}}\times]$.
