# Chapter 9: Radiometric and Satellite Navigation

Radiometric navigation determines position, velocity, and time by measuring the travel time and Doppler frequency shift of radio waves between known emitters (GNSS satellites, lunar beacons, or Deep Space Network ground stations) and the vehicle. `InertialNavigationFilter` supports both **one-way** and **two-way** radiometric measurement models in any Cartesian reference frame (ADR-0018).

## Technical names

- **Pseudorange** $\rho$ is a one-way range measurement corrupted by the time offset between the receiver clock and the transmitter clock.
- **Pseudorange rate** $\dot{\rho}$ is the time derivative of pseudorange, corrupted by the frequency drift of the receiver oscillator.
- **Two-way range** is a round-trip range measurement from a ground station to a spacecraft transponder and back, which cancels the vehicle clock offset.
- **Shapiro delay** $\Delta\rho_{\text{Shapiro}}$ is the relativistic extra light travel path caused by the gravitational potential of a massive body such as the Sun.

## 1. The physical problem and observability

In a one-way system such as GPS, Galileo, or a lunar navigation satellite system, the receiver timestamps signal arrival against its local crystal or atomic oscillator. A clock offset $\delta t$ of $1\,\mu\text{s}$ produces a range bias $c\delta t \approx 300\text{ m}$.

- Without clock states in the filter, a single pseudorange cannot separate position error along the line of sight from clock bias.
- With receiver clock bias $c\delta t$ and clock drift $c\delta\dot{t}$ in the state vector, three geometric effects make the full state observable:
  1. Simultaneous pseudoranges from multiple emitters have different unit line-of-sight vectors $\mathbf{u}_i$, whereas $c\delta t$ enters every measurement with coefficient $+1$.
  2. Over time, orbital or vehicle motion rotates the line-of-sight vector $\mathbf{u}(t)$ while the IMU or orbital force model constrains the trajectory.
  3. In deep space, two-way coherent transponding from an Earth station eliminates the vehicle clock altogether and observes radial distance and velocity across interplanetary distances.

## 2. Mathematical formulation

### Receiver clock dynamics (`ClockNoiseModel`)

States 20 and 21 of `InertialNavigationFilter` track the receiver clock range bias $x_{20} = c\delta t$ ($\text{m}$) and range-rate drift $x_{21} = c\delta\dot{t}$ ($\text{m/s}$). Driven by white frequency noise spectral density $S_f$ ($\text{m}^2/\text{s}$) and random-walk frequency noise spectral density $S_g$ ($\text{m}^2/\text{s}^3$), the discrete transition and process covariance over step $\Delta t$ are:

$$\boldsymbol{\Phi}_{\text{clk}}(\Delta t) = \begin{bmatrix} 1 & \Delta t \\ 0 & 1 \end{bmatrix}, \qquad \mathbf{Q}_{\text{clk}}(\Delta t) = \begin{bmatrix} S_f\Delta t + \frac{1}{3}S_g\Delta t^3 & \frac{1}{2}S_g\Delta t^2 \\ \frac{1}{2}S_g\Delta t^2 & S_g\Delta t \end{bmatrix}$$

### One-way pseudorange and pseudorange rate

Given vehicle position $\mathbf{p}$, velocity $\mathbf{v}$, and emitter position $\mathbf{p}_e$ and velocity $\mathbf{v}_e$ in the active filter frame, let $\mathbf{d} = \mathbf{p} - \mathbf{p}_e$, $r = \|\mathbf{d}\|$, and unit line-of-sight vector $\mathbf{u} = \mathbf{d}/r$.

- **`ObservationValue::Pseudorange`**:
  With Sagnac and atmospheric media correction $\Delta\rho_{\text{corr}}$ (`sagnac_and_media_correction_m`), the predicted measurement and non-zero Jacobian entries are:

  $$h_{\rho}(\mathbf{x}) = \|\mathbf{p} - \mathbf{p}_e\| + c\delta t + \Delta\rho_{\text{corr}}, \qquad \frac{\partial h_{\rho}}{\partial \delta\mathbf{p}} = \mathbf{u}^\top, \qquad \frac{\partial h_{\rho}}{\partial (c\delta t)} = 1$$

- **`ObservationValue::PseudorangeRate`**:
  The predicted measurement and non-zero Jacobian entries are:

  $$h_{\dot{\rho}}(\mathbf{x}) = \mathbf{u}^\top(\mathbf{v} - \mathbf{v}_e) + c\delta\dot{t}, \qquad \frac{\partial h_{\dot{\rho}}}{\partial \delta\mathbf{p}} = \frac{(\mathbf{v} - \mathbf{v}_e)^\top(\mathbf{I}_3 - \mathbf{u}\mathbf{u}^\top)}{r}, \qquad \frac{\partial h_{\dot{\rho}}}{\partial \delta\mathbf{v}} = \mathbf{u}^\top, \qquad \frac{\partial h_{\dot{\rho}}}{\partial (c\delta\dot{t})} = 1$$

### Two-way range with relativistic Shapiro delay

In `ObservationValue::TwoWayRange`, a ground station at $\mathbf{p}_{\text{tx}}(t_{\text{tx}})$ transmits an uplink signal, the spacecraft at $\mathbf{p}(t_{\text{turn}})$ transponds the signal with known turnaround delay $\Delta\rho_{\text{turn}}$, and a receiving station at $\mathbf{p}_{\text{rx}}(t_{\text{rx}})$ records arrival. Let $r_{\text{up}} = \|\mathbf{p} - \mathbf{p}_{\text{tx}}\|$ and $r_{\text{dn}} = \|\mathbf{p} - \mathbf{p}_{\text{rx}}\|$.

When a gravitating body with gravitational parameter $\mu_B$ at position $\mathbf{p}_B$ is supplied (`shapiro_body`), each leg adds the general-relativistic Shapiro delay:

$$\Delta\rho_{\text{Shapiro}}(\mathbf{r}_1, \mathbf{r}_2) = \frac{2\mu_B}{c^2}\ln\!\left(\frac{r_1 + r_2 + r_{12}}{r_1 + r_2 - r_{12}}\right)$$

where $r_1 = \|\mathbf{r}_1 - \mathbf{p}_B\|$, $r_2 = \|\mathbf{r}_2 - \mathbf{p}_B\|$, and $r_{12} = \|\mathbf{r}_1 - \mathbf{r}_2\|$. The predicted round-trip range is:

$$h_{2\rho}(\mathbf{x}) = r_{\text{up}} + r_{\text{dn}} + \Delta\rho_{\text{Shapiro,up}} + \Delta\rho_{\text{Shapiro,dn}} + \Delta\rho_{\text{turn}}$$

and the position Jacobian $\partial h_{2\rho}/\partial \delta\mathbf{p}$ includes the exact spatial gradient of the Shapiro delay on both legs. `ObservationValue::TwoWayRangeRate` similarly models the sum of the uplink and downlink Doppler range rates $\mathbf{u}_{\text{up}}^\top(\mathbf{v} - \mathbf{v}_{\text{tx}}) + \mathbf{u}_{\text{dn}}^\top(\mathbf{v} - \mathbf{v}_{\text{rx}})$.

## 3. Pathologies, integrity, and admission gates

- **Unmodeled clock bias (`RejectionReason::ClockStateDisabled`)**: If a caller feeds `Pseudorange` or `PseudorangeRate` into a filter configured with `clock_model: None`, the clock offset would corrupt the position state directly. `InertialNavigationFilter` refuses the observation and increments `RejectionCounters::clock_state_disabled`.
- **Degenerate emitter geometry (`ImplausibleCovariance`)**: If the vehicle position coincides with the emitter or station ($r \to 0$) or if the Shapiro denominator $r_1 + r_2 - r_{12} \le 0$ (collinear conjunction behind the gravitating body), the model rejects the observation.
- **Spoofing and multipath**: Outlier pseudoranges fail the 1-DOF $\chi^2$ innovation gate (`scalar_gate_chi2`), and sustained spoofing triggers consecutive gate-refusal quality degradation (ADR-0015).

## 4. Implementation in `Navigate`

Radiometric models live in `crates/navigate-fusion/src/inertial/radiometric.rs` and are exposed through `ObservationValue`:

- `ObservationValue::Pseudorange { emitter_position_m, pseudorange_m, sagnac_and_media_correction_m, variance_m2 }`
- `ObservationValue::PseudorangeRate { emitter_position_m, emitter_velocity_mps, pseudorange_rate_mps, variance_mps2 }`
- `ObservationValue::TwoWayRange { tx_position_m, rx_position_m, shapiro_body, round_trip_range_m, turnaround_delay_m, variance_m2 }`
- `ObservationValue::TwoWayRangeRate { tx_position_m, tx_velocity_mps, rx_position_m, rx_velocity_mps, round_trip_range_rate_mps, variance_mps2 }`
