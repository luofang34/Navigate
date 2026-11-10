# Chapter 14: Magnetic Anomaly Navigation (MagNav)

Crustal magnetic anomalies arise from variations in remanent and induced magnetization within rocks above the Curie temperature depth in a planet's crust. Unlike optical or radar TRN, crustal magnetic fields penetrate clouds, fog, darkness, water, and foliage without attenuation, and require no radiated energy. `navigate-magnav` and `navigate-fusion` implement a complete magnetic anomaly navigation stack inspired by Canciani (`[Canciani2016]`) and `MagNav.jl` (`[Taylor2024]`), adapted to zero-heap `#![no_std]` safety and integrity rules (ADR-0012, ADR-0014, ADR-0016).

## Technical names

- **Total-field scalar anomaly** $\Delta B$ is the residual magnitude of the local magnetic field after subtracting the core planetary main field (such as IGRF) and diurnal space-weather variations.
- **Tolles–Lawson model** is an 18-term physical linear model that separates the magnetic interference of the aircraft airframe into permanent, induced, and eddy-current components.
- **Information ratio** $\eta$ is the dimensionless ratio $(\mathbf{g}^\top \mathbf{P}_{\text{pos}} \mathbf{g}) / R$ that measures whether the local crustal gradient $\mathbf{g}$ is steep enough relative to sensor noise $R$ to provide valid position information.

## 1. The physical problem: Airframe interference and flat-field ambiguity

A scalar optically pumped magnetometer (such as a Cesium or Rubidium vapor sensor) measures magnetic field intensity at sub-nanotesla precision ($1\text{ nT} = 10^{-9}\text{ T}$). However, three physical challenges arise:

1. **Vehicle magnetic interference:** The ferromagnetic engines, airframe structure, and skin eddy currents of the vehicle generate magnetic fields that change with vehicle attitude and angular rate, often exceeding the crustal anomaly signal.
2. **Correlated map bias:** Every aeromagnetic survey map (`map_revision`) contains a long-wavelength level offset from diurnal drift and tie-line leveling. Without a dedicated bias state $b_{\text{mag}}$, this offset biases the horizontal position estimate by $b_{\text{mag}}/\|\mathbf{g}\|$.
3. **Zero-gradient plains and multi-modal contours:** A scalar anomaly measurement constrains only 1 dimension along the local horizontal gradient $\mathbf{g} = [\partial B/\partial p_N, \partial B/\partial p_E, 0]^\top$. Over magnetically quiet sedimentary basins, $\|\mathbf{g}\| \to 0$.

## 2. Mathematical formulation

### 18-term Tolles–Lawson aircraft compensation (`OnlineTlCalibrator`)

Let $\mathbf{B}_{\text{vec}} = [B_x, B_y, B_z]^\top$ be the vector magnetic field measured by a tri-axial fluxgate magnetometer in body FRD axes, with direction cosines $\cos\boldsymbol{\alpha} = \mathbf{B}_{\text{vec}} / \|\mathbf{B}_{\text{vec}}\| = [c_x, c_y, c_z]^\top$ and time derivatives $\dot{c}_x, \dot{c}_y, \dot{c}_z$. `tolles_lawson_basis` constructs the 18 basis terms $\mathbf{a}(\mathbf{B}_{\text{vec}}, \dot{\mathbf{B}}_{\text{vec}}) \in \mathbb{R}^{18}$ (`[Tolles1950]`):

$$B_{\text{aircraft}} = \underbrace{\sum_{i=1}^3 \beta_i c_i}_{\text{3 Permanent}} + \underbrace{\|\mathbf{B}_{\text{vec}}\|\sum_{i=1}^3 \sum_{j=i}^3 \beta_{ij} c_i c_j}_{\text{6 Induced}} + \underbrace{\|\mathbf{B}_{\text{vec}}\|\sum_{i=1}^3 \sum_{j=1}^3 \beta'_{ij} c_i \dot{c}_j}_{\text{9 Eddy-current}}$$

`OnlineTlCalibrator` estimates the 18 coefficients plus a residual bias ($\boldsymbol{\theta} \in \mathbb{R}^{19}$) online using a Joseph-stabilized recursive least-squares filter. Crucially, when $\|\dot{\mathbf{B}}_{\text{vec}}\| < \text{min\_b\_dot\_norm\_t\_per\_s}$ (straight-and-level flight), the calibrator freezes coefficient learning so that geological crustal gradients are never absorbed into the aircraft interference coefficients!

### Dual-path estimation: Linearized ESKF versus Marginalized Particle Filter

`Navigate` provides two complementary estimators that sample `BilinearGridMap`:

| Estimator | Measurement model & state | When to use |
|---|---|---|
| **22-State ESKF (`InertialNavigationFilter`)** | `ObservationValue::MagneticAnomaly`: Taylor expansion $h(\mathbf{p}) = m(\mathbf{p}_{\text{lin}}) + \mathbf{g}^\top(\mathbf{p} - \mathbf{p}_{\text{lin}}) + b_{\text{mag}}$, where $b_{\text{mag}}$ (state 16) is a `FogmBias` tied to `(SourceId, map_revision)` | Tightly bound prior uncertainty ($\sigma_{\text{pos}} \ll r_{\text{lin}}$) over moderate-to-high magnetic gradient regions |
| **Marginalized Particle Filter (`MarginalizedParticleFilter`)** | `ingest_scalar_map`: Direct nonlinear evaluation $m(\mathbf{x}^{n,(i)}) + \hat{b}_{\text{mag}}^{(i)}$ at each position particle without spatial linearization | Large initial position uncertainty, non-Gaussian multimodal anomaly ridges, or transitions across zero-gradient valleys |

## 3. Pathologies, integrity, and admission gates (ADR-0012, ADR-0014)

When using the linearized ESKF model, `navigate-fusion` enforces four specialized protections:

1. **SI unit scale freedom:** Magnetic variances are on the order of $(3\text{ nT})^2 = 9 \times 10^{-18}\text{ T}^2$. The admission gate enforces strict positivity ($R > 0$) without an arbitrary floor that would reject SI tesla units.
2. **Linearization region check (`RejectionReason::LinearizationRegionExceeded`):** The filter checks $\|\hat{\mathbf{p}} - \mathbf{p}_{\text{lin}}\| + k_\sigma \sigma_h \le r_{\text{valid}}$. If the uncertainty cloud extends beyond the valid radius of the local gradient, the observation is rejected.
3. **Dimensionless information ratio gate (`RejectionReason::UninformativeGradient`):** The filter evaluates:

   $$\eta = \frac{\mathbf{g}^\top \mathbf{P}_{\text{pos}} \mathbf{g}}{R}$$

   If $\eta < \text{min\_information\_ratio}$, the map cell is too flat relative to sensor noise. The filter refuses the observation so that flat-cell measurements cannot reset the observation silence timer or falsely claim multi-sensor redundancy.
4. **Map-revision bias reset:** When the vehicle crosses into a new survey sheet with a different `Observation::map_revision`, `InertialNavigationFilter` resets $b_{\text{mag}}$ to zero with its stationary variance $\sigma_{b_{\text{mag}}}^2$ upon the first admitted measurement.

## 4. Implementation in `Navigate`

- `navigate-magnav`: `OnlineTlCalibrator`, `TlCalibratorConfig`, `tolles_lawson_basis`, `BilinearGridMap`, `CrlbTracker`, `ReferencePoint`.
- `navigate-fusion`: `ObservationValue::MagneticAnomaly`, `ScalarFieldMap`, `MarginalizedParticleFilter`.
