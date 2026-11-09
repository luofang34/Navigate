# Chapter 16: Reduced-Dynamic Orbit Determination and Cislunar Navigation

In orbital and cislunar flight, a spacecraft spends hours or days coasting between short thruster burns. During a coast, nongravitational accelerations are micro-g or smaller, below the bias stability of tactical and navigation-grade accelerometers. Integrating noisy IMU samples during a multi-day coast degrades the orbit estimate rapidly. Instead, **Reduced-Dynamic Orbit Determination (RDOD)** propagates the state using high-precision analytical and numerical force models augmented with stochastic empirical accelerations (Dynamic Model Compensation, DMC), and fuses radiometric and optical navigation (OpNav) observations. `navigate-astrodynamics` and `navigate-fusion` implement this capability in `#![no_std]` with zero heap allocation (ADR-0016, ADR-0018).

## Technical names

- **Spherical harmonic gravity field** (`SphericalHarmonicField`) is a series expansion of a body's gravitational potential in solid spherical harmonics parameterized by Stokes coefficients $(C_{nm}, S_{nm})$ up to degree and order $N \le 36$.
- **Dynamic Model Compensation (DMC)** models unmodeled orbital perturbations (such as outgassing, thermal radiation imbalance, or atmospheric density fluctuations) as a 3-axis First-Order Gauss–Markov (`FogmBias`) empirical acceleration vector $\mathbf{a}_{\text{dmc}}$.
- **Optical Navigation (OpNav)** determines spacecraft position and attitude by measuring the unit line-of-sight vector to a planetary body or landmark at a known ephemeris position (`LineOfSightReference::TargetPosition`).

## 1. The physical problem: Non-Keplerian perturbations and force mismodeling

A two-body Keplerian orbit is only a first approximation:

- Near an oblate planet or a lumpy moon, higher-degree gravity harmonics ($J_2$, $C_{22}$, mascons) perturb the orbit by kilometers per revolution.
- In cislunar space, the Sun and the Earth exert strong third-body tidal accelerations on a lunar orbiter.
- In low orbit, atmospheric drag (`ExponentialDrag`) decays semi-major axis, while in high orbit or interplanetary cruise, Solar Radiation Pressure (`CannonballSrp`) perturbs eccentricity and inclination.
- Pure dynamical orbit determination (where process noise $\mathbf{Q} = \mathbf{0}$) becomes overconfident and diverges when drag or SRP models have a 5% scale error. Adding DMC empirical acceleration states (`17..20`) prevents filter divergence and estimates unmodeled forces in real time.

## 2. Mathematical formulation (`navigate-astrodynamics`)

### Pole-free Cartesian solid spherical harmonics (`SphericalHarmonicField`)

Classical spherical harmonic recurrences in spherical coordinates $(r, \varphi, \lambda)$ contain $\tan\varphi$ and $1/\cos\varphi$ terms in the gravity gradient and become singular at the poles. `navigate-astrodynamics::gravity` uses the Montenbruck and Gill recursion in Cartesian solid harmonics $V_{nm}(x, y, z)$ and $W_{nm}(x, y, z)$ in body-fixed axes:

$$V_{00} = \frac{R_{\text{ref}}}{r}, \qquad W_{00} = 0$$

Every partial derivative of a degree-$n$ solid harmonic is a linear combination of degree-$(n+1)$ solid harmonics. Evaluating the recursion up to degree $N + 2$ on a fixed-size stack table (`MAX_INLINE_HARMONIC_DEGREE = 36`) yields both the exact gravitational acceleration $\mathbf{a}_{\text{grav}}(\mathbf{p})$ and the exact $3\times 3$ gravity-gradient tensor $\mathbf{G}(\mathbf{p}) = \partial \mathbf{a}_{\text{grav}}/\partial \mathbf{p}$ with **no polar singularity**.

### Third-body tides, Solar Radiation Pressure, and atmospheric drag

`OrbitalForceModel` sums five accelerations and their exact position/velocity Jacobians:

$$\mathbf{a}_{\text{total}}(\mathbf{p}, \mathbf{v}) = \mathbf{a}_{\text{central}}(\mathbf{p}) + \mathbf{a}_{\text{3B}}(\mathbf{p}) + \mathbf{a}_{\text{rot}}(\mathbf{p}, \mathbf{v}) + \mathbf{a}_{\text{SRP}}(\mathbf{p}) + \mathbf{a}_{\text{drag}}(\mathbf{p}, \mathbf{v})$$

1. **Third-body direct and indirect tidal acceleration (`ThirdBodySet`):**
   For up to `MAX_THIRD_BODIES` bodies with gravitational parameter $\mu_k$ at position $\mathbf{r}_k$ relative to the frame center and $\mathbf{d}_k = \mathbf{r}_k - \mathbf{p}$:

   $$\mathbf{a}_{\text{3B}} = \sum_k \mu_k \left(\frac{\mathbf{d}_k}{\|\mathbf{d}_k\|^3} - \frac{\mathbf{r}_k}{\|\mathbf{r}_k\|^3}\right)$$

2. **Cannonball Solar Radiation Pressure (`CannonballSrp`):**
   With cylindrical central-body shadow testing (`in_shadow`), area-to-mass ratio $A/m$, and reflectivity coefficient $C_R$:

   $$\mathbf{a}_{\text{SRP}} = -\nu\, P_{1\text{AU}} C_R \frac{A}{m} \left(\frac{1\text{ AU}}{\|\mathbf{r}_\odot - \mathbf{p}\|}\right)^2 \frac{\mathbf{r}_\odot - \mathbf{p}}{\|\mathbf{r}_\odot - \mathbf{p}\|}$$

3. **Rotating exponential atmosphere drag (`ExponentialDrag`):**
   With atmosphere rotation $\boldsymbol{\omega}_{\text{atm}}$, air-relative velocity $\mathbf{v}_{\text{rel}} = \mathbf{v} - \boldsymbol{\omega}_{\text{atm}} \times \mathbf{p}$, and exponential density $\rho(h) = \rho_0 \exp(-(h - h_0)/H_s)$:

   $$\mathbf{a}_{\text{drag}} = -\frac{1}{2}\rho(h)\,\frac{C_D A}{m}\,\|\mathbf{v}_{\text{rel}}\|\mathbf{v}_{\text{rel}}$$

### Optical Navigation to a finite-distance target (`LineOfSightReference::TargetPosition`)

When an optical camera images a planet, moon, or rendezvous target at position $\mathbf{p}_t$ in the filter frame, let $\mathbf{d} = \mathbf{p}_t - \mathbf{p}$ be the relative vector and $\hat{\mathbf{u}}_b = \hat{\mathbf{C}}_f^b (\mathbf{d}/\|\mathbf{d}\|)$ be the predicted unit direction in body axes. Projecting onto the 2D tangent basis $\mathbf{B} \in \mathbb{R}^{3\times 2}$ orthogonal to $\hat{\mathbf{u}}_b$ gives both attitude and position Jacobians (ADR-0018):

$$\frac{\partial \mathbf{y}}{\partial \delta\boldsymbol{\theta}} = \mathbf{B}^\top [\hat{\mathbf{u}}_b\times], \qquad \frac{\partial \mathbf{y}}{\partial \delta\mathbf{p}} = -\frac{1}{\|\mathbf{d}\|}\mathbf{B}^\top \hat{\mathbf{C}}_f^b$$

## 3. Mode transitions and impulsive burns

- **`set_propagation_mode`**: Switches cleanly between `PropagationMode::ForceModel` (coast) and `PropagationMode::SensedImu` (powered flight / atmospheric entry) while preserving the single 22-state covariance and OOSM history ring.
- **`apply_impulsive_burn`**: Applies an instantaneous $\Delta\mathbf{v}$ and injects its execution covariance $\boldsymbol{\Sigma}_{\Delta v}$ into the velocity block during a force-model coast.
- **Mode mismatch gate (`RejectionReason::PropagationModeMismatch`)**: Feeding IMU samples into a force-model coast (or calling `coast_to` / `apply_impulsive_burn` in sensed-IMU mode) is rejected and counted.
