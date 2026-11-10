# Chapter 6: Nonlinear and Multimodal Estimation: Marginalized Particle Filtering

An Extended Kalman Filter (EKF) or Error-State Kalman Filter (ESKF) linearizes a measurement model $\mathbf{h}(\mathbf{x})$ with a first-order Taylor series around the current state estimate. When a vehicle navigates over a geophysical scalar map (such as a crustal magnetic anomaly grid or a terrain elevation digital map), first-order linearization fails in two common regimes:

1. **Zero-gradient plateaus:** In a flat map cell, $\nabla m(\mathbf{p}) \approx \mathbf{0}$, so the linearized innovation provides no position correction.
2. **Multimodal ambiguity:** When the prior position uncertainty spans several map ridges or anomalies with identical scalar values, the true posterior distribution splits into multiple distinct peaks. A Gaussian filter averages those peaks into a wrong position between them.

`navigate-fusion::particle` provides `MarginalizedParticleFilter` (MPF), also known as a Rao–Blackwellized Particle Filter (RBPF), following Schön, Gustafsson, and Nordlund (2005) `[Schon2005]` (ADR-0014, ADR-0016).

## Technical names

- **Rao–Blackwellization (marginalization)** is the exact factoring of a state space into a low-dimensional nonlinear subset sampled by particles and a conditionally linear Gaussian subset tracked analytically by a Kalman filter `[Schon2005]`.
- **Effective sample size** $N_{\text{eff}}$ is a measure of particle weight concentration, defined as $1 / \sum_{i=1}^N (w^{(i)})^2$.
- **Posterior Cramér–Rao Lower Bound (CRLB)** is the theoretical minimum mean-square error matrix achievable by any unbiased estimator along a trajectory.

## 8-dimensional state space factorization (`3 + 5`)

A full 8-dimensional particle filter over 3D position, 3D velocity, barometric bias, and magnetic map bias would require thousands of particles to avoid sample impoverishment. However, the scalar map measurement $z_{\text{map},k} = m(\mathbf{x}_k^n) + b_{\text{mag},k} + e_{\text{map},k}$ and the curved-ellipsoid altitude measurement $z_{\text{alt},k} = h(\mathbf{x}_k^n) + b_{\text{alt},k} + e_{\text{alt},k}$ are nonlinear **only** in the 3D tangent-plane position $\mathbf{x}_k^n = [p_N, p_E, p_D]^\top \in \mathbb{R}^3$. Conditioned on the trajectory of $\mathbf{x}_k^n$, the remaining five states (`const L: usize = 5`) remain strictly linear-Gaussian:

$$\mathbf{x}_k^n = \begin{bmatrix} p_N \\ p_E \\ p_D \end{bmatrix}_k \in \mathbb{R}^3 \quad (\text{nonlinear position particles}), \qquad \mathbf{x}_k^l = \begin{bmatrix} v_N \\ v_E \\ v_D \\ b_{\text{alt}} \\ b_{\text{mag}} \end{bmatrix}_k \in \mathbb{R}^5 \quad (\text{conditionally linear Gaussian})$$

Why is $p_D$ kept in the nonlinear particle vector $\mathbf{x}_k^n$ rather than the linear vector $\mathbf{x}_k^l$? Away from the tangent-plane origin, the true ellipsoidal height $h(\mathbf{x}_k^n)$ is a nonlinear function of $(p_N, p_E, p_D)$ due to planetary curvature (Chapter 2). Keeping $\mathbf{x}_k^n \in \mathbb{R}^3$ allows `ingest_altitude` to evaluate `plane.from_ned(&NedOffset::new(x.x, x.y, x.z)).altitude_m` at each particle without flat-Earth approximation error.

The joint posterior factorizes exactly as `[Schon2005]`:

$$p\bigl(\mathbf{x}_{0:k}^n, \mathbf{x}_k^l \mid \mathbf{z}_{1:k}\bigr) = p\bigl(\mathbf{x}_k^l \mid \mathbf{x}_{0:k}^n, \mathbf{z}_{1:k}\bigr)\, p\bigl(\mathbf{x}_{0:k}^n \mid \mathbf{z}_{1:k}\bigr)$$

In `MarginalizedParticleFilter`:

- Each `MpfParticle` holds a 3D tangent-plane position `position: Vector3<f64>`, a 5D conditional linear mean vector `linear: SVector<f64, 5>` (`[v_n, v_e, v_d, b_alt, b_mag]`), and a natural log-weight `log_weight: f64`.
- Because the linear transition matrices $\mathbf{A}_n = [\Delta t\,\mathbf{I}_3, \mathbf{0}_{3\times 2}] \in \mathbb{R}^{3\times 5}$, $\mathbf{A}_l = \operatorname{diag}(1, 1, 1, \phi_{\text{alt}}, \phi_{\text{mag}}) \in \mathbb{R}^{5\times 5}$, and linear measurement matrices $\mathbf{C}$ do not depend on the particle position $\mathbf{x}^{n,(i)}$, **all particles share a single $5 \times 5$ conditional linear covariance matrix** $\mathbf{P}^l$ (`LinearCov = SMatrix<f64, 5, 5>`). This property reduces memory and CPU cost and allows zero-heap execution inside a caller-owned `ParticlePool` (two ping-pong slices of `MpfParticle`).

## The marginalized recursion

1. **Nonlinear measurement updates (`ingest_scalar_map`, `ingest_altitude`)**:
   Each particle evaluates the exact map value $m(\mathbf{x}_k^{n,(i)})$ via `ScalarFieldMap` (or ellipsoidal height $h(\mathbf{x}_k^{n,(i)})$) without spatial derivatives. With predicted measurement $\hat{z}^{(i)} = m(\mathbf{x}_k^{n,(i)}) + \hat{b}_{\text{mag}}^{(i)}$ and scalar innovation variance $S = P_{b_{\text{mag}}} + R$, each particle updates its conditional mean $\hat{\mathbf{x}}^{l,(i)}$ and log-weight via log-sum-exp normalization, while the shared $5 \times 5$ covariance $\mathbf{P}^l$ updates once in Joseph form.
2. **Linear velocity updates (`ingest_velocity`)**:
   Applies a 3D Kalman update to the velocity block `0..3` of each particle's $\hat{\mathbf{x}}^{l,(i)}$ and updates log-weights with the Mahalanobis distance under $\mathbf{S} = \mathbf{C}\mathbf{P}^l\mathbf{C}^\top + \mathbf{R}$.
3. **Systematic resampling**:
   When $N_{\text{eff}} = 1/\sum (w^{(i)})^2$ drops below `resample_threshold_ratio * N`, the filter resamples particles into the scratch slice of `ParticlePool` using a deterministic `SplitMix64` PRNG and swaps the active and scratch slices in $O(1)$ time.
4. **Cross-coupled time update (`predict`)**:
   With $\mathbf{A}_n = [\Delta t\,\mathbf{I}_3, \mathbf{0}_{3\times 2}]$, $\mathbf{N}_t = \mathbf{A}_n \mathbf{P}^l \mathbf{A}_n^\top + \mathbf{Q}_n$, and gain $\mathbf{L} = \mathbf{A}_l \mathbf{P}^l \mathbf{A}_n^\top \mathbf{N}_t^{-1}$, each particle draws motion $\Delta\mathbf{x}_k^{n,(i)} \sim \mathcal{N}(\mathbf{A}_n \hat{\mathbf{x}}_k^{l,(i)}, \mathbf{N}_t)$ and updates its linear mean via Step 5 of `[Schon2005]`:

   $$\hat{\mathbf{x}}_{k+1}^{l,(i)} = \mathbf{A}_l \hat{\mathbf{x}}_k^{l,(i)} + \mathbf{L}\bigl(\Delta\mathbf{x}_k^{n,(i)} - \mathbf{A}_n \hat{\mathbf{x}}_k^{l,(i)}\bigr), \qquad \mathbf{P}_{k+1}^l = \mathbf{A}_l \mathbf{P}_k^l \mathbf{A}_l^\top + \mathbf{Q}_l - \mathbf{L}\mathbf{N}_t\mathbf{L}^\top$$

   Consequently, scalar map updates of position automatically correct the vehicle velocity estimate through the motion cross-coupling!

## Integrity rules and limitations of the particle filter

`MarginalizedParticleFilter` enforces two integrity rules and carries specific scope limitations (ADR-0014):

- **Quality cap (`SolutionQuality::Degraded`):** Because a single scalar map provides no independent cross-check, the published quality is at best `Degraded`.
- **Multimodal spread guard (`SolutionQuality::Unusable`):** If the empirical horizontal $1\sigma$ spread of the particle cloud exceeds `MpfConfig::unimodal_max_1sigma_m` (such as when the cloud splits across two anomaly ridges), the published solution is marked `Unusable` so that guidance refuses to steer on the ambiguous weighted mean.
- **No internal admission funnel:** Unlike `NavigationFilter` and `InertialNavigationFilter`, `MarginalizedParticleFilter` does not run clock-domain, sequence-ordering, or $\chi^2$ innovation gates internally; the host adapter must validate stamps and provenance before calling `ingest_scalar_map`.

## Pre-flight observability: The Posterior Cramér–Rao Lower Bound (`CrlbTracker`)

Before flying a mission over a magnetic or terrain map, a mission planner can evaluate whether the map gradients along the planned route are strong enough to bound inertial drift. `navigate_magnav::CrlbTracker` runs the 22-state error covariance recursion along a sequence of `ReferencePoint` states using the map gradients $\mathbf{g}(\mathbf{p}_k) = \nabla m(\mathbf{p}_k)$:

$$\mathbf{J}_{k+1} = \bigl(\boldsymbol{\Phi}_k \mathbf{J}_k^{-1} \boldsymbol{\Phi}_k^\top + \mathbf{Q}_k\bigr)^{-1} + \mathbf{H}_k^\top \mathbf{R}_k^{-1} \mathbf{H}_k$$

The inverse Fisher information matrix $\mathbf{P}_{\text{CRLB},k} = \mathbf{J}_k^{-1}$ yields the lower bound on the position error standard deviation $(\sigma_N, \sigma_E, \sigma_D)$ in local NED axes at every waypoint along the route.
