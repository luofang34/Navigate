# Chapter 6: Nonlinear and Multimodal Estimation: Marginalized Particle Filtering

An Extended Kalman Filter (EKF) or Error-State Kalman Filter (ESKF) linearizes a measurement model $\mathbf{h}(\mathbf{x})$ with a first-order Taylor series around the current state estimate. When a vehicle navigates over a geophysical scalar map (such as a crustal magnetic anomaly grid or a terrain elevation digital map), first-order linearization fails in two common regimes:

1. **Zero-gradient plateaus:** In a flat map cell, $\nabla m(\mathbf{p}) \approx \mathbf{0}$, so the linearized innovation provides no position correction.
2. **Multimodal ambiguity:** When the prior position uncertainty spans several map ridges or anomalies with identical scalar values, the true posterior distribution splits into multiple distinct peaks. A Gaussian filter averages those peaks into a wrong position between them.

`navigate-fusion::particle` provides `MarginalizedParticleFilter` (MPF), also known as a Rao–Blackwellized Particle Filter (RBPF), following Schön, Gustafsson, and Nordlund (2005) (ADR-0014).

## Technical names

- **Rao–Blackwellization (marginalization)** is the exact factoring of a state space into a low-dimensional nonlinear subset sampled by particles and a conditionally linear Gaussian subset tracked analytically by a Kalman filter.
- **Effective sample size** $N_{\text{eff}}$ is a measure of particle weight concentration, defined as $1 / \sum_{i=1}^N (w^{(i)})^2$.
- **Posterior Cramér–Rao Lower Bound (CRLB)** is the theoretical minimum mean-square error matrix achievable by any unbiased estimator along a trajectory.

## State space factorization

A full 7-dimensional particle filter over horizontal position, 3D velocity, barometric bias, and magnetic map bias would require thousands of particles to avoid sample impoverishment. However, the scalar map measurement $z_k = m(\mathbf{p}_{h,k}) + b_{\text{mag},k} + v_k$ is nonlinear **only** in the 2D horizontal position $\mathbf{p}_h = [p_N, p_E]^\top$. Conditioned on the trajectory of $\mathbf{p}_h$, the remaining five states remain strictly linear-Gaussian:

$$\mathbf{x}_k^n = \begin{bmatrix} p_N \\ p_E \end{bmatrix}_k \in \mathbb{R}^2 \quad (\text{nonlinear particles}), \qquad \mathbf{x}_k^l = \begin{bmatrix} p_D \\ v_N \\ v_E \\ v_D \\ b_{\text{alt}} \\ b_{\text{mag}} \end{bmatrix}_k \quad (\text{conditionally linear Gaussian})$$

The joint posterior factorizes as:

$$p\bigl(\mathbf{x}_{0:k}^n, \mathbf{x}_k^l \mid \mathbf{z}_{1:k}\bigr) = p\bigl(\mathbf{x}_k^l \mid \mathbf{x}_{0:k}^n, \mathbf{z}_{1:k}\bigr)\, p\bigl(\mathbf{x}_{0:k}^n \mid \mathbf{z}_{1:k}\bigr)$$

In `MarginalizedParticleFilter`:

- Each `MpfParticle` holds a 2D tangent-plane position $\mathbf{x}^{n,(i)}$, a conditional linear mean vector $\hat{\mathbf{x}}^{l,(i)} \in \mathbb{R}^5$ (`[v_n, v_e, v_d, b_alt, b_mag]`), and a log-weight.
- Because the linear transition matrices and measurement Jacobians with respect to $\mathbf{x}^l$ do not depend on the particle position $\mathbf{x}^{n,(i)}$, **all particles share one conditional linear covariance matrix** $\mathbf{P}^l$. This property reduces the memory and CPU cost by an order of magnitude and allows zero-heap execution inside a caller-owned `ParticlePool` (two ping-pong slices of `MpfParticle`).

## The marginalized recursion

1. **Nonlinear measurement update (`ingest_scalar_map`)**:
   Each particle evaluates the exact map value $m(\mathbf{x}_k^{n,(i)})$ via `ScalarFieldMap` (such as `navigate_magnav::BilinearGridMap`) without spatial derivatives. With predicted measurement $\hat{z}^{(i)} = m(\mathbf{x}_k^{n,(i)}) + \hat{b}_{\text{mag}}^{(i)}$ and innovation variance $S = P_{b_{\text{mag}}} + R$, the log-weights update via a numerically stable log-sum-exp normalization.
2. **Conditional linear measurement update**:
   Each particle updates its own conditional mean $\hat{\mathbf{x}}^{l,(i)}$ using the innovation $z_k - m(\mathbf{x}_k^{n,(i)}) - \hat{b}_{\text{mag}}^{(i)}$, while the shared covariance $\mathbf{P}^l$ updates once.
3. **Systematic resampling**:
   When $N_{\text{eff}} = 1/\sum (w^{(i)})^2$ drops below the configured threshold, the filter resamples particles into the scratch slice of `ParticlePool` using a deterministic `SplitMix64` PRNG.
4. **Cross-coupled time update**:
   As particles propagate according to $\mathbf{x}_{k+1}^{n,(i)} = \mathbf{x}_k^{n,(i)} + \mathbf{v}_{h,k}^{(i)}\Delta t + \mathbf{w}_k^{n,(i)}$, the realized displacement $\mathbf{x}_{k+1}^{n,(i)} - \mathbf{x}_k^{n,(i)}$ acts as a pseudo-measurement on the conditional velocity mean $\hat{\mathbf{v}}_h^{(i)}$ (Step 5 of Schön et al., 2005). Consequently, scalar map updates of position automatically correct the vehicle velocity estimate!

## Integrity rules for particle clouds

A single geophysical map provides no cross-check against a second independent sensor class. `MarginalizedParticleFilter` therefore enforces two integrity rules (ADR-0014):

- The published solution quality is at best `SolutionQuality::Degraded`.
- If the empirical particle spread exceeds `MpfConfig::unimodal_max_1sigma_m` (indicating a split or multimodal particle cloud), the filter marks the solution as `SolutionQuality::Unusable` so that guidance refuses to steer on the ambiguous mean.

## Pre-flight observability: The Posterior Cramér–Rao Lower Bound (`CrlbTracker`)

Before flying a mission over a magnetic or terrain map, a mission planner needs to know whether the map gradients along the planned route are strong enough to bound inertial drift. `navigate_magnav::CrlbTracker` runs the 22-state error covariance recursion along a sequence of `ReferencePoint` states using the true map gradients $\mathbf{g}(\mathbf{p}_k) = \nabla m(\mathbf{p}_k)$:

$$\mathbf{J}_{k+1} = \bigl(\boldsymbol{\Phi}_k \mathbf{J}_k^{-1} \boldsymbol{\Phi}_k^\top + \mathbf{Q}_k\bigr)^{-1} + \mathbf{H}_k^\top \mathbf{R}_k^{-1} \mathbf{H}_k$$

The inverse Fisher information matrix $\mathbf{P}_{\text{CRLB},k} = \mathbf{J}_k^{-1}$ yields the lower bound on the position error standard deviation $(\sigma_N, \sigma_E, \sigma_D)$ in local NED axes at every waypoint along the route.
