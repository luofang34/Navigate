# Appendix A: Mathematical Notation and Coordinate Conventions

This appendix summarizes the mathematical symbols, coordinate frames, quaternion conventions, and physical units used throughout this book and the `Navigate` codebase.

## 1. Canonical SI and angular units

All public APIs and internal equations in `Navigate` use canonical SI units unless a field suffix explicitly states otherwise:

| Quantity | Unit | Field / type suffix convention |
|---|---|---|
| Distance / Position | meter ($\text{m}$) | `_m`, `CartesianPosition`, `BodyFixedPosition`, `NedOffset` |
| Velocity / Range rate | meter per second ($\text{m/s}$) | `_mps`, `CartesianVelocity`, `NedVelocity` |
| Acceleration / Specific force | meter per second squared ($\text{m/s}^2$) | `_mps2` |
| Plane angle / Geodetic lat-lon | radian ($\text{rad}$) | `_rad`, `GeodeticPosition` |
| Angular rate | radian per second ($\text{rad/s}$) | `_rad_per_s`, `_rps` |
| Magnetic flux density | tesla ($\text{T}$) | `_t`, variance `_t2`, gradient `_t_per_m` |
| Monotonic time / Duration | nanosecond ($\text{ns}$) | `_ns`, `MonotonicNanos`, `DurationNanos` |
| Physical coordinate time | seconds + attoseconds ($10^{-18}\text{ s}$) from J2000 | `PhysicalInstant`, `SignedDuration` |

## 2. Matrix, vector, and Lie algebra notation

- Bold lowercase letters ($\mathbf{p}, \mathbf{v}, \mathbf{u}$) denote column vectors in $\mathbb{R}^n$.
- Bold uppercase letters ($\mathbf{C}, \mathbf{F}, \boldsymbol{\Phi}, \mathbf{P}, \mathbf{Q}, \mathbf{R}, \mathbf{H}$) denote matrices.
- For any vector $\boldsymbol{\omega} = [\omega_x, \omega_y, \omega_z]^\top \in \mathbb{R}^3$, the skew-symmetric cross-product matrix $[\boldsymbol{\omega}\times] \in \mathfrak{so}(3)$ is:

$$[\boldsymbol{\omega}\times] = \begin{bmatrix} 0 & -\omega_z & \omega_y \\ \omega_z & 0 & -\omega_x \\ -\omega_y & \omega_x & 0 \end{bmatrix}, \qquad [\boldsymbol{\omega}\times]\mathbf{a} = \boldsymbol{\omega} \times \mathbf{a}$$

- The exponential map $\exp([\delta\boldsymbol{\theta}\times]) \in SO(3)$ maps a rotation vector $\delta\boldsymbol{\theta} \in \mathbb{R}^3$ (with angle $\theta = \|\delta\boldsymbol{\theta}\|$ and axis $\hat{\mathbf{u}} = \delta\boldsymbol{\theta}/\theta$) via Rodrigues' formula:

$$\exp([\delta\boldsymbol{\theta}\times]) = \mathbf{I}_3 + \frac{\sin\theta}{\theta}[\delta\boldsymbol{\theta}\times] + \frac{1 - \cos\theta}{\theta^2}[\delta\boldsymbol{\theta}\times]^2$$

## 3. Quaternion and attitude perturbation convention

- Quaternions use the Hamilton convention $\mathbf{q} = [w, x, y, z]$ with scalar part $w$ and vector part $\mathbf{v}_q = [x, y, z]^\top$.
- In `InertialNavigationFilter`, the nominal attitude quaternion $\mathbf{q}_b^f$ rotates vectors from **body FRD axes** ($b$) into the **active filter frame axes** ($f$):

$$\mathbf{v}_f = \mathbf{C}(\mathbf{q}_b^f)\,\mathbf{v}_b$$

- In `NavigationSolution`, the published surface attitude rotates vectors from body FRD axes into **local NED axes** at the vehicle position.
- The 3-DOF attitude error state $\delta\boldsymbol{\theta}$ (indices `6..9` of `InertialNavigationFilter`) is a **right (body-frame) multiplicative perturbation**:

$$\mathbf{C}_b^f = \hat{\mathbf{C}}_b^f \exp([\delta\boldsymbol{\theta}\times]), \qquad \mathbf{q}_b^f = \hat{\mathbf{q}}_b^f \otimes \begin{bmatrix} \cos(\|\delta\boldsymbol{\theta}\|/2) \\ \frac{\sin(\|\delta\boldsymbol{\theta}\|/2)}{\|\delta\boldsymbol{\theta}\|}\delta\boldsymbol{\theta} \end{bmatrix}$$

## 4. Camera and body axis conventions (`navigate-camera`, ADR-0021)

- **Body frame (FRD):** $+X$ forward, $+Y$ right, $+Z$ down.
- **Optical camera frame (OpenCV convention):** $+X$ right across image columns, $+Y$ down along image rows, $+Z$ forward along the optical viewing axis.
- **Render eye frame (OpenGL convention):** $+X$ right, $+Y$ up, $+Z$ back (`OPTICAL_FROM_GL` converts between OpenGL eye axes and optical camera axes without loss).
- **Pixel coordinates:** Integer pixel centers $(u, v)$ with $(0, 0)$ at the top-left pixel; $u$ increases right, $v$ increases down.
