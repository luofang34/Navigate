# Chapter 2: Reference Frames, Time Scales, and Multi-Body Geodesy

A navigation equation has meaning only when its reference frame, its coordinate time scale, and its reference surface are explicit. Single-body implementations often assume the Earth WGS84 ellipsoid and have coordinate singularities near the geographic poles. `Navigate` defines explicit frames, physical time scales, and pole-free ellipsoidal geodesy for any planetary body (ADR-0013, ADR-0015, ADR-0017).

## Technical names

- A **reference frame** (`FrameId`) is a named origin and triad of orthogonal axes, such as `EARTH_ITRF`, `EARTH_J2000`, `MOON_PA`, or `MARS_IAU`.
- A **reference ellipsoid** (`Ellipsoid` / `BodyEllipsoid`) is an oblate spheroid defined by an equatorial radius $a$, a flattening $f$, and a body-fixed `FrameId`.
- **Geodetic coordinates** (`GeodeticPosition`) describe a point by geodetic latitude $\varphi$, longitude $\lambda$, and ellipsoidal height $h$ normal to the reference ellipsoid.
- **Orthometric height** $H$ is the height above an equipotential gravity surface (the geoid on the Earth, the selenoid on the Moon, or the areoid on Mars).
- **Geoid undulation** $N$ is the height of the equipotential surface above the reference ellipsoid, such that $h = H + N$.

## Coordinate frames in `Navigate`

Integrated navigation across planetary surfaces and cislunar space uses four primary coordinate triads:

| Frame | Origin | Axes | Primary crate and type |
|---|---|---|---|
| Inertial (`EARTH_J2000`, `MOON_J2000`, ICRS) | Body center or solar system barycenter | Non-rotating relative to distant stars | `navigate-contract::FrameId`, `navigate-astrodynamics` |
| Body-fixed (`EARTH_ITRF`, `MOON_PA`, `MOON_ME`, `MARS_IAU`) | Body center of mass | $+Z$ along spin pole, $+X$ prime meridian, $+Y$ completes right-handed triad | `navigate-geodesy::BodyFixedPosition` |
| Local North-East-Down (NED) | Vehicle or tangent anchor | $+N$ tangent north, $+E$ tangent east, $+D$ inward along ellipsoid normal | `navigate-geodesy::NedOffset`, `navigate-contract::NedVelocity` |
| Body Forward-Right-Down (FRD) and Optical Camera | Vehicle reference point / camera optical center | Body: $+X$ forward, $+Y$ right, $+Z$ down. Camera: $+X$ right, $+Y$ down, $+Z$ forward | `navigate-camera::CameraMount`, `navigate-camera::OpticalPose` |

## Pole-free local axes and polar azimuth conditioning

Spherical-coordinate mechanization equations compute meridian and prime-vertical curvature rates with terms proportional to $\tan\varphi$ or $\sec\varphi = 1/\cos\varphi$. Those rates divide by zero at $\varphi = \pm 90^\circ$.

In `Navigate`, the inertial filter stores its state in the body-centered Cartesian frame (`InertialNavigationFilter`, ADR-0017). Local NED axes are constructed directly from trigonometric functions of $(\varphi, \lambda)$ without dividing by $\cos\varphi$:

$$\mathbf{C}_{\text{NED}\to\text{BF}}(\varphi, \lambda) = \begin{bmatrix} -\sin\varphi\cos\lambda & -\sin\lambda & -\cos\varphi\cos\lambda \\ -\sin\varphi\sin\lambda & \cos\lambda & -\cos\varphi\sin\lambda \\ \cos\varphi & 0 & -\sin\varphi \end{bmatrix}$$

The columns of $\mathbf{C}_{\text{NED}\to\text{BF}}$ are the unit vectors $\mathbf{u}_N$, $\mathbf{u}_E$, and $\mathbf{u}_D$ in body-fixed coordinates. For any given $(\varphi, \lambda)$, this matrix is orthonormal at every latitude, including the north and south poles.

### Polar azimuth sensitivity and model selection (ADR-0017)

Although $\mathbf{C}_{\text{NED}\to\text{BF}}(\pm\pi/2, \lambda)$ is algebraically finite and orthonormal for any supplied $\lambda$, the geographic meridian itself is not unique at the pole: a tiny horizontal position offset $\delta\mathbf{p}$ near $\varphi = \pm 90^\circ$ rotates the geodetic longitude $\lambda$ and the local north/east axes by up to $\pi$ radians. Therefore (ADR-0017):

- **State propagation and position fixes:** The filter state and covariance remain in body-centered Cartesian axes (`EARTH_ITRF`, `MOON_PA`), which have no polar singularity. A `PositionFix` converts into body-fixed Cartesian coordinates with identity Jacobian $\mathbf{H} = \mathbf{I}_3$, so polar position updates remain well-conditioned.
- **Velocity and attitude near the pole:** Because a small position error near $\varphi = \pm 90^\circ$ rotates the local NED axes, local-level `VelocityFix` and `AttitudeFix` observations lose geometric value close to a pole. Polar missions use Cartesian velocity updates (`CartesianVelocityFix`), ECEF star lines of sight (`LineOfSight`), and tangent-plane visual frames (`LocalFrame::BodyTangentEnu`, ADR-0020) anchored at a fixed reference point rather than the instantaneous vehicle meridian.

## Multi-body ellipsoidal geodesy (`navigate-geodesy`)

Given an ellipsoid with semi-major axis $a$, flattening $f$, first eccentricity squared $e^2 = f(2 - f)$, and prime-vertical radius of curvature:

$$N(\varphi) = \frac{a}{\sqrt{1 - e^2\sin^2\varphi}}$$

the forward conversion (`geodetic_to_body_fixed`) from $(\varphi, \lambda, h)$ to body-fixed Cartesian coordinates $(x, y, z)$ is exact:

$$\begin{aligned} x &= \bigl(N(\varphi) + h\bigr)\cos\varphi\cos\lambda \\ y &= \bigl(N(\varphi) + h\bigr)\cos\varphi\sin\lambda \\ z &= \bigl(N(\varphi)(1 - e^2) + h\bigr)\sin\varphi \end{aligned}$$

The inverse conversion (`body_fixed_to_geodetic`) uses a deterministic two-pass Bowring scheme `[Bowring1985]`. On spherical bodies ($f = 0$) and on the polar axis ($\sqrt{x^2 + y^2} = 0$), the closed-form solution is exact. Over tested altitudes from $-400\text{ m}$ to $100\text{ km}$ on WGS84, the round-trip error is less than $10^{-6}\text{ m}$.

### Spherical great-circle guidance geometry versus ellipsoidal geodesics

`navigate-geodesy::great_circle` (`distance_m`, `initial_bearing_rad`, `cross_track_m`, `along_track_m`) models the body as a sphere of mean radius $R_{\text{mean}} = \text{BodyEllipsoid::mean\_radius\_m}()$ rather than solving the full ellipsoidal geodesic problem `[Karney2013]`.

- **Error budget and scope:** On the Earth ($f \approx 1/298.257$), the spherical great-circle approximation differs from the ellipsoidal geodesic by at most $\sim 0.5\%$ in distance and bearing. On Mars ($f \approx 1/169.8$), the difference is at most $\sim 0.9\%$. On a lunar sphere ($f = 0$), the spherical model is exact.
- **Why this suffices for closed-loop guidance:** In `navigate-guidance`, cross-track deviation $e_{\text{xtk}}$ feeds a closed-loop lateral controller that drives $e_{\text{xtk}} \to 0$ as the vehicle approaches the target fix. Survey-grade position truth always goes through the exact body-fixed ellipsoidal conversion path (`geodetic_to_body_fixed` / `LocalTangentPlane`), never through `great_circle`.

### Vertical datums

Altimeters and navigation filters operate on ellipsoidal height $h$, whereas digital elevation models (DEMs) often store orthometric height $H$. `navigate-geodesy::datum` and `navigate-terrain-safety::OrthometricTerrain` convert between the two:

$$h = H + N(\varphi, \lambda)$$

Every function in `navigate-geodesy` and `navigate-terrain-safety` requires the caller to pass the `BodyEllipsoid` explicitly. If a caller passes a lunar solution to an Earth terrain model, `assess_clearance` rejects the call with `ClearanceUnavailable::EllipsoidMismatch`.

## Monotonic and physical time scales

`Navigate` separates interval timing from astronomical coordinate time:

1. **Monotonic time (`MonotonicNanos`)**: A 64-bit integer count of nanoseconds on a declared `ClockDomainId`. Admission gates use `MonotonicNanos` to check measurement freshness and history ordering.
2. **Physical time (`PhysicalInstant`)**: A high-precision epoch represented as 64-bit signed seconds and 64-bit attoseconds ($10^{-18}\text{ s}$) from the J2000 epoch on an explicit `TimeScale` (ADR-0015).

`navigate-astrodynamics::time_scales` and `navigate-celestial` convert between the physical time scales required for astrometry, Earth rotation, and relativistic orbit determination:

| Time scale | Role in navigation |
|---|---|
| `Tai` (International Atomic Time) | Continuous atomic second scale on the rotating geoid |
| `Utc` (Coordinated Universal Time) | Civil time scale; tracks `Tai` with integer leap-second steps |
| `Ut1` (Universal Time) | Earth rotation angle ($\text{ERA}$) relative to the celestial frame; $\Delta T = \text{TT} - \text{UT1}$ |
| `Tt` (Terrestrial Time) | Apparent geocentric ephemeris time; $\text{TT} = \text{TAI} + 32.184\text{ s}$ |
| `Tcg` (Geocentric Coordinate Time) | Relativistic coordinate time for Earth-centered inertial dynamics |
| `Tdb` / `Tcb` (Barycentric Dynamical / Coordinate Time) | Solar system barycentric ephemeris time scales |
| `Tcl` (Lunar Coordinate Time) | Relativistic coordinate time scale for cislunar and lunar surface operations (`tt_to_tcl`, `tcl_to_tt`) |
