# Chapter 15: Altimetry, Slant-Range Terrain Clearance, and Terrain Safety

Vertical channel stability is critical in inertial navigation because the vertical gravity gradient $\partial g_D / \partial h \approx +2g/R_e > 0$ makes pure strapdown vertical integration exponentially unstable. Vehicles stabilize the vertical channel and aid horizontal position using barometric altimeters, radar/laser altimeters, and digital terrain elevation models (TERCOM / TAN). `navigate-fusion`, `navigate-geodesy`, and `navigate-terrain-safety` model both barometric altitude and attitude-coupled slant-range terrain clearance on any planetary body (ADR-0012, ADR-0014, ADR-0016, ADR-0017).

## Technical names

- **Ellipsoidal height** $h$ is the geometric distance along the outward normal of the reference ellipsoid (`BodyEllipsoid`).
- **Barometric altitude bias** $b_{\text{alt}}$ is the time-varying height offset caused by non-standard atmospheric pressure and temperature gradients.
- **Slant range** $\rho$ is the distance from the vehicle to the terrain surface along a body-fixed radar or laser beam vector $\mathbf{b}$.
- **Terrain clearance** is the vertical margin $h_{\text{vehicle}} - h_{\text{terrain}}$ between the vehicle and the underlying surface, evaluated in ellipsoidal height on a shared reference ellipsoid.

## 1. The physical problem: Curvature, pressure drift, and beam inclination

Altimeter measurement models must account for three physical effects:

1. **Flat-plane altitude error:** In a local tangent plane, the plane's $-p_D$ coordinate diverges from ellipsoidal height $h$ by $\Delta h \approx d^2 / (2R_e)$ as distance $d$ from the origin grows. At $d = 50\text{ km}$ on the Earth, $-p_D$ is in error by $196\text{ m}$ (ADR-0012).
2. **Isobaric drift:** A barometric altimeter measures static pressure, not geometric height. Weather fronts shift isobaric surfaces by hundreds of meters.
3. **Attitude-slope slant coupling:** A radar or laser altimeter measures slant range along a beam $\mathbf{b}$ fixed to the body frame. During a $30^\circ$ banked turn over a sloped mountain valley, treating slant range as vertical height above ground level (AGL) injects a large altitude and position error.

## 2. Mathematical formulation

### Curvature-free ellipsoidal altitude and barometric bias (`ObservationValue::Altitude`)

In both `NavigationFilter` and `InertialNavigationFilter`, `ObservationValue::Altitude` converts the current Cartesian position $\mathbf{p}$ into exact geodetic coordinates $(\hat{\varphi}, \hat{\lambda}, \hat{h})$ on the attached `BodyEllipsoid`. The outward ellipsoid normal in body-fixed axes is:

$$\mathbf{n}_{\text{up}}(\hat{\varphi}, \hat{\lambda}) = -\mathbf{u}_D = \begin{bmatrix} \cos\hat{\varphi}\cos\hat{\lambda} \\ \cos\hat{\varphi}\sin\hat{\lambda} \\ \sin\hat{\varphi} \end{bmatrix}$$

which is the exact spatial gradient $\partial h / \partial \mathbf{p}$. Consequently, the altitude model has **zero curvature error** at any distance across the globe.

When `Observation::composition` contains `SensorClass::Barometric`, `InertialNavigationFilter` adds the FOGM barometric bias state $b_{\text{alt}}$ (state 15):

$$h_{\text{baro}}(\mathbf{x}) = h(\mathbf{p}) + b_{\text{alt}}, \qquad \frac{\partial h_{\text{baro}}}{\partial \delta\mathbf{p}} = \mathbf{n}_{\text{up}}^\top, \qquad \frac{\partial h_{\text{baro}}}{\partial \delta b_{\text{alt}}} = 1$$

Non-barometric altitude fixes (such as geometric GNSS height) have $\partial h/\partial \delta b_{\text{alt}} = 0$, allowing geometric height fixes to calibrate $b_{\text{alt}}$ directly.

### Attitude- and slope-coupled slant range (`ObservationValue::TerrainClearance`)

Let the local terrain surface around the linearization point have ellipsoidal elevation $T_0$ and local north/east terrain slopes $s_N = \partial T / \partial p_N$ and $s_E = \partial T / \partial p_E$. Define the upward terrain normal vector in local NED axes as $\mathbf{m} = [s_N, s_E, 1]^\top$.

For a rangefinder beam with unit direction $\mathbf{b}$ in body FRD axes and body-to-local-NED rotation $\mathbf{C}_{bl}$, the effective vertical projection factor $\kappa$ and predicted slant range $\hat{\rho}$ are (ADR-0014):

$$\kappa = \mathbf{m}^\top (\mathbf{C}_{bl}\mathbf{b}), \qquad \hat{\rho}(\mathbf{x}) = \frac{h - T_0}{\kappa}$$

Differentiating $\hat{\rho}$ with respect to local position $\delta\mathbf{p}_{\text{NED}} = [\delta p_N, \delta p_E, \delta p_D]^\top$ and right body attitude error $\delta\boldsymbol{\theta}$ yields:

$$\begin{aligned} \frac{\partial \rho}{\partial \delta\mathbf{p}_{\text{NED}}} &= \frac{1}{\kappa}\begin{bmatrix} -s_N & -s_E & -1 \end{bmatrix} \\ \frac{\partial \rho}{\partial \delta\boldsymbol{\theta}} &= -\frac{h - T_0}{\kappa^2}\,\mathbf{m}^\top \mathbf{C}_{bl}[\mathbf{b}\times] \end{aligned}$$

Thus, a slant-range terrain clearance measurement simultaneously constrains **vertical height**, **horizontal position across terrain slopes (TERCOM)**, and **vehicle pitch/roll attitude**, without touching the barometric bias $b_{\text{alt}}$!

## 3. Pathologies, integrity, and terrain safety (`navigate-terrain-safety`)

- **Grazing beam and subterranean guard (`ImplausibleCovariance`):** If the beam points nearly parallel to or above the terrain slope ($\kappa \le 0.05$) or if the nominal state lies at or below the terrain ($h \le T_0$), `InertialNavigationFilter` rejects the `TerrainClearance` observation.
- **Datum-honest terrain clearance alerting (`assess_clearance`):**
  `navigate-terrain-safety` evaluates terrain clearance for flight safety alerting (`AlertSeverity::Nominal`, `Caution`, `Warning`). Because DEMs typically store orthometric height $H$ above the geoid, selenoid, or areoid while the filter outputs ellipsoidal height $h$, `OrthometricTerrain` requires a `GeoidUndulation` model ($h_{\text{terrain}} = H + N$). If terrain data, geoid coverage, clock synchronization, ellipsoid match, or solution integrity is missing, `assess_clearance` fails closed with a typed `ClearanceUnavailable` reason.
