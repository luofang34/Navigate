# Chapter 10: Celestial Mechanics, Ephemerides, and Star Catalogs

Celestial navigation requires two fundamental references: the apparent directions of solar system bodies (the Sun, the Moon, and planets) and the apparent directions of stars at the observer's epoch and position. Unlike radio navigation, celestial references cannot be jammed or spoofed. `Navigate` provides three standalone, deterministic crates for celestial astrometry: `navigate-celestial`, `flamsteed`, and `eclipse-canon`.

## Technical names

- **Barycentric direction** is the unit vector from the solar system barycenter to a star at a given epoch, corrected for proper motion and binary orbital motion.
- **Apparent place** is the unit direction of a celestial body as seen by a moving observer, corrected for light-time delay and special-relativistic stellar aberration.
- **Delta T ($\Delta T$)** is the difference $\text{TT} - \text{UT1}$ between uniform atomic dynamical time and the irregular rotation angle of the Earth.

## 1. The physical problem and observability

A celestial sight measures the angle of a star or solar system body relative to the vehicle body axes or the local gravity vertical. To compare that sight with a predicted direction in a body-fixed frame (`EARTH_ITRF`), the navigation computer must account for five physical effects:

1. **Star space motion and binary orbits:** Stars move across the sky at up to several arcseconds per year (proper motion), and close visual binaries wobble around their barycenter.
2. **Light time ($\tau$):** Light from the Sun takes approximately $499\text{ s}$ to reach the Earth; the observer sees the body where it was at emission time $t_{\text{em}} = t_{\text{rec}} - \tau$.
3. **Special-relativistic aberration:** An observer moving at barycentric velocity $\mathbf{v}_{\text{obs}}$ sees light rays tilted toward the velocity vector by up to $20.5''$ on Earth orbit ($\|\mathbf{v}_{\text{obs}}\|/c \approx 10^{-4}$).
4. **Precession and nutation:** Lunisolar torques cause the Earth's spin axis to precess and nutate relative to the inertial frame (ICRS / J2000).
5. **Earth rotation ($\text{UT1}$):** The Earth rotates by $15''$ of arc per second of time, so an error of $1\text{ s}$ in $\Delta T = \text{TT} - \text{UT1}$ shifts Earth-fixed longitudes by $460\text{ m}$ at the equator.

## 2. Mathematical formulation

### Linear 3D space motion and binary orbits (`flamsteed`)

Given a star's catalog right ascension $\alpha_0$, declination $\delta_0$, parallax $\varpi$, proper motion $(\mu_{\alpha*}, \mu_\delta)$, and radial velocity $v_r$ at epoch J2000, `flamsteed` constructs the initial barycentric position $\mathbf{r}_0$ and space velocity $\mathbf{v}$ and propagates the star on a straight line in 3D space over $t$ Julian years (TT) from J2000:

$$\mathbf{u}(t) = \frac{\mathbf{r}_0 + \mathbf{v}\,t}{\|\mathbf{r}_0 + \mathbf{v}\,t\|}$$

For 133 bright stars in binary systems with Sixth Catalog of Orbits of Visual Binary Stars (ORB6) elements, `Star::component_direction_at` solves Kepler's equation $M = E - e\sin E$ and projects the Thiele–Innes orbital offset onto the tangent plane to give the position of the primary luminous component.

### Reception light-time iteration and relativistic aberration (`navigate-celestial`)

Given an observer state $(\mathbf{r}_{\text{obs}}, \mathbf{v}_{\text{obs}})$ at reception time $t_{\text{rec}}$ in barycentric axes, `observe_body` solves for the retarded emission time $t_{\text{em}} = t_{\text{rec}} - \tau$ via Euclidean light-time iteration:

$$\tau_{k+1} = \frac{\|\mathbf{r}_{\text{target}}(t_{\text{rec}} - \tau_k) - \mathbf{r}_{\text{obs}}(t_{\text{rec}})\|}{c}$$

Let $\mathbf{u}_{\text{geom}}$ be the geometric unit vector from the observer to the retarded target position, and let $\boldsymbol{\beta} = \mathbf{v}_{\text{obs}}/c$ with Lorentz factor $\gamma = 1/\sqrt{1 - \|\boldsymbol{\beta}\|^2}$. With `Correction::Aberrated`, `navigate-celestial` applies the exact special-relativistic velocity addition law:

$$\mathbf{u}_{\text{app}} = \frac{\mathbf{u}_{\text{geom}} + \left(\gamma + \frac{\gamma^2}{\gamma + 1}(\boldsymbol{\beta}\cdot\mathbf{u}_{\text{geom}})\right)\boldsymbol{\beta}}{\gamma\bigl(1 + \boldsymbol{\beta}\cdot\mathbf{u}_{\text{geom}}\bigr)}$$

### Precession, nutation, and Earth rotation

To transform apparent celestial directions into `EARTH_ITRF` (`sun_ecef`, `moon_ecef`, `stars_ecef`), `navigate-celestial` applies long-range precession, the IAU nutation model, the equation of the equinoxes, and Greenwich Apparent Sidereal Time ($\text{GAST}$) computed from $\text{UT1} = \text{TT} - \Delta T$.

## 3. Verification against the Five Millennium Eclipse Canon (`eclipse-canon`)

How can an embedded analytic ephemeris be verified over thousands of years without external SPICE files? A solar or lunar eclipse occurs only when the Sun, the Moon, and the Earth align to within a fraction of a degree.

`eclipse-canon` embeds all **11,898 solar eclipses** and **12,064 lunar eclipses** from the NASA Five Millennium Canon of Espenak and Meeus ($-1999$ to $+3000$). The release gate (`cargo test --release -p navigate-celestial --test eclipse_canon`) tests `AnalyticEphemeris` (fitted to JPL DE431) against every eclipse in the 5,000-year canon. In the modern era, `AnalyticEphemeris` matches JPL DE431 for the Sun and the Moon to within **0.3 arcseconds**.

## 4. Implementation in `Navigate`

| Crate | Key types and functions | Properties |
|---|---|---|
| `flamsteed` | `Catalog::bright_star_catalogue`, `Star::direction_at`, `Star::component_direction_at`, `Motion` | `#![no_std]`, zero-heap, supports `f64` and `f32`, embeds Yale Bright Star Catalogue 5th rev. + 133 ORB6 binary orbits |
| `navigate-celestial` | `AnalyticEphemeris`, `AniseEphemeris` (`anise` feature), `observe_body`, `sun_ecef`, `moon_ecef`, `stars_ecef_tt_into`, `delta_t`, `tt_to_tcl` | Sans-IO; `long-range` feature covers $-2000$ to $+3000$; compact default-features-off build covers $1800$ to $2200$ |
| `eclipse-canon` | `solar_eclipses`, `lunar_eclipses`, `SolarEclipse`, `LunarEclipse` | `#![no_std]`, zero dependencies, zero-heap embedded 5,000-year reference tables |
