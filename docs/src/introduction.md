# Introduction

> ⚠️ **Work in progress (WIP) — experimental draft. No guarantees of any kind.**
>
> Both this book and the `Navigate` software workspace are active works in progress. Chapters, mathematical derivations, API contracts, and literature citations are incomplete and subject to change without notice. Nothing in this book or repository has been qualified or certified for operational use. Do not use this material to navigate or operate a real vehicle, and do not rely on it in any safety-critical context.

**Author:** Fang Luo, Sokoly Research

This book has two purposes:

1. Introduction of modern integrated navigation and guidance methods.
2. It is the engineering manual for the [`Navigate`](https://github.com/luofang34/Navigate) Rust workspace.

## Technical names

- **Navigation** is the estimation of the position, velocity, attitude, and time of a vehicle, together with the uncertainty and the integrity of that estimate.
- **Guidance** is the calculation of trajectory commands that steer the vehicle from its estimated state toward a flight plan or a mission target.
- **Control** is the high-rate stabilization of the vehicle attitude and the allocation of actuator forces and torques.
- **Integrity** is a measure of the trust that a guidance law or an operator can place in the correctness of a navigation solution.
- A **sans-IO core** is a deterministic software module that performs no input/output (I/O) and reads no system clock.

## Connecting navigation theory and implementation

Ideal filter equations assume stationary Gaussian noise and continuous sensor availability. In practice, a vehicle navigation system operates under physical and environmental constraints:

- Global Navigation Satellite System (GNSS) signals experience multipath interference, radio-frequency jamming, and spoofing.
- Inertial Measurement Units (IMUs) drift over time because of sensor biases and gravity model errors.
- Optical and star cameras become unavailable in clouds, fog, precipitation, smoke, or featureless terrain, and thermal infrared contrast degrades during rain or diurnal thermal crossover.
- Camera frames and geophysical map measurements arrive with processing latency after newer IMU samples.
- Geophysical and terrain maps contain flat regions with weak gradients and repetitive regions with multiple matching locations.
- Cascaded estimators create statistical correlations when one filter consumes the output of another filter.
- Long-range vehicles cross the geographic poles and move between planetary reference frames.

This book connects these physical models and failure conditions to the types, admission checks, and modules in the `Navigate` workspace. Equations in the text correspond to modules in the repository, and admission gates map to variants in `RejectionReason` or `GuidanceRefusal`.

## Structure of the book

- **Part I (Chapters 1–3)** defines the navigation problem, establishes reference frames, time scales, and multi-body geodesy, and states the architectural contracts of `Navigate`.
- **Part II (Chapters 4–8)** presents the theory of integrated navigation: Bayesian admission gates, the 22-state Cartesian Error-State Kalman Filter (`InertialNavigationFilter`), the Marginalized Particle Filter (`MarginalizedParticleFilter`), out-of-sequence measurement replay, and the progression from loose coupling to tight and deep coupling.
- **Part III (Chapters 9–16)** examines each specific navigation modality: radiometric ranging, celestial mechanics, star tracking, visible and thermal terrain-relative navigation (TRN), magnetic anomaly navigation (MagNav), slant-range terrain clearance, and reduced-dynamic orbit determination (RDOD).
- **Part IV (Chapters 17–18)** shows how integrity-gated navigation solutions drive atmospheric route execution and space trajectory laws from orbit to touchdown.
- **Part V (Chapters 19–20) and the Appendices** cover deterministic simulation (`navrun`), the visual dataset and neural model pipeline (`navigate-visual-pipeline`), coordinate conventions, the crate-to-ADR traceability matrix, and the [bibliography](appendix/c-references.md).

## Licence and attribution

- **Book text and diagrams:** © Fang Luo, Sokoly Research. Licensed under the [Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/). You may share and adapt this text for any purpose, provided that you give appropriate credit to **Fang Luo, Sokoly Research**, link to the license, indicate if changes were made, and distribute your contributions under the same CC BY-SA 4.0 license.
- **Software source code:** The Rust source code in the `Navigate` repository is licensed under the GNU Affero General Public License, version 3 or later (`AGPL-3.0-or-later`), unless a crate manifest specifies otherwise.

### How to cite this work

```bibtex
@book{luo2026navigate,
  author       = {Luo, Fang},
  institution  = {Sokoly Research},
  title        = {Navigate: Principles and Architecture of Modern Integrated Navigation},
  year         = {2026},
  note         = {Work in progress, licensed under CC BY-SA 4.0}
}
```

## How to build this book locally

Install `mdbook` and run the build command from the repository root:

```sh
mdbook build docs
```

The build writes HTML output to `target/book/`. Run `mdbook serve docs` to preview the book locally in a browser.
