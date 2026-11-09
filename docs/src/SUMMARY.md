# Summary

[Introduction](introduction.md)

# Part I: The Navigation Problem and Foundations

- [The Navigation Problem](part1-foundations/ch01-navigation-problem.md)
- [Reference Frames, Time Scales, and Multi-Body Geodesy](part1-foundations/ch02-frames-time-geodesy.md)
- [Architectural Contracts: Correlation, Integrity, and Determinism](part1-foundations/ch03-architectural-contracts.md)

# Part II: Integrated Navigation Architecture

- [Bayesian Filtering and the Admission Funnel](part2-integration/ch04-filtering-and-admission.md)
- [Strapdown Inertial Navigation and the 22-State Unified ESKF](part2-integration/ch05-strapdown-inertial-eskf.md)
- [Nonlinear and Multimodal Estimation: Marginalized Particle Filtering](part2-integration/ch06-marginalized-particle-filter.md)
- [Latency, Clocks, and Out-of-Sequence Measurements](part2-integration/ch07-latency-clocks-oosm.md)
- [Coupling Paradigms: Loose, Tight, and Deep Integration](part2-integration/ch08-coupling-paradigms.md)

# Part III: Specific Navigation Modalities

- [Radiometric and Satellite Navigation](part3-modalities/ch09-radiometric-navigation.md)
- [Celestial Mechanics, Ephemerides, and Star Catalogs](part3-modalities/ch10-celestial-mechanics-catalogs.md)
- [Star Tracking and Astro-Inertial Navigation](part3-modalities/ch11-star-tracking-astro-inertial.md)
- [Visual and Thermal Terrain-Relative Navigation I: Sensing, Rendering, and Matching](part3-modalities/ch12-visual-trn-sensing-matching.md)
- [Visual and Thermal Terrain-Relative Navigation II: Pose Solving, Corridor Tracking, and Fusion](part3-modalities/ch13-visual-trn-pose-corridor-fusion.md)
- [Magnetic Anomaly Navigation (MagNav)](part3-modalities/ch14-magnetic-anomaly-navigation.md)
- [Altimetry, Slant-Range Terrain Clearance, and Terrain Safety](part3-modalities/ch15-altimetry-terrain-clearance.md)
- [Reduced-Dynamic Orbit Determination and Cislunar Navigation](part3-modalities/ch16-orbit-determination-cislunar.md)

# Part IV: From Navigation to Guidance and Trajectory Execution

- [Flight Plans, Procedures, and Surface Guidance](part4-guidance/ch17-routes-procedures-surface-guidance.md)
- [Space Trajectory Laws and Multi-Phase Mission Orchestration](part4-guidance/ch18-space-trajectories-missions.md)

# Part V: Verification, Tooling, and Appendices

- [Deterministic Closed-Loop Verification (`navrun`)](part5-tooling/ch19-deterministic-verification-navrun.md)
- [Visual Pipeline, Model Conversion, and Web Deployment](part5-tooling/ch20-visual-pipeline-web-deployment.md)

---

[Appendix A: Mathematical Notation and Coordinate Conventions](appendix/a-notation-and-conventions.md)
[Appendix B: Crate Topology and ADR Traceability Matrix](appendix/b-crate-and-adr-matrix.md)
[Appendix C: Bibliography and References](appendix/c-references.md)
