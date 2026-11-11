//! The error-state layout.

use crate::inertial::{CORE_STATES, StateLayout};

#[test]
fn the_full_layout_has_every_block_after_the_core() {
    let full = StateLayout::FULL;
    assert_eq!(full.dimension(), 22);
    assert!(full.fits(22));
    assert!(!full.fits(21));
    assert_eq!(
        [
            full.position(),
            full.velocity(),
            full.attitude(),
            full.accel_bias(),
            full.gyro_bias()
        ],
        [0, 3, 6, 9, 12]
    );
    assert_eq!(full.alt_bias(), Some(CORE_STATES));
    assert_eq!(full.mag_bias(), Some(16));
    assert_eq!(full.dmc(), Some(17));
    assert_eq!(full.clock(), Some(20));
}

#[test]
fn the_filter_of_today_is_built_with_the_full_layout() {
    let filter = crate::inertial::InertialNavigationFilter::new_surface(
        crate::inertial::InertialConfig::default(),
        super::CLOCK,
        navigate_astrodynamics::PlanetaryBody::EARTH_WGS84,
    );
    assert_eq!(filter.config.layout, StateLayout::FULL);
}

mod reduced {
    use nalgebra::{Const, DimMin};
    use navigate_contract::SymmetricCov3;

    use super::super::*;
    use crate::inertial::{InertialNavigationFilter, LayoutError, StateLayout};
    use crate::observation::{Observation, ObservationValue};
    use crate::rejection::RejectionReason;

    fn config(layout: StateLayout) -> InertialConfig {
        InertialConfig {
            layout,
            ..InertialConfig::default()
        }
    }

    fn filter<const N: usize>(layout: StateLayout) -> InertialNavigationFilter<N>
    where
        Const<N>: DimMin<Const<N>, Output = Const<N>>,
    {
        let mut filter = InertialNavigationFilter::<N>::surface_with_layout(
            config(layout),
            CLOCK,
            PlanetaryBody::EARTH_WGS84,
        )
        .expect("the layout fits");
        filter
            .initialize_surface(
                earth_origin(),
                NedVelocity::new(0.0, 0.0, 0.0),
                identity_attitude(),
                SymmetricCov3::from_diagonal(25.0, 25.0, 25.0),
                SymmetricCov3::from_diagonal(0.01, 0.01, 0.01),
                SymmetricCov3::from_diagonal(1e-6, 1e-6, 1e-6),
                t(0),
                gnss(),
            )
            .expect("plausible initial state");
        filter
    }

    /// 60 s of rotating IMU data with a position fix every 3 s and an
    /// altitude every second when `baro` holds.
    fn fly<const N: usize>(filter: &mut InertialNavigationFilter<N>, baro: bool)
    where
        Const<N>: DimMin<Const<N>, Output = Const<N>>,
    {
        let core = filter.core.expect("initialized");
        let (g, _) = core.regime.gravity(&core.nominal.p);
        let c = core.nominal.c_b_to_f();
        let force: [f64; 3] = (c.transpose() * -g).into();
        let earth = c.transpose() * core.regime.frame_rotation();
        let rate = [earth.x, earth.y, earth.z + 0.02];
        for k in 1..=600_u32 {
            let at_ms = u64::from(k) * 100;
            assert_accepted(filter.ingest_imu(&imu(k, at_ms, (force, rate), 0.1), t(at_ms)));
            if k % 30 == 0 {
                let mut p = earth_origin();
                p.latitude_rad += f64::from(k % 7) * 1e-7;
                let fix = Observation::new(
                    stamp(1, k, at_ms),
                    ObservationValue::PositionFix {
                        position: p,
                        covariance: SymmetricCov3::from_diagonal(4.0, 4.0, 9.0),
                    },
                    gnss(),
                );
                assert_accepted(filter.ingest(&fix, t(at_ms)));
            }
            if baro && k % 10 == 0 {
                let altitude = Observation::new(
                    stamp(3, k, at_ms),
                    ObservationValue::Altitude {
                        altitude_m: earth_origin().altitude_m + 1.0,
                        variance_m2: 1.0,
                    },
                    SourceComposition::of(SensorClass::Barometric),
                );
                assert_accepted(filter.ingest(&altitude, t(at_ms)));
            }
        }
    }

    /// The nominal state, the covariance of the core block, and the
    /// variance of the altitude bias of the layout after one flight.
    type CoreResult = (
        crate::inertial::nominal::InertialNominal,
        nalgebra::SMatrix<f64, 15, 15>,
        Option<f64>,
    );

    fn run<const N: usize>(layout: StateLayout, baro: bool) -> CoreResult
    where
        Const<N>: DimMin<Const<N>, Output = Const<N>>,
    {
        let mut flight = std::boxed::Box::new(filter::<N>(layout));
        fly(&mut flight, baro);
        let core = flight.core.expect("initialized");
        (
            core.nominal,
            core.p.fixed_view::<15, 15>(0, 0).into_owned(),
            layout.alt_bias().map(|i| core.p[(i, i)]),
        )
    }

    #[test]
    fn a_layout_without_an_unused_block_gives_the_same_core_estimate() {
        let full = run::<22>(StateLayout::FULL, false);
        let aviation = run::<16>(StateLayout::AVIATION, false);
        let core = run::<15>(StateLayout::CORE, false);
        assert_eq!((aviation.0, aviation.1), (full.0, full.1));
        assert_eq!((core.0, core.1), (full.0, full.1));
        // With an altimeter the altitude bias block is used: the full and
        // the aviation layouts agree, the bias included.
        let full = run::<22>(StateLayout::FULL, true);
        let aviation = run::<16>(StateLayout::AVIATION, true);
        assert_eq!(aviation, full);
    }

    #[test]
    fn a_layout_refuses_a_dimension_or_a_model_it_does_not_have() {
        assert_eq!(
            InertialNavigationFilter::<16>::surface_with_layout(
                config(StateLayout::CORE),
                CLOCK,
                PlanetaryBody::EARTH_WGS84,
            )
            .err(),
            Some(LayoutError::Dimension {
                layout: 15,
                filter: 16
            })
        );
        let with_dmc = InertialConfig {
            dmc: Some(crate::kalman::FogmBias::new(100.0, 1e-6)),
            ..config(StateLayout::CORE)
        };
        assert_eq!(
            InertialNavigationFilter::<15>::surface_with_layout(
                with_dmc,
                CLOCK,
                PlanetaryBody::EARTH_WGS84
            )
            .err(),
            Some(LayoutError::MissingBlock { block: "dmc" })
        );
    }

    #[test]
    fn a_measurement_whose_block_is_absent_is_refused_or_reads_without_it() {
        let mut core = std::boxed::Box::new(filter::<15>(StateLayout::CORE));
        let magnetic = Observation::new(
            stamp(4, 1, 0),
            ObservationValue::MagneticAnomaly {
                measured_t: 1e-7,
                map_t_at_lin: 1e-7,
                gradient_ned_t_per_m: [1e-9, 0.0, 0.0],
                linearization_point: earth_origin(),
                valid_radius_m: 1_000.0,
                variance_t2: 1e-18,
            },
            SourceComposition::of(SensorClass::MagneticAnomaly),
        );
        assert!(matches!(
            core.ingest(&magnetic, t(0)),
            crate::rejection::IngestOutcome::Rejected(
                RejectionReason::UnsupportedMeasurement { .. }
            )
        ));
        // A barometric altitude reads as a height without a bias.
        fly(&mut core, true);
        assert_eq!(core.layout(), StateLayout::CORE);
    }
}
