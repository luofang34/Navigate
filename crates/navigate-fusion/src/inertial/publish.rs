//! Publication of the inertial solution (ADR-0004, ADR-0014, ADR-0017).

use navigate_contract::{
    CartesianPosition, CartesianSolution, CartesianVelocity, MonotonicNanos, NavigationSolution,
    NedVelocity, SolutionStamp, WrappingSequence,
};

use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use navigate_contract::SourceComposition;
use navigate_geodesy::BodyEllipsoid;

use super::InertialNavigationFilter;
use super::estimate::InertialCore;
use super::local_level::LocalLevel;
use super::models::vehicle_level;
use super::nominal::{ATT, POS, VEL, to_contract};
use crate::integrity;
use crate::kalman::{horizontal_1sigma_m, matrix_to_cov3, vertical_1sigma_m};
use navigate_astrodynamics::GravityField;
use navigate_contract::{IntegrityAssessment, SpaceIntegrityAssessment};

impl InertialNavigationFilter {
    /// Publishes the geodetic solution for `now` on the reference
    /// ellipsoid of the regime. The filter converts the body-fixed state
    /// at the vehicle: the position is geodetic, and the velocity, the
    /// attitude, and the covariances are in the local NED axes at the
    /// vehicle. Thus `v = R_fl·v_frame`, `q = q_frame→local ⊗ q`,
    /// `P = R_fl·P_frame·R_flᵀ`, and `P_att = C_bl·P_θ·C_blᵀ`. The
    /// horizontal and vertical 1-sigma values come from the local
    /// position covariance. The result is `None` before the
    /// initialization, in a regime without a reference ellipsoid, for a
    /// `now` before the state time, and for a `now` after the state time
    /// before the first IMU sample. Publication does not change the
    /// retained state.
    pub fn tick(&mut self, now: MonotonicNanos) -> Option<NavigationSolution> {
        let core = self.propagated(now)?;
        let ellipsoid = core.regime.surface?;
        let state = SurfaceState::of(&core, ellipsoid);
        let integrity = self.assess(
            horizontal_1sigma_m(&state.pos_cov),
            vertical_1sigma_m(&state.pos_cov),
            now,
        );
        let solution = state.solution(
            self.stamp(now),
            integrity,
            self.ledger.epoch_composition(),
            ellipsoid,
        );
        Some(self.sequencer.surface(solution))
    }

    /// Publishes the Cartesian solution for `now` in the active frame.
    /// Each regime publishes it, so one filter gives this solution in
    /// orbit and the geodetic solution of [`Self::tick`] near the
    /// surface. The attitude rotates the body axes into the frame axes,
    /// and `P_att = C·P_θ·Cᵀ`. The uncertainty uses the orbital `RTN`
    /// axes when `|r × v|` defines an orbit plane, and the principal axes
    /// otherwise (ADR-0015). The result is `None` before the
    /// initialization, for a `now` before the state time, and for a
    /// `now` after the state time before the first IMU sample.
    pub fn tick_space(&mut self, now: MonotonicNanos) -> Option<CartesianSolution> {
        let core = self.propagated(now)?;
        let frame = core.regime.frame();
        let x = &core.nominal;
        let pos_cov = core.p.fixed_view::<3, 3>(POS, POS).into_owned();
        let vel_cov = core.p.fixed_view::<3, 3>(VEL, VEL).into_owned();
        let c = x.c_b_to_f();
        let att_cov = c * core.p.fixed_view::<3, 3>(ATT, ATT) * c.transpose();
        let v_inertial = x.v + core.regime.frame_rotation().cross(&x.p);
        let uncertainty = integrity::space_uncertainty(x.p, v_inertial, pos_cov);
        let integrity = self.assess_space(uncertainty, now);
        let stamp = self.stamp(now);
        Some(
            self.sequencer.space(
                CartesianSolution::new(
                    stamp,
                    frame,
                    CartesianPosition::new(x.p.x, x.p.y, x.p.z),
                    CartesianVelocity::new(x.v.x, x.v.y, x.v.z),
                    matrix_to_cov3(&pos_cov),
                    matrix_to_cov3(&vel_cov),
                    integrity,
                    self.ledger.epoch_composition(),
                )
                .with_attitude(to_contract(&x.q))
                .with_attitude_cov(matrix_to_cov3(&((att_cov + att_cov.transpose()) * 0.5))),
            ),
        )
    }

    /// A copy of the state propagated to `now`.
    fn propagated(&self, now: MonotonicNanos) -> Option<InertialCore> {
        let mut core = self.core?;
        now.elapsed_since(core.state_at)?;
        core.propagate_with(now, None, &self.config).ok()?;
        Some(core)
    }

    fn assess(
        &mut self,
        horizontal_1sigma_m: f64,
        vertical_1sigma_m: f64,
        now: MonotonicNanos,
    ) -> IntegrityAssessment {
        let fusion = self.config.fusion;
        self.ledger.prune_window(now, fusion.assessment_window);
        integrity::derive(
            &self.ledger,
            now,
            horizontal_1sigma_m,
            vertical_1sigma_m,
            &fusion,
        )
    }

    fn assess_space(
        &mut self,
        uncertainty: navigate_contract::SpaceUncertainty,
        now: MonotonicNanos,
    ) -> SpaceIntegrityAssessment {
        let fusion = self.config.fusion;
        self.ledger.prune_window(now, fusion.assessment_window);
        integrity::derive_space(&self.ledger, now, uncertainty, &fusion)
    }

    /// The stamp for `now`. The sequencer sets its sequence.
    const fn stamp(&self, now: MonotonicNanos) -> SolutionStamp {
        SolutionStamp::new(self.epoch, WrappingSequence::new(0), now, self.clock)
    }
}

/// The geodetic state of one core at the vehicle: the position, and the
/// velocity, the attitude, and the covariances in the local NED axes.
pub(super) struct SurfaceState {
    /// The local axes and the geodetic position at the vehicle.
    pub(super) local: LocalLevel,
    /// Velocity in m/s, NED.
    pub(super) velocity: Vector3<f64>,
    /// Attitude from the body axes to the local NED axes.
    pub(super) attitude: UnitQuaternion<f64>,
    /// Position covariance in m², NED.
    pub(super) pos_cov: Matrix3<f64>,
    /// Velocity covariance in (m/s)², NED.
    pub(super) vel_cov: Matrix3<f64>,
    /// Attitude covariance in rad², about the NED axes.
    pub(super) att_cov: Matrix3<f64>,
    /// Cross-covariance of the NED position and the attitude about the
    /// NED axes, in m·rad.
    pub(super) pos_att_cov: Matrix3<f64>,
}

impl SurfaceState {
    /// `v = R_fl·v_frame`, `q = q_frame→local ⊗ q`, `P = R_fl·P_frame·R_flᵀ`,
    /// `P_att = C_bl·P_θ·C_blᵀ`, and `P_pos,att = R_fl·P_p,θ·C_blᵀ`.
    pub(super) fn of(core: &InertialCore, ellipsoid: BodyEllipsoid) -> Self {
        let local = vehicle_level(core, ellipsoid);
        let r = local.bf_to_ned();
        let attitude = local.bf_to_ned_quaternion() * core.nominal.q;
        let c_bl = attitude.to_rotation_matrix().into_inner();
        let p_theta = core.p.fixed_view::<3, 3>(ATT, ATT).into_owned();
        let att_cov = c_bl * p_theta * c_bl.transpose();
        let p_pos_theta = core.p.fixed_view::<3, 3>(POS, ATT).into_owned();
        Self {
            velocity: r * core.nominal.v,
            attitude,
            pos_cov: local.bf_to_ned_cov(&core.p.fixed_view::<3, 3>(POS, POS).into_owned()),
            vel_cov: local.bf_to_ned_cov(&core.p.fixed_view::<3, 3>(VEL, VEL).into_owned()),
            att_cov: (att_cov + att_cov.transpose()) * 0.5,
            pos_att_cov: r * p_pos_theta * c_bl.transpose(),
            local,
        }
    }

    /// The contract solution of this state.
    pub(super) fn solution(
        &self,
        stamp: SolutionStamp,
        integrity: IntegrityAssessment,
        composition: SourceComposition,
        ellipsoid: BodyEllipsoid,
    ) -> NavigationSolution {
        let v = self.velocity;
        NavigationSolution::new(
            stamp,
            self.local.position,
            NedVelocity::new(v.x, v.y, v.z),
            matrix_to_cov3(&self.pos_cov),
            matrix_to_cov3(&self.vel_cov),
            integrity,
            composition,
            ellipsoid.contract(),
        )
        .with_attitude(to_contract(&self.attitude))
        .with_attitude_cov(matrix_to_cov3(&self.att_cov))
    }
}
