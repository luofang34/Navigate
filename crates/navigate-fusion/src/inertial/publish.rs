//! Publication of the inertial solution (ADR-0004, ADR-0014).

use nalgebra::Matrix3;
use navigate_contract::{
    CartesianPosition, CartesianSolution, CartesianVelocity, MonotonicNanos, NavigationSolution,
    NedVelocity, SolutionStamp,
};

use super::InertialNavigationFilter;
use super::estimate::InertialCore;
use super::models::{plane_to_local_at, quaternion_of, vehicle_position};
use super::nominal::{ATT, POS, VEL, to_contract};
use crate::filter::{horizontal_1sigma_m, matrix_to_cov3, vertical_1sigma_m};
use crate::integrity;
use navigate_contract::{IntegrityAssessment, SpaceIntegrityAssessment};

impl InertialNavigationFilter {
    /// Publishes the body-surface solution for `now`. The position,
    /// velocity, attitude, and covariances are in the local NED axes at
    /// the vehicle: `q = q_plane→local ⊗ q_body→plane` and
    /// `P_att = C_bl·P_θ·C_blᵀ`. The result is `None` before the
    /// initialization, in a space filter, for a `now` before the state
    /// time, and for a `now` after the state time before the first IMU
    /// sample. Publication does not change the retained state.
    pub fn tick_surface(&mut self, now: MonotonicNanos) -> Option<NavigationSolution> {
        let core = self.propagated(now)?;
        let surface = *core.regime.surface()?;
        let position = vehicle_position(&core, &surface);
        let r = plane_to_local_at(&surface, position).ok()?;
        let rotate = |block: Matrix3<f64>| {
            let rotated = r * block * r.transpose();
            (rotated + rotated.transpose()) * 0.5
        };
        let pos_cov = rotate(core.p.fixed_view::<3, 3>(POS, POS).into_owned());
        let vel_cov = rotate(core.p.fixed_view::<3, 3>(VEL, VEL).into_owned());
        let v = r * core.nominal.v;
        let attitude = quaternion_of(&r) * core.nominal.q;
        let c_bl = attitude.to_rotation_matrix().into_inner();
        let p_theta = core.p.fixed_view::<3, 3>(ATT, ATT).into_owned();
        let att_cov = c_bl * p_theta * c_bl.transpose();
        let integrity = self.assess(
            horizontal_1sigma_m(&pos_cov),
            vertical_1sigma_m(&pos_cov),
            now,
        );
        let stamp = self.next_stamp(now);
        Some(
            NavigationSolution::new(
                stamp,
                position,
                NedVelocity::new(v.x, v.y, v.z),
                matrix_to_cov3(&pos_cov),
                matrix_to_cov3(&vel_cov),
                integrity,
                self.ledger.epoch_composition(),
                surface.body.ellipsoid.contract(),
            )
            .with_attitude(to_contract(&attitude))
            .with_attitude_cov(matrix_to_cov3(&((att_cov + att_cov.transpose()) * 0.5))),
        )
    }

    /// Publishes the space solution for `now` in the active frame. The
    /// attitude rotates the body axes into the frame axes, and
    /// `P_att = C·P_θ·Cᵀ`. The uncertainty uses the orbital `RTN` axes
    /// when `|r × v|` defines an orbit plane, and the principal axes
    /// otherwise (ADR-0015). The result is `None` under the same
    /// conditions as [`Self::tick_surface`], and in a surface filter.
    pub fn tick_space(&mut self, now: MonotonicNanos) -> Option<CartesianSolution> {
        let core = self.propagated(now)?;
        if core.regime.surface().is_some() {
            return None;
        }
        let frame = core.regime.frame();
        let x = &core.nominal;
        let pos_cov = core.p.fixed_view::<3, 3>(POS, POS).into_owned();
        let vel_cov = core.p.fixed_view::<3, 3>(VEL, VEL).into_owned();
        let c = x.c_b_to_f();
        let att_cov = c * core.p.fixed_view::<3, 3>(ATT, ATT) * c.transpose();
        let uncertainty = integrity::space_uncertainty(x.p, x.v, pos_cov);
        let integrity = self.assess_space(uncertainty, now);
        let stamp = self.next_stamp(now);
        Some(
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
        )
    }

    /// A copy of the state propagated to `now`.
    fn propagated(&self, now: MonotonicNanos) -> Option<InertialCore> {
        let mut core = self.core.clone()?;
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

    fn next_stamp(&mut self, now: MonotonicNanos) -> SolutionStamp {
        let sequence = self.next_solution_sequence;
        self.next_solution_sequence = sequence.next();
        SolutionStamp::new(self.epoch, sequence, now, self.clock)
    }
}
