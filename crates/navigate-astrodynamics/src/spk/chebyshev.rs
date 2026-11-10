//! Exact fixed-interval Chebyshev polynomial evaluation up to second derivatives.

/// Why evaluating a Chebyshev record failed.
#[derive(Clone, Copy, Debug, PartialEq, thiserror::Error)]
pub enum ChebyshevError {
    /// Record half-span radius `radius_s` was zero, negative, or non-finite.
    #[error("invalid Chebyshev interval radius {radius_s} s")]
    InvalidRadius {
        /// Supplied half-interval radius in seconds.
        radius_s: f64,
    },
    /// Coefficient count was empty.
    #[error("Chebyshev polynomial has zero coefficients")]
    EmptyCoefficients,
    /// Normalized coordinate `tau` fell outside `[-1 - tol, 1 + tol]`.
    #[error("normalized Chebyshev coordinate tau={tau} outside [-1, 1]")]
    OutOfDomain {
        /// Computed normalized coordinate.
        tau: f64,
    },
}

/// Evaluated 3D `(position, velocity, acceleration)` tuple.
pub type ChebyshevPva3 = ([f64; 3], [f64; 3], [f64; 3]);

#[inline]
const fn mul_add_unfused(a: f64, b: f64, c: f64) -> f64 {
    a * b + c
}

/// Evaluates `(p(t), dp/dt, d2p/dt2)` for one scalar Chebyshev series
/// `p(tau) = sum_{k=0}^{N-1} c_k T_k(tau)` where `tau = ((t_whole_s - mid_s) + t_sub_s) / radius_s`.
///
/// # Errors
/// Returns [`ChebyshevError`] if `radius_s` is non-positive, `coeffs` is empty, or `tau` is out of domain.
pub fn eval_chebyshev_pva(
    coeffs: &[f64],
    mid_s: f64,
    radius_s: f64,
    t_whole_s: f64,
    t_sub_s: f64,
) -> Result<(f64, f64, f64), ChebyshevError> {
    if !radius_s.is_finite() || radius_s <= 0.0 {
        return Err(ChebyshevError::InvalidRadius { radius_s });
    }
    let Some(&c0) = coeffs.first() else {
        return Err(ChebyshevError::EmptyCoefficients);
    };
    let dt = (t_whole_s - mid_s) + t_sub_s;
    let tau_raw = dt / radius_s;
    if !tau_raw.is_finite() || tau_raw.abs() > 1.0 + 1e-11 {
        return Err(ChebyshevError::OutOfDomain { tau: tau_raw });
    }
    let tau = tau_raw.clamp(-1.0, 1.0);
    if coeffs.len() == 1 {
        return Ok((c0, 0.0, 0.0));
    }

    let two_tau = 2.0 * tau;
    let mut t_prev2 = 1.0_f64;
    let mut t_prev1 = tau;
    let mut dt_prev2 = 0.0_f64;
    let mut dt_prev1 = 1.0_f64;
    let mut ddt_prev2 = 0.0_f64;
    let mut ddt_prev1 = 0.0_f64;

    let c1 = coeffs.get(1).copied().unwrap_or(0.0);
    let mut sum_p = mul_add_unfused(c1, t_prev1, c0);
    let mut sum_dp = c1;
    let mut sum_ddp = 0.0_f64;

    let mut k = 2_usize;
    while let Some(&ck) = coeffs.get(k) {
        let tk = mul_add_unfused(two_tau, t_prev1, -t_prev2);
        let dtk = mul_add_unfused(two_tau, dt_prev1, 2.0 * t_prev1 - dt_prev2);
        let ddtk = mul_add_unfused(two_tau, ddt_prev1, 4.0 * dt_prev1 - ddt_prev2);

        sum_p = mul_add_unfused(ck, tk, sum_p);
        sum_dp = mul_add_unfused(ck, dtk, sum_dp);
        sum_ddp = mul_add_unfused(ck, ddtk, sum_ddp);

        t_prev2 = t_prev1;
        t_prev1 = tk;
        dt_prev2 = dt_prev1;
        dt_prev1 = dtk;
        ddt_prev2 = ddt_prev1;
        ddt_prev1 = ddtk;
        k = k.wrapping_add(1);
    }

    let inv_r = 1.0 / radius_s;
    let inv_r2 = inv_r * inv_r;
    Ok((sum_p, sum_dp * inv_r, sum_ddp * inv_r2))
}

/// Evaluates a 3-vector Type-2 Chebyshev record `[mid, radius, x_0..x_{m-1}, y_0..y_{m-1}, z_0..z_{m-1}]`
/// returning `(pos, vel, acc)` scaled by `unit_scale`.
///
/// # Errors
/// Returns [`ChebyshevError`] if the record layout or normalized coordinate is invalid.
pub fn eval_type2_record_3d(
    record: &[f64],
    t_whole_s: f64,
    t_sub_s: f64,
    unit_scale: f64,
) -> Result<ChebyshevPva3, ChebyshevError> {
    if record.len() < 5 || !(record.len() - 2).is_multiple_of(3) {
        return Err(ChebyshevError::EmptyCoefficients);
    }
    let mid_s = record.first().copied().unwrap_or(0.0);
    let radius_s = record.get(1).copied().unwrap_or(0.0);
    let n_coeff = (record.len() - 2) / 3;

    let mut pos = [0.0_f64; 3];
    let mut vel = [0.0_f64; 3];
    let mut acc = [0.0_f64; 3];
    let mut axis = 0_usize;
    while axis < 3 {
        let start = 2_usize.wrapping_add(axis.wrapping_mul(n_coeff));
        let end = start.wrapping_add(n_coeff);
        let coeffs = record
            .get(start..end)
            .ok_or(ChebyshevError::EmptyCoefficients)?;
        let (p, v, a) = eval_chebyshev_pva(coeffs, mid_s, radius_s, t_whole_s, t_sub_s)?;
        if let (Some(ps), Some(vs), Some(as_)) =
            (pos.get_mut(axis), vel.get_mut(axis), acc.get_mut(axis))
        {
            *ps = p * unit_scale;
            *vs = v * unit_scale;
            *as_ = a * unit_scale;
        }
        axis = axis.wrapping_add(1);
    }
    Ok((pos, vel, acc))
}

/// Evaluates a 6-component Type-3 Chebyshev record (`[mid, radius, x.., y.., z.., vx.., vy.., vz..]`)
/// returning `(pos, vel, acc)` where `vel` and `acc = d(vel)/dt` come from the explicit velocity series.
///
/// # Errors
/// Returns [`ChebyshevError`] if the record layout or normalized coordinate is invalid.
pub fn eval_type3_record_3d(
    record: &[f64],
    t_whole_s: f64,
    t_sub_s: f64,
    unit_scale: f64,
) -> Result<ChebyshevPva3, ChebyshevError> {
    if record.len() < 8 || !(record.len() - 2).is_multiple_of(6) {
        return Err(ChebyshevError::EmptyCoefficients);
    }
    let mid_s = record.first().copied().unwrap_or(0.0);
    let radius_s = record.get(1).copied().unwrap_or(0.0);
    let n_coeff = (record.len() - 2) / 6;

    let mut pos = [0.0_f64; 3];
    let mut vel = [0.0_f64; 3];
    let mut acc = [0.0_f64; 3];
    let mut axis = 0_usize;
    while axis < 3 {
        let p_start = 2_usize.wrapping_add(axis.wrapping_mul(n_coeff));
        let p_end = p_start.wrapping_add(n_coeff);
        let v_start = 2_usize.wrapping_add((axis.wrapping_add(3)).wrapping_mul(n_coeff));
        let v_end = v_start.wrapping_add(n_coeff);
        let p_coeffs = record
            .get(p_start..p_end)
            .ok_or(ChebyshevError::EmptyCoefficients)?;
        let v_coeffs = record
            .get(v_start..v_end)
            .ok_or(ChebyshevError::EmptyCoefficients)?;
        let (p, _, _) = eval_chebyshev_pva(p_coeffs, mid_s, radius_s, t_whole_s, t_sub_s)?;
        let (v, a, _) = eval_chebyshev_pva(v_coeffs, mid_s, radius_s, t_whole_s, t_sub_s)?;
        if let (Some(ps), Some(vs), Some(as_)) =
            (pos.get_mut(axis), vel.get_mut(axis), acc.get_mut(axis))
        {
            *ps = p * unit_scale;
            *vs = v * unit_scale;
            *as_ = a * unit_scale;
        }
        axis = axis.wrapping_add(1);
    }
    Ok((pos, vel, acc))
}
