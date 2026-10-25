"""Fit one elevated image region jointly with independently matched ground."""
import cv2
import numpy as np
from scipy.optimize import least_squares

from photo_pose import spatial_holdout


def fit_relief(regions, eye, rotation, intrinsics, size, up):
    """Return a joint camera/height proposal, retaining each region's holdout score.

    The first region is allowed a constant height correction along local gravity;
    the second supplies independent ground correspondences. Input heights remain
    unchanged. Callers must verify a fresh render before accepting the proposal.
    """
    if len(regions) != 2:
        raise ValueError('relief fitting requires elevated and ground regions')
    groups = []
    for points, pixels in regions:
        points, pixels = np.asarray(points, float), np.asarray(pixels, float)
        if points.shape != (len(pixels), 3) or pixels.shape != (len(points), 2):
            raise ValueError('relief correspondences must be aligned 3D/2D arrays')
        if not np.isfinite(points).all() or not np.isfinite(pixels).all():
            raise ValueError('relief correspondences must be finite')
        held = spatial_holdout(pixels, size)
        if len(points) < 30 or held.sum() < 6 or (~held).sum() < 24:
            return None
        groups.append((points, pixels, held))
    up = np.asarray(up, float)
    if up.shape != (3,) or not np.isfinite(up).all() or not np.isclose(np.linalg.norm(up), 1):
        raise ValueError('relief height direction must be a finite unit vector')
    eye = np.asarray(eye, float)
    initial_r = cv2.Rodrigues(np.diag([1, -1, -1]) @ rotation.T)[0].ravel()

    def project(x, index):
        points = groups[index][0] + (x[6] * up if index == 0 else 0)
        local = points - eye - x[:3]
        vector = initial_r + x[3:6]
        uv = cv2.projectPoints(local, vector, np.zeros(3), intrinsics, None)[0].reshape(-1, 2)
        depth = (local @ cv2.Rodrigues(vector)[0].T)[:, 2]
        return uv, depth

    def residual(x):
        errors = []
        for index, (_, pixels, held) in enumerate(groups):
            uv, _ = project(x, index)
            # A dense roof must not outweigh a smaller but geometrically distinct
            # parking lot simply because the matcher sampled it more frequently.
            errors.append((uv - pixels)[~held].ravel() / np.sqrt((~held).sum() / 300))
        return np.concatenate(errors)

    result = least_squares(residual, np.r_[np.zeros(6), 10.],
        x_scale=[100, 100, 100, .1, .1, .1, 10],
        bounds=([-200] * 3 + [-.5] * 3 + [-40], [200] * 3 + [.5] * 3 + [80]),
        loss='soft_l1', f_scale=2, max_nfev=500, diff_step=1e-4)
    scores = []
    for index, (_, pixels, held) in enumerate(groups):
        uv, depth = project(result.x, index)
        errors = np.linalg.norm(uv - pixels, axis=1)
        scores.append({'pairs': len(pixels), 'held_count': int(held.sum()),
            'held_median_px': float(np.median(errors[held])),
            'held_inlier_fraction': float(np.mean(errors[held] < 4)),
            'positive_depth_fraction': float(np.mean(depth > 0))})
    matrix = cv2.Rodrigues(initial_r + result.x[3:6])[0]
    return {'eye': eye + result.x[:3], 'rotation': matrix.T @ np.diag([1, -1, -1]),
        'height_offset_m': float(result.x[6]), 'scores': scores,
        'optimizer_converged': bool(result.success), 'requires_render_verification': True}
