"""Research camera bundles with direct map anchors and spatially held-out photo edges."""
import math

import cv2
import numpy as np
from scipy.optimize import least_squares

from photo_pair_depth import forest_check
from photo_pose import prior_radius, spatial_holdout
from photo_surface import basis
from photo_terrain_envelope import envelope_residual, project
from photo_water_pair import epipolar


def parameters(views, surface):
    bases = []; frames = []; vectors = []; intrinsics = []
    initial = []; lower = []; upper = []
    for view in views:
        lat, lon, altitude = surface.geodetic(view['eye'])
        position = view['prior'].get('position') or {}
        reported = position.get('reported_horizontal_error_m')
        bounded = reported is not None and 0 < reported < 100
        if bounded:
            lat, lon = position['latitude'], position['longitude']
        origin = surface.local(lat, lon, altitude)
        local = surface.basis.T @ basis(lat, lon)
        offset = (view['eye'] - origin) @ local
        # An epsilon-only initial vector gives the trust-region solver an epsilon-sized
        # first step and can satisfy ftol before correcting any camera error.
        offset[np.abs(offset) < 1e-6] = 0.
        # EXIF accuracy is a sensor claim; preserve the same uncertainty floor
        # as map acquisition instead of clipping image-supported camera seeds.
        radius = prior_radius(position)
        offset[:2] = np.clip(offset[:2], -radius + 1e-6, radius - 1e-6)
        bases.append(origin); frames.append(local)
        vectors.append(cv2.Rodrigues(np.diag([1, -1, -1]) @ view['rotation'].T)[0].ravel())
        K = view['K'].copy()
        focal = view['prior']['focal_35mm_mm'] * math.hypot(*view['size']) / math.hypot(36, 24)
        K[0, 0] = K[1, 1] = focal; intrinsics.append(K)
        initial.extend([*offset, 0, 0, 0])
        lower.extend([-radius, -radius, -500, -.15, -.15, -.15])
        upper.extend([radius, radius, 500, .15, .15, .15])
    initial.append(0); lower.append(math.log(.9)); upper.append(math.log(1.1))

    def cameras(values):
        result = []
        for i in range(len(views)):
            eye = bases[i] + frames[i] @ values[i * 6:i * 6 + 3]
            vector = vectors[i] + values[i * 6 + 3:i * 6 + 6]
            K = intrinsics[i].copy(); K[0, 0] *= math.exp(values[-1]); K[1, 1] *= math.exp(values[-1])
            result.append((eye, vector, cv2.Rodrigues(vector)[0], K))
        return result

    return cameras, np.array(initial), (np.array(lower), np.array(upper))


def support(views, edges):
    training = []; validation = []; maps = []
    for a, b, pairs, clear in edges:
        held = spatial_holdout(pairs[:, 1], tuple(views[b]['size']))
        validation.append((a, b, pairs[held], pairs[held & clear]))
        selected = pairs[(~held) & clear]
        if len(selected) < 24 or held.sum() < 12:
            raise ValueError(f'insufficient photo support on bundle edge {a}-{b}')
        cv2.setRNGSeed(0)
        _, mask = cv2.findFundamentalMat(selected[:, 0], selected[:, 1], cv2.USAC_MAGSAC, 1., .999, 10000)
        if mask is None or mask.sum() < 24:
            raise ValueError(f'no training consensus on bundle edge {a}-{b}')
        selected = selected[mask.ravel() > 0]
        index = np.linspace(0, len(selected) - 1, min(120, len(selected)), dtype=int)
        training.append((a, b, selected[index]))
    for i, view in enumerate(views):
        if not view['direct_map_anchor']:
            continue
        held = spatial_holdout(view['pixels'], tuple(view['size']))
        indices = np.flatnonzero((~held) & view['inliers'])
        if len(indices) < 24 or held.sum() < 12:
            raise ValueError(f'insufficient direct map support for bundle view {i}')
        indices = indices[np.linspace(0, len(indices) - 1, min(120, len(indices)), dtype=int)]
        maps.append((i, view['points'][indices], view['pixels'][indices],
                     view['points'][held], view['pixels'][held]))
    if len(maps) < 2:
        raise ValueError('a terrain bundle requires at least two direct map anchors')
    return training, validation, maps


def evaluate(cameras, validation, maps, surface):
    held = {}; map_held = {}
    for a, b, pairs, clear_pairs in validation:
        error = abs(epipolar([cameras[a], cameras[b]], pairs))
        residual, covered = envelope_residual([cameras[a], cameras[b]], clear_pairs, surface)
        held[f'{a}-{b}'] = {'pairs': len(pairs), 'median_epipolar_px': float(np.median(error)),
            'inlier_fraction': float(np.mean(error < 2)),
            'clear_terrain': forest_check([cameras[a], cameras[b]], clear_pairs, surface),
            'terrain_rays_covered': int(covered.sum()),
            'terrain_reprojection_px_quantiles': (np.quantile(np.linalg.norm(residual, axis=1),
                                                   [.1, .5, .9]).tolist() if len(residual) else None)}
    for i, _, _, points, pixels in maps:
        error = np.linalg.norm(project(cameras[i], points) - pixels, axis=1)
        map_held[str(i)] = {'pairs': len(points), 'median_px': float(np.median(error)),
                            'inlier_fraction': float(np.mean(error < 4))}
    return {'held': held, 'anchor_map_held': map_held,
            'positions': [list(map(float, surface.geodetic(c[0]))) for c in cameras]}


def solve(views, edges, surface):
    cameras, initial, bounds = parameters(views, surface)
    training, validation, maps = support(views, edges)

    def residual(values):
        selected = cameras(values); terms = []
        for a, b, pairs in training:
            pair = [selected[a], selected[b]]
            terms.extend([epipolar(pair, pairs) * 2, envelope_residual(pair, pairs, surface)[0].ravel()])
        for i, points, pixels, _, _ in maps:
            terms.append((project(selected[i], points) - pixels).ravel())
        return np.concatenate([*terms, [values[-1] / .1]])

    steps = np.r_[np.tile([.5, .5, .5, 1e-4, 1e-4, 1e-4], len(views)), 1e-4]

    def jacobian(values):
        columns = []
        for i, step in enumerate(steps):
            delta = np.zeros(len(values)); delta[i] = step
            columns.append((residual(values + delta) - residual(values - delta)) / (2 * step))
        return np.column_stack(columns)

    result = least_squares(residual, initial, jac=jacobian,
        x_scale=np.r_[np.tile([100, 100, 100, .05, .05, .05], len(views)), .1],
        bounds=bounds, loss='soft_l1', f_scale=2, max_nfev=100)
    fitted = cameras(result.x)
    return fitted, {**evaluate(fitted, validation, maps, surface),
        'status': 'research_proposal' if result.success else 'optimization_incomplete',
        'geometry_accepted': False, 'absolute_pose_verified': False, 'requires_map_verification': True,
        'optimizer_converged': bool(result.success), 'evaluations': int(result.nfev), 'cost': float(result.cost),
        'focal_factor': math.exp(result.x[-1]), 'training_points': sum(len(p) for _, _, p in training),
        'holdout_filtered_by_fitted_geometry': False,
        'model': 'shared focal; terrain 0/60/120m envelope; direct map anchors; shared position uncertainty bounds'}
