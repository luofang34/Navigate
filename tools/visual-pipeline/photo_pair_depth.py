"""Terrain clearance checks for triangulated forest correspondences."""
import cv2
import numpy as np


# Broad clearance priors allow DEM error and tall trees; they are not surveyed heights.
FOREST_BOUNDS_M = (-5., 120.)


def forest_pairs(data, pairs):
    selected = np.ones(len(pairs), bool)
    for i, view in enumerate(data):
        if 'labels' not in view:
            return np.zeros(len(pairs), bool)
        xy = np.rint(pairs[:, i]).astype(int)
        height, width = view['labels'].shape
        inside = (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
        forest = np.zeros(len(pairs), bool)
        forest[inside] = view['labels'][xy[inside, 1], xy[inside, 0]] == 4
        selected &= forest
    return selected


def forest_training_pairs(data, consensus):
    """Retain rare semantic support before applying the depth-computation budget."""
    forest = consensus[forest_pairs(data, consensus)]
    indices = np.linspace(0, len(forest) - 1, min(120, len(forest)), dtype=int)
    return forest[indices]


def clearance(cameras, pairs, surface):
    heights = np.full(len(pairs), np.nan)
    valid = np.zeros(len(pairs), bool)
    if not len(pairs):
        return heights, valid
    e1, _, R1, K1 = cameras[0]
    e2, _, R2, K2 = cameras[1]
    if np.linalg.norm(e1 - e2) < 1e-6:
        return heights, valid
    P1 = K1 @ np.column_stack([R1, np.zeros(3)])
    P2 = K2 @ np.column_stack([R2, R2 @ (e1 - e2)])
    hom = cv2.triangulatePoints(P1, P2, pairs[:, 0].T, pairs[:, 1].T)
    finite = np.isfinite(hom).all(axis=0) & (abs(hom[3]) > 1e-12)
    points = (hom[:3, finite] / hom[3, finite]).T + e1
    if not len(points):
        return heights, valid
    a, b = points - e1, points - e2
    cosine = np.sum(a * b, axis=1) / np.maximum(1e-15, np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    usable = ((a @ R1.T)[:, 2] > 0) & ((b @ R2.T)[:, 2] > 0) & (cosine < np.cos(np.radians(.1)))
    lat, lon, altitude = surface.geodetic(points)
    dem, covered = surface.sample('dem', lat, lon)
    heights[finite] = altitude - dem
    valid[finite] = usable & covered & np.isfinite(altitude - dem)
    return heights, valid


def forest_residual(cameras, pairs, surface):
    heights, valid = clearance(cameras, pairs, surface)
    low, high = FOREST_BOUNDS_M
    # Missing coverage and degenerate depth must not let the optimizer evade the check.
    return np.r_[np.where(valid, np.minimum(heights - low, 0) / 5, 1000),
                 np.where(valid, np.maximum(heights - high, 0) / 10, 1000)]


def forest_check(cameras, pairs, surface):
    heights, valid = clearance(cameras, pairs, surface)
    low, high = FOREST_BOUNDS_M
    supported = valid & (heights >= low) & (heights <= high)
    fraction = float(np.mean(supported)) if len(pairs) else 0.
    return {'status': ('insufficient_support' if len(pairs) < 30 else
                       'consistent_with_clearance_prior' if fraction >= .9 else 'terrain_conflict'),
            'pairs': len(pairs), 'valid_depth_pairs': int(valid.sum()),
            'bounds_m': list(FOREST_BOUNDS_M), 'minimum_parallax_deg': .1,
            'within_bounds_fraction': fraction,
            'height_quantiles_m': np.quantile(heights[valid], [.1, .5, .9]).tolist() if valid.any() else None,
            'held_pairs_filtered_by_fit': False, 'absolute_pose_verified': False}
