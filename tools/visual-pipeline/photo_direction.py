"""Transfer an orientation proposal between overlapping photographs."""
import cv2
import numpy as np

from photo_pose import spatial_holdout


def orientation_proposal(pairs, source_K, target_K, source_rotation, source_size, target_size):
    """Use a planar image transform only as a heading seed, never as a located pose.

    Translation and relief also contribute to a homography, so its nearest
    rotation needs independent map verification before camera admission.
    """
    pairs = np.asarray(pairs, float).reshape(-1, 2, 2)
    if not np.isfinite(pairs).all():
        raise ValueError('orientation correspondences must be finite')
    if len(pairs) < 60:
        return None
    held = spatial_holdout(pairs[:, 0], source_size)
    if held.sum() < 12 or (~held).sum() < 48:
        return None
    cv2.setRNGSeed(0)
    transform, consensus = cv2.findHomography(pairs[~held, 0], pairs[~held, 1],
        cv2.USAC_MAGSAC, 2., maxIters=10000, confidence=.999)
    if transform is None or consensus is None or consensus.sum() < 40:
        return None
    predicted = cv2.perspectiveTransform(pairs[:, :1], transform).reshape(-1, 2)
    error = np.linalg.norm(predicted - pairs[:, 1], axis=1)
    inliers = error < 4
    coverage = [float(cv2.contourArea(cv2.convexHull(pairs[inliers, i].astype(np.float32))) /
                      np.prod(size)) if inliers.sum() > 3 else 0
                for i, size in enumerate([source_size, target_size])]
    score = {'pairs': len(pairs), 'inliers': int(inliers.sum()), 'held_count': int(held.sum()),
             'held_median_px': float(np.median(error[held])),
             'held_inlier_fraction': float(np.mean(error[held] < 4)), 'coverage': coverage}
    if min(coverage) < .1 or score['held_median_px'] > 4 or score['held_inlier_fraction'] < .6:
        return None
    normalized = np.linalg.inv(target_K) @ transform @ source_K
    if np.linalg.det(normalized) < 0:
        normalized = -normalized
    u, singular, vt = np.linalg.svd(normalized)
    if singular.min() <= 0 or singular.max() / singular.min() > 3:
        return None
    relative = u @ vt
    optical = relative @ np.diag([1, -1, -1]) @ source_rotation.T
    return {'rotation': optical.T @ np.diag([1, -1, -1]), 'transform': transform,
            'score': score, 'inliers': inliers, 'geometry_accepted': False,
            'requires_map_verification': True}
