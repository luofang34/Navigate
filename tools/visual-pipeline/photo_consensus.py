"""Diagnose coherent image correspondences without correcting their pose error."""
import cv2
import numpy as np

from photo_pose import spatial_holdout


def consensus_alignment(pairs, size):
    """Select associations on training cells, then measure unchanged pixel offsets.

    This diagnostic does not replace fixed-render acceptance. A coherent plane
    alone cannot establish that the entire three-dimensional scene is aligned.
    """
    pairs = np.asarray(pairs, float).reshape(-1, 2, 2)
    result = {'aligned': False, 'pairs': 0, 'held_pairs': 0, 'median_px': None,
              'held_median_px': None, 'inlier_fraction': 0., 'held_inlier_fraction': 0.,
              'image_support_fraction': 0., 'pose_refitted': False,
              'selection_uses_identity_error': False}
    held = spatial_holdout(pairs[:, 1], size)
    if (~held).sum() < 24 or held.sum() < 6:
        return result
    parameters = cv2.UsacParams()
    parameters.confidence = .999
    parameters.maxIterations = 10000
    parameters.randomGeneratorState = 0
    parameters.isParallel = False
    parameters.threshold = 3.
    parameters.score = cv2.SCORE_METHOD_MAGSAC
    model, _ = cv2.findHomography(pairs[~held, 0], pairs[~held, 1], parameters)
    if model is None or not np.isfinite(model).all() or np.linalg.cond(model) > 1e8:
        return result
    predicted = cv2.perspectiveTransform(pairs[:, 0][None], model)[0]
    selected = np.linalg.norm(predicted - pairs[:, 1], axis=1) < 3
    error = np.linalg.norm(pairs[:, 0] - pairs[:, 1], axis=1)
    tested = selected & held
    support = pairs[selected & (error < 4), 1].astype(np.float32)
    coverage = (float(cv2.contourArea(cv2.convexHull(support)) / np.prod(size))
                if len(support) >= 3 else 0.)
    result.update(pairs=int(selected.sum()), held_pairs=int(tested.sum()),
                  association_homography=model.tolist(), image_support_fraction=coverage)
    if selected.any():
        result.update(median_px=float(np.median(error[selected])),
                      inlier_fraction=float(np.mean(error[selected] < 4)))
    if tested.any():
        result.update(held_median_px=float(np.median(error[tested])),
                      held_inlier_fraction=float(np.mean(error[tested] < 4)))
    result['aligned'] = bool(selected.sum() >= 30 and tested.sum() >= 6 and coverage >= .05
                             and result['inlier_fraction'] >= .6
                             and result['held_inlier_fraction'] >= .6
                             and result['held_median_px'] <= 4)
    return result
