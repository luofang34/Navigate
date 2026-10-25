"""Check a fixed camera against fresh rendered-image matches without refitting it."""
from pathlib import Path

import cv2
import numpy as np

from photo_tracking import matching_pixels
from photo_consensus import consensus_alignment


def alignment(pairs, query_mask, reference_mask):
    """Measure identity alignment only where both images have observed surface pixels."""
    pairs = np.asarray(pairs, float).reshape(-1, 2, 2)
    height, width = query_mask.shape
    if reference_mask.shape != query_mask.shape:
        raise ValueError('fixed-pose images must have the same dimensions')
    inside = (np.isfinite(pairs).all(axis=(1, 2)) & (pairs >= 0).all(axis=(1, 2))
              & (pairs[:, :, 0] < width - .5).all(axis=1)
              & (pairs[:, :, 1] < height - .5).all(axis=1))
    pairs = pairs[inside]
    xy = pairs.round().astype(int)
    keep = ((query_mask[xy[:, 1, 1], xy[:, 1, 0]] > 0)
            & (reference_mask[xy[:, 0, 1], xy[:, 0, 0]] > 0))
    pairs = pairs[keep]
    error = np.linalg.norm(pairs[:, 0] - pairs[:, 1], axis=1)
    inliers = error < 4
    coverage = (float(cv2.contourArea(cv2.convexHull(pairs[inliers, 1].astype(np.float32)))
                      / (width * height)) if inliers.sum() >= 3 else 0)
    median = float(np.median(error)) if len(error) else None
    fraction = float(inliers.mean()) if len(error) else 0
    passed = len(pairs) >= 30 and coverage >= .05 and fraction >= .6 and median <= 4
    return {'pairs': len(pairs), 'inliers': int(inliers.sum()), 'median_px': median,
            'inlier_fraction': fraction, 'image_support_fraction': coverage,
            'passed': bool(passed), 'pose_refitted': False,
            'consensus': consensus_alignment(pairs, (width, height))}, pairs, error


def verify_pose(surface, eye, rotation, intrinsics, size, matcher, query, mask, directory,
                contrast=False):
    """Persist a fresh match against the final render, separate from PnP fitting evidence."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    color, depth, _ = surface.render(eye, rotation, intrinsics, size)
    reference_mask = depth > 0
    reference = directory / 'reference.png'
    if not cv2.imwrite(str(reference), matching_pixels(color, reference_mask, matcher, contrast)):
        raise OSError(f'cannot save fixed-pose render: {reference}')
    response = matcher.match(reference, query, refinement=getattr(matcher, 'patch_refinement', True))
    report, pairs, errors = alignment(response['pairs'], mask, reference_mask)
    np.savez_compressed(directory / 'matches.npz', pairs=pairs, errors=errors,
                        eye=eye, rotation=rotation, K=intrinsics, size=size)
    return {**report, 'backend': response['backend']}


def merit(report):
    """Rank supported final renders; fitting scores cannot compensate for a failed render."""
    if not report['passed']:
        return None
    return report['inliers'] * report['image_support_fraction'] / (1 + report['median_px'])
