"""Image-only matching and native-pixel flow refinement for research camera bundles."""
from itertools import combinations

import cv2
import numpy as np

from photo_tracking import matching_pixels, match_pair, prepared


def matching_window(mask):
    """Exclude empty borders without letting small reflection islands retain the sky."""
    _, labels, regions, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8))
    if len(regions) == 1:
        raise ValueError('photo has no valid matching pixels')
    largest = 1 + int(np.argmax(regions[1:, cv2.CC_STAT_AREA]))
    # Disconnected terrain through separate windows must remain available. Only
    # discard outlying islands when one component contains almost all valid pixels.
    dominant = regions[largest, cv2.CC_STAT_AREA] >= .9 * regions[1:, cv2.CC_STAT_AREA].sum()
    y, x = np.where(labels == largest if dominant else mask > 0)
    return int(x.min()), int(y.min()), int(x.max() + 1), int(y.max() + 1)


def refine_flow(images, sizes, pairs):
    if len(pairs) < 24 or not np.isfinite(pairs).all():
        return np.empty((0, 2, 2))
    transform, _ = cv2.findHomography(pairs[:, 0], pairs[:, 1], cv2.USAC_MAGSAC,
                                    3., maxIters=10000, confidence=.999)
    if transform is None or not np.isfinite(transform).all() or np.linalg.cond(transform) > 1e8:
        return np.empty((0, 2, 2))
    scales = [np.array(image.shape[::-1]) / size for image, size in zip(images, sizes)]
    centers = [np.array([[s[0], 0, (s[0] - 1) / 2], [0, s[1], (s[1] - 1) / 2],
                         [0, 0, 1.]]) for s in scales]
    native_transform = centers[1] @ transform @ np.linalg.inv(centers[0])
    inverse = np.linalg.inv(native_transform)
    warped = cv2.warpPerspective(images[1], inverse, images[0].shape[::-1])
    points = [((pairs[:, i] + .5) * scales[i] - .5).astype(np.float32).reshape(-1, 1, 2)
              for i in range(2)]
    predicted = cv2.perspectiveTransform(points[1], inverse)
    # Image-only prealignment removes lens scale and rotation. Coarse pyramid
    # windows would mix the limited overlap with the warped image's empty border.
    settings = dict(winSize=(31, 31), maxLevel=1,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 40, .001),
        flags=cv2.OPTFLOW_USE_INITIAL_FLOW)
    forward, good, error = cv2.calcOpticalFlowPyrLK(
        images[0], warped, points[0], predicted.copy(), **settings)
    backward, returned, reverse_error = cv2.calcOpticalFlowPyrLK(
        warped, images[0], forward, points[0].copy(), **settings)
    native_query = cv2.perspectiveTransform(forward, native_transform)
    cycle = np.linalg.norm(backward - points[0], axis=(1, 2))
    shift = np.linalg.norm((native_query - points[1]) / scales[1], axis=(1, 2))
    keep = (good.ravel().astype(bool) & returned.ravel().astype(bool) &
            (cycle <= .5) & (shift <= 3) & (error.ravel() < 20) & (reverse_error.ravel() < 20))
    result = pairs[keep].copy()
    result[:, 1] = (native_query[keep, 0] + .5) / scales[1] - .5
    return result


def selected_pixels(pairs, masks):
    selected = np.isfinite(pairs).all(axis=(1, 2))
    for i, mask in enumerate(masks):
        finite = np.isfinite(pairs[:, i]).all(axis=1)
        xy = np.rint(np.where(np.isfinite(pairs[:, i]), pairs[:, i], -1)).astype(int)
        height, width = mask.shape
        inside = finite & (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
        indices = np.flatnonzero(inside)
        inside[indices] &= mask[xy[indices, 1], xy[indices, 0]] > 0
        selected &= inside
    return selected


def match_edges(cases, root, output, matcher, exclude_bright_clouds):
    paths = []; masks = []; clear = []; native = []; sizes = []; windows = []
    for i, case in enumerate(cases):
        image, mask, size = prepared(case, root)
        left, top, right, bottom = matching_window(mask)
        windows.append([left, top, right, bottom])
        path = output / f'photo-{i}.png'
        if not cv2.imwrite(str(path), matching_pixels(image[top:bottom, left:right],
                                                      mask[top:bottom, left:right], matcher)):
            raise ValueError(f'cannot write matching photo: {path}')
        paths.append(path); masks.append(mask); sizes.append(size)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        bright = ((hsv[:, :, 1] < 90) & (hsv[:, :, 2] > 180)).astype(np.uint8)
        bright = cv2.dilate(bright, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
        clear.append(((mask > 0) & ((bright == 0) | (not exclude_bright_clouds))).astype(np.uint8))
        full = cv2.imread(str(root / case['image']), 0)
        ratio = 1600 / max(full.shape)
        native.append(cv2.resize(full, (round(full.shape[1] * ratio), round(full.shape[0] * ratio))))
    edges = []; reports = []
    order = [(i, 0) for i in range(1, len(cases))] + list(combinations(range(1, len(cases)), 2))
    for a, b in order:
        directory = output / f'edge-{a}-{b}'; directory.mkdir()
        raw = np.asarray(match_pair(matcher, paths[a], paths[b], directory), np.float64).reshape(-1, 2, 2)
        raw += np.array([windows[a][:2], windows[b][:2]])
        valid = raw[selected_pixels(raw, [masks[a], masks[b]])]
        if len(valid) < 24:
            raise ValueError(f'insufficient masked photo matches on bundle edge {a}-{b}')
        refined = refine_flow([native[a], native[b]], [sizes[a], sizes[b]], valid)
        refined = refined[selected_pixels(refined, [masks[a], masks[b]])]
        ground = selected_pixels(refined, [clear[a], clear[b]])
        np.savez_compressed(directory / 'pairs.npz', pairs=refined, clear=ground, raw_pairs=raw)
        edges.append((a, b, refined, ground))
        reports.append({'edge': [a, b], 'raw_pairs': len(raw), 'valid_pairs': len(valid),
                        'crop_xyxy': [windows[a], windows[b]],
                        'refined_pairs': len(refined), 'clear_pairs': int(ground.sum())})
    return edges, reports
