"""Extract long road-like line proposals without fixed image-region cutoffs."""
import math

import cv2
import numpy as np


def samples(line, shape, step=4):
    endpoints = np.asarray(line, float).reshape(2, 2)
    count = max(2, round(np.linalg.norm(endpoints[1] - endpoints[0]) / step) + 1)
    points = np.linspace(*endpoints, count)
    height, width = shape[:2]
    inside = ((points >= 0).all(axis=1) & (points[:, 0] < width - 1)
              & (points[:, 1] < height - 1))
    return points[inside]


def reference_line(image, valid):
    """Propose the centre of a long neutral-coloured strip in rendered imagery."""
    h, w = valid.shape
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 80)
            & (hsv[:, :, 2] < 240) & (valid > 0)).astype(np.uint8) * 255
    lines = cv2.HoughLinesP(mask, 1, np.pi / 720, 80,
                            minLineLength=round(min(h, w) * .6),
                            maxLineGap=round(min(h, w) * .05))
    if lines is None:
        return None
    # Hough endpoints seed a strip centre; averaging across the road width avoids
    # choosing an arbitrary shoulder solely because its contrast is stronger.
    line = max(lines.reshape(-1, 4), key=lambda p: np.hypot(p[2] - p[0], p[3] - p[1]))
    points = samples(line, image.shape)
    direction = points[-1] - points[0]
    direction /= np.linalg.norm(direction)
    normal = np.array([-direction[1], direction[0]])
    offsets = np.arange(-20, 21)
    centred = []
    for point in points:
        xy = (point + offsets[:, None] * normal).round().astype(int)
        inside = ((xy >= 0).all(axis=1) & (xy[:, 0] < w) & (xy[:, 1] < h))
        xy, shift = xy[inside], offsets[inside]
        found = shift[mask[xy[:, 1], xy[:, 0]] > 0]
        if len(found):
            centred.append(point + np.median(found) * normal)
    if len(centred) < 30:
        return None
    fitted = cv2.fitLine(np.array(centred, np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
    direction, origin = fitted[:2], fitted[2:]
    along = (np.asarray(centred) - origin) @ direction
    result = samples(np.r_[origin + along.min() * direction, origin + along.max() * direction], image.shape)
    xy = result.round().astype(int)
    result = result[valid[xy[:, 1], xy[:, 0]] > 0]
    return result if len(result) >= 30 else None


def query_line(image, valid, expected):
    """Find a supported edge near a projected strip, retaining only observed extent."""
    h, w = valid.shape
    fitted = cv2.fitLine(np.asarray(expected, np.float32), cv2.DIST_HUBER, 0, .01, .01).ravel()
    direction, origin = fitted[:2], fitted[2:]
    normal = np.array([-direction[1], direction[0]])
    gray = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8)).apply(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    edges = cv2.Canny(cv2.GaussianBlur(gray, (3, 3), 0), 30, 80)
    safe = cv2.erode((valid > 0).astype(np.uint8), np.ones((5, 5), np.uint8))
    edges[safe == 0] = 0
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, 100,
                            minLineLength=round(min(h, w) * .5), maxLineGap=20)
    candidates = []
    for line in [] if lines is None else lines.reshape(-1, 4):
        endpoints = line.reshape(2, 2)
        axis = endpoints[1] - endpoints[0]
        axis = axis / np.linalg.norm(axis)
        if abs(axis @ direction) < math.cos(math.radians(10)):
            continue
        if abs((endpoints.mean(axis=0) - origin) @ normal) > min(h, w) * .1:
            continue
        xy = samples(line, image.shape, step=1).round().astype(int)
        strength = int(np.count_nonzero(edges[xy[:, 1], xy[:, 0]]))
        if strength >= .35 * len(xy):
            candidates.append((strength, line))
    if not candidates:
        return None
    return samples(max(candidates, key=lambda item: item[0])[1], image.shape)
