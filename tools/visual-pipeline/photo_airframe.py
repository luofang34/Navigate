"""Complete partially labelled aircraft supports using their image-border edges."""
import itertools
import math

import cv2
import numpy as np


LINE_POLICY = {'long_edge_pixels': 640, 'minimum_length_fraction': .45,
               'endpoint_border_fraction': .07, 'maximum_gap_fraction': .08,
               'minimum_width_fraction': .035, 'maximum_width_fraction': .35,
               'maximum_angle_deg': 12, 'minimum_airplane_fraction': .10,
               'maximum_edge_density': .10, 'confidence_min': .35,
               'padding_fraction': .008, 'maximum_lines': 128,
               'selection': 'maximum_airplane_fraction'}


def border_lines(edges):
    height, width = edges.shape
    short = min(height, width)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, 70,
                           minLineLength=round(short * LINE_POLICY['minimum_length_fraction']),
                           maxLineGap=round(short * LINE_POLICY['maximum_gap_fraction']))
    if lines is None:
        return []
    selected = []
    for line in lines.reshape(-1, 4):
        a, b = line.reshape(2, 2).astype(float)
        edge = lambda point: min(point[0], point[1], width - 1 - point[0], height - 1 - point[1])
        if max(edge(a), edge(b)) < short * LINE_POLICY['endpoint_border_fraction']:
            samples = np.linspace(a, b, math.ceil(np.linalg.norm(b - a)) + 1).round().astype(int)
            strength = np.count_nonzero(edges[samples[:, 1], samples[:, 0]])
            selected.append((strength, a, b))
    # Long accidental alignments in textured terrain must not crowd the genuine
    # support boundaries out of the bounded set of candidate lines.
    selected.sort(key=lambda item: -item[0])
    return [(a, b) for _, a, b in selected[:LINE_POLICY['maximum_lines']]]


def border_support_mask(image, labels, confidence):
    if image is None or image.shape[:2] != labels.shape or confidence.shape != labels.shape:
        raise ValueError('airframe edge masks require aligned image, labels and confidence')
    height, width = labels.shape
    scale = min(1., LINE_POLICY['long_edge_pixels'] / max(height, width))
    size = (round(width * scale), round(height * scale))
    pixels = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    semantic = cv2.resize(labels, size, interpolation=cv2.INTER_NEAREST)
    certainty = cv2.resize(confidence, size, interpolation=cv2.INTER_LINEAR)
    # A road or roof can have the same long parallel edges. Only an existing
    # airplane label can seed an expansion; sky, walls and fences cannot.
    airplane = (semantic == 90) & (certainty >= LINE_POLICY['confidence_min'])
    if not airplane.any():
        return np.zeros(labels.shape, np.uint8)
    edges = cv2.Canny(cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY), 40, 120)
    mask = np.zeros(edges.shape, np.uint8)
    best_fraction = LINE_POLICY['minimum_airplane_fraction']
    short = min(edges.shape)
    for (a, b), (c, d) in itertools.combinations(border_lines(edges), 2):
        u, v = (b - a) / np.linalg.norm(b - a), (d - c) / np.linalg.norm(d - c)
        angle = math.degrees(math.acos(np.clip(abs(float(u @ v)), 0, 1)))
        separation = abs(float(((c + d - a - b) / 2) @ np.array([-u[1], u[0]])))
        if angle > LINE_POLICY['maximum_angle_deg'] or not (
                short * LINE_POLICY['minimum_width_fraction'] < separation
                < short * LINE_POLICY['maximum_width_fraction']):
            continue
        polygon = cv2.convexHull(np.array([a, b, c, d], np.float32)).round().astype(np.int32)
        area = np.zeros(edges.shape, np.uint8)
        cv2.fillConvexPoly(area, polygon, 255)
        selected = area > 0
        fraction = np.mean(airplane[selected])
        if (fraction > best_fraction
                and np.mean(edges[selected] > 0) < LINE_POLICY['maximum_edge_density']):
            # Several pairs can bracket the same support. Combining them also
            # includes strips of nearby ground; prefer the tightest semantic fit.
            best_fraction = fraction
            mask = area
    padding = max(1, round(short * LINE_POLICY['padding_fraction']))
    mask = cv2.dilate(mask, np.ones((2 * padding + 1, 2 * padding + 1), np.uint8))
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
