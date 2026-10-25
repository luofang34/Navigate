"""Reflective water landmarks with independent photo pairing and map footprints."""
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from photo_shoreline import photo_water, reference_water
from photo_surface import RADIUS


def reflective_regions(view):
    image, valid, K = view['image'], view['valid'], view['K']
    main = photo_water(image, view['labels'], valid)
    rows = np.nonzero(main)[0]
    if not len(rows):
        raise ValueError('distant water needs a supported main water outline')
    height, width = valid.shape
    horizon = view['horizon']; roll = np.radians(horizon['roll_deg'])
    line = (K[1, 2] - np.tan(np.radians(horizon['down_deg'])) * K[0, 0] / np.cos(roll)
            + np.tan(roll) * (np.arange(width) - K[0, 2]))
    region = (valid > 0) & (np.arange(height)[:, None] > line[None] + 2)
    region[max(0, int(rows.min()) - 5):] = False
    if not region.any():
        raise ValueError('no covered ground between the main water and horizon')
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    threshold = float(np.quantile(hsv[:, :, 2][region], .98))
    bright = (hsv[:, :, 2] >= threshold) & (hsv[:, :, 1] < 50) & region
    count, labels, stats, centers = cv2.connectedComponentsWithStats(bright.astype(np.uint8))
    regions = []
    for key in range(1, count):
        if stats[key, 4] >= 20:
            y, x = np.where(labels == key)
            regions.append({'center': centers[key], 'pixels': np.column_stack([x, y]).astype(float)})
    return regions


def pair_regions(regions, homography):
    """Use image-only cycle agreement before consulting any terrain pose."""
    if min(map(len, regions)) < 2:
        raise ValueError('at least two distant water regions are required in each photo')
    a, b = [np.array([item['center'] for item in items]) for items in regions]
    forward = cv2.perspectiveTransform(a[None], homography)[0]
    reverse = cv2.perspectiveTransform(b[None], np.linalg.inv(homography))[0]
    errors = np.maximum(np.linalg.norm(forward[:, None] - b[None], axis=2),
                        np.linalg.norm(a[:, None] - reverse[None], axis=2))
    selected = []
    for i in range(len(a)):
        j = int(errors[i].argmin())
        ratio = len(regions[0][i]['pixels']) / len(regions[1][j]['pixels'])
        if errors[i, j] <= 12 and errors[:, j].argmin() == i and .25 <= ratio <= 4:
            selected.append((float(errors[i, j]), i, j))
    selected.sort()
    if len(selected) < 2:
        raise ValueError('distant water regions did not form two independent photo pairs')
    selected = selected[:2]
    observations = [[regions[side][item[side + 1]]['pixels'] for item in selected] for side in range(2)]
    return observations, [{'cycle_max_error_px': e, 'reference_center': a[i].tolist(),
                           'query_center': b[j].tolist()} for e, i, j in selected]


def map_crop(surface, camera, size):
    eye, vector, rotation, K = camera
    color, depth, world = surface.render(eye, rotation.T @ np.diag([1, -1, -1]), K, size)
    main = reference_water(color, depth)
    rows = np.nonzero(main)[0]
    if not len(rows):
        raise ValueError('reference camera has no supported main water outline')
    cut = int(rows.min()) - 5
    band = depth > 0; band[max(0, cut):] = False
    if not band.any():
        raise ValueError('context catalog does not cover the distant ground')
    lat, lon, _ = surface.geodetic(world[band])
    grid, valid, z, x0, y0 = surface.context.grids['imagery']
    px = ((lon + 180) / 360 * 2**z - x0) * 256
    py = ((1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * 2**z - y0) * 256
    left, right = max(0, int(px.min()) - 16), min(grid.shape[1], int(px.max()) + 17)
    top, bottom = max(0, int(py.min()) - 16), min(grid.shape[0], int(py.max()) + 17)
    if right <= left or bottom <= top:
        raise ValueError('distant footprint does not intersect context imagery')
    return grid[top:bottom, left:right], valid[top:bottom, left:right], (z, x0, y0, left, top), cut


def shoreline_world(surface, contour, origin):
    z, x0, y0, left, top = origin
    pixels = cv2.approxPolyDP(contour, 1.5, True).reshape(-1, 2).astype(float) + [left, top]
    lon = ((pixels[:, 0] + .5) / 256 + x0) / 2**z * 360 - 180
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 *
        ((pixels[:, 1] + .5) / 256 + y0) / 2**z))))
    altitude, covered = surface.sample('dem', lat, lon)
    if not covered.all() or len(pixels) < 3:
        return None
    la, lo = np.radians(lat), np.radians(lon)
    unit = np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])
    return (unit * (RADIUS + altitude[:, None]) - surface.origin) @ surface.basis


def associate_footprints(footprints, targets, observed_areas):
    """Reject water footprints too small to explain even partial reflected regions."""
    centers = np.array([points.mean(axis=0) for points in footprints])
    areas = np.array([cv2.contourArea(points.astype(np.float32)) for points in footprints])
    cost = np.linalg.norm(targets[:, None] - centers[None], axis=2)
    # Reflection may occupy only part of a lake. Allow a factor of two in focal
    # scale, but a subpixel footprint cannot explain a multi-pixel water region.
    cost[areas[None] * 4 < np.asarray(observed_areas)[:, None]] = 1e6
    rows, columns = linear_sum_assignment(cost)
    if len(rows) != len(targets) or np.any(cost[rows, columns] > 40):
        raise ValueError('no plausible one-to-one distant water association')
    return columns, cost[rows, columns], areas[columns]


def map_regions(surface, camera, size, targets, observed_areas):
    image, valid, origin, cut = map_crop(surface, camera, size)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    water = (hsv[:, :, 0] > 65) & (hsv[:, :, 2] < 65) & (valid > 0)
    water = cv2.morphologyEx(water.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(water)
    eye, vector, rotation, K = camera
    candidates, footprints, records = [], [], []
    for key in range(1, count):
        x, y, w, h, area = map(int, stats[key])
        if area < 64:
            continue
        region = (labels[y:y+h, x:x+w] == key).astype(np.uint8)
        contours, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shifted = (*origin[:3], origin[3] + x, origin[4] + y)
        points = shoreline_world(surface, max(contours, key=cv2.contourArea), shifted)
        if points is None or not (((points - eye) @ rotation.T)[:, 2] > 0).all():
            continue
        uv = cv2.projectPoints(points - eye, vector, np.zeros(3), K, None)[0].reshape(-1, 2)
        center = uv.mean(axis=0)
        if not (-40 < center[0] < size[0] + 40 and 0 < center[1] < cut):
            continue
        if np.linalg.norm(targets - center, axis=1).min() > 40:
            continue
        candidates.append(points); footprints.append(uv)
        records.append({'orthophoto_area': area, 'center': center.tolist(),
                        'source_bbox': [origin[3] + x, origin[4] + y, w, h]})
    if len(candidates) < len(targets):
        raise ValueError('context imagery has insufficient plausible water footprints')
    columns, errors, areas = associate_footprints(footprints, targets, observed_areas)
    return [candidates[j] for j in columns], [dict(records[j], query_component=int(i),
        initial_centroid_error_px=float(errors[i]), projected_area_px2=float(areas[i]),
        observed_area_px2=int(observed_areas[i])) for i, j in enumerate(columns)]


def polygon_distance(points, polygon):
    """Distance to the filled concave footprint; bays must remain outside water."""
    a, b = polygon, np.roll(polygon, -1, axis=0)
    ab, ap = b - a, points[:, None] - a[None]
    t = np.clip(np.sum(ap * ab[None], axis=2) / np.maximum(1e-12, np.sum(ab * ab, axis=1)), 0, 1)
    distance = np.linalg.norm(ap - t[:, :, None] * ab[None], axis=2).min(axis=1)
    crossing = (a[None, :, 1] > points[:, None, 1]) != (b[None, :, 1] > points[:, None, 1])
    denominator = np.where(abs(ab[:, 1]) < 1e-12, 1e-12, ab[:, 1])
    right = points[:, None, 0] < (ab[None, :, 0] * (points[:, None, 1] - a[None, :, 1])
                                  / denominator[None] + a[None, :, 0])
    inside = np.logical_xor.reduce(crossing & right, axis=1)
    return np.where(inside, 0, distance)


def landmark_residual(cameras, observations, polygons, selection=(0,)):
    terms = []
    for i, (eye, vector, _, K) in enumerate(cameras):
        for index in selection:
            uv = cv2.projectPoints(polygons[index] - eye, vector, np.zeros(3), K, None)[0].reshape(-1, 2)
            terms.extend(polygon_distance(observations[i][index], uv))
    return np.array(terms)
