"""Explicit, hash-checked local height corrections over immutable terrain tiles."""
import json
import math
from pathlib import Path

import cv2
import numpy as np

from photo_surface import RADIUS, Surface
from photo_testset import digest, write_json


def region_mask(surface, points):
    """Segment a bounded imagery region from independently matched surface seeds."""
    image, _, zoom, tile_x, tile_y = surface.grids['imagery']
    points = np.asarray(points, float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('surface seeds must be finite 3D points')
    if len(points) < 3:
        return None
    lat, lon, _ = surface.geodetic(points)
    x = ((lon + 180) / 360 * 2**zoom - tile_x) * 256 - .5
    y = ((1 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2 * 2**zoom - tile_y) * 256 - .5
    if (min(x.min(), y.min()) < 3 or x.max() >= image.shape[1] - 3
            or y.max() >= image.shape[0] - 3):
        return None
    meters_per_pixel = 2 * np.pi * RADIUS * np.cos(np.radians(np.mean(lat))) / (2**zoom * 256)
    if max(np.ptp(x), np.ptp(y)) * meters_per_pixel > 1000:
        return None
    padding = max(4, math.ceil(50 / meters_per_pixel))
    left, top = max(0, math.floor(x.min()) - padding), max(0, math.floor(y.min()) - padding)
    right = min(image.shape[1], math.ceil(x.max()) + padding)
    bottom = min(image.shape[0], math.ceil(y.max()) + padding)
    if right <= left or bottom <= top or (right - left) * (bottom - top) > 16_000_000:
        return None
    crop = image[top:bottom, left:right].copy()
    labels = np.full(crop.shape[:2], cv2.GC_PR_BGD, np.uint8)
    labels[:2] = labels[-2:] = cv2.GC_BGD
    labels[:, :2] = labels[:, -2:] = cv2.GC_BGD
    for point in np.column_stack([x - left, y - top]).round().astype(int):
        cv2.circle(labels, tuple(point), 3, cv2.GC_FGD, -1)
    cv2.setRNGSeed(0)
    cv2.grabCut(crop, labels, None, np.zeros((1, 65)), np.zeros((1, 65)), 5, cv2.GC_INIT_WITH_MASK)
    mask = np.isin(labels, [cv2.GC_FGD, cv2.GC_PR_FGD]).astype(np.uint8) * 255
    if not mask.any() or np.mean(mask > 0) > .9:
        return None
    return mask, [left, top, right, bottom], [zoom, tile_x, tile_y]


def save_correction(directory, mask, window, origin, height, catalog_hash):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / 'relief-mask.png'
    if not cv2.imwrite(str(path), mask):
        raise OSError(f'cannot save relief mask: {path}')
    model = {'format': 'sokoly-photo-surface-relief', 'version': 1,
             'height_offset_m': float(height), 'window': window, 'xyz_origin': origin,
             'mask': path.name, 'mask_sha256': digest(path), 'catalog_sha256': catalog_hash}
    destination = directory / 'surface-correction.json'
    write_json(destination, model)
    return destination


class ReliefSurface(Surface):
    """Ray cast a local surface correction without changing the shared base grids."""
    def __init__(self, base, path, catalog_hash):
        self.__dict__.update(base.__dict__)
        self.base = base
        path = Path(path)
        model = json.loads(path.read_text())
        if (model.get('format') != 'sokoly-photo-surface-relief' or model.get('version') != 1
                or model.get('catalog_sha256') != catalog_hash):
            raise ValueError('surface correction requires its exact base catalog')
        relative = Path(model['mask'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('surface correction mask must remain inside its directory')
        mask_path = (path.parent / relative).resolve(strict=True)
        mask_path.relative_to(path.parent.resolve())
        if digest(mask_path) != model['mask_sha256']:
            raise ValueError('surface correction mask checksum mismatch')
        self.relief_mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        self.relief_window = model['window']
        self.relief_origin = model['xyz_origin']
        self.relief_height = model['height_offset_m']
        if (not isinstance(self.relief_window, list) or len(self.relief_window) != 4
                or any(type(value) is not int for value in self.relief_window)):
            raise ValueError('surface correction window must contain four integer bounds')
        left, top, right, bottom = self.relief_window
        height, width = base.grids['imagery'][0].shape[:2]
        if (not (0 <= left < right <= width and 0 <= top < bottom <= height)
                or self.relief_mask is None or self.relief_mask.shape != (bottom - top, right - left)
                or self.relief_origin != list(base.grids['imagery'][2:])
                or type(self.relief_height) not in (int, float)
                or not math.isfinite(self.relief_height) or not -40 <= self.relief_height <= 80
                or not np.isin(self.relief_mask, [0, 255]).all()):
            raise ValueError('invalid surface correction dimensions, frame, height or mask')

    def sample(self, kind, latitude, longitude):
        values, covered = self.base.sample(kind, latitude, longitude)
        if kind != 'dem':
            return values, covered
        return values + self.relief_height * self.region(latitude, longitude), covered

    def region(self, latitude, longitude):
        """Identify the corrected footprint independently of its inferred height."""
        zoom, tile_x, tile_y = self.relief_origin
        left, top, _, _ = self.relief_window
        x = (((np.asarray(longitude) + 180) / 360 * 2**zoom - tile_x) * 256 - .5 - left).astype(np.float32)
        y = (((1 - np.arcsinh(np.tan(np.radians(latitude))) / np.pi) / 2 * 2**zoom - tile_y) * 256 - .5 - top).astype(np.float32)
        return cv2.remap(self.relief_mask, np.atleast_2d(x), np.atleast_2d(y),
                           cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT).reshape(np.shape(x)) > 0
