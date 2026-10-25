"""Pixel residuals against a terrain-height envelope without triangulated-depth weights."""
import cv2
import numpy as np


def project(camera, points):
    eye, vector, _, intrinsics = camera
    return cv2.projectPoints(points - eye, vector, np.zeros(3), intrinsics, None)[0].reshape(-1, 2)


def terrain_hits(camera, pixels, surface, clearance_m):
    """Intersect rays with the first covered terrain crossing at the supplied clearance."""
    eye, _, rotation, intrinsics = camera
    rays = np.c_[pixels, np.ones(len(pixels))] @ np.linalg.inv(intrinsics).T @ rotation
    near = np.zeros(len(pixels)); far = np.full(len(pixels), 40000.)
    hit = np.zeros(len(pixels), bool); previous = 0.
    previous_above = np.zeros(len(pixels), bool)
    for distance in np.geomspace(10, 40000, 75):
        lat, lon, altitude = surface.geodetic(eye + distance * rays)
        dem, covered = surface.sample('dem', lat, lon)
        # Entering a catalog below its surface is a coverage edge, not a terrain hit.
        found = previous_above & covered & (altitude <= dem + clearance_m) & ~hit
        near[found] = previous; far[found] = distance; hit |= found; previous = distance
        previous_above = covered & (altitude > dem + clearance_m)
    for _ in range(14):
        mid = (near + far) / 2
        lat, lon, altitude = surface.geodetic(eye + mid[:, None] * rays)
        dem, covered = surface.sample('dem', lat, lon)
        above = (altitude > dem + clearance_m) | ~covered
        near = np.where(above, mid, near); far = np.where(above, far, mid)
    points = eye + ((near + far) / 2)[:, None] * rays
    return points, hit


def envelope_residual(cameras, pairs, surface):
    """Measure distance to projected 0/60/120 m clearance segments; missing rays fail."""
    if not len(pairs):
        return np.empty((0, 2)), np.empty(0, bool)
    samples = []; valid = np.ones(len(pairs), bool)
    eye, _, rotation, _ = cameras[1]
    for height in [0., 60., 120.]:
        points, hit = terrain_hits(cameras[0], pairs[:, 0], surface, height)
        pixels = project(cameras[1], points)
        depth = ((points - eye) @ rotation.T)[:, 2]
        valid &= hit & (depth > 0) & np.isfinite(pixels).all(axis=1)
        samples.append(pixels)
    residuals = []
    for a, b in zip(samples[:-1], samples[1:]):
        direction = b - a
        position = np.clip(np.sum((pairs[:, 1] - a) * direction, axis=1) /
                           np.maximum(np.sum(direction * direction, axis=1), 1e-12), 0, 1)
        residuals.append(a + position[:, None] * direction - pairs[:, 1])
    errors = np.stack(residuals)
    best = np.argmin(np.linalg.norm(errors, axis=2), axis=0)
    return np.where(valid[:, None], errors[best, np.arange(len(pairs))], 100.), valid
