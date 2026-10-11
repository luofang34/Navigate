"""Bounded camera-position hypotheses around an explicitly uncertain location."""
import math

from photo_pose import prior_radius


def offsets(position, step_m=0., radius_m=0.):
    if not math.isfinite(step_m) or not math.isfinite(radius_m) or min(step_m, radius_m) < 0:
        raise ValueError('position grid distances must be finite and nonnegative')
    if not step_m:
        if radius_m:
            raise ValueError('position radius requires a positive grid step')
        return [(0., 0.)]
    radius = radius_m or position.get('reported_horizontal_error_m') or prior_radius(position)
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError('position search requires positive uncertainty')
    extent = math.floor(radius / step_m)
    if extent > 8:
        raise ValueError('position grid is too dense; increase its step')
    result = [(x * step_m, y * step_m) for y in range(-extent, extent + 1)
              for x in range(-extent, extent + 1) if math.hypot(x * step_m, y * step_m) <= radius]
    return sorted(result, key=lambda value: (math.hypot(*value), value))


def location(latitude, longitude, east_m, north_m):
    if east_m == 0 and north_m == 0:
        return latitude, longitude
    distance = math.hypot(east_m, north_m) / 6371008.8
    bearing = math.atan2(east_m, north_m)
    lat, lon = map(math.radians, [latitude, longitude])
    destination = math.asin(math.sin(lat) * math.cos(distance)
                            + math.cos(lat) * math.sin(distance) * math.cos(bearing))
    lon += math.atan2(math.sin(bearing) * math.sin(distance) * math.cos(lat),
                     math.cos(distance) - math.sin(lat) * math.sin(destination))
    return math.degrees(destination), (math.degrees(lon) + 180) % 360 - 180
