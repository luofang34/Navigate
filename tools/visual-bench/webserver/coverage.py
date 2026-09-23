"""Bound area and route requests before provider access."""
import math


def tile_xy(lon, lat, zoom):
    n = 2**zoom
    return ((lon + 180) / 360 * n,
            (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * n)


def tile_bounds(z, x, y):
    n = 2**z
    lat = lambda v: math.degrees(math.atan(math.sinh(math.pi * (1 - 2*v/n))))
    return [x/n*360-180, lat(y+1), (x+1)/n*360-180, lat(y)]


def validate_point(point):
    if len(point) != 2 or not all(isinstance(v, (int, float)) and math.isfinite(v) for v in point):
        raise ValueError('Coordinates must be finite [longitude, latitude] pairs')
    if not -180 <= point[0] <= 180 or not -85 <= point[1] <= 85:
        raise ValueError('Coordinates are outside supported Mercator coverage')


def segment_distance(p, a, b):
    dx, dy = b[0]-a[0], b[1]-a[1]
    denominator = dx*dx + dy*dy
    t = max(0, min(1, ((p[0]-a[0])*dx+(p[1]-a[1])*dy)/denominator)) if denominator else 0
    return math.hypot(p[0]-a[0]-t*dx, p[1]-a[1]-t*dy)


def plan(value):
    zoom = value.get('zoom', 16)
    if type(zoom) is not int or not 14 <= zoom <= 17:
        raise ValueError('Imagery zoom must be an integer from 14 to 17')
    route = value.get('route')
    radius = 0
    if route is not None:
        if not isinstance(route, list) or not 2 <= len(route) <= 128:
            raise ValueError('A route requires 2 to 128 longitude/latitude points')
        for p in route: validate_point(p)
        buffer = value.get('buffer_m', 1000)
        if not isinstance(buffer, (int, float)) or not math.isfinite(buffer) or not 100 <= buffer <= 10000:
            raise ValueError('Route buffer must be 100 to 10000 metres')
        points = [tile_xy(*p, zoom) for p in route]
        latitude = max(abs(p[1]) for p in route)
        radius = buffer/(40075016.68557849*math.cos(math.radians(latitude))/2**zoom)
        limits = [min(p[0] for p in points)-radius, min(p[1] for p in points)-radius,
                  max(p[0] for p in points)+radius, max(p[1] for p in points)+radius]
    else:
        bounds = value.get('bounds')
        if not isinstance(bounds, list) or len(bounds) != 4:
            raise ValueError('An area requires [west, south, east, north]')
        w,s,e,n = bounds
        validate_point([w,s]); validate_point([e,n])
        if w >= e or s >= n:
            raise ValueError('Empty or dateline-crossing areas are unsupported')
        x0,y0=tile_xy(w,n,zoom); x1,y1=tile_xy(e,s,zoom)
        limits=[x0,y0,x1,y1]
    x0,y0,x1,y1=map(math.floor,limits)
    if (x1-x0+1)*(y1-y0+1)>100000:
        raise ValueError('Selection is too large; use smaller regional packages')
    tiles=[]
    for y in range(y0,y1+1):
        for x in range(x0,x1+1):
            if not 0 <= x < 2**zoom or not 0 <= y < 2**zoom:
                raise ValueError('Selection crosses supported map bounds')
            if route and min(segment_distance((x+.5,y+.5),a,b) for a,b in zip(points,points[1:]))>radius+math.sqrt(.5):
                continue
            tiles.append([zoom,x,y])
            if len(tiles)>400:
                raise ValueError('Selection exceeds 400 imagery tiles. Reduce the area or zoom.')
    all_bounds=[tile_bounds(*t) for t in tiles]
    bounds=[min(b[0] for b in all_bounds),min(b[1] for b in all_bounds),max(b[2] for b in all_bounds),max(b[3] for b in all_bounds)]
    return dict(provider='Microsoft Planetary Computer / USDA NAIP',bounds=bounds,requested=value,
                imagery_tiles=tiles,estimated_max_bytes=len(tiles)*512*512*4,
                imagery_zoom=zoom,terrain_zoom=14,offline_use='Public domain; retain attribution')
