"""Read verified offline imagery and terrain for regional candidate search."""

from pathlib import Path
import hashlib, json, math
import cv2
import numpy as np


class MapData:
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.manifest = json.loads((self.path / "map.json").read_text())
        if self.manifest["schema_version"] not in (1, 2):
            raise ValueError("Unsupported map package schema")
        self.tiles = self.manifest["tiles"]
        self.dem = {}
        for tile in self.tiles:
            for key in ("imagery", "elevation"):
                if not tile.get(key):
                    continue
                artifact = tile[key]
                file = (self.path / artifact["path"]).resolve()
                if not file.is_relative_to(self.path):
                    raise ValueError("Map artifact leaves package")
                if hashlib.sha256(file.read_bytes()).hexdigest() != artifact["sha256"]:
                    raise ValueError("Map artifact digest mismatch: " + str(file))
        self.zoom = max(t["xyz"][0] for t in self.tiles if t.get("imagery"))
        top = [t for t in self.tiles if t.get("imagery") and t["xyz"][0] == self.zoom]
        self.x0 = min(t["xyz"][1] for t in top)
        self.y0 = min(t["xyz"][2] for t in top)
        width = (max(t["xyz"][1] for t in top) - self.x0 + 1) * 512
        height = (max(t["xyz"][2] for t in top) - self.y0 + 1) * 512
        if width * height > 100_000_000:
            raise ValueError(
                "Use a regional package smaller than 100 million imagery pixels"
            )
        self.mosaic = np.zeros((height, width), np.uint8)
        self.alpha = np.zeros_like(self.mosaic)
        for tile in top:
            image = cv2.imread(
                str(self.path / tile["imagery"]["path"]), cv2.IMREAD_UNCHANGED
            )
            if image is None or image.shape[:2] != (512, 512):
                raise ValueError("Invalid imagery tile dimensions")
            x = (tile["xyz"][1] - self.x0) * 512
            y = (tile["xyz"][2] - self.y0) * 512
            self.mosaic[y : y + 512, x : x + 512] = cv2.cvtColor(
                image,
                cv2.COLOR_BGRA2GRAY if image.shape[2] == 4 else cv2.COLOR_BGR2GRAY,
            )
            self.alpha[y : y + 512, x : x + 512] = (
                image[:, :, 3] if image.shape[2] == 4 else 255
            )
        self.lat0, self.lon0 = self.manifest["anchor_lat_lon"]
        self.scale = 6371008.8 * math.cos(math.radians(self.lat0))
        self.north0 = math.asinh(math.tan(math.radians(self.lat0)))

    def pixel_lat_lon(self, points):
        p = np.asarray(points, dtype=np.float64)
        x = self.x0 + p[..., 0] / 512
        y = self.y0 + p[..., 1] / 512
        lon = x / 2**self.zoom * 360 - 180
        lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / 2**self.zoom))))
        return lat, lon

    def lat_lon_pixel(self, lat, lon):
        x = ((lon + 180) / 360 * 2**self.zoom - self.x0) * 512
        y = (
            (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * 2**self.zoom
            - self.y0
        ) * 512
        return np.array([x, y], np.float64)

    def elevation(self, lat, lon):
        for tile in sorted(self.tiles, key=lambda t: t["xyz"][0], reverse=True):
            if not tile.get("elevation"):
                continue
            z, x, y = tile["xyz"]
            xx = (lon + 180) / 360 * 2**z
            yy = (1 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2 * 2**z
            if int(xx) != x or int(yy) != y:
                continue
            key = (z, x, y)
            if key not in self.dem:
                image = cv2.imread(str(self.path / tile["elevation"]["path"])).astype(
                    np.float32
                )
                self.dem[key] = (
                    image[:, :, 2] * 256 + image[:, :, 1] + image[:, :, 0] / 256 - 32768
                )
            return float(
                cv2.getRectSubPix(
                    self.dem[key], (1, 1), ((xx - x) * 256 - 0.5, (yy - y) * 256 - 0.5)
                )[0, 0]
            )
        raise ValueError(f"No DEM coverage at {lat:.6f}, {lon:.6f}")

    def world_points(self, pixels):
        lat, lon = self.pixel_lat_lon(pixels)
        return np.column_stack(
            [
                self.scale * np.radians(lon - self.lon0),
                self.scale * (np.arcsinh(np.tan(np.radians(lat))) - self.north0),
                [self.elevation(a, b) for a, b in zip(lat, lon)],
            ]
        )

    def coordinate(self, position):
        return math.degrees(
            math.atan(math.sinh(self.north0 + position[1] / self.scale))
        ), self.lon0 + math.degrees(position[0] / self.scale)
