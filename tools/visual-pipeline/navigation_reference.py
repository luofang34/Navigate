"""Host reference renderer for a pinned Navigate imagery/DEM package."""
import hashlib
import json
import math
from pathlib import Path

import cv2
import numpy as np

RADIUS = 6371008.8


def mercator(lat, lon):
    return np.array([(lon + 180) / 360, (1 - np.arcsinh(np.tan(np.radians(lat))) / math.pi) / 2])


def rotation(quaternion):
    x, y, z, w = quaternion
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


class ReferenceMap:
    def __init__(self, manifest, centre, radius_m=700):
        self.manifest = json.loads(Path(manifest).read_text())
        self.anchor = self.manifest["anchor_lat_lon"]
        self.origin = mercator(*self.anchor)
        self.scale = 2 * math.pi * RADIUS * math.cos(math.radians(self.anchor[0]))
        point = mercator(*centre)
        root = Path(manifest).parent.parent
        self.chunks = {}
        self.grids = {}
        for kind, side in [("imagery", 512), ("elevation", 256)]:
            tiles = [t for t in self.manifest["tiles"] if t.get(kind)]
            zoom = max(t["xyz"][0] for t in tiles)
            span = radius_m / self.scale * 2**zoom
            cx, cy = point * 2**zoom
            tiles = [t for t in tiles if t["xyz"][0] == zoom and
                     abs(t["xyz"][1] + .5 - cx) < span + 1 and
                     abs(t["xyz"][2] + .5 - cy) < span + 1]
            west, north = [min(t["xyz"][i] for t in tiles) for i in [1, 2]]
            east, south = [max(t["xyz"][i] for t in tiles) for i in [1, 2]]
            grid = np.zeros(((south-north+1)*side, (east-west+1)*side, 4), np.uint8)
            for tile in tiles:
                asset = tile[kind]
                digest = asset["chunk"]
                if digest not in self.chunks:
                    chunk = (root / "chunks" / (digest + ".bin")).read_bytes()
                    if hashlib.sha256(chunk).hexdigest() != digest:
                        raise ValueError("map chunk digest mismatch")
                    self.chunks[digest] = chunk
                data = self.chunks[digest][asset["offset"]:asset["offset"]+asset["length"]]
                image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
                if image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
                _, x, y = tile["xyz"]
                grid[(y-north)*side:(y-north+1)*side, (x-west)*side:(x-west+1)*side] = image
            if kind == "elevation":
                valid = grid[:, :, 3].astype(np.float32) / 255
                grid = grid[:, :, 2].astype(np.float32)*256 + grid[:, :, 1] + grid[:, :, 0]/256 - 32768
                self.grids["terrain_valid"] = (valid, zoom, side, west, north)
            self.grids[kind] = (grid, zoom, side, west, north)
        self.chunks.clear()

    def local(self, lat, lon, altitude):
        xy = (mercator(lat, lon) - self.origin) * self.scale
        return [float(xy[0]), float(-xy[1]), float(altitude)]

    def sample(self, kind, east, north):
        grid, zoom, side, x0, y0 = self.grids[kind]
        x = ((east / self.scale + self.origin[0]) * 2**zoom - x0) * side - .5
        y = ((-north / self.scale + self.origin[1]) * 2**zoom - y0) * side - .5
        return cv2.remap(grid, np.asarray(x, np.float32), np.asarray(y, np.float32), cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def ground(self, lat, lon):
        """Local east, north and bare-earth height of geodetic grids, with validity."""
        point = mercator(np.asarray(lat, np.float64), np.asarray(lon, np.float64))
        east = (point[0] - self.origin[0]) * self.scale
        north = -(point[1] - self.origin[1]) * self.scale
        east32, north32 = east.astype(np.float32), north.astype(np.float32)
        height = self.sample("elevation", east32, north32)
        valid = self.sample("terrain_valid", east32, north32) > .99
        return np.stack((east, north, height), axis=-1), valid

    def terrain(self, position):
        return float(self.sample("elevation", np.array([[position[0]]]), np.array([[position[1]]]))[0, 0])

    def intersect(self, camera, pose, pixels):
        fx, fy = camera["focal"]
        cx, cy = camera["principal"]
        pixels = np.asarray(pixels)
        rays = np.stack(((pixels[..., 0]-cx)/fx, (cy-pixels[..., 1])/fy, -np.ones(pixels.shape[:-1])), axis=-1)
        rays = rays @ rotation(pose["rotation"]).T
        eye = np.array(pose["position"])
        depth = np.full(pixels.shape[:-1], max(1, eye[2] - self.terrain(eye)))
        for _ in range(8):
            point = eye + depth[..., None] * rays
            height = self.sample("elevation", point[..., 0], point[..., 1])
            depth = (height-eye[2]) / np.minimum(rays[..., 2], -.001)
        point = eye + depth[..., None] * rays
        mask = (rays[..., 2] < -.01) & (depth > 0) & (depth < 5000)
        mask &= self.sample("terrain_valid", point[..., 0], point[..., 1]) > .99
        return point, depth, mask

    def render(self, directory, name, camera, pose):
        width, height = camera["size"]
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        point, depth, mask = self.intersect(camera, pose, np.stack((xx, yy), axis=-1))
        color = self.sample("imagery", point[..., 0], point[..., 1])
        mask &= color[:, :, 3] > 250
        color[~mask] = 0
        depth[~mask] = 0
        gray = cv2.cvtColor(color, cv2.COLOR_BGRA2GRAY)
        cv2.imwrite(str(directory / (name + ".png")), gray)
        depth.astype("<f4").tofile(directory / (name + ".depth"))
        return {"image":name+".png", "depth":name+".depth", "pose":pose}, gray

    def footprint(self, camera, pose):
        width, height = camera["size"]
        corners = [[[-.5, -.5], [width-.5, -.5], [width-.5, height-.5], [-.5, height-.5]]]
        points, _, valid = self.intersect(camera, pose, corners)
        if not valid.all():
            return None
        xy = points[0, :, :2] / self.scale
        mx = self.origin[0] + xy[:, 0]
        my = self.origin[1] - xy[:, 1]
        lat = np.degrees(np.arctan(np.sinh(math.pi*(1-2*my))))
        lon = mx*360-180
        return np.column_stack((lon, lat)).tolist()
