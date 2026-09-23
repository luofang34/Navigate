"""Search the location prior before proposing camera poses."""

import math, time
import cv2, numpy as np
from .contracts import ImageMatcher


class RegionalSearch:
    def __init__(self, map_data, matcher: ImageMatcher, camera, prior):
        self.map = map_data
        self.matcher = matcher
        self.camera = camera
        self.prior = prior
        self.crops = []
        if "geodetic_lat_lon_alt_m" in prior:
            self.latitude, self.longitude, altitude = prior["geodetic_lat_lon_alt_m"]
        else:
            self.latitude, self.longitude = self.map.coordinate(prior["position_enu_m"])
            altitude = prior["position_enu_m"][2]
        agl = altitude - self.map.elevation(self.latitude, self.longitude)
        if not 10 <= agl <= 2000:
            raise ValueError(
                "Prior must give camera altitude above local terrain between 10 and 2000 metres"
            )
        mpp = 2 * math.pi * self.map.scale / (2**self.map.zoom * 512)
        self.crop_size = max(
            256, round(1.5 * agl * camera["width"] / camera["fx"] / mpp)
        )
        self.radius = prior["position_radius_m"]
        center = self.map.lat_lon_pixel(self.latitude, self.longitude)
        height, width = self.map.mosaic.shape
        size = self.crop_size
        if size > min(width, height):
            raise ValueError(
                "Map imagery extent is too small for this camera footprint"
            )
        positions = lambda n: sorted(
            set([*range(0, max(1, n - size + 1), max(1, size // 2)), n - size])
        )
        for y in positions(height):
            for x in positions(width):
                if (
                    np.linalg.norm(np.array([x + size / 2, y + size / 2]) - center)
                    * mpp
                    > self.radius + size * mpp
                ):
                    continue
                if np.mean(self.map.alpha[y : y + size, x : x + size] == 255) < 0.35:
                    continue
                image = cv2.resize(
                    self.map.mosaic[y : y + size, x : x + size],
                    (640, 640),
                    interpolation=cv2.INTER_AREA,
                )
                self.crops.append((x, y, image))
        if not self.crops:
            raise ValueError("No usable map crops overlap the location prior")

    def candidates(self, image):
        started = time.monotonic()
        height, width = image.shape
        candidates = []
        for angle in range(0, 360, 45):
            transform = cv2.getRotationMatrix2D(
                ((width - 1) / 2, (height - 1) / 2), angle, 1
            )
            corners = np.float32(
                [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
            )
            rotated = cv2.transform(corners[None], transform)[0]
            low = rotated.min(0)
            high = rotated.max(0)
            transform[:, 2] -= low
            rotated = cv2.warpAffine(
                image, transform, tuple(np.ceil(high - low + 1).astype(int))
            )
            query = rotated
            for x, y, reference in self.crops:
                pairs = self.matcher.match_images(query, reference)
                a, b = pairs.first, pairs.second
                if len(a) < 12:
                    continue
                H, mask = cv2.findHomography(
                    a, b, cv2.USAC_MAGSAC, 3, maxIters=5000, confidence=0.999
                )
                if H is None:
                    continue
                keep = mask[:, 0] > 0
                if keep.sum() < 10:
                    continue
                a = cv2.transform(a[None], cv2.invertAffineTransform(transform))[0]
                coverage = cv2.contourArea(cv2.convexHull(a[keep])) / (width * height)
                mp = (
                    (b[keep].astype(np.float64) + 0.5) * self.crop_size / 640
                    - 0.5
                    + np.array([x, y], np.float64)
                )
                try:
                    pose = self.propose(a[keep].astype(np.float64), mp)
                except (ValueError, cv2.error):
                    continue
                if pose is not None:
                    candidates.append(
                        dict(
                            candidate=pose,
                            inliers=int(keep.sum()),
                            coverage=coverage,
                            score=float(keep.sum()) * min(1, coverage / 0.3),
                            angle=angle,
                            crop=[x, y],
                        )
                    )
        candidates.sort(key=lambda r: r["score"], reverse=True)
        return candidates, dict(
            algorithm="regional_homography_terrain_pnp",
            stage="retrieval_only",
            search_seconds=time.monotonic() - started,
            map_crops=len(self.crops),
            pose_candidates=len(candidates),
        )

    def propose(self, query, map_points):
        obj = self.map.world_points(map_points)
        c = self.camera
        K = np.array(
            [[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]], np.float64
        )
        ok, rv, tv, indices = cv2.solvePnPRansac(
            obj,
            query,
            K,
            None,
            iterationsCount=2000,
            reprojectionError=4,
            confidence=0.999,
            flags=cv2.SOLVEPNP_EPNP,
        )
        if not ok or indices is None or len(indices) < 10:
            return None
        keep = indices[:, 0]
        rv, tv = cv2.solvePnPRefineLM(obj[keep], query[keep], K, None, rv, tv)
        rotation = cv2.Rodrigues(rv)[0]
        position = (-rotation.T @ tv).ravel()
        lat, lon = self.map.coordinate(position)
        agl = position[2] - self.map.elevation(lat, lon)
        if not np.isfinite(position).all() or not 10 < agl < 2000:
            return None
        prior_position = np.array(
            self.prior.get(
                "position_enu_m",
                [
                    self.map.scale * math.radians(self.longitude - self.map.lon0),
                    self.map.scale
                    * (
                        math.asinh(math.tan(math.radians(self.latitude)))
                        - self.map.north0
                    ),
                    self.prior.get("geodetic_lat_lon_alt_m", [0, 0, 0])[2],
                ],
            )
        )
        if np.linalg.norm(position - prior_position) > self.radius:
            return None
        eye = rotation.T @ np.diag([1, -1, -1])
        axis = cv2.Rodrigues(eye)[0].ravel()
        angle = np.linalg.norm(axis)
        quaternion = (
            np.r_[axis / angle * math.sin(angle / 2), math.cos(angle / 2)]
            if angle > 1e-10
            else np.array([0, 0, 0, 1])
        )
        return dict(
            position_enu_m=position.tolist(),
            eye_to_enu_xyzw=quaternion.tolist(),
        )
