"""Quality-limited ground-image composition; camera geometry comes from Navigate."""
import json
import math
from pathlib import Path

import cv2
import numpy as np

from navigation_reference import rotation


def suitability(frame, camera):
    corners = np.asarray(frame.get("corners"), dtype=np.float64)
    if corners.shape != (4, 2) or not np.isfinite(corners).all():
        return None
    local = corners - corners.mean(axis=0)
    local *= [math.cos(math.radians(corners[:, 1].mean())), 1.0]
    if not cv2.isContourConvex(local.astype(np.float32)):
        return None
    navigation = frame.get("navigation")
    if navigation is None:
        return 1.0
    support = navigation.get("support")
    if support is not None and not support["ground_projection"]:
        return None
    pose, height = navigation.get("pose"), navigation.get("camera_agl")
    if pose is None or height is None or height[0] <= 0 or height[1] > height[0] / 2:
        return None
    if camera is None:
        raise ValueError("Navigate mosaic requires its camera configuration")
    width, height_px = camera["size"]
    fx, fy = camera["focal"]
    cx, cy = camera["principal"]
    pixels = np.array([[-.5, -.5], [width-.5, -.5], [width-.5, height_px-.5], [-.5, height_px-.5]])
    rays = np.column_stack(((pixels[:, 0]-cx)/fx, (cy-pixels[:, 1])/fy, -np.ones(4)))
    rays = rays @ rotation(pose["rotation"]).T
    incidence = -rays[:, 2] / np.linalg.norm(rays, axis=1)
    # Grazing rays amplify pose/depth error and cannot support this ground preview.
    if incidence.min() < math.cos(math.radians(70)):
        return None
    return float(incidence.min()**2 / (1 + (height[1]/height[0])**2))


def orthorectify(picture, pose, camera, ground, valid):
    """Image pixels seen at bare-earth ground points through a pinhole camera."""
    h, w = picture.shape[:2]
    fx, fy = camera["focal"]
    cx, cy = camera["principal"]
    eye = (ground - np.asarray(pose["position"])) @ rotation(pose["rotation"])
    depth = -eye[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = (cx + fx*eye[..., 0]/depth).astype(np.float32)
        v = (cy - fy*eye[..., 1]/depth).astype(np.float32)
    inside = valid & (depth > 1) & (u >= -.5) & (u <= w-.5) & (v >= -.5) & (v <= h-.5)
    edge = np.minimum.reduce([(u+.5)/w, (w-.5-u)/w, (v+.5)/h, (h-.5-v)/h])
    warped = cv2.remap(cv2.cvtColor(picture, cv2.COLOR_BGR2BGRA), u, v, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REPLICATE)
    return warped, np.where(inside, np.minimum(1, edge*8), 0).astype(np.float32)


def mosaic(output, frames, surface=None, *, camera=None, image_name="mosaic.png", max_size=2048):
    """Compose supported frames. With `surface`, each pixel is the camera ray to
    the bare-earth model at the frame's Navigate pose; otherwise four corners."""
    configuration = output / "configuration.json"
    if camera is None and configuration.exists():
        camera = json.loads(configuration.read_text())["camera"]
    accepted = [(frame, score) for frame in frames
                if (score := suitability(frame, camera)) is not None]
    if not accepted:
        raise ValueError("no frame has sufficient support for a ground mosaic")
    # Navigate evaluates position, tilt and recent anchor support for each projected frame.
    mapped = [(frame, score) for frame, score in accepted
              if frame.get("navigation") is None or (frame["navigation"].get("support") or {}).get("ground_projection", frame["navigation"].get("map_observation_accepted", False))]
    if not mapped:
        raise ValueError("no admitted map observations support a geographic mosaic")
    chosen = mapped if camera is not None else mapped[::5]
    points = np.asarray([frame["corners"] for frame, _ in chosen]).reshape(-1, 2)
    west, south = points.min(axis=0)
    east, north = points.max(axis=0)
    aspect = (east-west) * math.cos(math.radians((north+south)/2)) / (north-south)
    width, height = (max_size, max(1, round(max_size/aspect))) if aspect > 1 else (max(1, round(max_size*aspect)), max_size)
    canvas = np.zeros((height, width, 4), np.uint8)
    best = np.zeros((height, width), np.float32)
    if surface is not None and camera is not None:
        columns = west + (np.arange(width)+.5) / width * (east-west)
        rows = north - (np.arange(height)+.5) / height * (north-south)
        ground, valid = surface.ground(*np.meshgrid(rows, columns, indexing="ij"))
    for frame, score in chosen:
        if surface is not None and camera is not None and (frame.get("navigation") or {}).get("pose"):
            corners = np.asarray(frame["corners"])
            x = (corners[:, 0]-west)/(east-west)*width - .5
            y = (north-corners[:, 1])/(north-south)*height - .5
            x0, x1 = max(0, int(x.min())-8), min(width, int(np.ceil(x.max()))+9)
            y0, y1 = max(0, int(y.min())-8), min(height, int(np.ceil(y.max()))+9)
            picture = cv2.imread(str(output / frame["image"]))
            if picture is None:
                raise ValueError(f"unreadable mosaic source {frame['image']}")
            warped, support = orthorectify(picture, frame["navigation"]["pose"], camera,
                                           ground[y0:y1, x0:x1], valid[y0:y1, x0:x1])
            weight = score * support
            region, kept = canvas[y0:y1, x0:x1], best[y0:y1, x0:x1]
            mask = (support > 0) & (weight > kept)
            region[mask] = warped[mask]
            kept[mask] = weight[mask]
            continue
        picture = cv2.imread(str(output / frame["image"]))
        if picture is None:
            raise ValueError(f"unreadable mosaic source {frame['image']}")
        h, w = picture.shape[:2]
        src = np.float32([[-.5, -.5], [w-.5, -.5], [w-.5, h-.5], [-.5, h-.5]])
        corners = np.asarray(frame["corners"])
        dst = np.float32(np.column_stack(((corners[:, 0]-west)/(east-west)*width-.5,
                                         (north-corners[:, 1])/(north-south)*height-.5)))
        transform = cv2.getPerspectiveTransform(src, dst)
        yy, xx = np.mgrid[:h, :w]
        edge = np.minimum.reduce([(xx+.5)/w, (w-.5-xx)/w, (yy+.5)/h, (h-.5-yy)/h])
        confidence = (score * np.minimum(1, edge*8)).astype(np.float32)
        weight = cv2.warpPerspective(confidence, transform, (width, height))
        warped = cv2.warpPerspective(cv2.cvtColor(picture, cv2.COLOR_BGR2BGRA), transform, (width, height),
                                     borderMode=cv2.BORDER_REPLICATE)
        support = cv2.warpPerspective(np.ones((h, w), np.uint8), transform, (width, height),
                                      flags=cv2.INTER_NEAREST)
        mask = (support > 0) & (weight > best)
        canvas[mask] = warped[mask]
        best[mask] = weight[mask]
    if not cv2.imwrite(str(output / image_name), canvas):
        raise ValueError(f"cannot write mosaic {output / image_name}")
    return {"image": image_name, "corners": [[west, north], [east, north], [east, south], [west, south]],
            "coverage_fraction": float((best > 0).mean()), "source_frames": len(chosen),
            "rejected_frames": len(frames)-len(mapped),
            "composition": ("Navigate-supported projections onto the bare-earth model" if surface is not None
                            else "Navigate-supported four-corner projections")
                           + "; grazing or uncertain geometry excluded. Roofs and trees keep parallax."}
