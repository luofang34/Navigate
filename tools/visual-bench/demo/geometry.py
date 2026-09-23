"""Filter correspondences with rendered 3D depth before Rust pose verification."""

import hashlib
import time
from pathlib import Path
import cv2
import numpy as np
from .contracts import ImageMatcher, PixelPairs


def depth_inliers(pairs: PixelPairs, depth, camera):
    a, b = pairs.first.astype(np.float64), pairs.second.astype(np.float64)
    height, width = depth.shape
    inside = (
        (a[:, 0] >= 0)
        & (a[:, 0] <= width - 1)
        & (a[:, 1] >= 0)
        & (a[:, 1] <= height - 1)
    )
    inside &= (
        (b[:, 0] >= 0)
        & (b[:, 0] <= camera["width"] - 1)
        & (b[:, 1] >= 0)
        & (b[:, 1] <= camera["height"] - 1)
    )
    a, b = a[inside], b[inside]
    xy = np.rint(a).astype(int)
    d = depth[xy[:, 1], xy[:, 0]]
    usable = np.isfinite(d) & (d > 0)
    a, b, d = a[usable], b[usable], d[usable]
    valid_count = len(a)
    if valid_count < 6:
        return a[:0], b[:0], valid_count
    c = camera
    obj = np.column_stack(
        ((a[:, 0] - c["cx"]) / c["fx"] * d, (a[:, 1] - c["cy"]) / c["fy"] * d, d)
    )
    k = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]], np.float64)
    ok, _, _, indices = cv2.solvePnPRansac(
        obj,
        b,
        k,
        None,
        iterationsCount=2000,
        reprojectionError=3,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    keep = indices[:, 0] if ok and indices is not None else np.array([], dtype=int)
    return a[keep], b[keep], valid_count


def match_reference(
    matcher: ImageMatcher, reference_path, query_path, depth_path, camera
):
    reference = cv2.imread(str(reference_path), cv2.IMREAD_GRAYSCALE)
    query = cv2.imread(str(query_path), cv2.IMREAD_GRAYSCALE)
    shape = (camera["height"], camera["width"])
    if (
        reference is None
        or query is None
        or reference.shape != shape
        or query.shape != shape
    ):
        raise ValueError("Reference and query dimensions must match calibration")
    raw_depth = Path(depth_path).read_bytes()
    depth = np.frombuffer(raw_depth, dtype="<f4").reshape(shape)
    started = time.monotonic()
    pairs = matcher.match_images(reference, query)
    a, b, valid = depth_inliers(pairs, depth, camera)
    payload = dict(
        backend_identity=pairs.backend_identity,
        reference_image_sha256=hashlib.sha256(reference.tobytes()).hexdigest(),
        query_image_sha256=hashlib.sha256(query.tobytes()).hexdigest(),
        reference_depth_sha256=hashlib.sha256(raw_depth).hexdigest(),
        matches=[dict(reference=p.tolist(), query=q.tolist()) for p, q in zip(a, b)],
    )
    return payload, dict(
        tentative_matches=len(pairs.first),
        valid_depth_matches=valid,
        depth_pnp_inliers=len(a),
        pair_seconds=time.monotonic() - started,
        geometric_filter="rendered_depth_pnp",
        backend=pairs.backend_identity,
    )
