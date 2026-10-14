"""Give every video frame a camera pose by tracking rotation between map-matched anchors.

The camera moves a few metres per frame while it views terrain kilometres away,
so consecutive frames differ by a rotation; parallax is far below a pixel.
Rotation chains run forward from one anchor and backward from the next, and
are blended so that both anchors are met exactly; drift does not accumulate
past an anchor. Camera position is interpolated linearly between anchors.
"""
import cv2
import numpy as np

TRACK_POLICY = {'features': 3000, 'ratio': .75, 'ransac_deg': .15, 'iterations': 400,
                'minimum_inliers': 40, 'mask_erode_pixels': 7}


def rays(points, K):
    """Unit view directions in the camera frame: x right, y up, the camera looks along -z."""
    d = np.stack([(points[:, 0] - K[0, 2]) / K[0, 0], -(points[:, 1] - K[1, 2]) / K[1, 1],
                  -np.ones(len(points))], axis=1)
    return d / np.linalg.norm(d, axis=1, keepdims=True)


def kabsch(a, b):
    """Rotation M that minimizes |b - M a| over unit ray pairs."""
    u, _, vt = np.linalg.svd(b.T @ a)
    fix = np.diag([1., 1., np.sign(np.linalg.det(u @ vt))])
    return u @ fix @ vt


def relative_rotation(a, b, rng, policy=TRACK_POLICY):
    """Robust M with b ≈ M a; returns (M, inlier count) or (None, 0)."""
    if len(a) < policy['minimum_inliers']:
        return None, 0
    threshold = np.cos(np.radians(policy['ransac_deg']))
    best = None
    for _ in range(policy['iterations']):
        pick = rng.choice(len(a), 2, replace=False)
        M = kabsch(a[pick], b[pick])
        inliers = np.einsum('ij,ij->i', (M @ a.T).T, b) > threshold
        if best is None or inliers.sum() > best.sum():
            best = inliers
    if best.sum() < policy['minimum_inliers']:
        return None, int(best.sum())
    return kabsch(a[best], b[best]), int(best.sum())


class FeatureTracker:
    def __init__(self, K, policy=TRACK_POLICY):
        self.K, self.policy = K, policy
        self.sift = cv2.SIFT_create(nfeatures=policy['features'])
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.rng = np.random.default_rng(0)

    def features(self, frame, valid):
        size = 2 * self.policy['mask_erode_pixels'] + 1
        valid = cv2.erode(valid.astype(np.uint8) * 255, np.ones((size, size), np.uint8))
        return self.sift.detectAndCompute(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), valid)

    def rotation(self, first, second):
        """M with rays(second) ≈ M rays(first), from two (keypoints, descriptors) sets."""
        (k1, d1), (k2, d2) = first, second
        if d1 is None or d2 is None or len(k1) < 2 or len(k2) < 2:
            return None, 0
        good = [m for m, n in (p for p in self.matcher.knnMatch(d1, d2, k=2) if len(p) == 2)
                if m.distance < self.policy['ratio'] * n.distance]
        if not good:
            return None, 0
        a = rays(np.float64([k1[m.queryIdx].pt for m in good]), self.K)
        b = rays(np.float64([k2[m.trainIdx].pt for m in good]), self.K)
        return relative_rotation(a, b, self.rng, self.policy)


def slerp(a, b, t):
    """Rotation matrix on the shortest arc from a (t=0) to b (t=1)."""
    delta = cv2.Rodrigues(a.T @ b)[0].ravel()
    return a @ cv2.Rodrigues(delta * t)[0]


def dense_rotations(anchors, steps, count):
    """Per-frame camera-to-local rotations.

    `anchors` maps frame index to its matched rotation (columns are camera axes in
    the local frame). `steps[i]` is M with rays(i+1) ≈ M rays(i), or None where
    tracking failed. Frames joined to no anchor by tracked steps get None.
    """
    def chain(start, rotation, direction):
        out, frame = {start: rotation}, start
        while 0 <= frame + direction < count:
            step = steps[frame] if direction > 0 else steps[frame - 1]
            if step is None:
                break
            # Shared view directions satisfy R_i d_i = R_j d_j, so R_j = R_i M^T for d_j = M d_i.
            rotation = rotation @ (step.T if direction > 0 else step)
            frame += direction
            out[frame] = rotation
        return out

    forward = {a: chain(a, R, 1) for a, R in anchors.items()}
    backward = {a: chain(a, R, -1) for a, R in anchors.items()}
    ordered = sorted(anchors)
    result = [None] * count
    for frame in range(count):
        before = max((a for a in ordered if a <= frame), default=None)
        after = min((a for a in ordered if a >= frame), default=None)
        f = forward[before].get(frame) if before is not None else None
        b = backward[after].get(frame) if after is not None else None
        if f is not None and b is not None and after != before:
            result[frame] = slerp(f, b, (frame - before) / (after - before))
        else:
            result[frame] = f if f is not None else b
    return result


def interpolate_eyes(anchors, count):
    """Linear camera positions between anchors; held constant outside the anchored span."""
    ordered = sorted(anchors)
    frames = np.arange(count)
    return np.stack([np.interp(frames, ordered, [anchors[a][axis] for a in ordered]) for axis in range(3)], axis=1)
