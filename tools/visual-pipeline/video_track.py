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


# OpenCV cameras look along +z with y down; these cameras look along -z with y up.
FLIP = np.diag([1., -1., -1.])


def refit_fixed_focal(state, focal_ratio):
    """Re-solve one anchor pose with the clip's shared focal length.

    A video lens does not zoom between frames, so per-anchor focal estimates
    make the overlay breathe. Returns (rotation, eye, rms_px) at the anchor's
    matching resolution, or None when the anchor kept no inlier evidence.
    """
    if 'points' not in state or int(np.sum(state['inliers'])) < 6:
        return None
    width, height = (int(v) for v in state['size'])
    focal = focal_ratio * width
    K = np.array([[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1.]])
    keep = np.asarray(state['inliers'], bool)
    points = np.asarray(state['points'], np.float64)[keep]
    pixels = np.asarray(state['pixels'], np.float64)[keep]
    rotation_cv = FLIP @ state['rotation'].T
    rvec = cv2.Rodrigues(rotation_cv)[0]
    tvec = -rotation_cv @ state['eye']
    ok, rvec, tvec = cv2.solvePnP(points, pixels, K, None, rvec, tvec, useExtrinsicGuess=True,
                                  flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    rotation_cv = cv2.Rodrigues(rvec)[0]
    projected = cv2.projectPoints(points, rvec, tvec, K, None)[0].reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((projected - pixels) ** 2, axis=1))))
    return rotation_cv.T @ FLIP, (-rotation_cv.T @ tvec.ravel()), rms


def rotation_for_eye(state, eye, focal_ratio):
    """Best rotation for a fixed eye and focal from the anchor's inlier correspondences; returns (R, rms_px)."""
    width, height = (int(v) for v in state['size'])
    focal = focal_ratio * width
    keep = np.asarray(state['inliers'], bool)
    points, pixels = np.asarray(state['points'], float)[keep], np.asarray(state['pixels'], float)[keep]
    K = np.array([[focal, 0, (width - 1) / 2], [0, focal, (height - 1) / 2], [0, 0, 1.]])
    toward = points - eye
    R = kabsch(rays(pixels, K), toward / np.linalg.norm(toward, axis=1, keepdims=True))
    camera = toward @ R
    projected = np.column_stack([focal * camera[:, 0] / -camera[:, 2] + K[0, 2],
                                 -focal * camera[:, 1] / -camera[:, 2] + K[1, 2]])
    return R, float(np.sqrt(np.mean(np.sum((projected - pixels) ** 2, axis=1))))


def shared_focal(states, ratios=np.linspace(.6, 1.1, 51)):
    """Focal length (as a fraction of image width) that best fits every anchor's map correspondences.

    Anchor solves can stop before refining focal, so their individual focal
    values are not evidence; the reprojection error of all anchors together is.
    """
    def total(ratio):
        fits = [refit_fixed_focal(state, ratio) for state in states.values()]
        return sum(f[2] ** 2 for f in fits if f is not None)
    coarse = min(ratios, key=total)
    step = ratios[1] - ratios[0]
    return float(min(np.linspace(coarse - step, coarse + step, 21), key=total))


def consistent_anchors(states, fps):
    """One focal, one straight constant-velocity path, and per-anchor rotations that agree with both.

    Near-level views constrain position along the view direction poorly, and
    each anchor fits its own focal. Free per-anchor solutions therefore jump
    by tens of metres and several percent of focal between anchors, which
    the overlay shows as stretching. A few seconds of flight is close to
    straight and steady, so a line through the anchor positions is a better
    estimate than any single anchor.
    """
    ratio = shared_focal(states)
    frames = sorted(states)
    eyes = []
    for frame in frames:
        refit = refit_fixed_focal(states[frame], ratio)
        eyes.append(refit[1] if refit else states[frame]['eye'])
    times = np.array(frames, float) / fps
    design = np.column_stack([np.ones_like(times), times])
    path, *_ = np.linalg.lstsq(design, np.array(eyes), rcond=None)
    anchors = {}
    for frame, row in zip(frames, design):
        eye = row @ path
        rotation, rms = rotation_for_eye(states[frame], eye, ratio)
        anchors[frame] = (rotation, eye, rms)
    return ratio, path, anchors


REGISTER_POLICY = {'width': 480, 'highpass_sigma': 6, 'minimum_response': .05, 'window': 9,
                   'outlier_px': 8., 'minimum_valid_fraction': .05}


def residual_shift(video, render, valid, policy=REGISTER_POLICY):
    """Translation of the rendered terrain against the video frame, in pixels, and the correlation peak."""
    if valid.mean() < policy['minimum_valid_fraction']:
        return None
    weight = valid.astype(np.float32)
    def detail(image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return (gray - cv2.GaussianBlur(gray, (0, 0), policy['highpass_sigma'])) * weight
    window = cv2.createHanningWindow(valid.shape[::-1], cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(detail(video), detail(render), window)
    return (dx, dy) if response >= policy['minimum_response'] else None


def smooth_shifts(shifts, policy=REGISTER_POLICY):
    """Robust running median; registration failures and isolated outliers take their neighbours' value."""
    count = len(shifts)
    known = np.array([s if s is not None else (np.nan, np.nan) for s in shifts], float)
    half = policy['window'] // 2
    centre = np.array([np.nanmedian(known[max(0, i - half):i + half + 1], axis=0) if
                       np.isfinite(known[max(0, i - half):i + half + 1]).any() else (0., 0.) for i in range(count)])
    centre = np.nan_to_num(centre)
    keep = np.isfinite(known).all(axis=1) & (np.linalg.norm(np.nan_to_num(known) - centre, axis=1) <= policy['outlier_px'])
    cleaned = np.where(keep[:, None], known, centre)
    return np.array([np.median(cleaned[max(0, i - half):i + half + 1], axis=0) for i in range(count)])


def corrected(rotation, shift, focal):
    """Turn the camera so that the terrain moves by `shift` pixels in the image (x right, y down)."""
    pan = np.arctan2(shift[0], focal)
    tilt = np.arctan2(shift[1], focal)
    # A rendered feature that sits right of the video feature needs the camera turned right (about +y up axis is left).
    turn = cv2.Rodrigues(np.array([0., -pan, 0.]))[0] @ cv2.Rodrigues(np.array([-tilt, 0., 0.]))[0]
    return rotation @ turn
