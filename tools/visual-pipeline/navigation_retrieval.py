"""Camera-frame matches propose search poses; only a fresh map match can anchor them."""
import cv2
import numpy as np


class Retrieval:
    def __init__(self, root):
        self.root = root
        self.detector = cv2.SIFT_create(nfeatures=1600)
        self.matcher = cv2.BFMatcher()
        self.frames = []

    def features(self, path):
        image = cv2.imread(str(self.root / path), cv2.IMREAD_GRAYSCALE)
        image = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8)).apply(image)
        return self.detector.detectAndCompute(image, None)

    def add(self, sequence, image, pose):
        points, descriptors = self.features(image)
        if descriptors is not None:
            self.frames.append((sequence, image, pose, points, descriptors))
            if len(self.frames) > 48:
                del self.frames[1]

    def candidates(self, image):
        points, descriptors = self.features(image)
        if descriptors is None:
            return []
        ranked = []
        for sequence, source, pose, earlier, features in self.frames:
            neighbours = self.matcher.knnMatch(features, descriptors, k=2)
            good = [pair[0] for pair in neighbours if len(pair) == 2 and pair[0].distance < .7 * pair[1].distance]
            if len(good) < 25:
                continue
            a = np.float32([earlier[m.queryIdx].pt for m in good])
            b = np.float32([points[m.trainIdx].pt for m in good])
            _, mask = cv2.findHomography(a, b, cv2.RANSAC, 2)
            if mask is None:
                continue
            mask = mask.ravel().astype(bool)
            if mask.sum() < 20 or np.prod(np.ptp(b[mask], axis=0)) < 5000:
                continue
            ranked.append((int(mask.sum()), {"sequence": sequence, "image": source, "pose": pose,
                                             "pairs": np.stack((a[mask], b[mask]), axis=1).tolist()}))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [item[1] for item in ranked[:2]]
