"""Navigate-owned image and candidate interfaces for the local demo."""

from dataclasses import dataclass
from typing import Protocol, TypedDict, NotRequired
import numpy as np


@dataclass(frozen=True)
class PixelPairs:
    """Pixel coordinates have no model score or confidence interpretation."""

    first: np.ndarray
    second: np.ndarray
    backend_identity: str

    def __post_init__(self):
        if (
            self.first.shape != self.second.shape
            or self.first.ndim != 2
            or self.first.shape[1] != 2
        ):
            raise ValueError("Pixel pairs must have matching N by 2 shapes")
        if not np.isfinite(self.first).all() or not np.isfinite(self.second).all():
            raise ValueError("Pixel pairs must be finite")


class ImageMatcher(Protocol):
    def match_images(self, first: np.ndarray, second: np.ndarray) -> PixelPairs:
        """Accept grayscale uint8 images and return coordinates in their pixel frames."""
        ...


class CandidatePose(TypedDict):
    """Position uses local ENU metres. Rotation maps camera axes to ENU."""

    position_enu_m: list[float]
    eye_to_enu_xyzw: list[float]


class SearchCandidate(TypedDict):
    """A proposal carries retrieval diagnostics, not geographic confidence."""

    candidate: CandidatePose
    inliers: NotRequired[int]
    coverage: NotRequired[float]
    score: NotRequired[float]
    angle: NotRequired[int]
    crop: NotRequired[list[int]]


class CandidateSearch(Protocol):
    def candidates(self, image: np.ndarray) -> tuple[list[SearchCandidate], dict]:
        """Return pose proposals and retrieval diagnostics, without acceptance claims."""
        ...
