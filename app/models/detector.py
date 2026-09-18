from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class BoundingBox:
    """
    Represents a rectangular bounding box.

    Coordinates follow the standard image convention:

        (x1, y1) = top-left
        (x2, y2) = bottom-right
    """

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        """Return the width of the bounding box."""
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        """Return the height of the bounding box."""
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        """Return the area of the bounding box."""
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """Return the center point of the bounding box."""
        return (
            (self.x1 + self.x2) / 2,
            (self.y1 + self.y2) / 2,
        )


@dataclass
class PersonDetection:
    """
    Represents a person detected by the person detector.
    """

    bbox: BoundingBox
    confidence: float


@dataclass
class FaceDetection:
    """
    Represents a face detected by InsightFace.

    InsightFace already calculates the face recognition
    embedding during the same inference, so we retain it here.

    This prevents us from running another face model later.
    """

    bbox: BoundingBox
    confidence: float
    landmarks: Optional[object] = None
    embedding: Optional[np.ndarray] = None

    # Coarse head-pose estimate derived from InsightFace landmarks.
    # Values are in degrees and are intended for identity matching
    # and representative selection, not medical/measurement use.
    yaw: Optional[float] = None
    pitch: Optional[float] = None
    roll: Optional[float] = None
    pose: Optional[str] = None
