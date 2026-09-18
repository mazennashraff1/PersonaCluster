from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.models.detector import BoundingBox


@dataclass
class PersonObservation:
    """
    Represents one detected occurrence of a person in one image.

    An observation does NOT mean a confirmed identity.

    It represents:
        "Something that appears to be one person in this image."

    An observation may contain:
        - face + body
        - face only
        - body only
    """

    # ---------------------------------------------------------
    # Identity of the observation
    # ---------------------------------------------------------

    observation_id: int
    image_id: str

    # ---------------------------------------------------------
    # Person / body information
    # ---------------------------------------------------------

    person_bbox: Optional[BoundingBox]
    person_detection_confidence: Optional[float]

    # ---------------------------------------------------------
    # Face information
    # ---------------------------------------------------------

    face_bbox: Optional[BoundingBox]
    face_detection_confidence: Optional[float]

    # ---------------------------------------------------------
    # Face pose
    # ---------------------------------------------------------
    # Coarse head-pose estimate derived from InsightFace landmarks.
    face_yaw: Optional[float] = None
    face_pitch: Optional[float] = None
    face_roll: Optional[float] = None
    face_pose: Optional[str] = None

    # ---------------------------------------------------------
    # Face/body geometric association
    # ---------------------------------------------------------

    association_score: float = 0.0

    # ---------------------------------------------------------
    # Learned representations
    # ---------------------------------------------------------

    face_embedding: Optional[np.ndarray] = None
    body_embedding: Optional[np.ndarray] = None

    # ---------------------------------------------------------
    # Quality
    # ---------------------------------------------------------

    face_quality: float = 0.0
    body_quality: float = 0.0

    # ---------------------------------------------------------
    # Embedding validation
    # ---------------------------------------------------------

    face_embedding_valid: bool = False
    body_embedding_valid: bool = False

    # ---------------------------------------------------------
    # Temporal information
    # ---------------------------------------------------------

    moment_id: Optional[int] = None

    # ---------------------------------------------------------
    # Clustering information
    # ---------------------------------------------------------

    cluster_id: Optional[int] = None

    # =========================================================
    # Properties
    # =========================================================

    @property
    def has_face(self) -> bool:
        """Return True when this observation contains a face embedding."""
        return self.face_embedding is not None

    @property
    def has_body(self) -> bool:
        """Return True when this observation contains a body embedding."""
        return self.body_embedding is not None

    @property
    def observation_type(self) -> str:
        """
        Return the observation type.

        Possible values:
            face_and_body
            face_only
            body_only
            empty
        """

        if self.has_face and self.has_body:
            return "face_and_body"

        if self.has_face:
            return "face_only"

        if self.has_body:
            return "body_only"

        return "empty"
