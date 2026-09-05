from typing import Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis


class FaceEncoder:

    def __init__(
        self,
        modelName: str = "buffalo_l",
        detection_size: tuple[int, int] = (640, 640),
    ):

        self.app = FaceAnalysis(name=modelName)

        self.app.prepare(
            ctx_id=0,
            det_size=detection_size,
        )

    def encode(
        self,
        image: np.ndarray,
        face_bbox,
    ) -> Optional[np.ndarray]:

        if face_bbox is None:
            return None

        # --------------------------------------------------
        # Extract face coordinates
        # --------------------------------------------------

        x1 = max(0, face_bbox.x1)
        y1 = max(0, face_bbox.y1)

        x2 = min(image.shape[1], face_bbox.x2)
        y2 = min(image.shape[0], face_bbox.y2)

        # --------------------------------------------------
        # Validate crop
        # --------------------------------------------------

        if x2 <= x1 or y2 <= y1:
            return None

        face_crop = image[y1:y2, x1:x2]

        if face_crop.size == 0:
            return None

        # --------------------------------------------------
        # Run InsightFace
        # --------------------------------------------------

        faces = self.app.get(face_crop)

        if not faces:
            return None

        # --------------------------------------------------
        # Select the best detected face
        # --------------------------------------------------

        best_face = max(
            faces,
            key=lambda face: float(face.det_score),
        )

        # --------------------------------------------------
        # Get ArcFace embedding
        # --------------------------------------------------

        embedding = best_face.embedding

        if embedding is None:
            return None

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        # --------------------------------------------------
        # Normalize embedding
        # --------------------------------------------------

        norm = np.linalg.norm(embedding)

        if norm == 0:
            return None

        embedding = embedding / norm

        return embedding
