from typing import List

import numpy as np
from insightface.app import FaceAnalysis

from app import configuration as config
from app.models.detector import BoundingBox, FaceDetection
from app.identity.facePoseEstimator import FacePoseEstimator


class FaceDetector:
    """
    Detects faces and extracts their face embeddings.

    InsightFace performs both operations during the same inference.
    """

    def __init__(
        self,
        modelName: str,
        detection_size: tuple[int, int],
        detection_threshold: float,
        ctx_id: int,
    ):
        self.app = FaceAnalysis(
            name=modelName,
        )

        self.app.prepare(
            ctx_id=ctx_id,
            det_size=detection_size,
            det_thresh=detection_threshold,
        )

        self.pose_estimator = FacePoseEstimator()

    def detect(self, image) -> List[FaceDetection]:
        """
        Detect all faces in an image.

        Returns:
            List[FaceDetection]

        Each FaceDetection contains:

            - bounding box
            - detection confidence
            - landmarks
            - normalized face embedding
        """

        faces = self.app.get(image)

        detections = []

        for face in faces:

            x1, y1, x2, y2 = face.bbox.astype(int)

            embedding = None

            if face.embedding is not None:

                embedding = np.asarray(
                    face.embedding,
                    dtype=np.float32,
                )

                # L2 normalization.
                norm = np.linalg.norm(embedding)

                if norm > 0:
                    embedding = embedding / norm

            yaw, pitch, roll, pose = self.pose_estimator.estimate(
                landmarks=face.kps,
                image_shape=image.shape,
            )

            detections.append(
                FaceDetection(
                    bbox=BoundingBox(
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    ),
                    confidence=float(face.det_score),
                    landmarks=face.kps,
                    embedding=embedding,
                    yaw=yaw,
                    pitch=pitch,
                    roll=roll,
                    pose=pose,
                )
            )

        return detections
