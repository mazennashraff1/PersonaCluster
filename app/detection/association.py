from typing import List

from app.models.detector import (
    BoundingBox,
    FaceDetection,
    PersonDetection,
)
from app.models.observation import PersonObservation
from app import configuration as config
from app.identity.facePoseEstimator import FacePoseEstimator


class AssociationEngine:
    """
    Associates detected faces with detected people.

    The engine supports three valid observation types:

        1. face_and_body
        2. face_only
        3. body_only

    A face is NEVER discarded simply because no body was found.

    A person is NEVER discarded simply because no face was found.
    """

    def __init__(self):
        self.min_score = config.ASSOCIATION_MIN_SCORE

    # ---------------------------------------------------------
    # Geometry
    # ---------------------------------------------------------

    @staticmethod
    def bbox_area(
        bbox: BoundingBox,
    ) -> int:
        """Return bounding-box area."""

        return bbox.area

    @staticmethod
    def point_inside_bbox(
        point: tuple[float, float],
        bbox: BoundingBox,
    ) -> bool:
        """Check whether a point lies inside a bounding box."""

        x, y = point

        return bbox.x1 <= x <= bbox.x2 and bbox.y1 <= y <= bbox.y2

    @staticmethod
    def intersection_area(
        bbox_a: BoundingBox,
        bbox_b: BoundingBox,
    ) -> int:
        """Calculate intersection area between two bounding boxes."""

        x1 = max(bbox_a.x1, bbox_b.x1)
        y1 = max(bbox_a.y1, bbox_b.y1)

        x2 = min(bbox_a.x2, bbox_b.x2)
        y2 = min(bbox_a.y2, bbox_b.y2)

        width = max(0, x2 - x1)
        height = max(0, y2 - y1)

        return width * height

    # ---------------------------------------------------------
    # Association score
    # ---------------------------------------------------------

    def calculate_association_score(
        self,
        face_bbox: BoundingBox,
        body_bbox: BoundingBox,
    ) -> float:
        """
        Calculate how strongly a face belongs to a person box.

        We combine:

            1. Face overlap with person box.
            2. Whether the face center is inside the person box.

        A face completely contained inside the person bounding box
        receives a score of 1.0.
        """

        if body_bbox.area <= 0:
            return 0.0

        intersection = self.intersection_area(
            face_bbox,
            body_bbox,
        )

        overlap_ratio = intersection / face_bbox.area if face_bbox.area > 0 else 0.0

        face_center_inside = self.point_inside_bbox(
            face_bbox.center,
            body_bbox,
        )

        if face_center_inside:

            return 0.5 + (0.5 * overlap_ratio)

        return overlap_ratio

    # ---------------------------------------------------------
    # Find best body
    # ---------------------------------------------------------

    def find_best_body(
        self,
        face_bbox: BoundingBox,
        body_detections: List[PersonDetection],
        used_body_ids: set[int],
    ):
        """
        Find the best unused person detection for a face.

        Returns:

            (body_index, score)

        or:

            (None, 0.0)
        """

        best_body_index = None
        best_score = 0.0

        for body_index, body_detection in enumerate(body_detections):

            # One body can only belong to one face.
            if body_index in used_body_ids:
                continue

            score = self.calculate_association_score(
                face_bbox=face_bbox,
                body_bbox=body_detection.bbox,
            )

            if score > best_score:

                best_score = score
                best_body_index = body_index

        if best_body_index is not None and best_score >= self.min_score:
            return best_body_index, best_score

        return None, 0.0

    # ---------------------------------------------------------
    # Main association
    # ---------------------------------------------------------

    def associate(
        self,
        image_id: str,
        face_detections: List[FaceDetection],
        body_detections: List[PersonDetection],
        observation_id_start: int = 0,
    ) -> List[PersonObservation]:
        """
        Create observations from face and person detections.

        The output can contain:

            Face + Body observations
            Face-only observations
            Body-only observations

        No valid detection is silently discarded.
        """

        observations = []

        used_body_ids: set[int] = set()

        next_observation_id = observation_id_start

        # =====================================================
        # STEP 1
        # Process every detected face.
        # =====================================================

        for face_detection in face_detections:

            body_index, association_score = self.find_best_body(
                face_bbox=face_detection.bbox,
                body_detections=body_detections,
                used_body_ids=used_body_ids,
            )

            face_yaw, face_pitch, face_roll, face_pose = FacePoseEstimator.estimate(
                face_detection.landmarks,
                frontal_yaw_degrees=config.FACE_POSE_FRONTAL_YAW_DEGREES,
                profile_yaw_degrees=config.FACE_POSE_PROFILE_YAW_DEGREES,
            )

            # -------------------------------------------------
            # Case A: Face + Body
            # -------------------------------------------------

            if body_index is not None:

                body_detection = body_detections[body_index]

                used_body_ids.add(body_index)

                observation = PersonObservation(
                    observation_id=next_observation_id,
                    image_id=image_id,
                    person_bbox=body_detection.bbox,
                    person_detection_confidence=(body_detection.confidence),
                    face_bbox=face_detection.bbox,
                    face_detection_confidence=(face_detection.confidence),
                    association_score=association_score,
                    face_yaw=face_yaw,
                    face_pitch=face_pitch,
                    face_roll=face_roll,
                    face_pose=face_pose,
                    # Face embedding comes directly from
                    # InsightFace.
                    face_embedding=(face_detection.embedding),
                )

            # -------------------------------------------------
            # Case B: Face only
            # -------------------------------------------------

            else:

                observation = PersonObservation(
                    observation_id=next_observation_id,
                    image_id=image_id,
                    person_bbox=None,
                    person_detection_confidence=None,
                    face_bbox=face_detection.bbox,
                    face_detection_confidence=(face_detection.confidence),
                    association_score=0.0,
                    face_yaw=face_yaw,
                    face_pitch=face_pitch,
                    face_roll=face_roll,
                    face_pose=face_pose,
                    # Still keep the face embedding.
                    face_embedding=(face_detection.embedding),
                )

            observations.append(observation)

            next_observation_id += 1

        # =====================================================
        # STEP 2
        # Create body-only observations for people that
        # were not associated with any face.
        # =====================================================

        for body_index, body_detection in enumerate(body_detections):

            if body_index in used_body_ids:
                continue

            observation = PersonObservation(
                observation_id=next_observation_id,
                image_id=image_id,
                person_bbox=body_detection.bbox,
                person_detection_confidence=(body_detection.confidence),
                face_bbox=None,
                face_detection_confidence=None,
                association_score=0.0,
                face_embedding=None,
            )

            observations.append(observation)

            next_observation_id += 1

        return observations
