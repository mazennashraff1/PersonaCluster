"""
Same-image duplicate detection for identity clustering.

This module handles a very specific failure mode:

    one source image
        -> person detector/association produces two observations
        -> both observations actually describe the same person

Those duplicate observations must not become separate identity evidence.

The suppressor is intentionally conservative. It only suppresses an
observation when BOTH conditions are satisfied:

    1. very high face embedding similarity
    2. meaningful bounding-box overlap in the same image

The stronger-quality observation is retained as the canonical observation.
No external/reference image is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from app.models.detector import BoundingBox
from app.models.observation import PersonObservation


@dataclass(frozen=True)
class DuplicateSuppressionResult:
    """Result of same-image duplicate suppression."""

    kept_observations: list[PersonObservation]
    duplicate_to_canonical: dict[int, int]

    @property
    def suppressed_count(self) -> int:
        return len(self.duplicate_to_canonical)


class SameImageDuplicateSuppressor:
    """
    Conservative same-image duplicate suppressor.

    The input is expected to contain observations with valid face
    embeddings. The class is still defensive and ignores observations
    without usable face embeddings.
    """

    def __init__(
        self,
        face_similarity_threshold: float,
        person_iou_threshold: float,
        face_iou_threshold: float,
    ) -> None:
        self.face_similarity_threshold = float(face_similarity_threshold)
        self.person_iou_threshold = float(person_iou_threshold)
        self.face_iou_threshold = float(face_iou_threshold)

        if not -1.0 <= self.face_similarity_threshold <= 1.0:
            raise ValueError(
                "SAME_IMAGE_DUPLICATE_FACE_SIMILARITY must be between -1 and 1."
            )

        if not 0.0 <= self.person_iou_threshold <= 1.0:
            raise ValueError("SAME_IMAGE_DUPLICATE_PERSON_IOU must be between 0 and 1.")

        if not 0.0 <= self.face_iou_threshold <= 1.0:
            raise ValueError("SAME_IMAGE_DUPLICATE_FACE_IOU must be between 0 and 1.")

    @staticmethod
    def _normalise_embedding(
        embedding: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        if embedding is None:
            return None

        try:
            vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            return None

        if vector.size == 0 or not np.all(np.isfinite(vector)):
            return None

        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0.0:
            return None

        return vector / norm

    @classmethod
    def _face_similarity(
        cls,
        first: PersonObservation,
        second: PersonObservation,
    ) -> Optional[float]:
        first_embedding = cls._normalise_embedding(first.face_embedding)
        second_embedding = cls._normalise_embedding(second.face_embedding)

        if first_embedding is None or second_embedding is None:
            return None

        if first_embedding.shape != second_embedding.shape:
            return None

        similarity = float(np.dot(first_embedding, second_embedding))
        if not np.isfinite(similarity):
            return None

        return float(np.clip(similarity, -1.0, 1.0))

    @staticmethod
    def _iou(
        first: Optional[BoundingBox],
        second: Optional[BoundingBox],
    ) -> float:
        if first is None or second is None:
            return 0.0

        intersection_x1 = max(first.x1, second.x1)
        intersection_y1 = max(first.y1, second.y1)
        intersection_x2 = min(first.x2, second.x2)
        intersection_y2 = min(first.y2, second.y2)

        intersection_width = max(0, intersection_x2 - intersection_x1)
        intersection_height = max(0, intersection_y2 - intersection_y1)
        intersection_area = intersection_width * intersection_height

        if intersection_area <= 0:
            return 0.0

        union_area = first.area + second.area - intersection_area
        if union_area <= 0:
            return 0.0

        return float(intersection_area / union_area)

    @staticmethod
    def _quality(observation: PersonObservation) -> float:
        face_quality = float(np.clip(observation.face_quality, 0.0, 1.0))
        detection_confidence = float(
            np.clip(observation.face_detection_confidence or 0.0, 0.0, 1.0)
        )

        return 0.70 * face_quality + 0.30 * detection_confidence

    def _is_duplicate_pair(
        self,
        first: PersonObservation,
        second: PersonObservation,
    ) -> bool:
        if first.image_id != second.image_id:
            return False

        face_similarity = self._face_similarity(first, second)
        if face_similarity is None:
            return False

        if face_similarity < self.face_similarity_threshold:
            return False

        person_iou = self._iou(first.person_bbox, second.person_bbox)
        face_iou = self._iou(first.face_bbox, second.face_bbox)

        # Bounding-box overlap is the second safety gate. This prevents
        # nearby but different people with accidentally similar embeddings
        # from being suppressed.
        return (
            person_iou >= self.person_iou_threshold
            or face_iou >= self.face_iou_threshold
        )

    def suppress(
        self,
        observations: list[PersonObservation],
    ) -> DuplicateSuppressionResult:
        """
        Suppress duplicate observations within each source image.

        The highest-quality observation in each duplicate group becomes
        canonical. Suppressed observations map back to that canonical ID
        so the final clustering assignment can be copied to them later.
        """

        # Union-find over observation IDs. Each connected component is one
        # conservative same-image duplicate group.
        parent: dict[int, int] = {
            observation.observation_id: observation.observation_id
            for observation in observations
        }

        def find(value: int) -> int:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        def union(first_id: int, second_id: int) -> None:
            first_root = find(first_id)
            second_root = find(second_id)
            if first_root != second_root:
                # Deterministic root.
                if first_root < second_root:
                    parent[second_root] = first_root
                else:
                    parent[first_root] = second_root

        by_image: dict[str, list[PersonObservation]] = {}
        for observation in observations:
            by_image.setdefault(observation.image_id, []).append(observation)

        for image_observations in by_image.values():
            # Pairwise comparison is intentionally local to one image.
            # Duplicate detections per image are normally small, so this is
            # much cheaper than an event-wide O(N²) operation.
            for index, first in enumerate(image_observations):
                if not first.face_embedding_valid or first.face_embedding is None:
                    continue

                for second in image_observations[index + 1 :]:
                    if not second.face_embedding_valid or second.face_embedding is None:
                        continue

                    if self._is_duplicate_pair(first, second):
                        union(first.observation_id, second.observation_id)

        components: dict[int, list[PersonObservation]] = {}
        for observation in observations:
            root = find(observation.observation_id)
            components.setdefault(root, []).append(observation)

        duplicate_to_canonical: dict[int, int] = {}
        kept_ids: set[int] = set()

        for component in components.values():
            ranked = sorted(
                component,
                key=lambda observation: (
                    -self._quality(observation),
                    -float(observation.face_detection_confidence or 0.0),
                    observation.observation_id,
                ),
            )

            canonical = ranked[0]
            kept_ids.add(canonical.observation_id)

            for duplicate in ranked[1:]:
                duplicate_to_canonical[duplicate.observation_id] = (
                    canonical.observation_id
                )

        kept_observations = [
            observation
            for observation in observations
            if observation.observation_id in kept_ids
        ]

        return DuplicateSuppressionResult(
            kept_observations=kept_observations,
            duplicate_to_canonical=duplicate_to_canonical,
        )
