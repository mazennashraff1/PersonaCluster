from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.models.observation import PersonObservation


@dataclass(frozen=True)
class BestImageCandidate:
    observation_id: int
    image_id: str
    cluster_id: int
    score: float
    pose: str


@dataclass(frozen=True)
class BestImageSelection:
    cluster_id: int
    candidates: list[BestImageCandidate]

    @property
    def observation_ids(self) -> list[int]:
        return [candidate.observation_id for candidate in self.candidates]

    @property
    def image_ids(self) -> list[str]:
        return [candidate.image_id for candidate in self.candidates]


class BestImageSelector:
    """
    Select the strongest event images for each discovered identity.

    This selector works AFTER clustering. It does not alter identity
    assignments. It only ranks observations that already belong to a
    confirmed cluster.

    Selection principles:

        1. Prefer strong face quality.
        2. Prefer strong face detection confidence.
        3. Prefer strong person detection confidence.
        4. Prefer strong face/body association when available.
        5. Avoid selecting the same source image twice for one identity.
        6. Prefer pose diversity when good candidates are available.

    There is intentionally no fabricated "occlusion detector" here.
    Existing face_quality is used as the project's current proxy for
    face clarity / visibility. A future dedicated occlusion score can
    be added without changing the selector interface.
    """

    def __init__(
        self,
        max_images_per_cluster: int = 5,
        min_pose_diversity: bool = True,
    ) -> None:
        if max_images_per_cluster < 1:
            raise ValueError("max_images_per_cluster must be at least 1.")

        self.max_images_per_cluster = int(max_images_per_cluster)
        self.min_pose_diversity = bool(min_pose_diversity)

    @staticmethod
    def _clip01(value: object) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return 0.0

        if numeric != numeric:  # NaN
            return 0.0

        return max(0.0, min(1.0, numeric))

    def _quality_score(self, observation: PersonObservation) -> float:
        """Rank one observation using only already-stored signals."""
        face_quality = self._clip01(observation.face_quality)
        face_confidence = self._clip01(observation.face_detection_confidence)
        person_confidence = self._clip01(observation.person_detection_confidence)
        association = self._clip01(observation.association_score)

        # Face quality is the strongest signal because the resulting image
        # is intended to be useful for visually identifying the person.
        return (
            0.50 * face_quality
            + 0.25 * face_confidence
            + 0.15 * person_confidence
            + 0.10 * association
        )

    @staticmethod
    def _normalise_pose(observation: PersonObservation) -> str:
        pose = observation.face_pose
        if pose in {"frontal", "left", "right", "profile"}:
            return str(pose)
        return "unknown"

    def _candidate(
        self,
        observation: PersonObservation,
        cluster_id: int,
    ) -> BestImageCandidate:
        return BestImageCandidate(
            observation_id=observation.observation_id,
            image_id=observation.image_id,
            cluster_id=int(cluster_id),
            score=float(self._quality_score(observation)),
            pose=self._normalise_pose(observation),
        )

    def select(
        self,
        observations: list[PersonObservation],
        assignments: dict[int, Optional[int]],
    ) -> dict[int, BestImageSelection]:
        """
        Return the best source-image observations for every confirmed cluster.

        At most one observation from a source image is selected for a cluster.
        This matters because Patch 3 suppresses duplicate detections for
        clustering, while the original observations are intentionally retained
        in the event store.
        """

        grouped: dict[int, list[PersonObservation]] = {}

        for observation in observations:
            cluster_id = assignments.get(observation.observation_id)

            if cluster_id is None:
                continue

            grouped.setdefault(int(cluster_id), []).append(observation)

        results: dict[int, BestImageSelection] = {}

        for cluster_id, cluster_observations in sorted(grouped.items()):
            # ------------------------------------------------------------
            # Keep only the strongest observation for each source image.
            # This prevents duplicate detections from creating duplicate
            # "best" images even when those observations share the cluster.
            # ------------------------------------------------------------
            best_per_image: dict[str, PersonObservation] = {}

            for observation in cluster_observations:
                current = best_per_image.get(observation.image_id)

                if current is None or self._quality_score(
                    observation
                ) > self._quality_score(current):
                    best_per_image[observation.image_id] = observation

            candidates = [
                self._candidate(observation, cluster_id)
                for observation in best_per_image.values()
            ]

            candidates.sort(
                key=lambda item: (
                    -item.score,
                    item.observation_id,
                )
            )

            selected: list[BestImageCandidate] = []
            selected_images: set[str] = set()
            selected_poses: set[str] = set()

            # ------------------------------------------------------------
            # Pass 1: guarantee useful pose diversity when possible.
            # We only take a pose if its candidate is already one of the
            # strongest candidates. A weak image is never selected merely
            # because it has a different pose.
            # ------------------------------------------------------------
            if self.min_pose_diversity:
                for pose in ("frontal", "left", "right", "profile"):
                    if len(selected) >= self.max_images_per_cluster:
                        break

                    pose_candidates = [
                        candidate
                        for candidate in candidates
                        if candidate.pose == pose
                        and candidate.image_id not in selected_images
                    ]

                    if not pose_candidates:
                        continue

                    best_pose_candidate = pose_candidates[0]

                    selected.append(best_pose_candidate)
                    selected_images.add(best_pose_candidate.image_id)
                    selected_poses.add(best_pose_candidate.pose)

            # ------------------------------------------------------------
            # Pass 2: fill remaining slots by score.
            # ------------------------------------------------------------
            for candidate in candidates:
                if len(selected) >= self.max_images_per_cluster:
                    break

                if candidate.image_id in selected_images:
                    continue

                selected.append(candidate)
                selected_images.add(candidate.image_id)
                selected_poses.add(candidate.pose)

            # Highest-quality first in final output, regardless of the
            # order in which pose-diverse candidates were selected.
            selected.sort(
                key=lambda item: (
                    -item.score,
                    item.observation_id,
                )
            )

            results[cluster_id] = BestImageSelection(
                cluster_id=cluster_id,
                candidates=selected,
            )

        return results
