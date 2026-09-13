"""
Pose-aware identity profile built from observations already assigned to a cluster.

This module does not perform clustering by itself. It only answers:
    "Which observations are the strongest representatives of this identity,
     while keeping useful pose diversity?"

The profile is rebuilt whenever a cluster changes. Representatives are real
observations; no external/reference image is required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from app.models.observation import PersonObservation

POSE_ORDER = ("frontal", "left", "right", "profile", "unknown")


@dataclass
class IdentityProfile:
    """Self-built identity representation for one candidate cluster."""

    pose_representatives: dict[str, list[PersonObservation]] = field(
        default_factory=dict
    )

    @staticmethod
    def _quality(observation: PersonObservation) -> float:
        face_quality = float(np.clip(observation.face_quality, 0.0, 1.0))
        face_confidence = float(
            np.clip(
                observation.face_detection_confidence or 0.0,
                0.0,
                1.0,
            )
        )

        # Keep the same philosophy as the existing clustering code:
        # quality first, detection confidence second.
        return 0.70 * face_quality + 0.30 * face_confidence

    @staticmethod
    def pose_key(observation: PersonObservation) -> str:
        pose = observation.face_pose
        if pose in POSE_ORDER:
            return pose
        return "unknown"

    @classmethod
    def build(
        cls,
        observations: list[PersonObservation],
        total_representatives: int,
        max_per_pose: int,
    ) -> "IdentityProfile":
        """Build a pose-diverse profile from real observations."""

        if total_representatives < 1:
            return cls()

        grouped: dict[str, list[PersonObservation]] = {pose: [] for pose in POSE_ORDER}

        for observation in observations:
            if not observation.face_embedding_valid:
                continue
            if observation.face_embedding is None:
                continue
            grouped[cls.pose_key(observation)].append(observation)

        for pose in grouped:
            grouped[pose].sort(
                key=lambda observation: (
                    -cls._quality(observation),
                    observation.observation_id,
                )
            )

        selected: dict[str, list[PersonObservation]] = {pose: [] for pose in POSE_ORDER}

        # First guarantee pose coverage. We deliberately prefer known poses.
        for pose in ("frontal", "left", "right", "profile"):
            if grouped[pose]:
                selected[pose].append(grouped[pose][0])

        selected_count = sum(len(items) for items in selected.values())

        # Fill remaining slots by globally strongest observations that are not
        # already selected, while respecting max_per_pose.
        selected_ids = {
            observation.observation_id
            for pose_observations in selected.values()
            for observation in pose_observations
        }

        if selected_count < total_representatives:
            remaining = sorted(
                (
                    observation
                    for pose in POSE_ORDER
                    for observation in grouped[pose]
                    if observation.observation_id not in selected_ids
                ),
                key=lambda observation: (
                    -cls._quality(observation),
                    observation.observation_id,
                ),
            )

            for observation in remaining:
                pose = cls.pose_key(observation)
                if len(selected[pose]) >= max_per_pose:
                    continue
                if (
                    sum(len(items) for items in selected.values())
                    >= total_representatives
                ):
                    break
                selected[pose].append(observation)

        return cls(pose_representatives=selected)

    def representatives(self) -> list[PersonObservation]:
        """Return deterministic flattened representatives."""
        result: list[PersonObservation] = []
        for pose in POSE_ORDER:
            result.extend(self.pose_representatives.get(pose, []))
        return result

    def observations_for_pose(self, pose: Optional[str]) -> list[PersonObservation]:
        """Return representatives matching a pose label."""
        key = pose if pose in POSE_ORDER else "unknown"
        return list(self.pose_representatives.get(key, []))
