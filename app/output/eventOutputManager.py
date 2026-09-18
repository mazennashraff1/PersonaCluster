from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

from app import configuration as config
from app.models.observation import PersonObservation


class EventOutputManager:
    """Build the final output using reference-matched known people.

    The clustering stage has already happened before this class is called.
    A known person may own multiple cluster IDs. All of those clusters are
    aggregated into ONE person directory.

    Final structure:

        output/
        └── Person Name/
            ├── Best Images/
            ├── All Images/
            └── representative Image.jpg
    """

    VALID_MODES = {
        "REFERENCE_MATCHED",
    }

    def __init__(
        self,
        event_path: str | Path,
        output_directory_name: str = "output",
        mode: str = "REFERENCE_MATCHED",
        clean_before_run: bool = True,
    ) -> None:
        self.event_path = Path(event_path)
        self.mode = str(mode).upper().strip()
        self.clean_before_run = bool(clean_before_run)

        if self.mode not in self.VALID_MODES:
            raise ValueError(
                f"Unsupported EVENT_OUTPUT_MODE={mode!r}. "
                f"Expected one of: {', '.join(sorted(self.VALID_MODES))}."
            )

        self.output_root = self.event_path / output_directory_name

    @staticmethod
    def _project_root() -> Path:
        """Return the project root from this module's location."""
        return Path(__file__).resolve().parents[2]

    def _resolve_source_path(self, image_id: str) -> Path | None:
        """Resolve the portable image ID stored in EventStore."""
        candidate = Path(str(image_id))
        candidates = (
            [candidate]
            if candidate.is_absolute()
            else [self._project_root() / candidate]
        )

        for path in candidates:
            try:
                if path.is_file():
                    return path.resolve(strict=False)
            except OSError:
                continue

        return None

    @staticmethod
    def _safe_output_name(image_id: str, observation_id: int) -> str:
        """Keep the original filename and add observation ID on collision."""
        name = Path(str(image_id)).name
        return name or f"observation_{observation_id}.jpg"

    @staticmethod
    def _safe_person_directory_name(person_name: str) -> str:
        """Make the reference person's name safe for use as a Windows folder."""
        invalid = '<>:"/\\|?*'
        cleaned = "".join("_" if char in invalid else char for char in str(person_name))
        cleaned = " ".join(cleaned.split()).strip().rstrip(".")
        return cleaned or "Unknown Person"

    @staticmethod
    def _unique_by_image(
        observations: Iterable[PersonObservation],
        assignments: dict[int, int | None],
    ) -> dict[int, list[PersonObservation]]:
        """Return one strongest observation per source image in each cluster."""
        grouped: dict[int, dict[str, PersonObservation]] = defaultdict(dict)

        for observation in observations:
            cluster_id = assignments.get(observation.observation_id)
            if cluster_id is None:
                continue

            cluster_id = int(cluster_id)
            current = grouped[cluster_id].get(observation.image_id)

            if current is None:
                grouped[cluster_id][observation.image_id] = observation
                continue

            current_score = (
                float(current.face_quality or 0.0),
                float(current.face_detection_confidence or 0.0),
                int(current.observation_id),
            )
            new_score = (
                float(observation.face_quality or 0.0),
                float(observation.face_detection_confidence or 0.0),
                int(observation.observation_id),
            )

            if new_score > current_score:
                grouped[cluster_id][observation.image_id] = observation

        return {
            cluster_id: list(image_map.values())
            for cluster_id, image_map in grouped.items()
        }

    @staticmethod
    def _best_image_score(observation: PersonObservation) -> float:
        """Rank a source image using the existing quality signals."""

        def clip(value: object) -> float:
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                return 0.0
            if not np.isfinite(numeric):
                return 0.0
            return max(0.0, min(1.0, numeric))

        return (
            0.50 * clip(observation.face_quality)
            + 0.25 * clip(observation.face_detection_confidence)
            + 0.15 * clip(observation.person_detection_confidence)
            + 0.10 * clip(observation.association_score)
        )

    @staticmethod
    def _normalise_pose(observation: PersonObservation) -> str:
        pose = observation.face_pose
        if pose in {"frontal", "left", "right", "profile"}:
            return str(pose)
        return "unknown"

    def _select_best_images(
        self,
        cluster_observations: list[PersonObservation],
    ) -> list[tuple[PersonObservation, float, str]]:
        """Select strong unique images from one cluster."""
        max_images = int(config.BEST_IMAGES_PER_CLUSTER)
        require_pose_diversity = bool(config.BEST_IMAGES_REQUIRE_POSE_DIVERSITY)

        best_per_image: dict[str, PersonObservation] = {}
        for observation in cluster_observations:
            current = best_per_image.get(observation.image_id)
            if current is None or self._best_image_score(
                observation
            ) > self._best_image_score(current):
                best_per_image[observation.image_id] = observation

        candidates = [
            (
                observation,
                self._best_image_score(observation),
                self._normalise_pose(observation),
            )
            for observation in best_per_image.values()
        ]
        candidates.sort(key=lambda item: (-item[1], item[0].observation_id))

        selected: list[tuple[PersonObservation, float, str]] = []
        selected_images: set[str] = set()

        if require_pose_diversity:
            for pose in ("frontal", "left", "right", "profile"):
                if len(selected) >= max_images:
                    break
                pose_candidates = [
                    candidate
                    for candidate in candidates
                    if candidate[2] == pose
                    and candidate[0].image_id not in selected_images
                ]
                if pose_candidates:
                    selected.append(pose_candidates[0])
                    selected_images.add(pose_candidates[0][0].image_id)

        for candidate in candidates:
            if len(selected) >= max_images:
                break
            if candidate[0].image_id in selected_images:
                continue
            selected.append(candidate)
            selected_images.add(candidate[0].image_id)

        selected.sort(key=lambda item: (-item[1], item[0].observation_id))
        return selected

    @staticmethod
    def _representative_score(observation: PersonObservation) -> float:
        """Score a valid face observation for the person representative."""
        if not observation.face_embedding_valid or observation.face_bbox is None:
            return -1.0

        return config.REPRESENTATIVE_FACE_QUALITY_WEIGHT * float(
            observation.face_quality or 0.0
        ) + config.REPRESENTATIVE_FACE_DETECTION_WEIGHT * float(
            observation.face_detection_confidence or 0.0
        )

    @classmethod
    def _select_best_representative(
        cls,
        observations: list[PersonObservation],
    ) -> Optional[PersonObservation]:
        candidates = [
            observation
            for observation in observations
            if observation.face_embedding_valid and observation.face_bbox is not None
        ]
        if not candidates:
            return None
        return max(candidates, key=cls._representative_score)

    @staticmethod
    def _crop_face(
        image: np.ndarray,
        observation: PersonObservation,
    ) -> Optional[np.ndarray]:
        """Crop a face with the configured padding."""
        if image is None or image.size == 0 or observation.face_bbox is None:
            return None

        bbox = observation.face_bbox
        image_height, image_width = image.shape[:2]

        x1, y1 = int(bbox.x1), int(bbox.y1)
        x2, y2 = int(bbox.x2), int(bbox.y2)

        if x2 <= x1 or y2 <= y1:
            return None

        width = x2 - x1
        height = y2 - y1
        padding_x = int(width * config.FACE_PADDING_RATIO)
        padding_y = int(height * config.FACE_PADDING_RATIO)

        x1 = max(0, x1 - padding_x)
        y1 = max(0, y1 - padding_y)
        x2 = min(image_width, x2 + padding_x)
        y2 = min(image_height, y2 + padding_y)

        face = image[y1:y2, x1:x2]
        return face if face.size else None

    def _save_person_representative(
        self,
        person_name: str,
        observations: list[PersonObservation],
        person_directory: Path,
    ) -> Optional[Path]:
        """Save exactly one representative face for the whole person."""
        best = self._select_best_representative(observations)
        if best is None:
            print(f"[WARNING] No valid representative for {person_name}.")
            return None

        source_path = self._resolve_source_path(best.image_id)
        if source_path is None:
            print(f"[WARNING] Missing representative source: {best.image_id}")
            return None

        image = cv2.imread(str(source_path))
        if image is None:
            return None

        try:
            face = self._crop_face(image, best)
            if face is None:
                return None

            output_path = person_directory / "representative Image.jpg"
            if not cv2.imwrite(str(output_path), face):
                print(f"[WARNING] Failed to save representative: {output_path}")
                return None

            print(
                f"  {person_name} representative: "
                f"observation={best.observation_id}, "
                f"image={best.image_id}"
            )
            return output_path
        finally:
            del image

    def _copy_unique(
        self,
        source_path: Path,
        destination_directory: Path,
        image_id: str,
        observation_id: int,
    ) -> Path:
        destination = destination_directory / self._safe_output_name(
            image_id,
            observation_id,
        )
        if destination.exists():
            destination = destination_directory / (
                f"{destination.stem}_{observation_id}{destination.suffix}"
            )
        shutil.copy2(source_path, destination)
        return destination

    def _prepare_output_root(self) -> None:
        self.output_root.mkdir(parents=True, exist_ok=True)
        if not self.clean_before_run:
            return

        for child in self.output_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()

    def write(
        self,
        observations: list[PersonObservation],
        assignments: dict[int, int | None],
        cluster_person_matches: Optional[dict[int, str]] = None,
    ) -> Path:
        """Build person folders from already matched cluster IDs.

        The same person can appear in many clusters. Every matched cluster is
        collected under that person's single directory.
        """
        self._prepare_output_root()

        if self.mode != "REFERENCE_MATCHED":
            raise ValueError(
                "The current output pipeline requires "
                "EVENT_OUTPUT_MODE='REFERENCE_MATCHED'."
            )

        grouped = self._unique_by_image(observations, assignments)
        cluster_person_matches = cluster_person_matches or {}

        person_clusters: dict[str, list[int]] = defaultdict(list)
        for cluster_id, person_name in cluster_person_matches.items():
            cluster_id = int(cluster_id)
            if cluster_id in grouped:
                person_clusters[str(person_name)].append(cluster_id)

        copied_files = 0
        missing_files: list[str] = []

        for person_name in sorted(person_clusters, key=str.casefold):
            person_directory = self.output_root / self._safe_person_directory_name(
                person_name
            )
            person_directory.mkdir(parents=True, exist_ok=True)

            all_directory = person_directory / "All Images"
            best_directory = person_directory / "Best Images"
            all_directory.mkdir(parents=True, exist_ok=True)
            best_directory.mkdir(parents=True, exist_ok=True)

            person_observations: list[PersonObservation] = []

            for cluster_id in sorted(person_clusters[person_name]):
                cluster_observations = grouped.get(cluster_id, [])
                person_observations.extend(cluster_observations)

                # ----------------------------------------------------
                # ALL IMAGES
                # ----------------------------------------------------
                for observation in sorted(
                    cluster_observations,
                    key=lambda item: (item.image_id, item.observation_id),
                ):
                    source_path = self._resolve_source_path(observation.image_id)
                    if source_path is None:
                        missing_files.append(observation.image_id)
                        continue

                    self._copy_unique(
                        source_path,
                        all_directory,
                        observation.image_id,
                        observation.observation_id,
                    )
                    copied_files += 1

                # ----------------------------------------------------
                # BEST IMAGES
                # ----------------------------------------------------
                for candidate, score, pose in self._select_best_images(
                    cluster_observations
                ):
                    source_path = self._resolve_source_path(candidate.image_id)
                    if source_path is None:
                        missing_files.append(candidate.image_id)
                        continue

                    self._copy_unique(
                        source_path,
                        best_directory,
                        candidate.image_id,
                        candidate.observation_id,
                    )
                    copied_files += 1

            representative = self._save_person_representative(
                person_name=person_name,
                observations=person_observations,
                person_directory=person_directory,
            )
        print()
        print("=" * 60)
        print("REFERENCE-MATCHED FINAL OUTPUT")
        print("=" * 60)
        print(f"Output directory: {self.output_root}")
        matched_cluster_count = sum(
            len(cluster_ids) for cluster_ids in person_clusters.values()
        )
        print(f"Known people written: {len(person_clusters)}")
        print(f"Matched clusters aggregated: {matched_cluster_count}")
        print(f"Images copied: {copied_files}")
        if missing_files:
            print(f"Missing source images: {len(set(missing_files))}")

        return self.output_root
