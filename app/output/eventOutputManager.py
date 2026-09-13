from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from app.models.observation import PersonObservation
from app.selection.bestImageSelector import BestImageSelection


class EventOutputManager:
    """
    Build the final on-disk output for a processed event.

    Supported modes:

        CLUSTERING_ONLY
            clusterXX/allImages/

        BEST_IMAGES_ONLY
            clusterXX/bestImages/

        BOTH
            clusterXX/allImages/
            clusterXX/bestImages/

    The manager never changes clustering assignments. It only copies the
    already-discovered source images into the requested output structure.
    """

    VALID_MODES = {
        "CLUSTERING_ONLY",
        "BEST_IMAGES_ONLY",
        "BOTH",
    }

    def __init__(
        self,
        event_path: str | Path,
        output_directory_name: str = "output",
        mode: str = "BOTH",
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
    def _canonical_image_id(path: str | Path) -> str:
        return str(Path(path).resolve(strict=False))

    def _resolve_source_path(
        self,
        image_id: str,
        source_images: dict[str, Path] | None = None,
    ) -> Path | None:
        """Resolve an observation image to the real source file.

        The EventStore now stores the canonical full path as ``image_id``.
        Therefore the primary lookup is the ID itself. The ``source_images``
        index is only a compatibility fallback for legacy/relative IDs.
        """

        # ------------------------------------------------------------
        # 1. New format: image_id IS the canonical full path.
        # ------------------------------------------------------------
        candidate = Path(str(image_id))

        try:
            if candidate.is_file():
                return candidate.resolve(strict=False)
        except OSError:
            pass

        # ------------------------------------------------------------
        # 2. Compatibility with the source-image index.
        # ------------------------------------------------------------
        if source_images:
            canonical_id = self._canonical_image_id(image_id)
            source_path = source_images.get(canonical_id)
            if source_path is not None:
                source_path = Path(source_path)
                try:
                    if source_path.is_file():
                        return source_path.resolve(strict=False)
                except OSError:
                    pass

            # Compare normalized absolute paths instead of raw strings.
            for key, path in source_images.items():
                try:
                    if self._canonical_image_id(key) == canonical_id:
                        path = Path(path)
                        if path.is_file():
                            return path.resolve(strict=False)
                except OSError:
                    continue

        # ------------------------------------------------------------
        # 3. Legacy relative image_id.
        # ------------------------------------------------------------
        relative_candidate = self.event_path / candidate
        try:
            if relative_candidate.is_file():
                return relative_candidate.resolve(strict=False)
        except OSError:
            pass

        # ------------------------------------------------------------
        # 4. Last-resort basename search. Only accept an unambiguous
        #    match because duplicate filenames are explicitly allowed.
        # ------------------------------------------------------------
        basename = candidate.name
        if basename:
            matches = []

            if source_images:
                for path in source_images.values():
                    path = Path(path)
                    if path.name == basename and path.is_file():
                        matches.append(path.resolve(strict=False))
            else:
                valid_extensions = {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp",
                    ".tif",
                    ".tiff",
                }
                for path in self.event_path.rglob("*"):
                    if (
                        path.is_file()
                        and path.suffix.lower() in valid_extensions
                        and path.name == basename
                    ):
                        matches.append(path.resolve(strict=False))

            # Deduplicate normalized paths.
            unique_matches = {str(path): path for path in matches}
            if len(unique_matches) == 1:
                return next(iter(unique_matches.values()))

        return None

    @staticmethod
    def _safe_output_name(image_id: str, observation_id: int) -> str:
        """
        Keep the original filename whenever possible.

        Observation ID is only appended when the caller needs a collision-safe
        fallback. Normal event processing uses image_id as the unique image key.
        """
        name = Path(str(image_id)).name
        if not name:
            return f"observation_{observation_id}.jpg"
        return name

    @staticmethod
    def _unique_by_image(
        observations: Iterable[PersonObservation],
        assignments: dict[int, int | None],
    ) -> dict[int, list[PersonObservation]]:
        """Return one source observation per image for each confirmed cluster."""
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

            # Prefer the strongest stored face-quality signal when duplicate
            # observations from the same image survive in the event store.
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
        best_selections: dict[int, BestImageSelection] | None,
        source_images: dict[str, Path] | None = None,
    ) -> Path:
        """
        Write the requested final event output.

        Args:
            observations:
                All observations loaded from EventStore.
            assignments:
                Final cluster assignment by observation ID.
            best_selections:
                Patch 4's best-image selections. Required for BEST_IMAGES_ONLY
                and BOTH.
            source_images:
                image_id -> actual source file path.
        """
        if self.mode in {"BEST_IMAGES_ONLY", "BOTH"} and best_selections is None:
            raise ValueError(
                "best_selections is required for BEST_IMAGES_ONLY/BOTH output."
            )

        self._prepare_output_root()

        grouped = self._unique_by_image(observations, assignments)
        cluster_ids = sorted(grouped)

        if self.mode == "BEST_IMAGES_ONLY":
            cluster_ids = sorted(best_selections or {})

        copied_files = 0
        missing_files: list[str] = []
        manifest: dict[str, object] = {
            "event": self.event_path.name,
            "mode": self.mode,
            "clusters": {},
        }

        for cluster_id in cluster_ids:
            cluster_directory = self.output_root / f"cluster{cluster_id + 1:02d}"
            cluster_directory.mkdir(parents=True, exist_ok=True)

            cluster_manifest: dict[str, object] = {
                "cluster_id": cluster_id,
                "allImages": [],
                "bestImages": [],
            }

            # ------------------------------------------------------------
            # ALL IMAGES
            # ------------------------------------------------------------
            if self.mode in {"CLUSTERING_ONLY", "BOTH"}:
                all_directory = cluster_directory / "allImages"
                all_directory.mkdir(parents=True, exist_ok=True)

                for observation in sorted(
                    grouped.get(cluster_id, []),
                    key=lambda item: (item.image_id, item.observation_id),
                ):
                    source_path = self._resolve_source_path(
                        observation.image_id,
                        source_images,
                    )
                    if source_path is None or not source_path.is_file():
                        missing_files.append(observation.image_id)
                        continue

                    destination = all_directory / self._safe_output_name(
                        observation.image_id,
                        observation.observation_id,
                    )
                    shutil.copy2(source_path, destination)
                    copied_files += 1

                    cluster_manifest["allImages"].append(
                        {
                            "image_id": observation.image_id,
                            "observation_id": observation.observation_id,
                            "pose": observation.face_pose or "unknown",
                        }
                    )

            # ------------------------------------------------------------
            # BEST IMAGES
            # ------------------------------------------------------------
            if self.mode in {"BEST_IMAGES_ONLY", "BOTH"}:
                best_directory = cluster_directory / "bestImages"
                best_directory.mkdir(parents=True, exist_ok=True)

                selection = (best_selections or {}).get(cluster_id)
                if selection is not None:
                    for candidate in selection.candidates:
                        source_path = self._resolve_source_path(
                            candidate.image_id,
                            source_images,
                        )
                        if source_path is None or not source_path.is_file():
                            missing_files.append(candidate.image_id)
                            continue

                        destination = best_directory / self._safe_output_name(
                            candidate.image_id,
                            candidate.observation_id,
                        )
                        shutil.copy2(source_path, destination)
                        copied_files += 1

                        cluster_manifest["bestImages"].append(
                            {
                                "image_id": candidate.image_id,
                                "observation_id": candidate.observation_id,
                                "pose": candidate.pose,
                                "score": round(float(candidate.score), 6),
                            }
                        )

            manifest["clusters"][str(cluster_id + 1)] = cluster_manifest

        manifest["copied_files"] = copied_files
        manifest["missing_source_files"] = sorted(set(missing_files))

        manifest_path = self.output_root / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print()
        print("=" * 60)
        print("FINAL EVENT OUTPUT")
        print("=" * 60)
        print(f"Output mode: {self.mode}")
        print(f"Output directory: {self.output_root}")
        print(f"Clusters written: {len(cluster_ids)}")
        print(f"Images copied: {copied_files}")

        if missing_files:
            print(f"Missing source images: {len(set(missing_files))}")
            for image_id in sorted(set(missing_files)):
                print(f"  - {image_id}")

        print(f"Manifest: {manifest_path}")

        return self.output_root
