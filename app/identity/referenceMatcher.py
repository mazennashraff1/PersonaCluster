from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app import configuration as config
from app.detection.faceDetector import FaceDetector


@dataclass(frozen=True)
class ReferenceImage:
    """One reference image and its normalized face embedding."""

    person_name: str
    phone_number: str
    image_path: Path
    embedding: np.ndarray


class ReferenceMatcher:
    """Match final discovered clusters to known people.

    Reference images are named:

        Person Name - Phone Number.ext

    Everything before the first `` - `` is the person's name. The phone
    number is ignored. Multiple reference images for one person are supported.

    This class does not modify the clustering algorithm. It runs only after
    clustering and can assign many cluster IDs to the same person.
    """

    VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    def __init__(self, reference_directory: str | Path | None = None) -> None:
        self.project_root = Path(__file__).resolve().parents[2]
        self.reference_directory = (
            self.project_root / Path(reference_directory or config.REFERENCES_PATH)
        ).resolve(strict=False)

        self.threshold = float(config.REFERENCE_MATCH_THRESHOLD)
        self.min_margin = float(config.REFERENCE_MATCH_MIN_MARGIN)
        self.person_contacts: dict[str, str] = {}

        self.face_detector = FaceDetector(
            modelName=config.FACE_MODEL_NAME,
            detection_size=config.FACE_DETECTION_SIZE,
            detection_threshold=config.FACE_DETECTION_THRESHOLD,
            ctx_id=config.FACE_CTX_ID,
        )

    @staticmethod
    def _person_name_from_filename(path: Path) -> str:
        stem = path.stem.strip()
        if " - " in stem:
            return stem.split(" - ", 1)[0].strip()
        return stem

    @staticmethod
    def _phone_number_from_filename(path: Path) -> str:
        stem = path.stem.strip()
        if " - " in stem:
            return stem.split(" - ", 1)[1].strip()
        return ""

    @staticmethod
    def _normalise(embedding: np.ndarray) -> Optional[np.ndarray]:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0.0:
            return None
        return vector / norm

    @staticmethod
    def _select_reference_face(faces) -> Optional[object]:
        valid = [face for face in faces if getattr(face, "embedding", None) is not None]
        if not valid:
            return None

        def score(face) -> tuple[float, float]:
            confidence = float(getattr(face, "confidence", 0.0) or 0.0)
            bbox = getattr(face, "bbox", None)
            area = 0.0
            if bbox is not None:
                try:
                    x1, y1, x2, y2 = map(float, bbox)
                    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                except (TypeError, ValueError):
                    pass
            return confidence, area

        return max(valid, key=score)

    def load_references(self) -> dict[str, list[ReferenceImage]]:
        """Load and embed all reference images grouped by person name."""
        profiles: dict[str, list[ReferenceImage]] = {}

        if not self.reference_directory.exists():
            print(
                f"[WARNING] Reference directory does not exist: "
                f"{self.reference_directory}"
            )
            return profiles

        paths = sorted(
            path
            for path in self.reference_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in self.VALID_EXTENSIONS
        )

        print()
        print("=" * 60)
        print("LOADING REFERENCE PEOPLE")
        print("=" * 60)
        print(f"Reference directory: {self.reference_directory}")
        print(f"Reference images found: {len(paths)}")

        for image_path in paths:
            person_name = self._person_name_from_filename(image_path)
            phone_number = self._phone_number_from_filename(image_path)
            if not person_name:
                continue

            if phone_number and person_name not in self.person_contacts:
                self.person_contacts[person_name] = phone_number

            image = cv2.imread(str(image_path))
            if image is None:
                print(f"[WARNING] Could not read reference: {image_path}")
                continue

            try:
                faces = self.face_detector.detect(image)
                face = self._select_reference_face(faces)
                if face is None:
                    print(f"[WARNING] No usable face: {image_path.name}")
                    continue

                embedding = self._normalise(face.embedding)
                if embedding is None:
                    print(f"[WARNING] Invalid embedding: {image_path.name}")
                    continue

                profiles.setdefault(person_name, []).append(
                    ReferenceImage(
                        person_name=person_name,
                        phone_number=phone_number,
                        image_path=image_path,
                        embedding=embedding,
                    )
                )
                print(f"  {person_name}: {image_path.name}")
            finally:
                del image

        print(f"Known people loaded: {len(profiles)}")
        return profiles

    def get_person_contacts(self) -> dict[str, str]:
        """Return reference person -> phone number mappings loaded from filenames."""
        return dict(self.person_contacts)

    def match_clusters(
        self,
        observations,
        assignments: dict[int, int | None],
    ) -> tuple[dict[int, str], dict[int, dict[str, object]]]:
        """Match each discovered cluster to at most one known person.

        A single person can receive any number of cluster IDs. This is the
        intended behavior for separate frontal, side, profile, etc. clusters.
        """
        profiles = self.load_references()
        if not profiles:
            return {}, {}

        clusters: dict[int, list[object]] = {}
        for observation in observations:
            cluster_id = assignments.get(observation.observation_id)
            if cluster_id is None:
                continue
            if not observation.face_embedding_valid:
                continue
            if observation.face_embedding is None:
                continue
            clusters.setdefault(int(cluster_id), []).append(observation)

        matches: dict[int, str] = {}
        details: dict[int, dict[str, object]] = {}

        # Normalize event embeddings once instead of once per reference.
        normalized_observations: dict[int, np.ndarray] = {}
        for cluster_observations in clusters.values():
            for observation in cluster_observations:
                vector = self._normalise(observation.face_embedding)
                if vector is not None:
                    normalized_observations[observation.observation_id] = vector

        for cluster_id, cluster_observations in sorted(clusters.items()):
            ranked: list[tuple[float, str, Path]] = []

            for person_name, references in profiles.items():
                best_score = -1.0
                best_reference: Optional[Path] = None

                for reference in references:
                    for observation in cluster_observations:
                        event_embedding = normalized_observations.get(
                            observation.observation_id
                        )
                        if event_embedding is None:
                            continue

                        score = float(np.dot(reference.embedding, event_embedding))
                        if score > best_score:
                            best_score = score
                            best_reference = reference.image_path

                if best_reference is not None:
                    ranked.append((best_score, person_name, best_reference))

            ranked.sort(key=lambda item: (-item[0], item[1].casefold()))
            if not ranked:
                continue

            best_score, best_person, best_reference = ranked[0]
            second_score = ranked[1][0] if len(ranked) > 1 else -1.0
            margin = best_score - second_score if second_score >= -1.0 else 1.0

            accepted = best_score >= self.threshold and (
                len(ranked) == 1 or margin >= self.min_margin
            )

            details[cluster_id] = {
                "person_name": best_person if accepted else None,
                "score": round(float(best_score), 6),
                "margin": round(float(margin), 6),
                "reference_image": str(best_reference) if best_reference else None,
                "accepted": accepted,
                "candidate_count": len(ranked),
            }

            if accepted:
                matches[cluster_id] = best_person
                print(
                    f"Cluster {cluster_id + 1:02d} -> {best_person} "
                    f"(score={best_score:.3f}, margin={margin:.3f})"
                )
            else:
                print(
                    f"Cluster {cluster_id + 1:02d} -> UNKNOWN/AMBIGUOUS "
                    f"(best={best_score:.3f}, margin={margin:.3f})"
                )

        return matches, details
