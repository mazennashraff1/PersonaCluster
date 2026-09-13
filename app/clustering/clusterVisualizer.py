"""
Cluster Visualization
=====================

Visual validation tool for the event-based person clustering system.

Output structure:

<event_path>/cluster_visualization/
│
├── representatives/
│   ├── cluster_000.jpg
│   ├── cluster_001.jpg
│   └── ...
│
├── cluster_000/
│   ├── IMG_001.jpg
│   ├── IMG_004.jpg
│   └── ...
│
├── cluster_001/
│   ├── IMG_002.jpg
│   └── ...
│
└── noise/
    ├── IMG_010.jpg
    └── ...

IMPORTANT:

1. Original images are copied into their cluster folders.
2. The representatives folder contains ONE FACE per cluster.
3. Each representative is a SINGLE cropped face.
4. NO contact sheets are generated.
5. NO combined images are generated.
6. Clustering itself is NOT performed here.
7. This file only visualizes the clustering result.
"""

from __future__ import annotations

import shutil
from collections import defaultdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from app.models.observation import PersonObservation

# ============================================================
# Configuration
# ============================================================

FACE_PADDING_RATIO = 0.25


# ============================================================
# Image lookup
# ============================================================


def build_image_index(
    event_path: str | Path,
) -> dict[str, Path]:
    """
    Build a full-path image index.

    The current EventStore uses the canonical absolute path as image_id,
    so the index must use the same identity. Basenames are intentionally
    not used as the primary key because duplicate filenames are allowed.
    """

    event_path = Path(event_path).resolve(strict=False)
    image_index: dict[str, Path] = {}

    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

    for path in event_path.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() not in valid_extensions:
            continue

        canonical = path.resolve(strict=False)
        image_index[str(canonical)] = canonical

    print(f"Indexed {len(image_index)} image file(s) recursively.")
    return image_index


def resolve_source_image(
    image_id: str,
    event_path: str | Path,
    image_index: dict[str, Path],
) -> Optional[Path]:
    """Resolve a stored image_id to the exact source image."""
    candidate = Path(str(image_id))

    if candidate.is_file():
        return candidate.resolve(strict=False)

    event_path = Path(event_path).resolve(strict=False)
    canonical = str(candidate.resolve(strict=False))
    indexed = image_index.get(canonical)
    if indexed is not None and indexed.is_file():
        return indexed

    relative = event_path / candidate
    if relative.is_file():
        return relative.resolve(strict=False)

    basename = candidate.name
    matches = [
        path
        for path in image_index.values()
        if path.name == basename and path.is_file()
    ]
    if len(matches) == 1:
        return matches[0]

    return None


# ============================================================
# Crop face
# ============================================================


def crop_face(
    image: np.ndarray,
    observation: PersonObservation,
) -> Optional[np.ndarray]:
    """
    Crop the detected face from an observation with padding.

    Returns None when the image or face bounding box is invalid.
    """
    if image is None or image.size == 0:
        return None

    bbox = observation.face_bbox
    if bbox is None:
        return None

    image_height, image_width = image.shape[:2]

    x1 = int(bbox.x1)
    y1 = int(bbox.y1)
    x2 = int(bbox.x2)
    y2 = int(bbox.y2)

    if x2 <= x1 or y2 <= y1:
        return None

    face_width = x2 - x1
    face_height = y2 - y1

    padding_x = int(face_width * FACE_PADDING_RATIO)
    padding_y = int(face_height * FACE_PADDING_RATIO)

    x1 = max(0, x1 - padding_x)
    y1 = max(0, y1 - padding_y)
    x2 = min(image_width, x2 + padding_x)
    y2 = min(image_height, y2 + padding_y)

    if x2 <= x1 or y2 <= y1:
        return None

    face = image[y1:y2, x1:x2]

    if face.size == 0:
        return None

    return face


# ============================================================
# Representative score
# ============================================================


def calculate_representative_score(
    observation: PersonObservation,
) -> float:
    """
    Calculate how suitable an observation is as the
    representative of its cluster.

    Higher score = better representative.

    Current weighting:

        70% -> face quality
        30% -> face detection confidence
    """

    if not observation.face_embedding_valid:
        return -1.0

    if observation.face_bbox is None:
        return -1.0

    if observation.face_quality is not None:

        face_quality = float(observation.face_quality)

    else:

        face_quality = 0.0

    if observation.face_detection_confidence is not None:

        detection_confidence = float(observation.face_detection_confidence)

    else:

        detection_confidence = 0.0

    score = 0.70 * face_quality + 0.30 * detection_confidence

    return score


# ============================================================
# Select best representative
# ============================================================


def select_best_representative(
    cluster_observations: list[PersonObservation],
) -> Optional[PersonObservation]:

    candidates = [
        observation
        for observation in cluster_observations
        if (observation.face_embedding_valid and observation.face_bbox is not None)
    ]

    if not candidates:
        return None

    best_observation = max(
        candidates,
        key=calculate_representative_score,
    )

    return best_observation


# ============================================================
# Save ONE representative face
# ============================================================


def save_cluster_representative(
    cluster_id: int,
    cluster_observations: list[PersonObservation],
    image_index: dict[str, Path],
    representatives_dir: Path,
    event_path: str | Path,
) -> Optional[Path]:

    best_observation = select_best_representative(cluster_observations)

    if best_observation is None:

        print(f"[WARNING] Cluster {cluster_id} " f"has no valid face representative.")

        return None

    image_path = resolve_source_image(
        best_observation.image_id, event_path, image_index
    )

    if image_path is None:

        print(
            f"[WARNING] Could not find original image "
            f"'{best_observation.image_id}' "
            f"for cluster {cluster_id}."
        )

        return None

    # --------------------------------------------------------
    # Read the EXACT original image.
    # --------------------------------------------------------

    image = cv2.imread(str(image_path))

    if image is None:

        print(f"[WARNING] Could not read image: " f"{image_path}")

        return None

    face = crop_face(
        image=image,
        observation=best_observation,
    )

    if face is None:

        print(
            f"[WARNING] Could not crop face from "
            f"observation "
            f"{best_observation.observation_id}."
        )

        del image

        return None

    output_path = representatives_dir / f"cluster_{cluster_id:03d}.jpg"

    success = cv2.imwrite(
        str(output_path),
        face,
    )

    if not success:

        print(f"[WARNING] Failed to save representative: " f"{output_path}")

        del face
        del image

        return None

    score = calculate_representative_score(best_observation)

    print(f"  Cluster {cluster_id} representative:")

    print(f"      observation : " f"{best_observation.observation_id}")

    print(f"      source image: " f"{best_observation.image_id}")

    print(f"      source path : " f"{image_path}")

    print(
        f"      face quality: " f"{float(best_observation.face_quality):.3f}"
        if best_observation.face_quality is not None
        else "      face quality: None"
    )

    print(f"      score       : " f"{score:.3f}")

    print(f"      saved       : " f"{output_path}")

    # ========================================================
    # Release memory.
    # ========================================================

    del face
    del image

    return output_path


# ============================================================
# Copy original images into cluster folder
# ============================================================


def copy_cluster_images(
    cluster_id: Optional[int],
    cluster_observations: list[PersonObservation],
    image_index: dict[str, Path],
    output_dir: Path,
    event_path: str | Path,
) -> Path:
    """
    Copy the ORIGINAL image files into their cluster folder.

    IMPORTANT:

    The original folder structure is NOT reproduced.

    Example source:

        EVENT_PATH/
            camera_1/
                morning/
                    IMG_001.jpg

    Destination:

        cluster_visualization/
            cluster_000/
                IMG_001.jpg

    Only the actual image file is copied.

    The destination is deliberately constructed as:

        cluster_dir / image_path.name

    rather than using the complete source path.
    """

    # --------------------------------------------------------
    # Determine destination folder.
    # --------------------------------------------------------

    if cluster_id is None:

        cluster_dir = output_dir / "noise"

    else:

        cluster_dir = output_dir / f"cluster_{cluster_id:03d}"

    cluster_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Keep track of images already copied.
    # --------------------------------------------------------

    copied_images: set[str] = set()

    for observation in cluster_observations:

        image_id = observation.image_id

        if image_id in copied_images:
            continue

        # ----------------------------------------------------
        # Find the EXACT source image.
        # ----------------------------------------------------

        image_path = resolve_source_image(image_id, event_path, image_index)

        if image_path is None:

            print(f"  [WARNING] Original image not found: " f"{image_id}")

            continue

        # ====================================================
        # IMPORTANT
        #
        # ONLY THE IMAGE NAME is used here.
        #
        # We do NOT do:
        #
        #     cluster_dir / image_path
        #
        # because that could reproduce the complete source
        # directory structure.
        #
        # Instead:
        #
        #     image_path.name
        #
        # gives only:
        #
        #     IMG_001.jpg
        #
        # ====================================================

        destination = cluster_dir / image_path.name

        try:

            # ------------------------------------------------
            # copy2 copies the actual file.
            #
            # It does NOT copy the parent folders.
            # ------------------------------------------------

            shutil.copy2(
                image_path,
                destination,
            )

            copied_images.add(image_id)

            print(f"  Copied image: " f"{image_path.name}")

        except Exception as exc:

            print(f"  [WARNING] Could not copy " f"{image_id}: {exc}")

    return cluster_dir


# ============================================================
# Main visualization function
# ============================================================


def visualizeClusters(
    event_path: str | Path,
    observations: list[PersonObservation],
    cluster_assignments: dict[int, Optional[int]],
    output_dir: str | Path | None = None,
    columns: int = 5,
    max_items_per_cluster: int = 100,
) -> Path:
    """
    Create visual validation output.

    There are NO contact sheets.

    Each cluster contains the original image files.

    Representatives contain exactly ONE cropped face.
    """

    event_path = Path(event_path)

    if output_dir is None:

        output_dir = event_path / "cluster_visualization"

    else:

        output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Representatives directory.
    # ========================================================

    representatives_dir = output_dir / "representatives"

    representatives_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Start logging.
    # ========================================================

    print()
    print("=" * 60)
    print("CREATING CLUSTER VISUALIZATION")
    print("=" * 60)

    print()
    print("Representative mode:")

    print("  ONE FACE PER CLUSTER")

    print("  NO CONTACT SHEETS")

    print("  NO COMBINED IMAGES")

    print()

    # ========================================================
    # Build recursive image index.
    # ========================================================

    image_index = build_image_index(event_path)

    # ========================================================
    # Group observations by cluster.
    # ========================================================

    grouped: dict[
        Optional[int],
        list[PersonObservation],
    ] = defaultdict(list)

    for observation in observations:

        cluster_id = cluster_assignments.get(observation.observation_id)

        grouped[cluster_id].append(observation)

    # ========================================================
    # Get cluster IDs.
    #
    # Normal clusters first.
    # Noise last.
    # ========================================================

    cluster_ids = [
        cluster_id for cluster_id in grouped.keys() if cluster_id is not None
    ]

    cluster_ids.sort()

    if None in grouped:

        cluster_ids.append(None)

    # ========================================================
    # Process every cluster.
    # ========================================================

    representative_count = 0

    for cluster_id in cluster_ids:

        cluster_observations = grouped[cluster_id]

        # ----------------------------------------------------
        # Sort observations.
        # ----------------------------------------------------

        cluster_observations.sort(
            key=lambda observation: (
                observation.image_id,
                observation.observation_id,
            )
        )

        # ====================================================
        # STEP 1
        #
        # Copy ORIGINAL images.
        #
        # Only the image file is copied.
        # The source directory structure is NOT copied.
        # ====================================================

        cluster_folder = copy_cluster_images(
            cluster_id=cluster_id,
            cluster_observations=cluster_observations,
            image_index=image_index,
            output_dir=output_dir,
            event_path=event_path,
        )

        # ====================================================
        # STEP 2
        #
        # Create ONE face representative.
        #
        # Noise does not get a representative.
        # ====================================================

        if cluster_id is not None:

            representative_path = save_cluster_representative(
                cluster_id=cluster_id,
                cluster_observations=cluster_observations,
                image_index=image_index,
                representatives_dir=representatives_dir,
                event_path=event_path,
            )

            if representative_path is not None:

                representative_count += 1

        # ====================================================
        # STEP 3
        # Logging.
        # ====================================================

        unique_images = len(
            {observation.image_id for observation in cluster_observations}
        )

        label = "noise" if cluster_id is None else f"cluster_{cluster_id:03d}"

        print(
            f"{label:15s} | "
            f"observations={len(cluster_observations):3d} | "
            f"images={unique_images:3d} | "
            f"folder={cluster_folder}"
        )

    # ========================================================
    # Finish.
    # ========================================================

    print()
    print("=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)

    print(f"Representatives created: " f"{representative_count}")

    print("Representative directory:")

    print(f"  {representatives_dir}")

    print("Cluster directory:")

    print(f"  {output_dir}")

    print("=" * 60)

    return output_dir
