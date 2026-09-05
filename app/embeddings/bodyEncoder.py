from typing import List, Optional

import numpy as np

import torch
from torchreid.utils import FeatureExtractor

from app.models.detector import BoundingBox


class BodyEncoder:
    """
    Generates a body/appearance embedding using OSNet.

    Pipeline:

        Person bounding box
                ↓
        Upper-body crop
                ↓
        OSNet Re-ID model
                ↓
        L2-normalized appearance embedding

    This is NOT face recognition.

    Face identity information:
        InsightFace

    Body/appearance information:
        OSNet
    """

    def __init__(
        self,
        model_name: str,
        model_path: Optional[str],
        device: Optional[str],
        upper_body_ratio: float,
    ):
        """
        Initialize the body encoder.

        Args:
            model_name:
                Torchreid model name.

            model_path:
                Optional path to custom pretrained weights.

            device:
                "cuda" or "cpu".

                If None, automatically use CUDA when
                PyTorch CUDA is available, otherwise CPU.

            upper_body_ratio:
                Portion of the person bounding box used
                for the appearance crop.

                0.60 means upper 60%.
        """

        if not 0.0 < upper_body_ratio <= 1.0:
            raise ValueError("upper_body_ratio must be between 0 and 1.")

        self.upper_body_ratio = upper_body_ratio

        # -----------------------------------------------------
        # Select device automatically.
        # -----------------------------------------------------

        if device is None:

            device = "cuda" if torch.cuda.is_available() else "cpu"

        if device == "cuda" and not torch.cuda.is_available():

            print(
                "CUDA requested but PyTorch CUDA is unavailable. "
                "Falling back to CPU."
            )

            device = "cpu"

        self.device = device

        print(f"BodyEncoder device: {self.device}")

        # -----------------------------------------------------
        # Create Torchreid feature extractor.
        # -----------------------------------------------------

        extractor_kwargs = {
            "model_name": model_name,
            "device": self.device,
        }

        if model_path is not None:

            extractor_kwargs["model_path"] = model_path

        self.extractor = FeatureExtractor(**extractor_kwargs)

    # ---------------------------------------------------------
    # Crop extraction
    # ---------------------------------------------------------

    def extract_upper_body(
        self,
        image: np.ndarray,
        person_bbox: BoundingBox,
    ) -> Optional[np.ndarray]:
        """
        Extract the upper portion of the detected person.

        The crop is clipped to the image boundaries.

        Returns:
            BGR NumPy image or None if invalid.
        """

        if image is None or image.size == 0:
            return None

        image_height, image_width = image.shape[:2]

        # -----------------------------------------------------
        # Clamp bbox to image boundaries.
        # -----------------------------------------------------

        x1 = max(
            0,
            min(
                int(person_bbox.x1),
                image_width,
            ),
        )

        y1 = max(
            0,
            min(
                int(person_bbox.y1),
                image_height,
            ),
        )

        x2 = max(
            0,
            min(
                int(person_bbox.x2),
                image_width,
            ),
        )

        y2 = max(
            0,
            min(
                int(person_bbox.y2),
                image_height,
            ),
        )

        if x2 <= x1 or y2 <= y1:
            return None

        person_height = y2 - y1

        # -----------------------------------------------------
        # Extract upper-body region.
        # -----------------------------------------------------

        crop_height = max(
            1,
            int(person_height * self.upper_body_ratio),
        )

        crop_y2 = min(
            image_height,
            y1 + crop_height,
        )

        crop = image[
            y1:crop_y2,
            x1:x2,
        ]

        if crop.size == 0:
            return None

        return crop

        # ---------------------------------------------------------

    # Batch encoding
    # ---------------------------------------------------------

    def encode_batch(
        self,
        image: np.ndarray,
        person_bboxes: List[Optional[BoundingBox]],
    ) -> List[Optional[np.ndarray]]:
        """
        Generate body embeddings for multiple person bounding boxes
        in one Torchreid inference call.

        This does NOT change the embedding logic.

        The only difference from encode() is that multiple crops
        are passed to Torchreid together instead of one at a time.
        """

        # -----------------------------------------------------
        # Prepare crops.
        #
        # Keep the original order so that:
        #
        #   result[i] -> person_bboxes[i]
        #
        # -----------------------------------------------------

        crops = []
        valid_indices = []

        for index, person_bbox in enumerate(person_bboxes):

            if person_bbox is None:
                continue

            crop = self.extract_upper_body(
                image=image,
                person_bbox=person_bbox,
            )

            if crop is None:
                continue

            crops.append(crop)
            valid_indices.append(index)

        # -----------------------------------------------------
        # Prepare output.
        #
        # Every input bbox gets one output position.
        # Invalid / missing crops remain None.
        # -----------------------------------------------------

        embeddings: List[Optional[np.ndarray]] = [None for _ in person_bboxes]

        if not crops:
            return embeddings

        # -----------------------------------------------------
        # ONE Torchreid inference call for all crops.
        # -----------------------------------------------------

        features = self.extractor(crops)

        # -----------------------------------------------------
        # Convert each feature to the same normalized format
        # produced by the original encode() method.
        # -----------------------------------------------------

        for feature_index, feature in enumerate(features):

            embedding = feature.detach().cpu().numpy()

            embedding = np.asarray(
                embedding,
                dtype=np.float32,
            )

            norm = np.linalg.norm(embedding)

            if norm <= 0:
                continue

            embedding = embedding / norm

            original_index = valid_indices[feature_index]

            embeddings[original_index] = embedding

        return embeddings

    # ---------------------------------------------------------
    # Encoding
    # ---------------------------------------------------------

    def encode(
        self,
        image: np.ndarray,
        person_bbox: Optional[BoundingBox],
    ) -> Optional[np.ndarray]:
        """
        Generate an L2-normalized body embedding.

        If there is no person bounding box, return None.

        This is what allows face-only observations to remain
        face-only.
        """

        if person_bbox is None:
            return None

        crop = self.extract_upper_body(
            image=image,
            person_bbox=person_bbox,
        )

        if crop is None:
            return None

        # -----------------------------------------------------
        # Torchreid handles the model preprocessing and
        # feature extraction.
        # -----------------------------------------------------

        features = self.extractor([crop])

        # Convert tensor → NumPy.
        embedding = features[0].detach().cpu().numpy()

        embedding = np.asarray(
            embedding,
            dtype=np.float32,
        )

        # -----------------------------------------------------
        # L2 normalize.
        #
        # This makes cosine similarity equivalent to the
        # dot product later.
        # -----------------------------------------------------

        norm = np.linalg.norm(embedding)

        if norm <= 0:
            return None

        embedding = embedding / norm

        return embedding
