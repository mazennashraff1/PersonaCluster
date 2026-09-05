from typing import Optional

import cv2
import numpy as np

from app import configuration as config
from app.models.detector import BoundingBox
from app.models.observation import PersonObservation


class QualityCalculator:
    """
    Calculates face and body embedding quality.

    Quality represents how trustworthy an embedding is for
    identity matching.

    It does NOT decide whether a person observation should
    be removed.

    Quality range:

        0.0 = very poor representation
        1.0 = very good representation

    Current quality signals:

        1. Detection confidence
        2. Bounding-box resolution
        3. Image sharpness
    """

    # ========================================================
    # Generic helpers
    # ========================================================

    @staticmethod
    def clamp(
        value: float,
        minimum: float = config.QUALITY_MIN,
        maximum: float = config.QUALITY_MAX,
    ) -> float:
        """
        Restrict a value to the configured quality range.
        """

        return max(
            minimum,
            min(
                maximum,
                float(value),
            ),
        )

    @staticmethod
    def normalize_range(
        value: float,
        minimum: float,
        maximum: float,
    ) -> float:
        """
        Convert a value from an arbitrary range into [0, 1].

        Values below minimum become 0.
        Values above maximum become 1.
        """

        if maximum <= minimum:
            raise ValueError(
                "Maximum normalization value must be greater " "than minimum."
            )

        if value <= minimum:
            return 0.0

        if value >= maximum:
            return 1.0

        return (value - minimum) / (maximum - minimum)

    # ========================================================
    # Crop extraction
    # ========================================================

    @staticmethod
    def extract_crop(
        image: np.ndarray,
        bbox: Optional[BoundingBox],
    ) -> Optional[np.ndarray]:
        """
        Safely extract a bounding-box crop from an image.
        """

        if image is None or image.size == 0:
            return None

        if bbox is None:
            return None

        image_height, image_width = image.shape[:2]

        x1 = max(0, min(int(bbox.x1), image_width))
        y1 = max(0, min(int(bbox.y1), image_height))

        x2 = max(0, min(int(bbox.x2), image_width))
        y2 = max(0, min(int(bbox.y2), image_height))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        return crop

    # ========================================================
    # Sharpness
    # ========================================================

    @staticmethod
    def calculate_sharpness(
        crop: Optional[np.ndarray],
        minimum: float,
        maximum: float,
    ) -> float:
        """
        Estimate sharpness using Laplacian variance.

        The raw variance is calibrated into [0, 1].

        This means the actual threshold values are controlled
        from configuration.py instead of being hard-coded here.
        """

        if crop is None or crop.size == 0:
            return 0.0

        # Convert to grayscale.
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(
                crop,
                cv2.COLOR_BGR2GRAY,
            )
        else:
            gray = crop

        # Laplacian variance.
        variance = cv2.Laplacian(
            gray,
            cv2.CV_64F,
        ).var()

        return QualityCalculator.normalize_range(
            value=variance,
            minimum=minimum,
            maximum=maximum,
        )

    # ========================================================
    # Face size quality
    # ========================================================

    @staticmethod
    def calculate_face_size_quality(
        face_bbox: Optional[BoundingBox],
    ) -> float:
        """
        Calculate face resolution quality.

        The smaller face dimension is used because a face
        should have sufficient resolution in both directions.
        """

        if face_bbox is None:
            return 0.0

        face_size = min(
            face_bbox.width,
            face_bbox.height,
        )

        return QualityCalculator.normalize_range(
            value=face_size,
            minimum=config.FACE_MIN_SIZE,
            maximum=config.FACE_REFERENCE_SIZE,
        )

    # ========================================================
    # Body size quality
    # ========================================================

    @staticmethod
    def calculate_body_size_quality(
        body_bbox: Optional[BoundingBox],
    ) -> float:
        """
        Calculate body resolution quality.
        """

        if body_bbox is None:
            return 0.0

        body_size = min(
            body_bbox.width,
            body_bbox.height,
        )

        return QualityCalculator.normalize_range(
            value=body_size,
            minimum=config.BODY_MIN_SIZE,
            maximum=config.BODY_REFERENCE_SIZE,
        )

    # ========================================================
    # Face quality
    # ========================================================

    def calculate_face_quality(
        self,
        image: np.ndarray,
        observation: PersonObservation,
    ) -> float:
        """
        Calculate the quality of the face representation.
        """

        if (
            observation.face_bbox is None
            or observation.face_detection_confidence is None
        ):
            return 0.0

        # -----------------------------------------------
        # 1. Detection confidence
        # -----------------------------------------------

        detection_quality = self.clamp(observation.face_detection_confidence)

        # -----------------------------------------------
        # 2. Face resolution
        # -----------------------------------------------

        size_quality = self.calculate_face_size_quality(observation.face_bbox)

        # -----------------------------------------------
        # 3. Face sharpness
        # -----------------------------------------------

        face_crop = self.extract_crop(
            image=image,
            bbox=observation.face_bbox,
        )

        sharpness_quality = self.calculate_sharpness(
            crop=face_crop,
            minimum=config.FACE_SHARPNESS_MIN,
            maximum=config.FACE_SHARPNESS_MAX,
        )

        # -----------------------------------------------
        # Combine
        # -----------------------------------------------

        quality = (
            config.FACE_QUALITY_DETECTION_WEIGHT * detection_quality
            + config.FACE_QUALITY_SIZE_WEIGHT * size_quality
            + config.FACE_QUALITY_SHARPNESS_WEIGHT * sharpness_quality
        )

        return self.clamp(quality)

    # ========================================================
    # Body quality
    # ========================================================

    def calculate_body_quality(
        self,
        image: np.ndarray,
        observation: PersonObservation,
    ) -> float:
        """
        Calculate the quality of the body representation.
        """

        if (
            observation.person_bbox is None
            or observation.person_detection_confidence is None
        ):
            return 0.0

        # -----------------------------------------------
        # 1. Detection confidence
        # -----------------------------------------------

        detection_quality = self.clamp(observation.person_detection_confidence)

        # -----------------------------------------------
        # 2. Body resolution
        # -----------------------------------------------

        size_quality = self.calculate_body_size_quality(observation.person_bbox)

        # -----------------------------------------------
        # 3. Body sharpness
        # -----------------------------------------------

        body_crop = self.extract_crop(
            image=image,
            bbox=observation.person_bbox,
        )

        sharpness_quality = self.calculate_sharpness(
            crop=body_crop,
            minimum=config.BODY_SHARPNESS_MIN,
            maximum=config.BODY_SHARPNESS_MAX,
        )

        # -----------------------------------------------
        # Combine
        # -----------------------------------------------

        quality = (
            config.BODY_QUALITY_DETECTION_WEIGHT * detection_quality
            + config.BODY_QUALITY_SIZE_WEIGHT * size_quality
            + config.BODY_QUALITY_SHARPNESS_WEIGHT * sharpness_quality
        )

        return self.clamp(quality)

    # ========================================================
    # Observation
    # ========================================================

    def calculate(
        self,
        image: np.ndarray,
        observation: PersonObservation,
    ) -> PersonObservation:
        """
        Calculate both face and body quality for one observation.
        """

        observation.face_quality = self.calculate_face_quality(
            image=image,
            observation=observation,
        )

        observation.body_quality = self.calculate_body_quality(
            image=image,
            observation=observation,
        )

        return observation

    # ========================================================
    # Batch
    # ========================================================

    def calculate_all(
        self,
        image: np.ndarray,
        observations: list[PersonObservation],
    ) -> list[PersonObservation]:
        """
        Calculate quality for every observation in one image.
        """

        for observation in observations:
            self.calculate(
                image=image,
                observation=observation,
            )

        return observations
