from typing import List

import numpy as np

from app import configuration as config

from app.detection.personDetector import PersonDetector
from app.detection.faceDetector import FaceDetector
from app.detection.association import AssociationEngine

from app.embeddings.bodyEncoder import BodyEncoder

from app.quality.qualityCalculator import QualityCalculator

from app.validation.embeddingValidator import EmbeddingValidator

from app.models.observation import PersonObservation


class PersonPipeline:
    """
    Processes ONE image through the complete person
    observation pipeline.

    Pipeline:

        Image
          ↓
        Person Detection
          ↓
        Face Detection + Face Embedding
          ↓
        Face/Body Association
          ↓
        PersonObservation
          ↓
        Body Embedding
          ↓
        Quality Calculation
          ↓
        Embedding Validation
          ↓
        Observations

    This class does NOT perform:

        - similarity calculation
        - person matching
        - clustering

    Those operations happen after observations have been
    persisted for the event.
    """

    def __init__(self):

        # -----------------------------------------------------
        # Person detector
        # -----------------------------------------------------

        self.person_detector = PersonDetector(
            modelName=config.PERSON_MODEL_NAME,
            confidence_threshold=(config.PERSON_DETECTION_THRESHOLD),
            device=config.PERSON_DETECTION_DEVICE,
        )

        # -----------------------------------------------------
        # Face detector + face embedding model
        # -----------------------------------------------------

        self.face_detector = FaceDetector(
            modelName=config.FACE_MODEL_NAME,
            detection_size=config.FACE_DETECTION_SIZE,
            detection_threshold=config.FACE_DETECTION_THRESHOLD,
            ctx_id=config.FACE_CTX_ID,
        )

        # -----------------------------------------------------
        # Face/body association
        # -----------------------------------------------------

        self.association_engine = AssociationEngine()

        # -----------------------------------------------------
        # Body embedding model
        # -----------------------------------------------------

        self.body_encoder = BodyEncoder(
            model_name=config.BODY_MODEL_NAME,
            model_path=config.BODY_MODEL_PATH,
            device=config.BODY_DEVICE,
            upper_body_ratio=(config.BODY_UPPER_BODY_RATIO),
        )

        # -----------------------------------------------------
        # Quality calculator
        # -----------------------------------------------------

        self.quality_calculator = QualityCalculator()

        # -----------------------------------------------------
        # Embedding validator
        # -----------------------------------------------------

        self.embedding_validator = EmbeddingValidator()

        # -----------------------------------------------------
        # Global observation ID
        # -----------------------------------------------------

        self.next_observation_id = 0

    # =========================================================
    # PROCESS ONE IMAGE
    # =========================================================

    def process_image(
        self,
        image: np.ndarray,
        image_id: str,
    ) -> List[PersonObservation]:
        """
        Process one image.

        Args:
            image:
                Original BGR image loaded by OpenCV.

            image_id:
                Identifier of the image.

        Returns:
            All observations detected in this image.
        """

        if image is None or image.size == 0:
            raise ValueError(f"Invalid image supplied for: {image_id}")

        # =====================================================
        # STEP 1 — PERSON DETECTION
        # =====================================================

        person_detections = self.person_detector.detect(image)

        # =====================================================
        # STEP 2 — FACE DETECTION + FACE EMBEDDINGS
        # =====================================================

        face_detections = self.face_detector.detect(image)

        # =====================================================
        # STEP 3 — FACE/BODY ASSOCIATION
        # =====================================================

        observations = self.association_engine.associate(
            image_id=image_id,
            face_detections=face_detections,
            body_detections=person_detections,
            observation_id_start=(self.next_observation_id),
        )

        # Update global observation ID.
        self.next_observation_id += len(observations)

        # =====================================================
        # STEP 4 — BODY EMBEDDINGS
        # =====================================================
        #
        # Performance optimization:
        #
        # Previously:
        #
        #     observation 1 -> OSNet
        #     observation 2 -> OSNet
        #     observation 3 -> OSNet
        #
        # Now:
        #
        #     observation 1 ┐
        #     observation 2 ├── ONE OSNet batch
        #     observation 3 ┘
        #
        # The embedding calculation itself is unchanged.
        # Only the inference batching is different.
        # =====================================================

        body_bboxes = [observation.person_bbox for observation in observations]

        body_embeddings = self.body_encoder.encode_batch(
            image=image,
            person_bboxes=body_bboxes,
        )

        for observation, body_embedding in zip(
            observations,
            body_embeddings,
        ):
            observation.body_embedding = body_embedding

        # =====================================================
        # STEP 5 — QUALITY CALCULATION
        # =====================================================

        observations = self.quality_calculator.calculate_all(
            image=image,
            observations=observations,
        )

        # =====================================================
        # STEP 6 — EMBEDDING VALIDATION
        # =====================================================

        validation_results = self.embedding_validator.validate_all(observations)

        # -----------------------------------------------------
        # Store validation directly in the observation.
        # -----------------------------------------------------

        for observation, validation in zip(
            observations,
            validation_results,
        ):

            observation.face_embedding_valid = validation["face_valid"]

            observation.body_embedding_valid = validation["body_valid"]

        # =====================================================
        # RETURN
        # =====================================================

        return observations
