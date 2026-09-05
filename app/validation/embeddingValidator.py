from typing import Optional

import numpy as np

from app.models.observation import PersonObservation


class EmbeddingValidator:
    """
    Validates face and body embeddings stored inside
    PersonObservation objects.

    This class does NOT:
        - calculate similarity
        - identify people
        - modify embeddings
        - calculate quality

    Its only responsibility is to determine whether
    an embedding is safe to use in later matching stages.
    """

    # ---------------------------------------------------------
    # Generic embedding validation
    # ---------------------------------------------------------

    @staticmethod
    def validate_embedding(
        embedding: Optional[np.ndarray],
    ) -> bool:
        """
        Validate a single embedding.

        An embedding is considered valid when:

            1. It exists.
            2. It is a NumPy array.
            3. It contains numeric values.
            4. It is not empty.
            5. It contains no NaN values.
            6. It contains no infinite values.
            7. Its L2 norm is greater than zero.

        The embedding is NOT modified.
        """

        # No embedding.
        if embedding is None:
            return False

        # Must be a NumPy array.
        if not isinstance(embedding, np.ndarray):
            return False

        # Must contain data.
        if embedding.size == 0:
            return False

        # Embedding must contain numeric values.
        if not np.issubdtype(embedding.dtype, np.number):
            return False

        # Reject NaN values.
        if not np.all(np.isfinite(embedding)):
            return False

        # Reject zero vectors.
        norm = np.linalg.norm(embedding)

        if not np.isfinite(norm):
            return False

        if norm <= 0:
            return False

        return True

    # ---------------------------------------------------------
    # Face embedding
    # ---------------------------------------------------------

    def validate_face_embedding(
        self,
        observation: PersonObservation,
    ) -> bool:
        """
        Validate the face embedding of one observation.
        """

        return self.validate_embedding(observation.face_embedding)

    # ---------------------------------------------------------
    # Body embedding
    # ---------------------------------------------------------

    def validate_body_embedding(
        self,
        observation: PersonObservation,
    ) -> bool:
        """
        Validate the body embedding of one observation.
        """

        return self.validate_embedding(observation.body_embedding)

    # ---------------------------------------------------------
    # Observation validation
    # ---------------------------------------------------------

    def validate(
        self,
        observation: PersonObservation,
    ) -> dict:
        """
        Validate all embeddings belonging to one observation.

        Returns a dictionary containing the validation
        result for the face and body embeddings.
        """

        face_valid = self.validate_face_embedding(observation)

        body_valid = self.validate_body_embedding(observation)

        return {
            "face_valid": face_valid,
            "body_valid": body_valid,
        }

    # ---------------------------------------------------------
    # Validate all observations
    # ---------------------------------------------------------

    def validate_all(
        self,
        observations: list[PersonObservation],
    ) -> list[dict]:
        """
        Validate embeddings for all observations.

        Returns one validation result dictionary
        for each observation.
        """

        results = []

        for observation in observations:

            result = self.validate(observation)

            results.append(result)

        return results
