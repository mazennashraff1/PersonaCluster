from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import heapq

import numpy as np

from app import configuration as config
from app.models.observation import PersonObservation

# ============================================================
# Identity Cluster
# ============================================================


@dataclass
class IdentityCluster:
    """
    Internal representation of one candidate real-world identity.

    The image_ids set is the hard constraint that prevents two
    observations from the same source image from becoming the
    same identity cluster.
    """

    cluster_id: int
    observations: list[PersonObservation] = field(default_factory=list)

    @property
    def image_ids(self) -> set[str]:
        return {observation.image_id for observation in self.observations}

    @property
    def size(self) -> int:
        return len(self.observations)


# ============================================================
# Statistics
# ============================================================


@dataclass
class ClusterRunStats:
    total_observations: int
    valid_face_observations: int
    clustered_observations: int
    unknown_observations: int
    cluster_count: int
    rejected_same_image_merges: int
    rejected_low_similarity_merges: int


# ============================================================
# Constrained Identity Clustering
# ============================================================


class ConstrainedIdentityClustering:
    """
    Quality-aware, constrained face + body identity clustering.

    IMPORTANT:

    This class intentionally preserves the existing clustering
    decision logic.

    The optimization is primarily computational:

        OLD:
            repeatedly scan every cluster pair

        NEW:
            calculate pair scores once and keep them in a
            priority queue.

    The following identity rules remain unchanged:

        1. Face similarity is the primary signal.
        2. Body similarity is supporting evidence.
        3. Quality is supporting evidence.
        4. Strong observations are used as representatives.
        5. Same-image observations can NEVER belong to the
           same identity cluster.
        6. The strongest compatible pair is merged first.
        7. MERGE_THRESHOLD is unchanged.
    """

    def __init__(self) -> None:

        # --------------------------------------------------------
        # Configuration
        # --------------------------------------------------------

        self.merge_threshold = float(config.MERGE_THRESHOLD)

        self.min_face_similarity = float(config.MIN_FACE_SIMILARITY)

        self.min_cluster_size = int(config.MIN_CLUSTER_SIZE)

        self.representative_count = int(config.REPRESENTATIVE_COUNT)

        self.face_weight = float(config.CLUSTER_FACE_WEIGHT)

        self.body_weight = float(config.CLUSTER_BODY_WEIGHT)

        self.quality_weight = float(config.CLUSTER_QUALITY_WEIGHT)

        self.face_only_weight = float(config.CLUSTER_FACE_ONLY_WEIGHT)

        self.face_only_quality_weight = float(config.CLUSTER_FACE_ONLY_QUALITY_WEIGHT)

        self.representative_face_quality_weight = float(
            config.REPRESENTATIVE_FACE_QUALITY_WEIGHT
        )

        self.representative_face_detection_weight = float(
            config.REPRESENTATIVE_FACE_DETECTION_WEIGHT
        )

        self.last_stats: Optional[ClusterRunStats] = None

        # --------------------------------------------------------
        # Performance caches.
        #
        # These caches DO NOT change any clustering rules.
        #
        # They simply prevent the same observation data from
        # being normalized / scored repeatedly.
        # --------------------------------------------------------

        self._face_embedding_cache: dict[
            int,
            Optional[np.ndarray],
        ] = {}

        self._body_embedding_cache: dict[
            int,
            Optional[np.ndarray],
        ] = {}

        self._quality_cache: dict[
            int,
            float,
        ] = {}

        # --------------------------------------------------------
        # Representative cache.
        #
        # Key:
        #     cluster_id
        #
        # Value:
        #     current representatives
        #
        # This is invalidated only when a cluster changes.
        # --------------------------------------------------------

        self._representative_cache: dict[
            int,
            list[PersonObservation],
        ] = {}

        self._validate_configuration()

    # ============================================================
    # Configuration validation
    # ============================================================

    def _validate_configuration(self) -> None:

        if not 0.0 <= self.merge_threshold <= 1.0:
            raise ValueError("MERGE_THRESHOLD must be between 0 and 1.")

        if not -1.0 <= self.min_face_similarity <= 1.0:
            raise ValueError("MIN_FACE_SIMILARITY must be between -1 and 1.")

        if self.min_cluster_size < 1:
            raise ValueError("MIN_CLUSTER_SIZE must be at least 1.")

        if self.representative_count < 1:
            raise ValueError("REPRESENTATIVE_COUNT must be at least 1.")

        for name, value in (
            (
                "CLUSTER_FACE_WEIGHT",
                self.face_weight,
            ),
            (
                "CLUSTER_BODY_WEIGHT",
                self.body_weight,
            ),
            (
                "CLUSTER_QUALITY_WEIGHT",
                self.quality_weight,
            ),
            (
                "CLUSTER_FACE_ONLY_WEIGHT",
                self.face_only_weight,
            ),
            (
                "CLUSTER_FACE_ONLY_QUALITY_WEIGHT",
                self.face_only_quality_weight,
            ),
            (
                "REPRESENTATIVE_FACE_QUALITY_WEIGHT",
                self.representative_face_quality_weight,
            ),
            (
                "REPRESENTATIVE_FACE_DETECTION_WEIGHT",
                self.representative_face_detection_weight,
            ),
        ):

            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")

        full_signal_total = self.face_weight + self.body_weight + self.quality_weight

        face_only_total = self.face_only_weight + self.face_only_quality_weight

        representative_total = (
            self.representative_face_quality_weight
            + self.representative_face_detection_weight
        )

        if not np.isclose(
            full_signal_total,
            1.0,
        ):
            raise ValueError(
                "CLUSTER_FACE_WEIGHT + "
                "CLUSTER_BODY_WEIGHT + "
                "CLUSTER_QUALITY_WEIGHT "
                "must equal 1.0."
            )

        if not np.isclose(
            face_only_total,
            1.0,
        ):
            raise ValueError(
                "CLUSTER_FACE_ONLY_WEIGHT + "
                "CLUSTER_FACE_ONLY_QUALITY_WEIGHT "
                "must equal 1.0."
            )

        if not np.isclose(
            representative_total,
            1.0,
        ):
            raise ValueError(
                "REPRESENTATIVE_FACE_QUALITY_WEIGHT + "
                "REPRESENTATIVE_FACE_DETECTION_WEIGHT "
                "must equal 1.0."
            )

    # ============================================================
    # Embedding helpers
    # ============================================================

    @staticmethod
    def _normalise_embedding(
        embedding: Optional[np.ndarray],
    ) -> Optional[np.ndarray]:
        """
        Convert an embedding to a finite unit vector.

        This is the same normalization logic used by the
        original implementation.
        """

        if embedding is None:
            return None

        try:
            array = np.asarray(
                embedding,
                dtype=np.float32,
            ).reshape(-1)

        except (TypeError, ValueError):
            return None

        if array.size == 0:
            return None

        if not np.all(np.isfinite(array)):
            return None

        norm = float(np.linalg.norm(array))

        if not np.isfinite(norm) or norm <= 0.0:
            return None

        return array / norm

    @classmethod
    def _cosine_similarity(
        cls,
        first: Optional[np.ndarray],
        second: Optional[np.ndarray],
    ) -> Optional[float]:
        """
        Original public cosine similarity behavior.

        Kept unchanged for compatibility.
        """

        first_normalised = cls._normalise_embedding(first)

        second_normalised = cls._normalise_embedding(second)

        if first_normalised is None or second_normalised is None:
            return None

        if first_normalised.shape != second_normalised.shape:
            return None

        similarity = float(
            np.dot(
                first_normalised,
                second_normalised,
            )
        )

        if not np.isfinite(similarity):
            return None

        return float(
            np.clip(
                similarity,
                -1.0,
                1.0,
            )
        )

    # ============================================================
    # Cached observation preparation
    # ============================================================

    def _prepare_observation(
        self,
        observation: PersonObservation,
    ) -> None:
        """
        Normalize embeddings and calculate quality exactly once
        for this observation.

        This does NOT change the values used by clustering.

        It only prevents repeating the same work.
        """

        observation_id = observation.observation_id

        if observation_id not in self._face_embedding_cache:

            self._face_embedding_cache[observation_id] = self._normalise_embedding(
                observation.face_embedding
            )

        if observation_id not in self._body_embedding_cache:

            if observation.body_embedding_valid:

                self._body_embedding_cache[observation_id] = self._normalise_embedding(
                    observation.body_embedding
                )

            else:

                self._body_embedding_cache[observation_id] = None

        if observation_id not in self._quality_cache:

            self._quality_cache[observation_id] = self._quality_score_uncached(
                observation
            )

    # ============================================================
    # Quality
    # ============================================================

    @staticmethod
    def _clip_quality(
        value: object,
    ) -> float:

        try:
            numeric_value = float(value)

        except (TypeError, ValueError):
            return 0.0

        if not np.isfinite(numeric_value):
            return 0.0

        return float(
            np.clip(
                numeric_value,
                0.0,
                1.0,
            )
        )

    def _quality_score_uncached(
        self,
        observation: PersonObservation,
    ) -> float:
        """
        Original quality calculation.
        """

        face_quality = self._clip_quality(observation.face_quality)

        detection_confidence = self._clip_quality(observation.face_detection_confidence)

        return float(
            self.representative_face_quality_weight * face_quality
            + self.representative_face_detection_weight * detection_confidence
        )

    def _quality_score(
        self,
        observation: PersonObservation,
    ) -> float:
        """
        Cached quality calculation.
        """

        observation_id = observation.observation_id

        if observation_id not in self._quality_cache:

            self._quality_cache[observation_id] = self._quality_score_uncached(
                observation
            )

        return self._quality_cache[observation_id]

    # ============================================================
    # Observation validation
    # ============================================================

    @classmethod
    def _is_valid_face_observation(
        cls,
        observation: PersonObservation,
    ) -> bool:
        """
        Only observations with a valid face embedding are
        eligible for identity clustering.

        Same behavior as the original implementation.
        """

        if not observation.face_embedding_valid:
            return False

        return cls._normalise_embedding(observation.face_embedding) is not None

    @classmethod
    def _select_valid_observations(
        cls,
        observations: list[PersonObservation],
    ) -> list[PersonObservation]:

        return [
            observation
            for observation in observations
            if cls._is_valid_face_observation(observation)
        ]

    # ============================================================
    # Pair scoring
    # ============================================================

    def pair_score(
        self,
        first: PersonObservation,
        second: PersonObservation,
    ) -> Optional[float]:
        """
        Public pair scoring method.

        The actual scoring formula is intentionally identical
        to the original implementation.
        """

        face_similarity = self._cosine_similarity(
            first.face_embedding,
            second.face_embedding,
        )

        if face_similarity is None:
            return None

        # --------------------------------------------------------
        # Hard face gate.
        # --------------------------------------------------------

        if face_similarity < self.min_face_similarity:
            return None

        quality_score = (self._quality_score(first) + self._quality_score(second)) / 2.0

        body_similarity: Optional[float] = None

        if first.body_embedding_valid and second.body_embedding_valid:

            body_similarity = self._cosine_similarity(
                first.body_embedding,
                second.body_embedding,
            )

        if body_similarity is not None:

            # ----------------------------------------------------
            # Convert cosine [-1, 1] to [0, 1].
            # ----------------------------------------------------

            body_score = float(
                np.clip(
                    (body_similarity + 1.0) / 2.0,
                    0.0,
                    1.0,
                )
            )

            score = (
                self.face_weight * face_similarity
                + self.body_weight * body_score
                + self.quality_weight * quality_score
            )

        else:

            score = (
                self.face_only_weight * face_similarity
                + self.face_only_quality_weight * quality_score
            )

        if not np.isfinite(score):
            return None

        return float(
            np.clip(
                score,
                0.0,
                1.0,
            )
        )

    # ============================================================
    # Cached pair scoring
    # ============================================================

    def _cached_pair_score(
        self,
        first: PersonObservation,
        second: PersonObservation,
    ) -> Optional[float]:
        """
        Performance-optimized version of pair_score().

        IMPORTANT:

        The mathematical formula is the same as pair_score().

        The difference is that normalized embeddings and quality
        values have already been calculated and are reused.
        """

        first_id = first.observation_id
        second_id = second.observation_id

        first_face = self._face_embedding_cache.get(first_id)

        second_face = self._face_embedding_cache.get(second_id)

        if first_face is None or second_face is None:
            return None

        if first_face.shape != second_face.shape:
            return None

        face_similarity = float(
            np.dot(
                first_face,
                second_face,
            )
        )

        if not np.isfinite(face_similarity):
            return None

        face_similarity = float(
            np.clip(
                face_similarity,
                -1.0,
                1.0,
            )
        )

        # --------------------------------------------------------
        # Hard face gate.
        # --------------------------------------------------------

        if face_similarity < self.min_face_similarity:
            return None

        quality_score = (
            self._quality_cache[first_id] + self._quality_cache[second_id]
        ) / 2.0

        body_similarity: Optional[float] = None

        if first.body_embedding_valid and second.body_embedding_valid:

            first_body = self._body_embedding_cache.get(first_id)

            second_body = self._body_embedding_cache.get(second_id)

            if (
                first_body is not None
                and second_body is not None
                and first_body.shape == second_body.shape
            ):

                body_similarity = float(
                    np.dot(
                        first_body,
                        second_body,
                    )
                )

                if np.isfinite(body_similarity):

                    body_similarity = float(
                        np.clip(
                            body_similarity,
                            -1.0,
                            1.0,
                        )
                    )

                else:

                    body_similarity = None

        if body_similarity is not None:

            # ----------------------------------------------------
            # Same body conversion as original implementation.
            # ----------------------------------------------------

            body_score = float(
                np.clip(
                    (body_similarity + 1.0) / 2.0,
                    0.0,
                    1.0,
                )
            )

            score = (
                self.face_weight * face_similarity
                + self.body_weight * body_score
                + self.quality_weight * quality_score
            )

        else:

            score = (
                self.face_only_weight * face_similarity
                + self.face_only_quality_weight * quality_score
            )

        if not np.isfinite(score):
            return None

        return float(
            np.clip(
                score,
                0.0,
                1.0,
            )
        )

    # ============================================================
    # Cluster representatives
    # ============================================================

    def _representative_score(
        self,
        observation: PersonObservation,
    ) -> float:

        return self._quality_score(observation)

    def _cluster_representatives(
        self,
        cluster: IdentityCluster,
    ) -> list[PersonObservation]:
        """
        Return the strongest observations in the cluster.

        Results are cached until this cluster changes.
        """

        cached = self._representative_cache.get(cluster.cluster_id)

        if cached is not None:
            return cached

        ranked = sorted(
            cluster.observations,
            key=lambda observation: (
                -self._representative_score(observation),
                observation.observation_id,
            ),
        )

        representatives = ranked[: self.representative_count]

        self._representative_cache[cluster.cluster_id] = representatives

        return representatives

    # ============================================================
    # Cluster compatibility
    # ============================================================

    def _cluster_pair_score(
        self,
        first: IdentityCluster,
        second: IdentityCluster,
    ) -> Optional[float]:
        """
        Calculate compatibility between two clusters.

        Same representative logic as the original implementation.
        """

        best_score: Optional[float] = None

        first_representatives = self._cluster_representatives(first)

        second_representatives = self._cluster_representatives(second)

        for first_observation in first_representatives:

            for second_observation in second_representatives:

                # ------------------------------------------------
                # Defensive same-image exclusion.
                # ------------------------------------------------

                if first_observation.image_id == second_observation.image_id:
                    continue

                score = self._cached_pair_score(
                    first_observation,
                    second_observation,
                )

                if score is None:
                    continue

                if best_score is None or score > best_score:
                    best_score = score

        return best_score

    # ============================================================
    # Same-image constraint
    # ============================================================

    @staticmethod
    def _can_merge(
        first: IdentityCluster,
        second: IdentityCluster,
    ) -> bool:
        """
        Hard identity constraint.

        Two clusters cannot merge if they contain observations
        from the same source image.
        """

        return first.image_ids.isdisjoint(second.image_ids)

    # ============================================================
    # Heap helpers
    # ============================================================

    @staticmethod
    def _pair_key(
        first_cluster_id: int,
        second_cluster_id: int,
    ) -> tuple[int, int]:
        """
        Produce deterministic pair ordering.

        The original implementation scans clusters in their
        deterministic list order.

        Cluster IDs preserve that ordering because the first
        cluster is retained during merges and the second is
        removed.

        Therefore:

            smaller cluster_id first
            larger cluster_id second

        reproduces the same deterministic tie ordering.
        """

        if first_cluster_id < second_cluster_id:
            return (
                first_cluster_id,
                second_cluster_id,
            )

        return (
            second_cluster_id,
            first_cluster_id,
        )

    def _push_cluster_pair(
        self,
        heap: list[tuple],
        first: IdentityCluster,
        second: IdentityCluster,
        versions: dict[int, int],
    ) -> None:
        """
        Calculate and insert one cluster pair into the priority
        queue.

        Invalid / same-image pairs are intentionally not inserted.
        """

        if not self._can_merge(
            first,
            second,
        ):
            return

        score = self._cluster_pair_score(
            first,
            second,
        )

        if score is None:
            return

        first_id, second_id = self._pair_key(
            first.cluster_id,
            second.cluster_id,
        )

        heapq.heappush(
            heap,
            (
                -score,
                first_id,
                second_id,
                versions[first_id],
                versions[second_id],
            ),
        )

    # ============================================================
    # Main clustering
    # ============================================================

    def cluster(
        self,
        observations: list[PersonObservation],
    ) -> dict[int, Optional[int]]:
        """
        Run constrained agglomerative identity clustering.

        PERFORMANCE OPTIMIZATION:

        The original implementation recalculated every cluster
        pair after every merge.

        This version:

            1. Calculates initial pair scores once.
            2. Stores them in a priority queue.
            3. Takes the strongest pair from the queue.
            4. After a merge, invalidates only pairs involving
               the changed cluster.
            5. Recalculates only those affected pairs.

        The actual identity decision rule remains:

            strongest compatible pair
                ->
            merge
                ->
            update representatives
                ->
            repeat
        """

        # --------------------------------------------------------
        # Always return an assignment entry for every observation.
        # --------------------------------------------------------

        assignments: dict[
            int,
            Optional[int],
        ] = {observation.observation_id: None for observation in observations}

        # --------------------------------------------------------
        # Clear caches.
        #
        # A clustering object can theoretically be reused for
        # multiple runs.
        # --------------------------------------------------------

        self._face_embedding_cache.clear()
        self._body_embedding_cache.clear()
        self._quality_cache.clear()
        self._representative_cache.clear()

        # --------------------------------------------------------
        # Select valid observations.
        #
        # This is intentionally the same selection rule.
        # --------------------------------------------------------

        valid_observations = self._select_valid_observations(observations)

        # --------------------------------------------------------
        # Prepare all valid observations once.
        #
        # This is a major performance improvement because
        # normalization and quality calculations no longer happen
        # repeatedly inside every pair comparison.
        # --------------------------------------------------------

        for observation in valid_observations:
            self._prepare_observation(observation)

        # --------------------------------------------------------
        # Every valid observation starts as its own cluster.
        # --------------------------------------------------------

        clusters: dict[
            int,
            IdentityCluster,
        ] = {}

        for index, observation in enumerate(valid_observations):

            clusters[index] = IdentityCluster(
                cluster_id=index,
                observations=[observation],
            )

        rejected_same_image_merges = 0
        rejected_low_similarity_merges = 0

        # --------------------------------------------------------
        # Version number for every active cluster.
        #
        # Whenever a cluster changes, its version increments.
        #
        # Heap entries containing an old version are discarded.
        # --------------------------------------------------------

        versions: dict[int, int] = {cluster_id: 0 for cluster_id in clusters}

        # --------------------------------------------------------
        # Priority queue.
        #
        # Python heap is a MIN heap, so score is stored as -score.
        #
        # Tuple:
        #
        #     (
        #         -score,
        #         first_cluster_id,
        #         second_cluster_id,
        #         first_version,
        #         second_version,
        #     )
        #
        # The cluster IDs provide deterministic tie ordering.
        # --------------------------------------------------------

        heap: list[tuple] = []

        cluster_ids = sorted(clusters.keys())

        # --------------------------------------------------------
        # Initial pair calculation.
        #
        # This is still O(N²), but it happens ONCE instead of
        # repeatedly after every merge.
        # --------------------------------------------------------

        for first_position in range(len(cluster_ids)):

            first_id = cluster_ids[first_position]

            first_cluster = clusters[first_id]

            for second_position in range(
                first_position + 1,
                len(cluster_ids),
            ):

                second_id = cluster_ids[second_position]

                second_cluster = clusters[second_id]

                if not self._can_merge(
                    first_cluster,
                    second_cluster,
                ):

                    rejected_same_image_merges += 1
                    continue

                score = self._cluster_pair_score(
                    first_cluster,
                    second_cluster,
                )

                if score is None:

                    rejected_low_similarity_merges += 1
                    continue

                heapq.heappush(
                    heap,
                    (
                        -score,
                        first_id,
                        second_id,
                        versions[first_id],
                        versions[second_id],
                    ),
                )

        # ========================================================
        # Greedy constrained agglomeration
        # ========================================================

        while len(clusters) > 1:

            best_pair = None
            best_score = None

            # ----------------------------------------------------
            # Find the strongest CURRENT valid pair.
            #
            # Stale heap entries are discarded here.
            # ----------------------------------------------------

            while heap:

                (
                    negative_score,
                    first_id,
                    second_id,
                    first_version,
                    second_version,
                ) = heapq.heappop(heap)

                # ------------------------------------------------
                # The clusters may have disappeared after a merge.
                # ------------------------------------------------

                if first_id not in clusters or second_id not in clusters:
                    continue

                # ------------------------------------------------
                # A cluster may have changed after a merge.
                # Old scores involving it are no longer valid.
                # ------------------------------------------------

                if (
                    versions[first_id] != first_version
                    or versions[second_id] != second_version
                ):
                    continue

                first_cluster = clusters[first_id]

                second_cluster = clusters[second_id]

                # ------------------------------------------------
                # Defensive same-image check.
                # ------------------------------------------------

                if not self._can_merge(
                    first_cluster,
                    second_cluster,
                ):

                    rejected_same_image_merges += 1
                    continue

                best_score = float(-negative_score)

                best_pair = (
                    first_id,
                    second_id,
                )

                break

            # ----------------------------------------------------
            # No compatible pair remains.
            # ----------------------------------------------------

            if best_pair is None:
                break

            # ----------------------------------------------------
            # The strongest compatible pair is still not good
            # enough.
            #
            # Same stopping rule as original implementation.
            # ----------------------------------------------------

            if best_score is None or best_score < self.merge_threshold:

                rejected_low_similarity_merges += 1
                break

            first_id, second_id = best_pair

            first_cluster = clusters[first_id]

            second_cluster = clusters[second_id]

            # ----------------------------------------------------
            # Defensive re-check immediately before mutation.
            # ----------------------------------------------------

            if not self._can_merge(
                first_cluster,
                second_cluster,
            ):

                rejected_same_image_merges += 1
                continue

            # ----------------------------------------------------
            # Preserve original merge direction.
            #
            # The original implementation keeps first_cluster
            # and extends it with second_cluster observations.
            # ----------------------------------------------------

            first_cluster.observations.extend(second_cluster.observations)

            # ----------------------------------------------------
            # The representatives of first_cluster may now have
            # changed.
            #
            # Therefore its representative cache MUST be removed.
            # ----------------------------------------------------

            self._representative_cache.pop(
                first_id,
                None,
            )

            # ----------------------------------------------------
            # Increment the version of the changed cluster.
            #
            # All old heap entries involving first_id now become
            # invalid automatically.
            # ----------------------------------------------------

            versions[first_id] += 1

            # ----------------------------------------------------
            # Remove the second cluster.
            # ----------------------------------------------------

            del clusters[second_id]

            # ----------------------------------------------------
            # Its representative cache is no longer needed.
            # ----------------------------------------------------

            self._representative_cache.pop(
                second_id,
                None,
            )

            # ----------------------------------------------------
            # Recalculate ONLY pairs involving the changed
            # first_cluster.
            #
            # Pairs between completely unrelated clusters have
            # exactly the same representatives and therefore the
            # same score as before the merge.
            # ----------------------------------------------------

            for other_id, other_cluster in clusters.items():

                if other_id == first_id:
                    continue

                if not self._can_merge(
                    first_cluster,
                    other_cluster,
                ):
                    continue

                score = self._cluster_pair_score(
                    first_cluster,
                    other_cluster,
                )

                if score is None:
                    continue

                pair_first_id, pair_second_id = self._pair_key(
                    first_id,
                    other_id,
                )

                heapq.heappush(
                    heap,
                    (
                        -score,
                        pair_first_id,
                        pair_second_id,
                        versions[pair_first_id],
                        versions[pair_second_id],
                    ),
                )

        # ========================================================
        # Convert surviving clusters into final compact IDs.
        # ========================================================

        final_cluster_id = 0
        confirmed_cluster_count = 0
        clustered_observations = 0

        # --------------------------------------------------------
        # Stable ordering.
        #
        # Same as original implementation.
        # --------------------------------------------------------

        surviving_clusters = list(clusters.values())

        surviving_clusters.sort(
            key=lambda cluster: min(
                observation.observation_id for observation in cluster.observations
            )
        )

        for cluster in surviving_clusters:

            if cluster.size < self.min_cluster_size:
                continue

            confirmed_cluster_count += 1

            for observation in cluster.observations:

                assignments[observation.observation_id] = final_cluster_id

                clustered_observations += 1

            final_cluster_id += 1

        unknown_observations = len(observations) - clustered_observations

        # ========================================================
        # Statistics
        # ========================================================

        self.last_stats = ClusterRunStats(
            total_observations=len(observations),
            valid_face_observations=len(valid_observations),
            clustered_observations=(clustered_observations),
            unknown_observations=(unknown_observations),
            cluster_count=(confirmed_cluster_count),
            rejected_same_image_merges=(rejected_same_image_merges),
            rejected_low_similarity_merges=(rejected_low_similarity_merges),
        )

        return assignments

    # ============================================================
    # Summary
    # ============================================================

    @staticmethod
    def summarize(
        observations: list[PersonObservation],
        assignments: dict[int, Optional[int]],
    ) -> dict[int, int]:
        """
        Return the number of observations assigned to each
        confirmed identity cluster.
        """

        summary: dict[int, int] = {}

        for observation in observations:

            cluster_id = assignments.get(observation.observation_id)

            if cluster_id is None:
                continue

            summary[cluster_id] = (
                summary.get(
                    cluster_id,
                    0,
                )
                + 1
            )

        return summary
