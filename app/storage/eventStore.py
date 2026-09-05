from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from app.models.detector import BoundingBox
from app.models.observation import PersonObservation


class EventStore:
    """
    Persistent storage for observations and image-processing jobs
    belonging to one event.

    SQLite is used as the persistent event database.

    Architecture:

        Image
          ↓
        Worker
          ↓
        PersonPipeline
          ↓
        PersonObservation
          ↓
        EventStore
          ↓
        SQLite
          ↓
        Final Event-Level Clustering

    The EventStore does NOT keep all observations in memory.

    Observations are persisted to SQLite and loaded only when
    required by the clustering stage or another query.
    """

    # =========================================================
    # Initialization
    # =========================================================

    def __init__(self, event_path: str | Path):
        """
        Initialize the EventStore.

        Args:
            event_path:
                Directory containing the event images.

        Database:

            <event_path>/event.db
        """

        self.event_path = Path(event_path)

        if not self.event_path.exists():
            self.event_path.mkdir(
                parents=True,
                exist_ok=True,
            )

        self.database_path = self.event_path / "event.db"

        # -----------------------------------------------------
        # SQLite connection
        # -----------------------------------------------------
        #
        # Each worker gets its OWN EventStore and therefore
        # its OWN SQLite connection.
        #
        # Connections are never shared between workers.
        # -----------------------------------------------------

        self.connection = sqlite3.connect(
            self.database_path,
            timeout=30.0,
        )

        # -----------------------------------------------------
        # SQLite concurrency configuration
        # -----------------------------------------------------

        # WAL allows readers and writers to work concurrently
        # much better than the default rollback journal.
        self.connection.execute("PRAGMA journal_mode = WAL")

        # NORMAL provides good performance while maintaining
        # appropriate SQLite durability for this workload.
        self.connection.execute("PRAGMA synchronous = NORMAL")

        # Wait up to 30 seconds if another worker temporarily
        # owns the SQLite write lock.
        self.connection.execute("PRAGMA busy_timeout = 30000")

        # Enable foreign-key enforcement.
        self.connection.execute("PRAGMA foreign_keys = ON")

        # -----------------------------------------------------
        # Legacy/batched observation commit support
        # -----------------------------------------------------
        #
        # Normal save_observations() can still use batched
        # commits.
        #
        # Worker processing uses the atomic
        # save_observations_and_complete_job() method instead.
        # -----------------------------------------------------

        self._pending_images = 0
        self._commit_every = 50

        # -----------------------------------------------------
        # Create database schema
        # -----------------------------------------------------

        self._create_schema()

    # =========================================================
    # SQLite commit helper
    # =========================================================

    def commit_if_needed(self) -> None:
        """
        Commit pending observation changes after a configured
        number of images.

        This is used by the normal save_observations() method.

        Worker processing uses a separate atomic transaction
        that commits observations and job completion together.
        """

        self._pending_images += 1

        if self._pending_images >= self._commit_every:
            self.connection.commit()
            self._pending_images = 0

    # =========================================================
    # Schema
    # =========================================================

    def _create_schema(self) -> None:
        """
        Create all required database tables and indexes.
        """

        # =====================================================
        # Observations
        # =====================================================

        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                observation_id INTEGER PRIMARY KEY,

                image_id TEXT NOT NULL,

                observation_type TEXT NOT NULL
                    CHECK (
                        observation_type IN (
                            'face_and_body',
                            'face_only',
                            'body_only',
                            'empty'
                        )
                    ),

                -- Person bounding box
                person_x1 INTEGER,
                person_y1 INTEGER,
                person_x2 INTEGER,
                person_y2 INTEGER,

                -- Face bounding box
                face_x1 INTEGER,
                face_y1 INTEGER,
                face_x2 INTEGER,
                face_y2 INTEGER,

                -- Detection / association
                person_detection_confidence REAL,
                face_detection_confidence REAL,
                association_score REAL NOT NULL,

                -- Quality
                face_quality REAL NOT NULL,
                body_quality REAL NOT NULL,

                -- Embedding validation
                face_embedding_valid INTEGER NOT NULL,
                body_embedding_valid INTEGER NOT NULL,

                -- Embeddings
                face_embedding BLOB,
                body_embedding BLOB,

                -- Embedding dimensions
                face_embedding_dimension INTEGER,
                body_embedding_dimension INTEGER,

                -- Temporal information
                moment_id INTEGER,

                -- Identity / clustering
                cluster_id INTEGER
            )
            """)

        # -----------------------------------------------------
        # Observation indexes
        # -----------------------------------------------------

        self.connection.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_observations_image_id
            ON observations(image_id)
            """)

        self.connection.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_observations_cluster_id
            ON observations(cluster_id)
            """)

        # =====================================================
        # Observation ID sequence
        # =====================================================
        #
        # Workers cannot safely use:
        #
        #     MAX(observation_id) + 1
        #
        # because multiple workers may ask for an ID at the
        # same time.
        #
        # Instead, observation_sequence acts as a centralized
        # atomic ID allocator.
        # =====================================================

        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS observation_sequence (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                next_id INTEGER NOT NULL
            )
            """)

        current_max_cursor = self.connection.execute("""
            SELECT COALESCE(
                MAX(observation_id),
                -1
            )
            FROM observations
            """)

        current_max = int(current_max_cursor.fetchone()[0])

        self.connection.execute(
            """
            INSERT OR IGNORE INTO observation_sequence (
                id,
                next_id
            )
            VALUES (
                1,
                ?
            )
            """,
            (current_max + 1,),
        )

        # =====================================================
        # Image processing jobs
        # =====================================================

        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS image_jobs (
                job_id INTEGER PRIMARY KEY AUTOINCREMENT,

                image_id TEXT NOT NULL UNIQUE,

                image_path TEXT NOT NULL,

                status TEXT NOT NULL
                    CHECK (
                        status IN (
                            'PENDING',
                            'PROCESSING',
                            'COMPLETED',
                            'FAILED'
                        )
                    ),

                worker_id TEXT,

                attempts INTEGER NOT NULL DEFAULT 0,

                created_at REAL NOT NULL,

                started_at REAL,

                completed_at REAL,

                error_message TEXT
            )
            """)

        # -----------------------------------------------------
        # Job indexes
        # -----------------------------------------------------

        self.connection.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_image_jobs_status
            ON image_jobs(status)
            """)

        self.connection.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_image_jobs_worker_id
            ON image_jobs(worker_id)
            """)

        self.connection.commit()

    # =========================================================
    # Bounding box serialization
    # =========================================================

    @staticmethod
    def _bbox_to_values(
        bbox: Optional[BoundingBox],
    ) -> tuple[
        Optional[int],
        Optional[int],
        Optional[int],
        Optional[int],
    ]:
        """
        Convert a BoundingBox into four database columns.
        """

        if bbox is None:
            return None, None, None, None

        return (
            int(bbox.x1),
            int(bbox.y1),
            int(bbox.x2),
            int(bbox.y2),
        )

    @staticmethod
    def _values_to_bbox(
        x1: Optional[int],
        y1: Optional[int],
        x2: Optional[int],
        y2: Optional[int],
    ) -> Optional[BoundingBox]:
        """
        Convert four database columns back into BoundingBox.
        """

        if x1 is None or y1 is None or x2 is None or y2 is None:
            return None

        return BoundingBox(
            x1=int(x1),
            y1=int(y1),
            x2=int(x2),
            y2=int(y2),
        )

    # =========================================================
    # Embedding serialization
    # =========================================================

    @staticmethod
    def _embedding_to_blob(
        embedding: Optional[np.ndarray],
    ) -> tuple[Optional[bytes], Optional[int]]:
        """
        Convert a NumPy embedding into SQLite BLOB storage.

        Embeddings are stored as float32.
        """

        if embedding is None:
            return None, None

        array = np.asarray(
            embedding,
            dtype=np.float32,
        )

        if array.size == 0:
            return None, None

        array = np.ascontiguousarray(array)

        return (
            array.tobytes(),
            int(array.size),
        )

    @staticmethod
    def _blob_to_embedding(
        blob: Optional[bytes],
        dimension: Optional[int],
    ) -> Optional[np.ndarray]:
        """
        Convert SQLite BLOB data back into a NumPy embedding.
        """

        if blob is None or dimension is None:
            return None

        embedding = np.frombuffer(
            blob,
            dtype=np.float32,
        ).copy()

        if embedding.size != dimension:
            raise ValueError(
                "Stored embedding dimension does not " "match the actual stored data."
            )

        return embedding

    # =========================================================
    # Save one observation
    # =========================================================

    def save_observation(
        self,
        observation: PersonObservation,
    ) -> None:
        """
        Save one observation.

        The observation can contain:

            face + body
            face only
            body only
            empty

        Missing values are stored as NULL.
        """

        person_bbox = self._bbox_to_values(observation.person_bbox)

        face_bbox = self._bbox_to_values(observation.face_bbox)

        face_blob, face_dimension = self._embedding_to_blob(observation.face_embedding)

        body_blob, body_dimension = self._embedding_to_blob(observation.body_embedding)

        self.connection.execute(
            """
            INSERT OR REPLACE INTO observations (
                observation_id,
                image_id,
                observation_type,

                person_x1,
                person_y1,
                person_x2,
                person_y2,

                face_x1,
                face_y1,
                face_x2,
                face_y2,

                person_detection_confidence,
                face_detection_confidence,
                association_score,

                face_quality,
                body_quality,

                face_embedding_valid,
                body_embedding_valid,

                face_embedding,
                body_embedding,

                face_embedding_dimension,
                body_embedding_dimension,

                moment_id,
                cluster_id
            )
            VALUES (
                :observation_id,
                :image_id,
                :observation_type,

                :person_x1,
                :person_y1,
                :person_x2,
                :person_y2,

                :face_x1,
                :face_y1,
                :face_x2,
                :face_y2,

                :person_detection_confidence,
                :face_detection_confidence,
                :association_score,

                :face_quality,
                :body_quality,

                :face_embedding_valid,
                :body_embedding_valid,

                :face_embedding,
                :body_embedding,

                :face_embedding_dimension,
                :body_embedding_dimension,

                :moment_id,
                :cluster_id
            )
            """,
            {
                "observation_id": observation.observation_id,
                "image_id": observation.image_id,
                "observation_type": observation.observation_type,
                "person_x1": person_bbox[0],
                "person_y1": person_bbox[1],
                "person_x2": person_bbox[2],
                "person_y2": person_bbox[3],
                "face_x1": face_bbox[0],
                "face_y1": face_bbox[1],
                "face_x2": face_bbox[2],
                "face_y2": face_bbox[3],
                "person_detection_confidence": observation.person_detection_confidence,
                "face_detection_confidence": observation.face_detection_confidence,
                "association_score": observation.association_score,
                "face_quality": observation.face_quality,
                "body_quality": observation.body_quality,
                "face_embedding_valid": int(observation.face_embedding_valid),
                "body_embedding_valid": int(observation.body_embedding_valid),
                "face_embedding": face_blob,
                "body_embedding": body_blob,
                "face_embedding_dimension": face_dimension,
                "body_embedding_dimension": body_dimension,
                "moment_id": observation.moment_id,
                "cluster_id": observation.cluster_id,
            },
        )

    # =========================================================
    # Save multiple observations
    # =========================================================

    def save_observations(
        self,
        observations: Iterable[PersonObservation],
    ) -> None:
        """
        Save all observations generated from one image.

        This method is kept for normal/non-worker usage.

        Worker processing should use:

            save_observations_and_complete_job()

        so that observation persistence and job completion
        happen atomically.
        """

        try:
            for observation in observations:
                self.save_observation(observation)

            self.commit_if_needed()

        except Exception:
            self.connection.rollback()
            self._pending_images = 0
            raise

    # =========================================================
    # Atomic worker save
    # =========================================================

    def save_observations_and_complete_job(
        self,
        observations: Iterable[PersonObservation],
        job_id: int,
        worker_id: str,
    ) -> None:
        """
        Atomically save observations and mark the image job
        as COMPLETED.

        This is the preferred method for ImageWorker.

        Why this matters:

            BAD:

                save observations
                    ↓
                commit
                    ↓
                worker crashes
                    ↓
                job still PROCESSING
                    ↓
                job recovered
                    ↓
                image processed again
                    ↓
                duplicate observations

            GOOD:

                BEGIN TRANSACTION
                    ↓
                save observations
                    ↓
                mark job COMPLETED
                    ↓
                COMMIT

        Therefore there is no successful database state where
        observations exist while the corresponding job remains
        PROCESSING.
        """

        try:
            # -------------------------------------------------
            # Start explicit transaction.
            # -------------------------------------------------

            self.connection.execute("BEGIN")

            # -------------------------------------------------
            # Save all observations generated by this image.
            # -------------------------------------------------

            for observation in observations:
                self.save_observation(observation)

            # -------------------------------------------------
            # Complete the corresponding job.
            #
            # The worker_id check prevents an old worker from
            # completing a job that has already been reclaimed.
            # -------------------------------------------------

            cursor = self.connection.execute(
                """
                UPDATE image_jobs

                SET
                    status = 'COMPLETED',
                    completed_at = ?

                WHERE
                    job_id = ?

                    AND worker_id = ?

                    AND status = 'PROCESSING'
                """,
                (
                    time.time(),
                    job_id,
                    worker_id,
                ),
            )

            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"Worker {worker_id} could not complete "
                    f"job {job_id}. "
                    f"The job may have been recovered or "
                    f"claimed by another worker."
                )

            # -------------------------------------------------
            # Both observations and job status become durable
            # together.
            # -------------------------------------------------

            self.connection.commit()

            self._pending_images = 0

        except Exception:
            self.connection.rollback()
            self._pending_images = 0
            raise

    # =========================================================
    # Get one observation
    # =========================================================

    def get_observation(
        self,
        observation_id: int,
    ) -> Optional[PersonObservation]:
        """
        Retrieve one observation by observation ID.
        """

        cursor = self.connection.execute(
            """
            SELECT
                observation_id,
                image_id,
                observation_type,

                person_x1,
                person_y1,
                person_x2,
                person_y2,

                face_x1,
                face_y1,
                face_x2,
                face_y2,

                person_detection_confidence,
                face_detection_confidence,
                association_score,

                face_quality,
                body_quality,

                face_embedding_valid,
                body_embedding_valid,

                face_embedding,
                body_embedding,

                face_embedding_dimension,
                body_embedding_dimension,

                moment_id,
                cluster_id

            FROM observations

            WHERE observation_id = ?
            """,
            (observation_id,),
        )

        row = cursor.fetchone()

        if row is None:
            return None

        return self._row_to_observation(row)

    # =========================================================
    # Convert DB row to observation
    # =========================================================

    def _row_to_observation(
        self,
        row: tuple,
    ) -> PersonObservation:
        """
        Convert one SQLite row into PersonObservation.
        """

        (
            observation_id,
            image_id,
            _observation_type,
            person_x1,
            person_y1,
            person_x2,
            person_y2,
            face_x1,
            face_y1,
            face_x2,
            face_y2,
            person_detection_confidence,
            face_detection_confidence,
            association_score,
            face_quality,
            body_quality,
            face_embedding_valid,
            body_embedding_valid,
            face_blob,
            body_blob,
            face_embedding_dimension,
            body_embedding_dimension,
            moment_id,
            cluster_id,
        ) = row

        person_bbox = self._values_to_bbox(
            person_x1,
            person_y1,
            person_x2,
            person_y2,
        )

        face_bbox = self._values_to_bbox(
            face_x1,
            face_y1,
            face_x2,
            face_y2,
        )

        face_embedding = self._blob_to_embedding(
            face_blob,
            face_embedding_dimension,
        )

        body_embedding = self._blob_to_embedding(
            body_blob,
            body_embedding_dimension,
        )

        return PersonObservation(
            observation_id=observation_id,
            image_id=image_id,
            person_bbox=person_bbox,
            person_detection_confidence=(person_detection_confidence),
            face_bbox=face_bbox,
            face_detection_confidence=(face_detection_confidence),
            association_score=association_score,
            face_embedding=face_embedding,
            body_embedding=body_embedding,
            face_quality=face_quality,
            body_quality=body_quality,
            face_embedding_valid=bool(face_embedding_valid),
            body_embedding_valid=bool(body_embedding_valid),
            moment_id=moment_id,
            cluster_id=cluster_id,
        )

    # =========================================================
    # Query observations for one image
    # =========================================================

    def get_observations_for_image(
        self,
        image_id: str,
    ) -> list[PersonObservation]:
        """
        Retrieve all observations belonging to one image.
        """

        cursor = self.connection.execute(
            """
            SELECT
                observation_id,
                image_id,
                observation_type,

                person_x1,
                person_y1,
                person_x2,
                person_y2,

                face_x1,
                face_y1,
                face_x2,
                face_y2,

                person_detection_confidence,
                face_detection_confidence,
                association_score,

                face_quality,
                body_quality,

                face_embedding_valid,
                body_embedding_valid,

                face_embedding,
                body_embedding,

                face_embedding_dimension,
                body_embedding_dimension,

                moment_id,
                cluster_id

            FROM observations

            WHERE image_id = ?

            ORDER BY observation_id
            """,
            (image_id,),
        )

        rows = cursor.fetchall()

        return [self._row_to_observation(row) for row in rows]

    # =========================================================
    # Get all observations
    # =========================================================

    def get_all_observations(
        self,
    ) -> list[PersonObservation]:
        """
        Retrieve all observations for the event.

        This is used by the final event-level clustering stage.

        Original images are NOT loaded here.

        Only observations and their stored embeddings are loaded.
        """

        cursor = self.connection.execute("""
            SELECT
                observation_id,
                image_id,
                observation_type,

                person_x1,
                person_y1,
                person_x2,
                person_y2,

                face_x1,
                face_y1,
                face_x2,
                face_y2,

                person_detection_confidence,
                face_detection_confidence,
                association_score,

                face_quality,
                body_quality,

                face_embedding_valid,
                body_embedding_valid,

                face_embedding,
                body_embedding,

                face_embedding_dimension,
                body_embedding_dimension,

                moment_id,
                cluster_id

            FROM observations

            ORDER BY observation_id
            """)

        rows = cursor.fetchall()

        return [self._row_to_observation(row) for row in rows]

    # =========================================================
    # Count observations
    # =========================================================

    def get_observation_count(self) -> int:
        """
        Return the total number of observations stored.
        """

        cursor = self.connection.execute("""
            SELECT COUNT(*)
            FROM observations
            """)

        row = cursor.fetchone()

        return int(row[0])

    # =========================================================
    # Get next observation ID
    # =========================================================

    def get_next_observation_id(self) -> int:
        """
        Return the current next observation ID.

        This method is kept for backward compatibility.

        IMPORTANT:

        Do NOT use this method for assigning IDs from multiple
        workers.

        Workers should use reserve_observation_ids().
        """

        cursor = self.connection.execute("""
            SELECT next_id
            FROM observation_sequence
            WHERE id = 1
            """)

        row = cursor.fetchone()

        if row is not None:
            return int(row[0])

        # -----------------------------------------------------
        # Defensive fallback for an old database.
        # -----------------------------------------------------

        cursor = self.connection.execute("""
            SELECT COALESCE(
                MAX(observation_id),
                -1
            ) + 1
            FROM observations
            """)

        row = cursor.fetchone()

        return int(row[0])

    # =========================================================
    # Reserve observation IDs
    # =========================================================

    def reserve_observation_ids(
        self,
        count: int,
    ) -> int:
        """
        Atomically reserve a contiguous block of observation IDs.

        Example:

            count = 3

            returns 100

            reserved IDs:

                100
                101
                102

            next worker receives:

                103 ...

        This makes observation ID assignment safe when multiple
        workers are processing images simultaneously.

        Returns:
            First reserved observation ID.
        """

        if count <= 0:
            raise ValueError("count must be greater than zero")

        try:
            # -------------------------------------------------
            # Lock the database for this short ID-allocation
            # transaction.
            # -------------------------------------------------

            self.connection.execute("BEGIN IMMEDIATE")

            cursor = self.connection.execute("""
                SELECT next_id
                FROM observation_sequence
                WHERE id = 1
                """)

            row = cursor.fetchone()

            if row is None:
                raise RuntimeError("observation_sequence is not initialized.")

            first_id = int(row[0])

            next_id = first_id + int(count)

            self.connection.execute(
                """
                UPDATE observation_sequence

                SET next_id = ?

                WHERE id = 1
                """,
                (next_id,),
            )

            self.connection.commit()

            return first_id

        except Exception:
            self.connection.rollback()
            raise

    # =========================================================
    # Save cluster assignments
    # =========================================================

    def save_cluster_assignments(
        self,
        assignments: dict[int, Optional[int]],
    ) -> None:
        """
        Persist final event-level cluster assignments.

        The clustering algorithm decides the assignments.

        EventStore only persists them.

        None means unknown/unassigned.
        """

        try:
            for observation_id, cluster_id in assignments.items():

                self.connection.execute(
                    """
                    UPDATE observations

                    SET cluster_id = ?

                    WHERE observation_id = ?
                    """,
                    (
                        cluster_id,
                        observation_id,
                    ),
                )

            self.connection.commit()

        except Exception:
            self.connection.rollback()
            raise

    # =========================================================
    # Create image processing jobs
    # =========================================================

    def create_image_jobs(
        self,
        image_paths: Iterable[str | Path],
    ) -> int:
        """
        Add images to the processing queue.

        Existing image IDs are ignored because image_id is UNIQUE.

        This makes the operation safe to run again after restart.

        Returns:
            Number of newly inserted jobs.
        """

        inserted = 0
        now = time.time()

        try:
            for image_path in image_paths:

                image_path = Path(image_path)

                image_id = image_path.name

                cursor = self.connection.execute(
                    """
                    INSERT OR IGNORE INTO image_jobs (
                        image_id,
                        image_path,
                        status,
                        worker_id,
                        attempts,
                        created_at,
                        started_at,
                        completed_at,
                        error_message
                    )
                    VALUES (
                        ?,
                        ?,
                        'PENDING',
                        NULL,
                        0,
                        ?,
                        NULL,
                        NULL,
                        NULL
                    )
                    """,
                    (
                        image_id,
                        str(image_path),
                        now,
                    ),
                )

                if cursor.rowcount == 1:
                    inserted += 1

            self.connection.commit()

        except Exception:
            self.connection.rollback()
            raise

        return inserted

    # =========================================================
    # Recover stale jobs
    # =========================================================

    def recover_stale_image_jobs(
        self,
        stale_timeout_seconds: int,
    ) -> int:
        """
        Return abandoned PROCESSING jobs to PENDING.

        A job is considered stale when its processing start time
        is older than stale_timeout_seconds.

        This should normally be called by the coordinator before
        workers start rather than on every job claim.
        """

        cutoff = time.time() - float(stale_timeout_seconds)

        try:
            cursor = self.connection.execute(
                """
                UPDATE image_jobs

                SET
                    status = 'PENDING',
                    worker_id = NULL,
                    started_at = NULL

                WHERE
                    status = 'PROCESSING'

                    AND started_at IS NOT NULL

                    AND started_at < ?
                """,
                (cutoff,),
            )

            self.connection.commit()

            return int(cursor.rowcount)

        except Exception:
            self.connection.rollback()
            raise

    # =========================================================
    # Claim next image job
    # =========================================================

    def claim_next_image_job(
        self,
        worker_id: str,
        stale_timeout_seconds: Optional[int] = None,
    ) -> Optional[dict]:
        """
        Atomically claim the next PENDING image.

        Multiple workers can safely call this method.

        SQLite's BEGIN IMMEDIATE guarantees that the SELECT
        and UPDATE happen while this connection owns the write
        transaction.

        stale_timeout_seconds is retained for backward
        compatibility.

        If supplied, stale jobs are recovered before claiming.
        For best performance, recover stale jobs once before
        starting the worker pool and omit this argument.
        """

        # -----------------------------------------------------
        # Optional compatibility behavior.
        #
        # New worker code should normally pass None here.
        # -----------------------------------------------------

        if stale_timeout_seconds is not None:
            self.recover_stale_image_jobs(stale_timeout_seconds)

        try:
            # -------------------------------------------------
            # Obtain a write transaction.
            # -------------------------------------------------

            self.connection.execute("BEGIN IMMEDIATE")

            # -------------------------------------------------
            # Select the oldest pending job.
            # -------------------------------------------------

            cursor = self.connection.execute("""
                SELECT
                    job_id,
                    image_id,
                    image_path,
                    attempts

                FROM image_jobs

                WHERE status = 'PENDING'

                ORDER BY job_id

                LIMIT 1
                """)

            row = cursor.fetchone()

            if row is None:
                self.connection.commit()
                return None

            (
                job_id,
                image_id,
                image_path,
                attempts,
            ) = row

            now = time.time()

            # -------------------------------------------------
            # Claim job.
            # -------------------------------------------------

            cursor = self.connection.execute(
                """
                UPDATE image_jobs

                SET
                    status = 'PROCESSING',
                    worker_id = ?,
                    attempts = attempts + 1,
                    started_at = ?,
                    completed_at = NULL,
                    error_message = NULL

                WHERE
                    job_id = ?

                    AND status = 'PENDING'
                """,
                (
                    worker_id,
                    now,
                    job_id,
                ),
            )

            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"Worker {worker_id} failed to claim " f"job {job_id}."
                )

            self.connection.commit()

            return {
                "job_id": int(job_id),
                "image_id": str(image_id),
                "image_path": str(image_path),
                "attempts": int(attempts) + 1,
            }

        except Exception:
            self.connection.rollback()
            raise

    # =========================================================
    # Complete image job
    # =========================================================

    def complete_image_job(
        self,
        job_id: int,
        worker_id: str,
    ) -> None:
        """
        Mark an image-processing job as COMPLETED.

        Normally the worker should use:

            save_observations_and_complete_job()

        instead.

        This method remains available for compatibility.
        """

        try:
            cursor = self.connection.execute(
                """
                UPDATE image_jobs

                SET
                    status = 'COMPLETED',
                    completed_at = ?

                WHERE
                    job_id = ?

                    AND worker_id = ?

                    AND status = 'PROCESSING'
                """,
                (
                    time.time(),
                    job_id,
                    worker_id,
                ),
            )

            self.connection.commit()

            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"Worker {worker_id} could not complete "
                    f"job {job_id}. "
                    f"The job may have been recovered or "
                    f"claimed by another worker."
                )

        except Exception:
            self.connection.rollback()
            raise

    # =========================================================
    # Fail image job
    # =========================================================

    def fail_image_job(
        self,
        job_id: int,
        worker_id: str,
        error_message: str,
    ) -> None:
        """
        Mark an image-processing job as FAILED.

        The error is persisted so that failed images can be
        diagnosed after processing.
        """

        try:
            cursor = self.connection.execute(
                """
                UPDATE image_jobs

                SET
                    status = 'FAILED',
                    completed_at = ?,
                    error_message = ?

                WHERE
                    job_id = ?

                    AND worker_id = ?

                    AND status = 'PROCESSING'
                """,
                (
                    time.time(),
                    error_message,
                    job_id,
                    worker_id,
                ),
            )

            self.connection.commit()

            if cursor.rowcount != 1:
                raise RuntimeError(
                    f"Worker {worker_id} could not mark " f"job {job_id} as FAILED."
                )

        except Exception:
            self.connection.rollback()
            raise

    # =========================================================
    # Job statistics
    # =========================================================

    def get_image_job_counts(
        self,
    ) -> dict[str, int]:
        """
        Return the number of jobs in every state.
        """

        cursor = self.connection.execute("""
            SELECT
                status,
                COUNT(*)

            FROM image_jobs

            GROUP BY status
            """)

        counts = {
            "PENDING": 0,
            "PROCESSING": 0,
            "COMPLETED": 0,
            "FAILED": 0,
        }

        for status, count in cursor.fetchall():
            counts[str(status)] = int(count)

        return counts

    # =========================================================
    # Close
    # =========================================================

    def close(self) -> None:
        """
        Commit remaining changes and close SQLite.
        """

        if self.connection is not None:

            try:
                self.connection.commit()

            finally:
                self.connection.close()
                self.connection = None

    # =========================================================
    # Context manager
    # =========================================================

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        self.close()
