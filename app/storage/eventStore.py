from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from app.models.detector import BoundingBox
from app.models.observation import PersonObservation
from app import configuration as config


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
        # Create database schema
        # -----------------------------------------------------

        self._create_schema()

    # =========================================================
    # Portable image ID / path
    # =========================================================

    @staticmethod
    def _project_root() -> Path:
        """
        Return the project root independently of the current working
        directory.

        eventStore.py lives under:
            <project>/app/storage/eventStore.py

        Therefore parents[2] is the project root.
        """
        return Path(__file__).resolve().parents[2]

    @classmethod
    def _events_root(cls) -> Path:
        """Return the absolute configured events root."""
        return (cls._project_root() / Path(config.EVENTS_PATH)).resolve(strict=False)

    @classmethod
    def _canonical_image_id(
        cls,
        image_path: str | Path,
    ) -> str:
        """
        Return a portable project-relative image ID.

        The stored ID always keeps the complete path beginning at
        ``data/events`` (or whatever EVENTS_PATH is configured to be).

        Example:

            data/events/Wedding/camera1/IMG_001.jpg
            data/events/Wedding/camera2/IMG_001.jpg

        remain different IDs even though both filenames are IMG_001.jpg.

        Absolute machine-specific paths are NEVER stored in SQLite.
        """
        path = Path(image_path)

        project_root = cls._project_root()
        events_root = cls._events_root()

        # ---------------------------------------------------------
        # Absolute input
        # ---------------------------------------------------------
        #
        # If the file is already inside the configured events tree,
        # convert it directly to a project-relative path.
        # ---------------------------------------------------------
        if path.is_absolute():
            absolute_path = path.resolve(strict=False)

            try:
                relative_to_events = absolute_path.relative_to(events_root)
            except ValueError:
                # Keep this strict. An image outside EVENTS_PATH should
                # not silently become a misleading project-relative ID.
                raise ValueError(
                    f"Image path '{absolute_path}' is outside "
                    f"the configured events root '{events_root}'."
                )

            return (Path(config.EVENTS_PATH) / relative_to_events).as_posix()

        # ---------------------------------------------------------
        # Relative input
        # ---------------------------------------------------------
        #
        # Normal expected form:
        #     data/events/Wedding/camera1/IMG_001.jpg
        #
        # If a caller supplies an event-root-relative path instead,
        # resolve it against EVENTS_PATH.
        # ---------------------------------------------------------
        normalized = path.as_posix()

        configured_events = Path(config.EVENTS_PATH).as_posix().rstrip("/")

        if normalized == configured_events or normalized.startswith(
            configured_events + "/"
        ):
            return Path(normalized).as_posix()

        # Compatibility with callers that pass:
        #     Wedding/camera1/IMG_001.jpg
        candidate = (Path(config.EVENTS_PATH) / path).as_posix()
        return candidate

    @classmethod
    def _resolve_image_path(
        cls,
        image_id: str | Path,
    ) -> Path:
        """
        Resolve a stored project-relative image ID to the current
        machine's physical path.

        Example:

            DB:
                data/events/Wedding/camera1/IMG_001.jpg

            Runtime:
                <current-project>/data/events/Wedding/camera1/IMG_001.jpg
        """
        path = Path(str(image_id))

        if path.is_absolute():
            return path.resolve(strict=False)

        return (cls._project_root() / path).resolve(strict=False)

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
                cluster_id INTEGER,

                -- Coarse face pose
                face_yaw REAL,
                face_pitch REAL,
                face_roll REAL,
                face_pose TEXT
            )
            """)

        # -----------------------------------------------------
        # Observation indexes
        # -----------------------------------------------------

        self._ensure_observation_pose_columns()

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
        # Reference-to-cluster matches
        # =====================================================

        # A known person may own multiple discovered clusters, so this
        # relationship is stored separately from observations.cluster_id.
        self.connection.execute("""
            CREATE TABLE IF NOT EXISTS cluster_person_matches (
                cluster_id INTEGER PRIMARY KEY,
                person_name TEXT NOT NULL,
                similarity REAL NOT NULL,
                margin REAL NOT NULL,
                reference_image TEXT,
                matched_at REAL NOT NULL
            )
            """)

        self.connection.execute("""
            CREATE INDEX IF NOT EXISTS
            idx_cluster_person_matches_person
            ON cluster_person_matches(person_name)
            """)

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
        # Legacy image-ID migration
        # -----------------------------------------------------
        # Older event databases used only image_path.name as image_id.
        # Convert those IDs to the same canonical full path used for
        # all new jobs.
        self._migrate_image_job_ids()

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
    # Image-job schema migration
    # =========================================================

    @classmethod
    def _legacy_image_id_to_portable(
        cls,
        image_path: str | Path,
    ) -> str:
        """
        Convert old absolute/relative image paths into the new
        project-relative ``data/events/...`` representation.

        This specifically handles old databases that stored paths such as:

            D:\\University\\...\\data\\events\\Wedding\\IMG_001.jpg

        The machine-specific prefix is discarded while the complete
        ``data/events/...`` hierarchy is preserved.
        """
        raw = str(image_path).strip()

        # Already in the new portable format.
        configured_events = Path(config.EVENTS_PATH).as_posix().rstrip("/")
        normalized = raw.replace("\\", "/")

        if normalized == configured_events or normalized.startswith(
            configured_events + "/"
        ):
            return Path(normalized).as_posix()

        # Locate the configured EVENTS_PATH anywhere inside an old
        # absolute path. This makes migration independent of the old
        # project location.
        marker = "/" + configured_events + "/"
        lower_normalized = normalized.lower()
        marker_lower = marker.lower()

        marker_index = lower_normalized.find(marker_lower)

        if marker_index >= 0:
            return normalized[marker_index + 1 :]

        # Fall back to the normal canonicalizer for paths that can
        # still be resolved unambiguously from the current project.
        return cls._canonical_image_id(raw)

    def _migrate_image_job_ids(self) -> None:
        """
        Normalize existing image jobs and observations to portable
        project-relative image IDs.

        Existing processing state, attempts, and timestamps are preserved.
        """
        rows = self.connection.execute(
            "SELECT job_id, image_id, image_path FROM image_jobs"
        ).fetchall()

        for job_id, current_image_id, image_path in rows:
            portable_id = self._legacy_image_id_to_portable(image_path)

            if str(current_image_id) != portable_id or str(image_path) != portable_id:
                self.connection.execute(
                    """
                    UPDATE image_jobs
                    SET image_id = ?,
                        image_path = ?
                    WHERE job_id = ?
                    """,
                    (portable_id, portable_id, job_id),
                )

        observation_rows = self.connection.execute(
            "SELECT observation_id, image_id FROM observations"
        ).fetchall()

        for observation_id, current_image_id in observation_rows:
            portable_id = self._legacy_image_id_to_portable(current_image_id)

            if str(current_image_id) != portable_id:
                self.connection.execute(
                    """
                    UPDATE observations
                    SET image_id = ?
                    WHERE observation_id = ?
                    """,
                    (portable_id, observation_id),
                )

        self.connection.commit()

    # =========================================================
    # Observation schema migration
    # =========================================================

    def _ensure_observation_pose_columns(self) -> None:
        """
        Add the new pose columns to an existing event database.

        SQLite CREATE TABLE IF NOT EXISTS does not alter an existing
        table, so older event.db files need a small additive migration.
        """

        cursor = self.connection.execute("PRAGMA table_info(observations)")
        existing_columns = {str(row[1]) for row in cursor.fetchall()}

        new_columns = {
            "face_yaw": "REAL",
            "face_pitch": "REAL",
            "face_roll": "REAL",
            "face_pose": "TEXT",
        }

        for column_name, sql_type in new_columns.items():
            if column_name in existing_columns:
                continue

            self.connection.execute(
                f"ALTER TABLE observations ADD COLUMN {column_name} {sql_type}"
            )

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
                cluster_id,

                face_yaw,
                face_pitch,
                face_roll,
                face_pose
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
                :cluster_id,

                :face_yaw,
                :face_pitch,
                :face_roll,
                :face_pose
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
                "face_yaw": observation.face_yaw,
                "face_pitch": observation.face_pitch,
                "face_roll": observation.face_roll,
                "face_pose": observation.face_pose,
            },
        )

    # =========================================================
    # Save multiple observations
    # =========================================================

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

        except Exception:
            self.connection.rollback()
            raise

    # =========================================================
    # Get one observation
    # =========================================================

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
            face_yaw,
            face_pitch,
            face_roll,
            face_pose,
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
            face_yaw=face_yaw,
            face_pitch=face_pitch,
            face_roll=face_roll,
            face_pose=face_pose,
        )

    # =========================================================
    # Query observations for one image
    # =========================================================

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
                cluster_id,

                face_yaw,
                face_pitch,
                face_roll,
                face_pose

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
    # Save reference-to-cluster matches
    # =========================================================

    def save_cluster_person_matches(
        self,
        match_details: dict[int, dict[str, object]],
    ) -> None:
        """Persist accepted reference-person assignments for clusters."""
        try:
            self.connection.execute("DELETE FROM cluster_person_matches")

            for cluster_id, details in match_details.items():
                if not bool(details.get("accepted")):
                    continue

                person_name = details.get("person_name")
                if not person_name:
                    continue

                self.connection.execute(
                    """
                    INSERT OR REPLACE INTO cluster_person_matches (
                        cluster_id, person_name, similarity, margin,
                        reference_image, matched_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(cluster_id),
                        str(person_name),
                        float(details.get("score", 0.0)),
                        float(details.get("margin", 0.0)),
                        details.get("reference_image"),
                        time.time(),
                    ),
                )

            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise

    def get_cluster_person_matches(self) -> dict[int, str]:
        """Load accepted cluster-to-person matches from SQLite."""
        cursor = self.connection.execute("""
            SELECT cluster_id, person_name
            FROM cluster_person_matches
            ORDER BY cluster_id
            """)
        return {
            int(cluster_id): str(person_name)
            for cluster_id, person_name in cursor.fetchall()
        }

    # =========================================================
    # Create image processing jobs
    # =========================================================

    def create_image_jobs(
        self,
        image_paths: Iterable[str | Path],
    ) -> int:
        """
        Add images to the processing queue.

        Existing image IDs are ignored because the canonical full image
        path is UNIQUE. Different directories may therefore contain files
        with the same filename without colliding.

        Returns:
            Number of newly inserted jobs.
        """

        inserted = 0
        now = time.time()

        try:
            for image_path in image_paths:

                image_path = Path(image_path)

                # Store a portable project-relative path in BOTH fields.
                #
                # The complete data/events/... hierarchy is preserved, so
                # duplicate filenames in different folders remain unique.
                image_id = self._canonical_image_id(image_path)
                image_path = Path(image_id)

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
    # Retry failed image jobs
    # =========================================================

    def retry_failed_image_jobs(self) -> int:
        """
        Move images that failed on their first attempt back to
        PENDING so they receive one additional processing attempt.

        Only jobs with attempts == 1 are retried. This guarantees
        that an image is never retried more than once by this
        mechanism. Jobs that fail again remain FAILED.
        """
        try:
            cursor = self.connection.execute("""
                UPDATE image_jobs
                SET
                    status = 'PENDING',
                    worker_id = NULL,
                    started_at = NULL,
                    completed_at = NULL,
                    error_message = NULL
                WHERE status = 'FAILED' AND attempts = 1
                """)
            self.connection.commit()
            return int(cursor.rowcount)
        except Exception:
            self.connection.rollback()
            raise

    # =========================================================
    # Get failed image jobs
    # =========================================================

    def get_failed_image_jobs(self) -> list[dict]:
        """Return image jobs that are still FAILED."""
        cursor = self.connection.execute("""
            SELECT job_id, image_id, image_path, attempts, error_message
            FROM image_jobs
            WHERE status = 'FAILED'
            ORDER BY job_id
            """)
        rows = cursor.fetchall()
        return [
            {
                "job_id": int(row[0]),
                "image_id": str(row[1]),
                "image_path": str(row[2]),
                "attempts": int(row[3]),
                "error_message": row[4],
            }
            for row in rows
        ]

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
                # image_path in SQLite is portable; workers need the
                # current machine's physical path for OpenCV.
                "image_path": str(self._resolve_image_path(image_path)),
                "attempts": int(attempts) + 1,
            }

        except Exception:
            self.connection.rollback()
            raise

    # =========================================================
    # Complete image job
    # =========================================================

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
