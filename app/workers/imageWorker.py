from __future__ import annotations

import gc
import os
import socket
import time
import traceback
from pathlib import Path

import cv2

from app.pipeline import PersonPipeline
from app.storage.eventStore import EventStore


class ImageWorker:
    """
    Worker responsible for processing images from the SQLite
    image-processing job queue.

    Each worker owns:

        - its own EventStore
        - its own SQLite connection
        - its own PersonPipeline
        - its own ML model instances

    Workers NEVER share:

        - SQLite connections
        - PersonPipeline instances
        - YOLO models
        - InsightFace models
        - OSNet models

    This design is suitable for multiprocessing.

    The worker does NOT perform event-level clustering.

    Its only responsibility is:

        claim image
            ↓
        read image
            ↓
        process image
            ↓
        reserve observation IDs
            ↓
        save observations
            ↓
        mark job completed
    """

    # =========================================================
    # Initialization
    # =========================================================

    def __init__(
        self,
        event_path: str | Path,
        worker_id: str,
    ) -> None:
        """
        Create one independent image worker.

        IMPORTANT:

        This constructor should execute inside the worker
        process so that every process creates its own ML models.
        """

        self.event_path = Path(event_path)

        self.worker_id = worker_id

        # -----------------------------------------------------
        # Each worker gets its own SQLite connection.
        # -----------------------------------------------------

        self.event_store = EventStore(self.event_path)

        # -----------------------------------------------------
        # Each worker gets its own pipeline.
        #
        # This means each worker owns its own:
        #
        #   YOLO
        #   InsightFace
        #   OSNet
        #
        # model instances.
        # -----------------------------------------------------

        self.pipeline = PersonPipeline()

        self.processed_images = 0
        self.failed_images = 0

        self.total_observations = 0

    # =========================================================
    # Worker loop
    # =========================================================

    def run(self) -> None:
        """
        Continuously claim and process image jobs until the
        queue is empty.
        """

        print()
        print("=" * 60)
        print(f"WORKER STARTED: {self.worker_id}")
        print("=" * 60)

        print(
            f"[{self.worker_id}] " f"PID={os.getpid()} " f"HOST={socket.gethostname()}"
        )

        worker_start = time.perf_counter()

        try:

            while True:

                # -------------------------------------------------
                # Claim one pending job.
                #
                # Stale-job recovery is intentionally NOT performed
                # here on every iteration.
                #
                # The coordinator/main process handles recovery.
                # -------------------------------------------------

                job = self.event_store.claim_next_image_job(worker_id=self.worker_id)

                # -------------------------------------------------
                # Queue is empty.
                # -------------------------------------------------

                if job is None:
                    break

                job_id = job["job_id"]
                image_id = job["image_id"]
                image_path = job["image_path"]

                print()
                print(f"[{self.worker_id}] " f"Processing: {image_id}")

                image_start = time.perf_counter()

                image = None
                observations = None

                try:

                    # =================================================
                    # STEP 1
                    # Read current image
                    # =================================================

                    image = cv2.imread(image_path)

                    if image is None:
                        raise ValueError(f"Could not read image: " f"{image_path}")

                    # =================================================
                    # STEP 2
                    # Existing single-image pipeline
                    # =================================================
                    #
                    # IMPORTANT:
                    #
                    # The identity/clustering logic is NOT changed.
                    #
                    # This is still the exact single-image pipeline.
                    # =================================================

                    observations = self.pipeline.process_image(
                        image=image,
                        image_id=image_id,
                    )

                    pipeline_time = time.perf_counter() - image_start

                    # =================================================
                    # STEP 3
                    # Reserve globally unique observation IDs
                    # =================================================
                    #
                    # Multiple workers cannot safely use:
                    #
                    #     MAX(id) + 1
                    #
                    # because two workers could receive the same ID.
                    #
                    # reserve_observation_ids() solves this.
                    # =================================================

                    if observations:

                        first_id = self.event_store.reserve_observation_ids(
                            len(observations)
                        )

                        for offset, observation in enumerate(observations):
                            observation.observation_id = first_id + offset

                    # =================================================
                    # STEP 4
                    # Save observations AND complete job atomically
                    # =================================================
                    #
                    # This is important.
                    #
                    # We do NOT do:
                    #
                    #     save_observations()
                    #     complete_image_job()
                    #
                    # separately.
                    #
                    # Instead:
                    #
                    #     save observations
                    #             +
                    #     mark job completed
                    #
                    # happen in ONE SQLite transaction.
                    #
                    # This prevents a crash between the two operations
                    # from causing the image to be processed twice.
                    # =================================================

                    self.event_store.save_observations_and_complete_job(
                        observations=observations,
                        job_id=job_id,
                        worker_id=self.worker_id,
                    )

                    # =================================================
                    # STEP 5
                    # Statistics
                    # =================================================

                    observation_count = len(observations)

                    self.processed_images += 1

                    self.total_observations += observation_count

                    total_time = time.perf_counter() - image_start

                    print(f"[{self.worker_id}] " f"COMPLETED: {image_id}")

                    print(
                        f"[{self.worker_id}] " f"Observations: " f"{observation_count}"
                    )

                    print(
                        f"[{self.worker_id}] "
                        f"Pipeline time: "
                        f"{pipeline_time:.2f}s"
                    )

                    print(
                        f"[{self.worker_id}] "
                        f"Total image time: "
                        f"{total_time:.2f}s"
                    )

                except Exception as exc:

                    # =================================================
                    # Image failed
                    # =================================================

                    self.failed_images += 1

                    elapsed = time.perf_counter() - image_start

                    error_message = (
                        f"{type(exc).__name__}: {exc}\n" f"{traceback.format_exc()}"
                    )

                    print()
                    print(f"[{self.worker_id}] " f"FAILED: {image_id}")

                    print(
                        f"[{self.worker_id}] "
                        f"Time before failure: "
                        f"{elapsed:.2f}s"
                    )

                    print(error_message)

                    # -------------------------------------------------
                    # Persist the failure in the database.
                    # -------------------------------------------------

                    try:

                        self.event_store.fail_image_job(
                            job_id=job_id,
                            worker_id=self.worker_id,
                            error_message=error_message,
                        )

                    except Exception as fail_exc:

                        # -------------------------------------------------
                        # If even recording the failure fails, print it.
                        #
                        # The worker should still continue/exit cleanly.
                        # -------------------------------------------------

                        print(
                            f"[{self.worker_id}] "
                            f"Could not record FAILED state "
                            f"for job {job_id}: "
                            f"{fail_exc}"
                        )

                finally:

                    # =================================================
                    # Release image memory
                    # =================================================

                    if image is not None:
                        del image

                    if observations is not None:
                        del observations

                    # -------------------------------------------------
                    # Explicit garbage collection.
                    #
                    # This is intentionally retained for now.
                    # We can benchmark later whether removing it
                    # improves throughput.
                    # -------------------------------------------------

                    gc.collect()

        finally:

            # =========================================================
            # Worker summary
            # =========================================================

            worker_time = time.perf_counter() - worker_start

            print()
            print("=" * 60)
            print(f"WORKER FINISHED: " f"{self.worker_id}")
            print("=" * 60)

            print(f"[{self.worker_id}] " f"Processed: " f"{self.processed_images}")

            print(f"[{self.worker_id}] " f"Failed: " f"{self.failed_images}")

            print(f"[{self.worker_id}] " f"Observations: " f"{self.total_observations}")

            print(f"[{self.worker_id}] " f"Worker time: " f"{worker_time:.2f}s")

            # -----------------------------------------------------
            # Close this worker's private SQLite connection.
            # -----------------------------------------------------

            self.event_store.close()


# =============================================================
# Standalone worker entry point
# =============================================================
#
# This function is useful when multiprocessing starts a worker
# process.
#
# It is intentionally defined at module level because Windows
# multiprocessing uses spawn.
# =============================================================


def run_image_worker(
    event_path: str | Path,
    worker_id: str,
) -> None:
    """
    Create and run one ImageWorker.

    This function is the multiprocessing entry point.
    """

    worker = ImageWorker(
        event_path=event_path,
        worker_id=worker_id,
    )

    worker.run()
