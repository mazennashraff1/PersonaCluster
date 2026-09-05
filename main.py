from __future__ import annotations

import gc
import multiprocessing
import os
import time
from pathlib import Path

import cv2

from app import configuration as config
from app.clustering.clusterVisualizer import visualizeClusters
from app.clustering.constrainedClustering import (
    ConstrainedIdentityClustering,
)
from app.storage.eventStore import EventStore
from app.workers.imageWorker import run_image_worker

# ============================================================
# Configuration
# ============================================================

EVENT_PATH = Path(config.EVENTS_PATH)


# ============================================================
# Get image paths
# ============================================================


def get_image_paths(
    event_path: str | Path,
):
    """
    Yield image paths from the event directory.

    Images are yielded one at a time rather than creating a
    large list in memory.
    """

    event_path = Path(event_path)

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
    }

    with os.scandir(event_path) as entries:

        for entry in entries:

            if not entry.is_file():
                continue

            extension = os.path.splitext(entry.name)[1].lower()

            if extension not in valid_extensions:
                continue

            yield entry.path


# ============================================================
# Create image jobs
# ============================================================


def prepare_image_jobs(
    event_store: EventStore,
) -> int:
    """
    Discover event images and create their processing jobs.

    Existing jobs are ignored because image_id is UNIQUE.

    Returns:
        Number of newly created jobs.
    """

    print()
    print("=" * 60)
    print("PREPARING IMAGE JOB QUEUE")
    print("=" * 60)

    image_paths = get_image_paths(EVENT_PATH)

    inserted = event_store.create_image_jobs(image_paths)

    counts = event_store.get_image_job_counts()

    print(f"New jobs created: {inserted}")

    print(f"PENDING:    {counts['PENDING']}")

    print(f"PROCESSING: {counts['PROCESSING']}")

    print(f"COMPLETED:  {counts['COMPLETED']}")

    print(f"FAILED:     {counts['FAILED']}")

    return inserted


# ============================================================
# Run workers
# ============================================================


def run_workers() -> None:
    """
    Start the configured worker processes.

    Each process creates:

        - its own EventStore
        - its own SQLite connection
        - its own PersonPipeline
        - its own ML models
    """

    worker_count = max(
        1,
        int(config.WORKER_COUNT),
    )

    print()
    print("=" * 60)
    print("STARTING IMAGE WORKERS")
    print("=" * 60)

    print(f"Configured workers: " f"{worker_count}")

    print(
        f"Worker mode: " f"{'PROCESSES' if config.WORKER_USE_PROCESSES else 'THREADS'}"
    )

    # ========================================================
    # PROCESS MODE
    # ========================================================
    #
    # Recommended for this project.
    #
    # Each process owns its own ML models.
    #
    # IMPORTANT:
    #
    # Multiple GPU processes can consume a lot of VRAM.
    #
    # Therefore:
    #
    #     RTX 3050 6GB → start with WORKER_COUNT = 1
    #
    # and benchmark before increasing it.
    # ========================================================

    if config.WORKER_USE_PROCESSES:

        # ----------------------------------------------------
        # Windows uses spawn.
        #
        # Explicitly using the spawn context makes the behavior
        # predictable across Windows/Linux.
        # ----------------------------------------------------

        multiprocessing_context = multiprocessing.get_context("spawn")

        processes = []

        worker_start = time.perf_counter()

        for index in range(worker_count):

            worker_id = f"worker-{index + 1}"

            process = multiprocessing_context.Process(
                target=run_image_worker,
                args=(
                    str(EVENT_PATH),
                    worker_id,
                ),
                name=worker_id,
            )

            processes.append(process)

            print(f"Starting {worker_id}")

            process.start()

        # ----------------------------------------------------
        # Wait for all workers.
        # ----------------------------------------------------

        for process in processes:
            process.join()

        worker_time = time.perf_counter() - worker_start

        print()
        print(f"All workers finished " f"in {worker_time:.2f}s")

        # ----------------------------------------------------
        # Check worker exit codes.
        # ----------------------------------------------------

        crashed_workers = []

        for process in processes:

            if process.exitcode != 0:

                crashed_workers.append(
                    (
                        process.name,
                        process.exitcode,
                    )
                )

        if crashed_workers:

            print()
            print("[WARNING] Some worker " "processes exited unexpectedly:")

            for worker_name, exit_code in crashed_workers:

                print(f"  {worker_name}: " f"exit code {exit_code}")

    # ========================================================
    # THREAD MODE
    # ========================================================
    #
    # This branch is intentionally kept simple.
    #
    # For ML inference, process mode is currently the safer
    # choice because model instances are isolated.
    #
    # Thread mode can be added/benchmarked later if required.
    # ========================================================

    else:

        raise RuntimeError(
            "WORKER_USE_PROCESSES=False is "
            "not enabled in this final configuration. "
            "Use process workers for the current "
            "GPU/ML pipeline."
        )


# ============================================================
# Print job statistics
# ============================================================


def print_job_statistics(
    event_store: EventStore,
) -> dict[str, int]:
    """
    Print the final image-job queue state.
    """

    counts = event_store.get_image_job_counts()

    print()
    print("=" * 60)
    print("IMAGE JOB RESULTS")
    print("=" * 60)

    print(f"PENDING:    {counts['PENDING']}")

    print(f"PROCESSING: {counts['PROCESSING']}")

    print(f"COMPLETED:  {counts['COMPLETED']}")

    print(f"FAILED:     {counts['FAILED']}")

    return counts


# ============================================================
# Constrained identity clustering
# ============================================================


def run_constrained_clustering(
    event_store: EventStore,
):
    """
    Run event-level constrained identity clustering.

    IMPORTANT:

    This stage happens ONLY after image workers have finished.

    The clustering logic itself is unchanged.

    It uses:

        - valid face embeddings
        - face similarity
        - body similarity
        - quality
        - face detection confidence
        - representative observations
        - same-image exclusion
    """

    print()
    print("=" * 60)
    print("CONSTRAINED IDENTITY CLUSTERING")
    print("=" * 60)

    # --------------------------------------------------------
    # Load observations from SQLite.
    #
    # Original images are NOT loaded.
    # --------------------------------------------------------

    observations = event_store.get_all_observations()

    print(f"Total observations in event: " f"{len(observations)}")

    # --------------------------------------------------------
    # Create clustering algorithm.
    #
    # DO NOT change the clustering logic here.
    # --------------------------------------------------------

    clusterer = ConstrainedIdentityClustering()

    assignments = clusterer.cluster(observations)

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    stats = clusterer.last_stats

    if stats is not None:

        print(f"Valid face observations: " f"{stats.valid_face_observations}")

        print(
            f"Observations assigned to identities: " f"{stats.clustered_observations}"
        )

        print(f"Unknown / unassigned observations: " f"{stats.unknown_observations}")

        print(f"Discovered identity clusters: " f"{stats.cluster_count}")

        print(f"Rejected same-image merges: " f"{stats.rejected_same_image_merges}")

        print(
            f"Rejected low-similarity candidates: "
            f"{stats.rejected_low_similarity_merges}"
        )

    # --------------------------------------------------------
    # Cluster sizes
    # --------------------------------------------------------

    summary = clusterer.summarize(
        observations,
        assignments,
    )

    print()
    print("Cluster sizes:")

    for cluster_id in sorted(summary):

        print(f"  Cluster {cluster_id}: " f"{summary[cluster_id]} observations")

    # --------------------------------------------------------
    # Persist assignments.
    # --------------------------------------------------------

    event_store.save_cluster_assignments(assignments)

    print()
    print("Cluster assignments saved " "to database.")

    return observations, assignments


# ============================================================
# Main
# ============================================================


def main():
    """
    Main event-processing coordinator.

    Responsibilities:

        1. Open event database
        2. Create image jobs
        3. Recover stale jobs
        4. Start workers
        5. Wait for workers
        6. Verify job state
        7. Run event-level clustering
        8. Run visualization
    """

    program_start = time.perf_counter()

    print("=" * 60)
    print("STARTING EVENT PERSON " "IDENTITY PIPELINE")
    print("=" * 60)

    print(f"Event path: {EVENT_PATH}")

    print(f"Worker count: " f"{config.WORKER_COUNT}")

    print(f"Worker mode: " f"{'PROCESS' if config.WORKER_USE_PROCESSES else 'THREAD'}")

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Main does NOT create PersonPipeline anymore.
    #
    # Workers create their own pipelines.
    # --------------------------------------------------------

    event_store = EventStore(EVENT_PATH)

    try:

        # ====================================================
        # PHASE 1
        # Prepare image jobs
        # ====================================================

        prepare_image_jobs(event_store)

        # ====================================================
        # PHASE 1.5
        # Recover stale jobs
        # ====================================================
        #
        # This happens BEFORE starting workers.
        #
        # We intentionally do not call stale recovery on every
        # worker claim because that would create unnecessary
        # database transactions.
        # ====================================================

        recovered = event_store.recover_stale_image_jobs(
            config.WORKER_STALE_TIMEOUT_SECONDS
        )

        if recovered > 0:

            print()
            print(f"Recovered {recovered} " f"stale image job(s).")

        # ====================================================
        # PHASE 2
        # Start workers
        # ====================================================

        run_workers()

        # ====================================================
        # PHASE 2.5
        # Check queue state
        # ====================================================

        counts = print_job_statistics(event_store)

        # ----------------------------------------------------
        # Normally this should be zero.
        #
        # If a worker crashed while processing an image,
        # its job can remain PROCESSING.
        # ----------------------------------------------------

        if counts["PROCESSING"] > 0:

            print()
            print(
                "[WARNING] "
                f"{counts['PROCESSING']} job(s) "
                "are still marked PROCESSING."
            )

            print("They were not included in " "the completed worker result.")

            # ------------------------------------------------
            # Do NOT blindly run clustering as if all images
            # were processed.
            # ------------------------------------------------

            raise RuntimeError(
                "Image processing did not finish cleanly. "
                "PROCESSING jobs remain in the database."
            )

        # ----------------------------------------------------
        # PENDING jobs should also normally be zero.
        # ----------------------------------------------------

        if counts["PENDING"] > 0:

            print()
            print("[WARNING] " f"{counts['PENDING']} image job(s) " "remain PENDING.")

            raise RuntimeError("Not all image jobs were processed.")

        # ====================================================
        # PHASE 3
        # Event-level constrained clustering
        # ====================================================

        observations, assignments = run_constrained_clustering(event_store)

        # ====================================================
        # PHASE 4
        # Final visualization
        # ====================================================

        print()
        print("=" * 60)
        print("VISUAL VALIDATION")
        print("=" * 60)

        visualization_dir = visualizeClusters(
            event_path=EVENT_PATH,
            observations=observations,
            cluster_assignments=assignments,
        )

        print()
        print("Cluster visualizations created at:")

        print(f"  {visualization_dir}")

        # ----------------------------------------------------
        # Release event-level observations.
        # ----------------------------------------------------

        del observations
        del assignments

        gc.collect()

        # ====================================================
        # Final summary
        # ====================================================

        total_time = time.perf_counter() - program_start

        final_observation_count = event_store.get_observation_count()

        print()
        print("=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)

        print(f"Total observations saved: " f"{final_observation_count}")

        print(f"Completed images: " f"{counts['COMPLETED']}")

        print(f"Failed images: " f"{counts['FAILED']}")

        print(f"Total execution time: " f"{total_time:.2f}s")

        print(f"Event database: " f"{event_store.database_path}")

    finally:

        # ----------------------------------------------------
        # Close database.
        # ----------------------------------------------------

        event_store.close()

        # ----------------------------------------------------
        # Close any OpenCV windows.
        # ----------------------------------------------------

        cv2.destroyAllWindows()


# ============================================================
# Windows / Python multiprocessing entry point
# ============================================================
#
# VERY IMPORTANT:
#
# This guard is required on Windows when using multiprocessing.
# Without it, child processes can recursively execute main.py.
# ============================================================

if __name__ == "__main__":

    multiprocessing.freeze_support()

    main()
