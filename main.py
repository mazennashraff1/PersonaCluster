from __future__ import annotations

import os
import gc
from openpyxl import Workbook
import multiprocessing
import time
from dotenv import load_dotenv
from pathlib import Path


from app import configuration as config
from app.clustering.constrainedClustering import (
    ConstrainedIdentityClustering,
)
from app.integrations.googleDriveUploader import GoogleDriveUploader
from app.output.eventOutputManager import EventOutputManager
from app.identity.referenceMatcher import ReferenceMatcher
from app.storage.eventStore import EventStore
from app.workers.imageWorker import run_image_worker

# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
EVENTS_ROOT = (PROJECT_ROOT / config.EVENTS_PATH).resolve()
EVENT_PATH: Path | None = None

load_dotenv()

GOOGLE_DRIVE_ENABLED = (
    os.getenv(
        "GOOGLE_DRIVE_ENABLED",
        "false",
    ).lower()
    == "true"
)

GOOGLE_DRIVE_CREDENTIALS_PATH = os.getenv(
    "GOOGLE_DRIVE_CREDENTIALS_PATH",
    "credentials/credentials.json",
)

GOOGLE_DRIVE_TOKEN_PATH = os.getenv(
    "GOOGLE_DRIVE_TOKEN_PATH",
    "data/google_drive/token.json",
)

GOOGLE_DRIVE_ROOT_FOLDER_ID = os.getenv(
    "GOOGLE_DRIVE_ROOT_FOLDER_ID",
    "",
)

GOOGLE_DRIVE_FOLDER_NAME = os.getenv(
    "GOOGLE_DRIVE_FOLDER_NAME",
    "Event Results",
)

GOOGLE_DRIVE_PUBLIC_LINK = (
    os.getenv(
        "GOOGLE_DRIVE_PUBLIC_LINK",
        "true",
    ).lower()
    == "true"
)

GOOGLE_DRIVE_RETRY_COUNT = int(
    os.getenv(
        "GOOGLE_DRIVE_RETRY_COUNT",
        "3",
    )
)

GOOGLE_DRIVE_HTTP_TIMEOUT_SECONDS = int(
    os.getenv(
        "GOOGLE_DRIVE_HTTP_TIMEOUT_SECONDS",
        "60",
    )
)


# ============================================================
# Get image paths
# ============================================================


def generate_final_excel(
    output_directory: str | Path,
    person_contacts: dict[str, str],
    person_folder_urls: dict[str, str],
) -> Path:
    """Create the final Excel sheet for the matched people."""
    output_directory = Path(output_directory)
    excel_path = output_directory / "final_results.xlsx"

    workbook = Workbook()
    worksheet = workbook.active
    worksheet.title = "Results"

    worksheet.append(
        [
            "Name of Person",
            "Phone Number",
            "Folder Shared Link",
        ]
    )

    for person_directory in sorted(
        (path for path in output_directory.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    ):
        person_name = person_directory.name
        link = person_folder_urls.get(person_name, "")
        worksheet.append(
            [
                person_name,
                person_contacts.get(person_name, ""),
                link,
            ]
        )

        if link:
            cell = worksheet.cell(row=worksheet.max_row, column=3)
            cell.hyperlink = link
            cell.style = "Hyperlink"

    worksheet.freeze_panes = "A2"
    worksheet.auto_filter.ref = worksheet.dimensions
    worksheet.column_dimensions["A"].width = 30
    worksheet.column_dimensions["B"].width = 22
    worksheet.column_dimensions["C"].width = 70

    workbook.save(excel_path)
    return excel_path


def get_image_paths(
    event_path: str | Path,
):
    """
    Recursively yield image paths from the event directory.

    Every subfolder is searched until an actual image file
    is found.

    Example:

        EVENT_PATH/
        ├── folder1/
        │   ├── image1.jpg
        │   └── subfolder/
        │       └── image2.jpg
        │
        ├── folder2/
        │   └── image3.png
        │
        └── image4.jpg

    All four images will be discovered.

    Images are yielded one at a time rather than creating a
    large list in memory.
    """

    event_path = Path(event_path)

    # Only the event Gallery is an input source. The event reference folder,
    # event.db, and any other event-level files must never be treated as
    # gallery images.
    gallery_path = event_path / "Gallery"
    if not gallery_path.is_dir():
        return
    event_path = gallery_path

    valid_extensions = {
        ".jpg",
        ".jpeg",
        ".avif",
        ".png",
        ".bmp",
        ".webp",
        ".tif",
        ".tiff",
    }

    # --------------------------------------------------------
    # rglob("*") recursively walks through ALL subdirectories.
    #
    # We only yield actual files with supported image
    # extensions.
    # --------------------------------------------------------

    for image_path in event_path.rglob("*"):

        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in valid_extensions:
            continue

        # ----------------------------------------------------
        # Yield the exact physical image path.
        #
        # EventStore converts this absolute runtime path into the
        # portable project-relative ID:
        #
        #     data/events/<folders>/<filename>
        #
        # The complete folder hierarchy is retained so duplicate
        # filenames remain distinguishable.
        # ----------------------------------------------------

        yield image_path.resolve()


# ============================================================
# Create image jobs
# ============================================================


def prepare_image_jobs(
    event_store: EventStore,
) -> int:
    """
    Discover event images recursively and create their
    processing jobs.

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

    print(f"Configured workers: {worker_count}")

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
        print(f"All workers finished in {worker_time:.2f}s")

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
            print("[WARNING] Some worker processes exited unexpectedly:")

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
    print("Cluster assignments saved to database.")

    return observations, assignments


# ============================================================
# Main
# ============================================================


def process_event(event_path: str | Path):
    """
    Process one event directory.

    The event directory must contain:
        Gallery/      recursively nested event images
        reference/    event-local reference images
    """

    global EVENT_PATH
    EVENT_PATH = Path(event_path).resolve()

    if not EVENT_PATH.is_dir():
        raise FileNotFoundError(f"Event directory does not exist: {EVENT_PATH}")

    gallery_path = EVENT_PATH / "Gallery"
    reference_path = EVENT_PATH / "reference"

    if not gallery_path.is_dir():
        raise FileNotFoundError(f"Missing Gallery directory: {gallery_path}")

    if not reference_path.is_dir():
        print(f"[WARNING] Reference directory does not exist: {reference_path}")

    program_start = time.perf_counter()

    print("=" * 60)
    print("STARTING EVENT PERSON IDENTITY PIPELINE")
    print("=" * 60)

    print(f"Event path: {EVENT_PATH}")

    print(f"Worker count: {config.WORKER_COUNT}")

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
        # Check first worker results
        # ====================================================

        counts = print_job_statistics(event_store)

        # ----------------------------------------------------
        # Normally this should be zero.
        #
        # If a worker crashed while processing an image,
        # its job can remain PROCESSING.
        # ----------------------------------------------------

        if counts["PROCESSING"] > 0:

            raise RuntimeError(
                "Image processing did not finish cleanly. "
                "PROCESSING jobs remain in the database."
            )

        # ====================================================
        # PHASE 2.6
        # Retry failed images once
        # ====================================================

        if counts["FAILED"] > 0:

            print()
            print("=" * 60)
            print("RETRYING FAILED IMAGES")
            print("=" * 60)

            print(
                f"Found {counts['FAILED']} failed image(s). "
                f"Retrying eligible images once."
            )

            retried = event_store.retry_failed_image_jobs()

            print(f"Images queued for retry: " f"{retried}")

            if retried > 0:

                run_workers()

                counts = print_job_statistics(event_store)

        # ====================================================
        # Final image-processing verification
        # ====================================================

        if counts["PROCESSING"] > 0:

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

        final_failed_jobs = event_store.get_failed_image_jobs()

        if final_failed_jobs:

            print()
            print("=" * 60)
            print("IMAGES THAT FAILED AFTER RETRY")
            print("=" * 60)

            for failed_job in final_failed_jobs:

                print(
                    f"  {failed_job['image_id']} "
                    f"(attempts={failed_job['attempts']})"
                )

                print(f"    Path: " f"{failed_job['image_path']}")

                print(
                    f"    Error: " f"{failed_job['error_message'] or 'Unknown error'}"
                )

        # ====================================================
        # PHASE 3
        # Event-level constrained clustering
        # ====================================================

        observations, assignments = run_constrained_clustering(event_store)

        # ====================================================
        # PHASE 4
        # Match discovered clusters against known references
        # ====================================================
        print()
        print("EVENT REFERENCE MATCHING")
        print("=" * 60)
        # The clustering algorithm is unchanged. A single known person may
        # own multiple cluster IDs (for example frontal + side/profile).
        reference_matcher = ReferenceMatcher(reference_path)
        cluster_person_matches, match_details = reference_matcher.match_clusters(
            observations=observations,
            assignments=assignments,
        )

        # Persist the cluster -> known-person relationship as well as the
        # existing observations.cluster_id values.
        event_store.save_cluster_person_matches(match_details)

        print()
        print(
            f"Accepted reference matches: {len(cluster_person_matches)} " f"cluster(s)"
        )

        # ====================================================
        # PHASE 5
        # Final reference-matched event output
        # ====================================================
        print()
        print("GENERATING FINAL OUTPUT")
        print("=" * 60)

        output_manager = EventOutputManager(
            event_path=EVENT_PATH,
            output_directory_name=config.EVENT_OUTPUT_DIRECTORY_NAME,
            mode=config.EVENT_OUTPUT_MODE,
            clean_before_run=config.EVENT_OUTPUT_CLEAN_BEFORE_RUN,
        )

        output_directory = output_manager.write(
            observations=observations,
            assignments=assignments,
            cluster_person_matches=cluster_person_matches,
        )

        print(f"Output directory: {output_directory}")

        # ====================================================
        # PHASE 6
        # Upload person folders/images and create the final Excel file locally
        # ====================================================

        person_contacts = reference_matcher.get_person_contacts()
        person_folder_urls: dict[str, str] = {}
        google_drive_result = None

        google_drive_uploader = None
        if GOOGLE_DRIVE_ENABLED:
            google_drive_uploader = GoogleDriveUploader(
                credentials_path=GOOGLE_DRIVE_CREDENTIALS_PATH,
                token_path=GOOGLE_DRIVE_TOKEN_PATH,
                root_folder_id=GOOGLE_DRIVE_ROOT_FOLDER_ID,
                root_folder_name=GOOGLE_DRIVE_FOLDER_NAME,
                public_link=GOOGLE_DRIVE_PUBLIC_LINK,
                public_link_role=getattr(
                    config,
                    "GOOGLE_DRIVE_PUBLIC_LINK_ROLE",
                    "writer",
                ),
                retry_count=GOOGLE_DRIVE_RETRY_COUNT,
                http_timeout_seconds=GOOGLE_DRIVE_HTTP_TIMEOUT_SECONDS,
            )

            google_drive_result = google_drive_uploader.upload_event(
                output_directory=output_directory,
                event_name=EVENT_PATH.name,
            )
            person_folder_urls = google_drive_result["person_folder_urls"]

        excel_path = generate_final_excel(
            output_directory=output_directory,
            person_contacts=person_contacts,
            person_folder_urls=person_folder_urls,
        )

        print(f"[OUTPUT] Final Excel: {excel_path}")

        # The Excel report is generated locally only.
        # Google Drive receives the person folders and their images, not the Excel file.
        if google_drive_result is not None:
            print("[DRIVE] Person folders and images uploaded successfully.")
            print(
                "[DRIVE] Event folder URL: "
                f"{google_drive_result['event_folder_url']}"
            )

        # ----------------------------------------------------
        # Release event-level observations.
        # ----------------------------------------------------

        del observations
        del assignments
        del output_directory

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

    print("=" * 60)
    print("PERSONACLUSTER EVENT BATCH")
    print("=" * 60)
    print(f"Events root: {EVENTS_ROOT}")

    if not EVENTS_ROOT.is_dir():
        raise FileNotFoundError(f"Events root does not exist: {EVENTS_ROOT}")

    event_directories = sorted(
        (path for path in EVENTS_ROOT.iterdir() if path.is_dir()),
        key=lambda path: path.name.casefold(),
    )

    if not event_directories:
        print("No event directories found.")
        raise SystemExit(0)

    print(f"Events found: {len(event_directories)}")
    for event_directory in event_directories:
        print(f"  - {event_directory.name}")

    failures = []
    for event_directory in event_directories:
        try:
            process_event(event_directory)
        except Exception as exc:
            failures.append((event_directory.name, exc))
            print()
            print("=" * 60)
            print(f"EVENT FAILED: {event_directory.name}")
            print("=" * 60)
            print(f"Error: {exc}")
            print("Continuing with the next event.")

    print()
    print("=" * 60)
    print("ALL EVENTS COMPLETE")
    print("=" * 60)
    print(f"Processed events: {len(event_directories) - len(failures)}")
    print(f"Failed events: {len(failures)}")
    for event_name, error in failures:
        print(f"  - {event_name}: {error}")

    if failures:
        raise SystemExit(1)
