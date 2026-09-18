# PersonaCluster Troubleshooting

This guide covers common installation, model, image-processing, worker, 
database, clustering, and output problems in the current PersonaCluster
implementation.

------------------------------------------------------------------------

## 1. `torch.cuda.is_available()` Returns `False`

Run:

```bash id="0f9w8u"
python -c "import torch; print(torch.cuda.is_available())"

```

Check:

- NVIDIA driver
- Active virtual environment
- PyTorch installation
- PyTorch CUDA build
- GPU visibility

Do not assume that the system's installed CUDA toolkit version must
exactly match the CUDA runtime bundled with the installed PyTorch build.

If CUDA is unavailable, the project can still use CPU execution where
the configured components support it, although processing may be
significantly slower.

------------------------------------------------------------------------

## 2. CUDA Out of Memory

A common cause is too many worker processes loading independent ML
models.

Reduce:

```python id="2awhgr"
WORKER_COUNT = 1
```

Each worker may instantiate:

```text id="3oh4pm"
YOLO
InsightFace
OSNet / TorchReID

```

and related runtime components.

The memory relationship is approximately:

```text id="x5duf0"
Worker count ↑
      ↓
Model instances ↑
      ↓
VRAM / RAM usage ↑
```

If one worker works but multiple workers fail, benchmark memory usage
before increasing the worker count.

Also check:

* GPU utilization
* VRAM usage
* Other applications using the GPU
* Image size
* Model configuration

------------------------------------------------------------------------

## 3. `cv2.imread()` Returns `None`

Check:

* Source path
* File existence
* File permissions
* File corruption
* File extension
* OpenCV decoding support

If the file exists but cannot be decoded, test it independently with
another image viewer or image-processing library.

Also verify that the path stored for the image job points to the intended
source file.

------------------------------------------------------------------------

## 4. TorchReID Import Fails

Test the installation:

```bash id="d9tvuj"
python -c "import torchreid; print('OK')"

```

If it fails, install Deep-Person-ReID:

```bash id="1v8qz4"
python -m pip install --no-build-isolation git+https://github.com/KaiyangZhou/deep-person-reid.git
```

Then restart the active Python environment and test again.

If the import still fails, check:

```text id="5d4yik"
Python version
PyTorch version
TorchVision version
TorchReID installation
Dependency conflicts

```

------------------------------------------------------------------------

## 5. Core Dependency Import Failure

Run:

```bash id="ys8e9g"
python -c "import numpy, cv2, PIL, scipy, torch, torchvision, ultralytics, insightface, onnxruntime, sklearn, skimage, matplotlib, yaml; print('All core dependencies imported successfully')"
```

If an import fails:

01. Identify the failing package.
02. Confirm the active virtual environment.
03. Check the installed package version.
04. Compare it with `requirements.txt`.
05. Resolve the dependency conflict.
06. Re-run the import test.

Avoid reinstalling the entire environment before identifying the
specific failing dependency.

------------------------------------------------------------------------

## 6. Model File Not Found

The person detector currently expects:

```text id="4guy9g"
models/yolo11n.pt

```

Check that the model exists at the path expected by the application.

The face and body models are resolved through their respective model
libraries and configuration.

For the current baseline:

```text id="c9t3si"
Person detector:
    models/yolo11n.pt

Face:
    buffalo_l

Body:
    osnet_x1_0
```

Do not commit large model weights unless their redistribution is
permitted by the applicable license.

------------------------------------------------------------------------

## 7. Workers Remain in `PROCESSING`

Inspect worker output for:

* Process termination
* CUDA errors
* Model initialization failures
* Image decoding failures
* Database errors
* Unexpected Python exceptions

If a worker terminates unexpectedly, its job may remain in:

```text id="j1wzj4"
PROCESSING

```

The event-processing logic can identify jobs that exceed:

```python id="4c4d1w"
WORKER_STALE_TIMEOUT_SECONDS
```

and recover them.

The current baseline is:

```text id="6m2zqi"
600 seconds

```

After recovery, the job can return to:

```text id="6p9lba"
PENDING
```

for another processing attempt.

------------------------------------------------------------------------

## 8. Jobs Remain `FAILED`

Inspect the error associated with the failed job.

Common causes include:

```text id="gqj0is"
Corrupt image
Unreadable image
Invalid crop
Model error
CUDA out-of-memory
Unexpected embedding shape
Dependency issue
Unexpected runtime exception

```

Fix the underlying problem before repeatedly retrying the same event.

A failed image should remain distinguishable from an image that was
successfully processed but produced no useful person observation.

------------------------------------------------------------------------

## 9. Duplicate Observations Appear

First determine whether the apparent duplicates are actually duplicate
observations or multiple valid observations produced from different
processing states.

Check:

- Whether the event database belongs to an older implementation
- Whether images were reprocessed
- Whether image jobs were recreated
- Whether the same event was run multiple times
- Whether an old database was reused after an architecture change

During development, use a fresh event database when changing the worker,
job, observation, or persistence architecture.

The current implementation does **not** contain a separate
same-image duplicate-suppression stage.

Do not attempt to solve database-level duplicate observations by adding
an obsolete duplicate-suppression configuration.

------------------------------------------------------------------------

## 10. Same Person Appears in Multiple Clusters

This is identity fragmentation.

Review:

```text id="l1t7bw"
MIN_FACE_SIMILARITY
MERGE_THRESHOLD
CROSS_POSE_MIN_FACE_SIMILARITY
CROSS_POSE_MIN_BODY_SIMILARITY
CROSS_POSE_MERGE_THRESHOLD
```

Also inspect:

```text id="3s4t9q"
Face quality
Face pose
Identity anchors
Body quality
Face/body embeddings

```

Do not immediately increase body weight.

Face remains the primary identity signal.

If the observations themselves are poor, fix the image-level pipeline
before making the clustering thresholds more permissive.

------------------------------------------------------------------------

## 11. Different People Are Merged

This is a false merge and should be treated as a serious clustering
failure.

Review:

```text id="i9f7aw"
MIN_FACE_SIMILARITY
MERGE_THRESHOLD
CROSS_POSE_MIN_FACE_SIMILARITY
CROSS_POSE_MIN_BODY_SIMILARITY
CROSS_POSE_MERGE_THRESHOLD
```

Also inspect:

* Face quality
* Face pose
* Body similarity
* Identity anchor behavior
* Same-image constraint
* Observation quality

Do not use the old duplicate-suppression configuration to troubleshoot
this problem; that component is not part of the current architecture.

A false merge generally deserves more attention than moderate
fragmentation because it directly combines different real identities.

------------------------------------------------------------------------

## 12. Profile-Only Observations Do Not Form Identities

This can be expected behavior.

PersonaCluster uses strong, approximately frontal observations as trusted
identity anchors.

An observation with:

```text id="c0yg0z"
Poor face quality
or
Large yaw
or
Small face

```

may contribute to an existing identity but should not independently
establish one when it does not satisfy the anchor requirements.

Review:

```text id="45b24r"
ANCHOR_MIN_FACE_QUALITY
ANCHOR_MIN_FACE_DETECTION_CONFIDENCE
ANCHOR_MIN_FACE_SIZE
ANCHOR_MAX_YAW_DEGREES
```

before changing general clustering thresholds.

------------------------------------------------------------------------

## 13. Unexpected Same-Image Merge

This is a clustering correctness issue.

The current clustering implementation has a hard same-image constraint:

> Two observations originating from the same source image cannot belong
> to the same identity cluster.

Check:

01. Source image identity/path.
02. Observation source-image IDs.
03. Cluster constraint logic.
04. Whether the observations represent different detections in the same
   image.
05. Whether a legacy database is being reused.

A valid final cluster should contain at most one observation from each
source image.

If a violation is reproducible on a clean database, inspect the
clustering implementation rather than attempting to suppress it at the
output stage.

------------------------------------------------------------------------

## 14. Clustering Produces Too Many Unknown Observations

Possible causes include:

* Face detection threshold too high
* Face quality too low
* Identity anchor requirements too strict
* Merge threshold too high
* Cross-pose thresholds too strict
* Poor body embeddings
* Poor image quality
* Insufficient compatible observations

Inspect the observations before changing thresholds.

Useful things to check include:

```text id="j0zqce"
Face availability
Face quality
Face size
Face pose
Body availability
Body quality
Embedding validity

```

If the observations are weak, threshold changes may only hide the
underlying problem.

------------------------------------------------------------------------

## 15. Clustering Is Too Slow

Measure where time is being spent before changing the architecture.

Separate:

```text id="yq27q3"
Image processing
Clustering
Output generation
```

Best-image selection and representative-face generation are performed by
`EventOutputManager` as part of the final output stage.

If image processing dominates, investigate:

* Model inference
* Worker count
* GPU utilization
* Image resolution
* Model initialization

If clustering dominates, investigate:

* Number of observations
* Candidate pair generation
* Pairwise similarity calculations
* Representative calculations
* Cache effectiveness
* Cluster merge operations

------------------------------------------------------------------------

## 16. Increasing Worker Count Makes the System Slower

This can happen because every worker can load its own ML models.

More workers can cause:

```text id="xjwm51"
VRAM pressure
CPU contention
RAM pressure
Model initialization overhead
GPU contention

```

Compare actual throughput with:

```text id="x25jyt"
1 worker
2 workers
...
```

Do not assume that additional workers will improve performance.

The best worker count depends on:

```text id="wq1wtt"
GPU memory
System RAM
CPU
Model sizes
Image resolution
Event size

```

------------------------------------------------------------------------

## 17. Old Results Appear in a New Run

Check:

```text id="g6n9sa"
Event database
output/
```

A stale database can contain observations from previous processing.

Previously generated output can also remain if output cleanup is
disabled.

For a clean development experiment, use:

```text id="z2t0o7"
Fresh event database
+
Fresh generated output

```

Then run the event again.

The current output configuration includes:

```python id="m6u1o7"
EVENT_OUTPUT_CLEAN_BEFORE_RUN = True
```

when enabled.

------------------------------------------------------------------------

## 18. Output Is Missing Images

Check:

* Original image still exists.
* Image path is correct.
* Image ID resolves correctly.
* Output directory is writable.
* Event output mode is correct.
* The source image was successfully processed.
* The cluster actually contains the expected observation.

`EventOutputManager` contains handling for image identifiers used by the
current storage/output flow.

The source image itself must still be available when the final output is
generated.

For output collisions, the output manager uses destination-name
handling so that images with identical basenames from different source
locations do not silently overwrite each other.

------------------------------------------------------------------------

## 19. `bestImages` Does Not Contain the Expected Images

Best-image selection occurs after identity clustering.

Check:

```text id="jpkp9w"
Cluster assignment
Face quality
Face detection confidence
Person detection confidence
Face/person association
Pose diversity
BEST_IMAGES_PER_CLUSTER
BEST_IMAGES_REQUIRE_POSE_DIVERSITY

```

The selection process does not change the cluster assignment.

Therefore, if the selected images look wrong while the cluster itself is
correct, investigate `EventOutputManager` and the image-selection
configuration rather than clustering first.

------------------------------------------------------------------------

## 20. `face.jpg` Is Missing or Looks Poor

The representative face is generated by `EventOutputManager`.

Check:

```text id="xk5f70"
Valid face observations
Face embedding availability
Face bounding box
Face quality
Face detection confidence
REPRESENTATIVE_FACE_QUALITY_WEIGHT
REPRESENTATIVE_FACE_DETECTION_WEIGHT
FACE_PADDING_RATIO
```

The expected output is:

```text id="n2r7qv"
output/
└── cluster01/

    ├── allImages/
    ├── bestImages/
    └── face.jpg

```

There is no separate representative-face directory in the current
architecture.

------------------------------------------------------------------------

## 1. Output Directory Structure Is Unexpected

The current output structure is:

```text
output/
├── Person Name/
│   ├── All Images/
│   ├── Best Images/
│   └── representative Image.jpg
└── final_results.xlsx
```

The application currently uses:

```python
EVENT_OUTPUT_MODE = "REFERENCE_MATCHED"
EVENT_OUTPUT_DIRECTORY_NAME = "output"
EVENT_OUTPUT_CLEAN_BEFORE_RUN = True
```

The old `CLUSTERING_ONLY`, `BEST_IMAGES_ONLY`, and `BOTH` modes are not part of
the active implementation.

If a person is missing, first check reference matching and the reference
filename format:

```text
Person Name - Phone Number.ext
```

Only clusters assigned to a known reference person are exported.

---

## 2. Final Excel Is Missing or Empty

The final report is created at:

```text
output/final_results.xlsx
```

It contains:

```text
Name of Person
Phone Number
Folder Shared Link
```

Check:

* `data/reference` contains correctly named reference files.
* Reference matching produced exported people.
* The output directory is writable.
* `openpyxl` is installed.
* The application reached the final output phase.

If Google Drive is disabled, the `Folder Shared Link` column can be blank.
This is expected.

---

## 3. Google Drive Folder Links Are Missing

The Excel links are generated from the actual person-folder IDs returned by
Google Drive.

Check:

```text
GOOGLE_DRIVE_ENABLED=true
GOOGLE_DRIVE_CREDENTIALS_PATH
GOOGLE_DRIVE_TOKEN_PATH
GOOGLE_DRIVE_ROOT_FOLDER_ID
GOOGLE_DRIVE_PUBLIC_LINK=true
```

The upload order is:

```text
Person folders
    ↓
Drive folder URLs
    ↓
final_results.xlsx
    ↓
Excel uploaded to event folder
```

If Drive upload is disabled or fails before folder URLs are returned, the
Excel report can still be generated locally, but its link column will not
contain Drive URLs.

When `GOOGLE_DRIVE_PUBLIC_LINK=true`, each person folder is given an
"Anyone with the link" viewer permission.

---

## 4. No Useful Clusters Are Produced

Before changing clustering thresholds, inspect the number and quality of
observations.

Check:

```text id="hrv5m0"
Total observations
Face observations
Body observations
Valid embeddings
Face quality
Body quality
Face pose
Trusted anchors

```

A useful diagnostic sequence is:

```text id="xj9x4z"
No clusters
    ↓
Are observations being created?
    ↓
Are face embeddings valid?
    ↓
Are there trusted anchors?
    ↓
Are observations sufficiently similar?
    ↓
Are clustering thresholds too strict?
```

If there are no reliable observations or no suitable anchors, changing
the merge threshold may not solve the problem.

------------------------------------------------------------------------

## 5. Clusters Contain Unexpected People

Inspect the strongest evidence connecting the observations.

Check:

```text id="2z5s5c"
Face similarity
Body similarity
Quality
Pose
Trusted anchor
Source images

```

Then determine whether the problem originated from:

```text id="j4z7z1"
Incorrect detection
Incorrect association
Poor embedding
Poor quality estimate
Incorrect pose
Clustering decision
```

This distinction is important because clustering should not compensate
for unreliable upstream observations.

------------------------------------------------------------------------

## 6. Before Reporting a Bug

Collect:

```text id="x6kq8b"
Python version
Operating system
PyTorch version
CUDA availability
GPU model
Worker count
Relevant configuration
Number of images
Number of observations
Cluster count
Job states
Error traceback

```

For reproducible issues, also provide:

```text id="h48t6s"
Smallest dataset that reproduces the problem
```

When possible, include the configuration values relevant to the failing
component.

Do not upload real people's photographs, face embeddings, or sensitive
event databases to a public issue tracker.

Use synthetic, consented, or appropriately sanitized examples.

------------------------------------------------------------------------

## 7. General Debugging Order

When something fails, investigate the active pipeline in this order:

```text id="t6wq1p"
01. Environment
02. Dependencies
03. Model loading
04. Image decoding
05. Person detection
06. Face detection
07. Face/person association
08. Face/body embeddings
09. Quality calculation
10. Embedding validation
11. Observation persistence
12. Worker/job state
13. Identity clustering
14. Best-image selection
15. Representative-face generation
16. Final output

```

This follows the current PersonaCluster execution path and helps isolate
the failing layer.

------------------------------------------------------------------------

## 8. Debugging Principle

When an identity result is incorrect, do not immediately modify the
clustering thresholds.

Trace the observation through the pipeline:

```text id="q4m1o4"
Source Image
     ↓
Detection
     ↓
Association
     ↓
Embeddings
     ↓
Quality
     ↓
Validation
     ↓
Persisted Observation
     ↓
Identity Clustering
     ↓
EventOutputManager
     ↓
Final Output
```

Ask:

> **Is the observation wrong, or is the identity decision wrong?**

If the observation is wrong, fix the image-level component.

If the observation is reliable but the identity assignment is wrong, 
investigate clustering.

If clustering is correct but the final images are wrong, investigate
`EventOutputManager` .

This separation keeps troubleshooting focused and prevents one component
from compensating for another component's failures.
