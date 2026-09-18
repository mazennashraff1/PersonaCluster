# PersonaCluster Configuration

PersonaCluster centralizes tunable project parameters in:

```text
app/configuration.py
```

This document describes the configuration groups and the current baseline
values used by the active implementation.

> These values are project-specific starting points. They are not
> universal biometric thresholds and should be calibrated against
> representative datasets.

------------------------------------------------------------------------

## 1. General Quality Range

```python
QUALITY_MIN = 0.0
QUALITY_MAX = 1.0
```

These values define the normalized quality range used by the quality
calculation components.

```text
0.0 → poor
1.0 → strong
```

------------------------------------------------------------------------

## 2. Person Detection

```python
PERSON_MODEL_NAME = "models/yolo11n.pt"

PERSON_DETECTION_THRESHOLD = 0.40

PERSON_DETECTION_DEVICE = None

PERSON_CLASS_ID = 0
```

### `PERSON_MODEL_NAME`

Specifies the YOLO person-detection model.

### `PERSON_DETECTION_THRESHOLD`

Controls the minimum confidence accepted for person detections.

Lower values can increase recall but may introduce more false detections.

Higher values can reduce false detections but may miss difficult or
partially visible people.

### `PERSON_DETECTION_DEVICE`

Controls the execution device when supported by the detector.

Examples include:

```text
None
cpu
cuda
cuda:0
```

`None` allows the underlying framework to select its default device.

### `PERSON_CLASS_ID`

The COCO class ID used for person detection.

The configured value is:

```text
0 → person
```

------------------------------------------------------------------------

## 3. Face Detection

```python
FACE_MODEL_NAME = "buffalo_l"

FACE_DETECTION_SIZE = (640, 640)

FACE_DETECTION_THRESHOLD = 0.40

FACE_CTX_ID = 0
```

The face detector and face-recognition components use the configured
InsightFace model.

### `FACE_DETECTION_THRESHOLD`

Controls the minimum confidence accepted for detected faces.

### `FACE_DETECTION_SIZE`

Defines the configured face-detection input size.

### `FACE_CTX_ID`

Controls the InsightFace execution context:

```text
0  → GPU
-1 → CPU
```

------------------------------------------------------------------------

## 4. Face/Person Association

```python
ASSOCIATION_MIN_SCORE = 0.30
```

Controls the minimum geometric association score used when assigning
detected faces to detected people.

Association considers geometric relationships such as:

* Face/person overlap
* Face-center containment

A missing face does not automatically discard a person observation.

------------------------------------------------------------------------

## 5. Body Encoder

```python
BODY_MODEL_NAME = "osnet_x1_0"

BODY_MODEL_PATH = None

BODY_DEVICE = None

BODY_UPPER_BODY_RATIO = 0.60
```

### `BODY_MODEL_NAME`

Specifies the TorchReID body-appearance model.

### `BODY_MODEL_PATH`

Controls an optional local model path.

`None` means the normal pretrained model is used.

### `BODY_DEVICE`

Controls the body-encoder execution device where supported.

Examples include:

```text
None
cpu
cuda
cuda:0
```

### `BODY_UPPER_BODY_RATIO`

Controls the proportion of the person bounding-box height used for the
body-appearance crop.

The baseline uses approximately:

```text
60%
```

of the detected person height.

------------------------------------------------------------------------

## 6. Face Quality

```python
FACE_QUALITY_DETECTION_WEIGHT = 0.40

FACE_QUALITY_SIZE_WEIGHT = 0.30

FACE_QUALITY_SHARPNESS_WEIGHT = 0.30

FACE_MIN_SIZE = 40

FACE_REFERENCE_SIZE = 120

FACE_SHARPNESS_MIN = 20.0

FACE_SHARPNESS_MAX = 150.0
```

These parameters control the calculation of face quality.

The quality calculation combines:

```text
Face detection confidence
Face size
Image sharpness
```

The configured weights are:

```text
Detection confidence → 40%
Face size            → 30%
Sharpness             → 30%
```

### Face size calibration

```text
FACE_MIN_SIZE = 40
```

defines the minimum useful face dimension.

```text
FACE_REFERENCE_SIZE = 120
```

defines the face dimension at which the size-quality component reaches
its maximum.

### Sharpness calibration

```text
FACE_SHARPNESS_MIN = 20.0
FACE_SHARPNESS_MAX = 150.0
```

define the configured sharpness range used when normalizing face
sharpness.

------------------------------------------------------------------------

## 7. Body Quality

```python
BODY_QUALITY_DETECTION_WEIGHT = 0.40

BODY_QUALITY_SIZE_WEIGHT = 0.30

BODY_QUALITY_SHARPNESS_WEIGHT = 0.30

BODY_MIN_SIZE = 100

BODY_REFERENCE_SIZE = 400

BODY_SHARPNESS_MIN = 20.0

BODY_SHARPNESS_MAX = 150.0
```

These parameters control the calculation of body quality.

The configured weights are:

```text
Detection confidence → 40%
Body size            → 30%
Sharpness             → 30%
```

### Body size calibration

```text
BODY_MIN_SIZE = 100
```

defines the minimum useful body dimension.

```text
BODY_REFERENCE_SIZE = 400
```

defines the body dimension at which the size-quality component reaches
its maximum.

### Sharpness calibration

```text
BODY_SHARPNESS_MIN = 20.0
BODY_SHARPNESS_MAX = 150.0
```

define the configured sharpness range used when normalizing body
sharpness.

------------------------------------------------------------------------

## 8. Standard Identity Clustering Weights

```python
CLUSTER_FACE_WEIGHT = 0.65

CLUSTER_BODY_WEIGHT = 0.20

CLUSTER_QUALITY_WEIGHT = 0.15
```

These weights define the baseline balance between:

```text
Face
Body
Quality
```

The intended hierarchy is:

```text
Face
  ↓
Primary identity signal

Body
  ↓
Supporting appearance evidence

Quality
  ↓
Evidence reliability
```

The weights normally add up to:

```text
1.0
```

------------------------------------------------------------------------

## 9. Face-Only Identity Clustering

```python
CLUSTER_FACE_ONLY_WEIGHT = 0.75

CLUSTER_FACE_ONLY_QUALITY_WEIGHT = 0.25
```

These weights are used when an observation contains valid face
information but does not have valid body information.

The score then relies on:

```text
Face    → 75%
Quality → 25%
```

------------------------------------------------------------------------

## 10. Identity Matching Thresholds

```python
MIN_FACE_SIMILARITY = 0.55

MERGE_THRESHOLD = 0.78
```

### `MIN_FACE_SIMILARITY`

A face similarity below this value cannot create an identity merge, 
regardless of other evidence.

This is a conservative guard against weak facial matches.

### `MERGE_THRESHOLD`

Defines the minimum combined identity score required for a compatible
candidate pair to be merged.

These values are project parameters and should not be interpreted as
universal biometric thresholds.

------------------------------------------------------------------------

## 11. Cluster Representation

```python
MIN_CLUSTER_SIZE = 2

REPRESENTATIVE_COUNT = 3
```

### `MIN_CLUSTER_SIZE`

Defines the minimum number of observations required for a cluster to be
considered a discovered identity.

### `REPRESENTATIVE_COUNT`

Controls the number of strong observations considered when representing
a cluster during the clustering process.

These representatives are internal clustering information and are not a
separate output component.

------------------------------------------------------------------------

## 12. Face Pose

```python
FACE_POSE_FRONTAL_YAW_DEGREES = 20.0

FACE_POSE_PROFILE_YAW_DEGREES = 55.0
```

Face pose is estimated from InsightFace's five facial landmarks using
OpenCV `solvePnP` .

The coarse interpretation is:

```text
|yaw| <= 20°
    → frontal

|yaw| >= 55°
    → profile
```

Intermediate values are categorized according to yaw direction.

The pose system is intended for coarse identity and image-selection
decisions rather than precise 3D head-pose measurement.

------------------------------------------------------------------------

## 13. Cross-Pose Identity Matching

```python
CROSS_POSE_MIN_FACE_SIMILARITY = 0.42

CROSS_POSE_MIN_BODY_SIMILARITY = 0.45

CROSS_POSE_MERGE_THRESHOLD = 0.66
```

Cross-pose matching has dedicated requirements because facial similarity
can decrease when the same person is viewed from different angles.

The cross-pose path is intentionally more permissive than standard
matching while remaining subject to the clustering constraints.

The configured requirements are:

```text
Minimum face similarity
Minimum body similarity
Cross-pose merge threshold
```

These values should be calibrated against representative event data.

------------------------------------------------------------------------

## 14. Identity Anchor Rules

```python
ANCHOR_MIN_FACE_QUALITY = 0.70

ANCHOR_MIN_FACE_DETECTION_CONFIDENCE = 0.70

ANCHOR_MIN_FACE_SIZE = 60

ANCHOR_MAX_YAW_DEGREES = 20.0
```

A trusted anchor is a strong full-face observation suitable for
establishing a discovered identity.

The anchor requirements combine:

```text
Face quality
Face detection confidence
Face size
Frontal pose
```

Conceptually:

```text
Clear
  +
Sufficiently large
  +
High-quality face
  +
Approximately frontal
        │
        ▼
  Trusted Anchor
```

The anchor logic is deliberately stricter than ordinary observation
matching.

Side-view and profile observations can contribute to an already anchored
identity but should not independently establish an identity when they do
not satisfy the anchor requirements.

------------------------------------------------------------------------

## 15. Cluster Representative Pose Diversity

```python
MAX_REPRESENTATIVES_PER_POSE = 2
```

Controls the maximum number of internal cluster representatives retained
for each pose category.

Supported coarse categories include:

```text
frontal
left
right
profile
unknown
```

This helps preserve useful pose diversity when representing a cluster
during identity matching.

------------------------------------------------------------------------

## 16. Best Image Selection

```python
BEST_IMAGES_PER_CLUSTER = 5

BEST_IMAGES_REQUIRE_POSE_DIVERSITY = True
```

Best-image selection is performed by `EventOutputManager` .

It does not modify identity assignments.

The current best-image score uses:

```text
50% face quality
25% face detection confidence
15% person detection confidence
10% face/person association
```

The selection process first keeps the strongest observation available
for each source image and can prefer pose diversity when configured.

The selected images are written to:

```text
clusterXX/bestImages/
```

There is no separate `BestImageSelector` module in the current
implementation.

------------------------------------------------------------------------

## 17. Representative Face Selection

```python
REPRESENTATIVE_FACE_QUALITY_WEIGHT = 0.70

REPRESENTATIVE_FACE_DETECTION_WEIGHT = 0.30

FACE_PADDING_RATIO = 0.25
```

These parameters control selection and cropping of the single
representative face generated by `EventOutputManager` .

The representative-face score uses:

```text
70% face quality
30% face detection confidence
```

Only observations with a valid face embedding and face bounding box can
be selected.

After selection, the face is cropped with the configured padding and
saved directly inside the cluster directory as:

```text
clusterXX/face.jpg
```

------------------------------------------------------------------------

## 18. Worker Configuration

```python
WORKER_COUNT = 2

WORKER_STALE_TIMEOUT_SECONDS = 600

WORKER_USE_PROCESSES = True
```

### `WORKER_COUNT`

Defines the number of image worker processes.

Each worker creates its own:

* SQLite connection
* PersonPipeline
* Person detector
* Face detector
* Body encoder
* Other required model state

Increasing the worker count can therefore increase memory and GPU VRAM
usage.

For a GPU with limited VRAM, a conservative starting point is:

```python
WORKER_COUNT = 1
```

The configured project baseline is currently:

```python
WORKER_COUNT = 2
```

### `WORKER_STALE_TIMEOUT_SECONDS`

Defines how long a `PROCESSING` image job can remain active before it can
be considered stale and recovered.

The current value is:

```text
600 seconds
```

### `WORKER_USE_PROCESSES`

Controls the worker execution architecture.

The current implementation uses multiprocessing:

```python
WORKER_USE_PROCESSES = True
```

The current `main.py` does not enable a thread-based worker path.

------------------------------------------------------------------------

## 19. Reference-Matched Event Output

```python
REFERENCES_PATH = "data/reference"
REFERENCE_MATCH_THRESHOLD = 0.55
REFERENCE_MATCH_MIN_MARGIN = 0.02

EVENT_OUTPUT_DIRECTORY_NAME = "output"
EVENT_OUTPUT_MODE = "REFERENCE_MATCHED"
EVENT_OUTPUT_CLEAN_BEFORE_RUN = True

BEST_IMAGES_PER_CLUSTER = 5
BEST_IMAGES_REQUIRE_POSE_DIVERSITY = True
```

The active implementation uses **reference-matched output only**. The older
`CLUSTERING_ONLY`, `BEST_IMAGES_ONLY`, and `BOTH` output modes are no longer
supported.

### Reference matching

Reference files use:

```text
Person Name - Phone Number.ext
```

The name becomes the final output folder name and the phone number is retained
as metadata for the final Excel report.

### `REFERENCE_MATCH_THRESHOLD`

Minimum similarity required for a discovered cluster to be assigned to a known
reference person.

### `REFERENCE_MATCH_MIN_MARGIN`

Minimum score margin required when comparing competing reference-person
matches. Ambiguous matches are not exported.

### Final output structure

```text
output/
├── Person Name/
│   ├── All Images/
│   ├── Best Images/
│   └── representative Image.jpg
└── final_results.xlsx
```

### Best-image selection

`BEST_IMAGES_PER_CLUSTER` controls how many strong images are retained from
each matched cluster. `BEST_IMAGES_REQUIRE_POSE_DIVERSITY` controls whether
the selector attempts to preserve different face poses.

### `final_results.xlsx`

The final Excel file is generated after person folders have been created:

```text
Name of Person | Phone Number | Folder Shared Link
```

The `Folder Shared Link` column is populated with the Google Drive folder URL
when Google Drive upload is enabled. If Google Drive is disabled, the link is
left blank.

---

## 20. Google Drive Output

Google Drive settings are read from `.env` by `main.py`, not from
`app/configuration.py`.

```text
GOOGLE_DRIVE_ENABLED
GOOGLE_DRIVE_CREDENTIALS_PATH
GOOGLE_DRIVE_TOKEN_PATH
GOOGLE_DRIVE_ROOT_FOLDER_ID
GOOGLE_DRIVE_FOLDER_NAME
GOOGLE_DRIVE_PUBLIC_LINK
GOOGLE_DRIVE_RETRY_COUNT
GOOGLE_DRIVE_HTTP_TIMEOUT_SECONDS
```

The active upload sequence is:

```text
Local person folders
        ↓
Google Drive event folder
        ↓
Upload each person folder
        ↓
Create read-access link when enabled
        ↓
Generate final_results.xlsx with those links
        ↓
Upload final_results.xlsx to the event folder
```

`GOOGLE_DRIVE_PUBLIC_LINK=true` makes each person's folder available through
an "Anyone with the link" viewer permission. This is the link written into
`final_results.xlsx`.

---

## 21. Event Input Path

```python
EVENTS_PATH = "data/events"
```

Defines the root directory containing event folders and images.

The main application recursively searches this location for supported
image files.

Supported image extensions include:

```text
.jpg
.jpeg
.png
.bmp
.webp
.tif
.tiff
```

------------------------------------------------------------------------

## 22. Model Summary

The current baseline model configuration is:

```text
Person detector:
    models/yolo11n.pt

Face:
    buffalo_l

Body:
    osnet_x1_0
```

Device configuration is controlled independently where supported:

```text
Person detector:
    PERSON_DETECTION_DEVICE

Face:
    FACE_CTX_ID

Body:
    BODY_DEVICE
```

------------------------------------------------------------------------

## 23. Configuration Principles

`app/configuration.py` is the single source of truth for tunable project
parameters.

Avoid redefining configurable thresholds, weights, model names, or
output settings independently inside other modules.

When tuning the system:

1. Change the corresponding value in `app/configuration.py`.
2. Run the system on representative event data.
3. Inspect the resulting clusters and selected images.
4. Measure false merges and fragmentation.
5. Compare the results against the previous configuration.
6. Keep changes that improve measured performance.

Thresholds should be changed systematically rather than independently
without evaluation.

------------------------------------------------------------------------

## 24. Recommended Baseline

For a new environment, begin conservatively with:

```python
WORKER_COUNT = 1
```

when GPU memory is limited.

The normal output configuration is:

```python
EVENT_OUTPUT_MODE = "REFERENCE_MATCHED"

EVENT_OUTPUT_CLEAN_BEFORE_RUN = True
```

Keep the identity-clustering thresholds unchanged until a representative
dataset has been evaluated.

Then calibrate:

```text
Face similarity
Merge threshold
Cross-pose thresholds
Anchor requirements
Quality thresholds
Worker count
```

using representative labeled event data.

------------------------------------------------------------------------

## 25. Configuration Responsibility Map

The major configuration groups map to the active components as follows:

```text
configuration.py
       │
       ├── Person Detection
       │       └── PersonDetector
       │
       ├── Face Detection
       │       └── FaceDetector
       │
       ├── Face/Person Association
       │       └── AssociationEngine
       │
       ├── Body Encoding
       │       └── BodyEncoder
       │
       ├── Face / Body Quality
       │       └── QualityCalculator
       │
       ├── Identity Matching
       │       └── ConstrainedIdentityClustering
       │
       ├── Face Pose
       │       └── FacePoseEstimator
       │
       ├── Best Images
       │       └── EventOutputManager
       │
       ├── Representative Face
       │       └── EventOutputManager
       │
       ├── Workers
       │       └── ImageWorker / main.py
       │
       └── Final Event Output
               └── EventOutputManager
```

This keeps the configuration model consistent with the current
execution path and avoids documenting components that are no longer part
of the project.
