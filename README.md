# PersonaCluster

### Event-Based Person Detection, Recognition, Clustering, and Best-Image Selection

PersonaCluster is an event-based computer-vision system that processes collections of photographs, detects people, extracts face and body appearance representations, creates quality-aware person observations, discovers anonymous identity clusters, and generates a structured output containing all images, selected best images, and a representative face for each discovered identity.

It is designed for **closed photographic events** such as:

* Conferences
* Weddings
* Parties
* Sports events
* Graduations
* Corporate events
* Other event-based photo collections

PersonaCluster does **not** require a permanent named-person gallery. Instead, it discovers anonymous person clusters within an event that can be reviewed and labeled afterward.

---

## Overview

The system separates **image processing** from **identity discovery**.

Each image is processed independently to produce structured person observations. After image processing is complete, the complete event observation set is passed to constrained identity clustering.

```text
Event Images
     │
     ▼
Recursive Image Discovery
     │
     ▼
SQLite Job Queue
     │
     ▼
Parallel Image Workers
     │
     ├── Person Detection
     ├── Face Detection + Embedding
     ├── Face/Person Association
     ├── Face Pose Estimation
     ├── Body/Re-ID Embedding
     ├── Quality Assessment
     └── Embedding Validation
     │
     ▼
Person Observations
     │
     ▼
SQLite Event Database
     │
     ▼
Constrained Identity Clustering
     │
     ├── Face Similarity
     ├── Body Similarity
     ├── Quality
     ├── Pose-Aware Matching
     ├── Identity Anchors
     └── Same-Image Constraints
     │
     ▼
Cluster Assignments
     │
     ▼
EventOutputManager
     │
     ├── All Images
     ├── Best Images
     └── Representative Face
     │
     ▼
Final Event Output
```

The core architectural principle is:

> **Image processing happens independently per image; identity clustering happens only after the event observations are available.**

---

# Key Features
* Recursive event image discovery
* Support for nested image directories
* YOLO-based person detection
* InsightFace face detection and face embeddings
* Face-to-person association
* Coarse face pose estimation
* TorchReID/OSNet body appearance embeddings
* Face and body quality scoring
* Embedding validation
* Face-only, body-only, and face+body observations
* SQLite-backed image processing queue
* Multiprocessing image workers
* Crash and stale-job recovery
* Globally unique observation IDs
* Atomic observation persistence
* Face-primary identity clustering
* Quality-aware identity matching
* Cross-pose matching
* Identity anchor requirements
* Same-image identity constraints
* Best-image selection
* Representative-face generation
* Configurable event output
* Human-review-ready anonymous identity clusters

---

# Architecture

PersonaCluster is divided into five logical layers:

```text
┌─────────────────────────────────────────────┐
│ Image Processing                            │
│ Detection → Association → Embeddings        │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│ Observation & Quality                       │
│ Pose + Quality + Validation                 │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│ Persistence                                 │
│ SQLite Jobs + Observations                  │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│ Identity Discovery                          │
│ Constrained Identity Clustering             │
└──────────────────────┬──────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────┐
│ Output                                      │
│ All Images + Best Images + Representative   │
│ Face                                        │
└─────────────────────────────────────────────┘
```

The system deliberately avoids maintaining shared identity state between
image workers.

```text
Worker 1 ──┐
Worker 2 ──┤
Worker 3 ──┼──► Person Observations ──► Event-Level Clustering
Worker N ──┘
```

This allows image processing to run in parallel while identity discovery
remains centralized at the event level.

---

# How It Works

## 1. Image Discovery

Images are discovered recursively inside the configured event directory.

Supported formats include:

```text
.jpg
.jpeg
.png
.bmp
.webp
.tif
.tiff
```

Nested directories are supported.

Example:

```text
event_001/
├── morning/
│   ├── IMG_001.jpg
│   └── IMG_002.jpg
├── ceremony/
│   └── IMG_003.jpg
└── reception/
    └── camera_A/
        └── IMG_004.jpg
```

---

## 2. Image Processing

Each image is processed independently by an `ImageWorker` .

The active image pipeline is:

```text
Image
  │
  ├──► Person Detection
  │
  ├──► Face Detection + Embedding
  │
  ├──► Face/Person Association
  │
  ├──► Face Pose Estimation
  │
  ├──► Body/Re-ID Embedding
  │
  ├──► Quality Calculation
  │
  └──► Embedding Validation
           │
           ▼
     PersonObservation
```

Workers do not perform global identity clustering.

---

## 3. Person Observations

A `PersonObservation` represents a detected occurrence of a person in a
single image.

It is **not** a confirmed identity.

An observation can contain:

```text
Face + Body
Face only
Body only
```

An observation stores information such as:

* Observation ID
* Source image
* Person bounding box
* Person detection confidence
* Face bounding box
* Face detection confidence
* Face pose
* Face embedding
* Body embedding
* Face quality
* Body quality
* Embedding validity
* Cluster assignment

This separation keeps detection, representation, quality assessment, and
identity discovery as separate responsibilities.

---

# Face and Body Representation

PersonaCluster uses two complementary appearance modalities:

```text
                  Person
                    │
           ┌────────┴────────┐
           ▼                 ▼
         Face              Body
           │                 │
           ▼                 ▼
    Face Embedding      OSNet Embedding
```

## Face

InsightFace provides:

* Face detection
* Facial landmarks
* Face embeddings

Face embeddings are L2-normalized before being used by the identity
clustering system.

The face representation is the **primary identity signal**.

## Body

The system extracts an upper-body region from the detected person
bounding box and generates an appearance embedding using TorchReID and
OSNet.

Body appearance is used as **supporting identity evidence**.

This is important because clothing and body appearance can change between
photographs.

---

# Face Pose Estimation

Face pose is estimated from the facial landmarks produced by InsightFace
using OpenCV's `solvePnP` .

The system estimates:

```text
Yaw
Pitch
Roll
```

and assigns a coarse pose category:

```text
frontal
left
right
profile
unknown
```

Pose information is used for:

* Pose-aware identity matching
* Cross-view comparison
* Representative-face selection
* Cluster refinement

Pose is treated as contextual evidence rather than a direct identity
signal.

---

# Quality Assessment

Face and body observations receive independent quality scores.

Quality considers factors such as:

```text
Detection confidence
Bounding-box resolution
Image sharpness
```

Quality values are normalized to:

```text
0.0 ─────────────── 1.0
poor                strong
```

Quality influences identity matching and output selection but does not
independently determine identity.

---

# Identity Clustering

After image processing is complete, PersonaCluster loads the persisted
event observations and performs constrained identity clustering.

```text
All Observations
       │
       ▼
Constrained Identity Clustering
       │
       ▼
Anonymous Identity Clusters
```

The clustering system uses:

* Face similarity
* Body similarity
* Observation quality
* Facial pose
* Identity anchors
* Same-image constraints

Face remains the primary identity signal.

The clustering strategy is intentionally conservative.

The goal is:

```text
High precision
     >
Maximum recall
```

A false merge can contaminate an entire identity cluster, so uncertain
observations may remain separated until stronger evidence becomes
available.

---

# Same-Image Constraint

Two different observations originating from the same source image cannot
be assigned to the same identity cluster.

For example:

```text
IMG_001.jpg

Person A
Person B
Person C
```

must not result in:

```text
Cluster 1
├── Observation A
└── Observation B
```

when those observations represent different detected people from the
same image.

This is a hard clustering constraint and an important correctness rule.

There is **no separate duplicate-suppression stage** in the current
architecture.

---

# Identity Anchors

PersonaCluster uses trusted identity anchors to establish strong
candidate identities.

The anchor concept is based on:

* Good face quality
* Strong face detection confidence
* Sufficient face resolution
* Approximately frontal face pose

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

This prevents weak profile or side-view observations from independently
establishing an identity when they do not satisfy the anchor
requirements.

Once an identity has a trusted anchor, observations from different poses
can contribute to that identity.

---

# Cross-Pose Matching

The clustering system supports matching observations across different
facial viewpoints.

Standard same-pose matching can use stronger requirements, while
cross-pose matching uses dedicated thresholds.

For example:

```text
Frontal
   │
   ├──► Left
   ├──► Right
   └──► Profile
```

This allows the system to connect observations of the same person across
different viewpoints without treating weak cross-pose evidence as
equivalent to a strong frontal comparison.

Cross-pose matching is controlled through the configuration values in:

```text
app/configuration.py
```

---

# Best Image Selection

After identity clustering, `EventOutputManager` selects strong source
images for each discovered identity.

The selection process considers:

* Face quality
* Face detection confidence
* Person detection confidence
* Face/person association
* Pose diversity

The selector does **not** change cluster assignments.

Conceptually:

```text
Cluster
   │
   ▼
Candidate Images
   │
   ▼
Quality + Confidence + Pose Diversity
   │
   ▼
Best Images
```

Selected images are written to:

```text
clusterXX/bestImages/
```

Best-image selection is part of `EventOutputManager` ; there is no
separate `BestImageSelector` module in the current project.

---

# Representative Face

`EventOutputManager` also selects a single representative face for each
cluster.

The representative face is selected using face quality and face
detection confidence.

The resulting crop is stored directly inside the cluster directory:

```text
cluster01/
└── face.jpg
```

The representative face is an output artifact and does not modify the
identity clustering result.

---

# Final Event Output

The active final-output mode is **reference matched**.

After identity clustering, the system matches discovered clusters against the
known people in `data/reference`. Only matched people are exported.

Reference files use:

```text
Person Name - Phone Number.ext
```

The person's name becomes the output folder name and the phone number is used
in the final Excel report.

The local output structure is:

```text
output/
├── Person Name/
│   ├── All Images/
│   ├── Best Images/
│   └── representative Image.jpg
│
└── final_results.xlsx
```

If multiple discovered clusters belong to the same reference person, their
images are combined into the same person directory.

## Final Excel Report

`final_results.xlsx` contains exactly these columns:

```text
Name of Person | Phone Number | Folder Shared Link
```

The report is generated after the person folders are built.

When Google Drive is enabled, the application uploads each person folder,
obtains its Drive folder link, writes that link into the Excel report, and
then uploads the completed Excel file to the event folder.

## Google Drive Output

The Google Drive structure is:

```text
Event Results/
└── Event Name/
    ├── Person Name/
    │   ├── All Images/
    │   ├── Best Images/
    │   └── representative Image.jpg
    ├── Another Person/
    │   ├── All Images/
    │   ├── Best Images/
    │   └── representative Image.jpg
    └── final_results.xlsx
```

When `GOOGLE_DRIVE_PUBLIC_LINK=true`, each person's folder is shared using an
"Anyone with the link" viewer permission. The resulting folder URL is written
to the `Folder Shared Link` column.

There is **no `manifest.json`** in the current output.

---

# Project Structure

The current active project structure is:

```text
PersonaCluster/
│
├── app/
│   ├── configuration.py
│   ├── pipeline.py
│   │
│   ├── models/
│   │   ├── detector.py
│   │   └── observation.py
│   │
│   ├── detection/
│   │   ├── personDetector.py
│   │   ├── faceDetector.py
│   │   └── association.py
│   │
│   ├── identity/
│   │   └── facePoseEstimator.py
│   │
│   ├── embeddings/
│   │   └── bodyEncoder.py
│   │
│   ├── quality/
│   │   └── qualityCalculator.py
│   │
│   ├── validation/
│   │   └── embeddingValidator.py
│   │
│   ├── storage/
│   │   └── eventStore.py
│   │
│   ├── workers/
│   │   └── imageWorker.py
│   │
│   ├── clustering/
│   │   └── constrainedClustering.py
│   │
│   └── output/
│       └── eventOutputManager.py
│
├── data/
│   └── events/
│
├── models/
│   └── yolo11n.pt
│
├── docs/
│   └── ...
│
├── main.py
├── requirements.txt
└── README.md
```

---

# Models and Technologies

| Component | Model / Technology |
|---|---|
| Person detection | Ultralytics YOLO11n |
| Face detection | InsightFace |
| Face embedding | InsightFace `buffalo_l` |
| Body/Re-ID | TorchReID `osnet_x1_0` |
| Face pose | OpenCV `solvePnP` |
| Persistence | SQLite |
| Image processing | OpenCV / NumPy |
| Identity clustering | Custom constrained clustering |

---

# Requirements

Recommended environment:

```text
OS:
    Windows 10/11 or Linux

Python:
    3.10 or 3.11

RAM:
    16 GB or more recommended

GPU:
    NVIDIA GPU recommended for large events

Storage:
    SSD recommended
```

CPU execution may be possible depending on the configured models, but
large image collections will generally be significantly slower.

---

# Installation

## 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd PersonaCluster
```

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Upgrade Packaging Tools

```bash
python -m pip install --upgrade pip setuptools wheel
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## 5. Install TorchReID

If TorchReID is not already installed:

```bash
python -m pip install --no-build-isolation git+https://github.com/KaiyangZhou/deep-person-reid.git
```

---

# Verify the Environment

## Verify PyTorch and CUDA

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Verify Core Dependencies

```bash
python -c "import numpy, cv2, PIL, scipy, torch, torchvision, ultralytics, insightface, onnxruntime, sklearn, skimage, matplotlib, yaml; print('All core dependencies imported successfully')"
```

Verify TorchReID separately:

```bash
python -c "import torchreid; print('TorchReID imported successfully')"
```

---

# Model Configuration

The person detector currently uses:

```text
models/yolo11n.pt
```

The configured appearance models are:

```text
InsightFace:
    buffalo_l

TorchReID:
    osnet_x1_0
```

Model and device settings are centralized in:

```text
app/configuration.py
```

---

# Configuration

Project parameters are centralized in:

```text
app/configuration.py
```

Configuration covers:

* Person detection
* Face detection
* Face/person association
* Body extraction
* Face quality
* Body quality
* Identity clustering
* Face pose
* Cross-pose matching
* Identity anchors
* Best-image selection
* Representative-face selection
* Worker configuration
* Event output

Example:

```python
PERSON_DETECTION_THRESHOLD = 0.40
FACE_DETECTION_THRESHOLD = 0.40
ASSOCIATION_MIN_SCORE = 0.30

CLUSTER_FACE_WEIGHT = 0.65
CLUSTER_BODY_WEIGHT = 0.20
CLUSTER_QUALITY_WEIGHT = 0.15

MIN_FACE_SIMILARITY = 0.55
MERGE_THRESHOLD = 0.78

WORKER_USE_PROCESSES = True

EVENT_OUTPUT_MODE = "REFERENCE_MATCHED"
```

Thresholds are **project-specific configuration values**, not universal
biometric thresholds. They should be calibrated against representative
datasets.

---

# Running the Project

Place event photographs inside the configured event directory and run:

```bash
python main.py
```

The application performs the complete event pipeline:

```text
Image Discovery
      ↓
Job Preparation
      ↓
Image Workers
      ↓
Observation Persistence
      ↓
Identity Clustering
      ↓
Cluster Assignments
      ↓
EventOutputManager
      ↓
All Images
Best Images
Representative Face
```

---

# Event Directory

An event can be organized as:

```text
data/
└── events/
    └── event_001/
        ├── IMG_0001.jpg
        ├── IMG_0002.jpg
        ├── IMG_0003.jpg
        └── ...
```

Nested directories are also supported:

```text
data/
└── events/
    └── event_001/
        ├── morning/
        │   ├── IMG_0001.jpg
        │   └── IMG_0002.jpg
        │
        └── reception/
            ├── IMG_0100.jpg
            └── IMG_0101.jpg
```

Each event maintains its own processing state and output.

---

# Multiprocessing

PersonaCluster uses multiprocessing for image processing.

Each worker owns its own:

```text
SQLite connection
PersonPipeline
YOLO model
InsightFace model
OSNet model
```

Workers communicate through the SQLite-backed image-job queue rather than
sharing ML model instances.

This provides process isolation but increases memory and VRAM usage.

For GPUs with limited memory, start with:

```python
WORKER_COUNT = 1
```

and benchmark before increasing the number of workers.

---

# Job Processing and Crash Recovery

Image jobs are tracked in SQLite.

The general lifecycle is:

```text
PENDING
   ↓
PROCESSING
   ↓
COMPLETED
```

or:

```text
PROCESSING
   ↓
FAILED
```

If a worker terminates unexpectedly, a job can remain in:

```text
PROCESSING
```

for longer than the configured stale-job timeout.

Stale jobs can then be recovered and returned to:

```text
PENDING
```

for processing again.

The timeout is controlled by:

```python
WORKER_STALE_TIMEOUT_SECONDS
```

---

# Data Persistence

Each event maintains its own SQLite database.

The database stores processing state including:

```text
Image jobs
Person observations
Cluster assignments
```

SQLite provides the persistent coordination layer between image workers
and the event-level identity-discovery stage.

---

# Evaluation

PersonaCluster should be evaluated at multiple levels.

### Detection

* Person detection precision
* Person detection recall
* Missed detections
* False detections

### Observation Quality

* Valid face rate
* Valid body rate
* Invalid embedding rate
* Face quality
* Body quality
* Association quality

### Identity Clustering

* Cluster purity
* False merge rate
* Identity fragmentation
* Unknown observations
* Same-image constraint violations

### Output Quality

* Best-image quality
* Pose diversity
* Representative-face quality
* Output completeness

### System Performance

* Processing time
* Images per second
* Clustering time
* GPU utilization
* VRAM usage
* RAM usage
* Failed job rate

A successful identity-clustering configuration should aim for:

```text
High cluster purity
+
Low false merges
+
Acceptable fragmentation
+
Acceptable unknown rate
```

rather than simply maximizing the number of merged observations.

See the detailed evaluation documentation for the complete evaluation
procedure.

---

# Limitations

PersonaCluster is intentionally designed as an **event-level identity
discovery system**.

Current limitations include:

* Identity clusters are anonymous until human labeling is applied.
* Cross-event identity recognition is not part of the core architecture.
* Body appearance can change significantly between images.
* Face visibility and image quality affect identity matching.
* Clustering thresholds require calibration for different datasets.
* Very large-scale approximate nearest-neighbor retrieval is not currently
  required by the core implementation.
* The system should not be treated as a universal biometric
  identification system.

---

# Design Principles

## 1. Observations Before Identities

```text
Detection
    ↓
Observation
    ↓
Identity
```

A raw detection is never treated as a confirmed identity.

## 2. Face Is the Primary Identity Signal

Face appearance is the primary identity evidence.

Body appearance provides supporting evidence.

## 3. Quality Matters

Strong observations should provide more reliable evidence than poor
observations.

## 4. Same-Image Merges Are Constrained

Different observations from the same source image cannot simply be
assigned to the same identity.

## 5. Clustering Happens After Image Processing

Workers generate observations.

The complete event observation set is then used for identity clustering.

## 6. Pose Provides Context

Pose helps with cross-view matching and representative selection but is
not itself an identity signal.

## 7. Identity Creation Is Stricter Than Identity Expansion

Strong approximately frontal observations can establish trusted anchors.

Additional observations from different viewpoints can subsequently
contribute to an established identity.

## 8. Output Does Not Change Identity Assignments

`EventOutputManager` operates on already-established clusters.

It selects images and generates the representative face without
changing the identity-clustering result.

## 9. Each Component Has One Responsibility

The current architecture deliberately avoids unused or duplicated
components.

```text
PersonPipeline
    → creates observations

EventStore
    → persists processing state and observations

ConstrainedIdentityClustering
    → discovers anonymous identity clusters

EventOutputManager
    → selects best images
    → creates representative faces
    → writes final output
```

---

# Security and Privacy

PersonaCluster can process photographs containing identifiable people and
generate face and body appearance embeddings.

Do not commit the following to a public repository:

```text
Real event photographs
Real event databases
Face embeddings
Body embeddings
Generated private output
API keys
Passwords
Access tokens
Private configuration
```

Use synthetic, consented, or appropriately licensed sample data when
demonstrating the project.

Also review the licenses and terms of the underlying models and
dependencies before distributing the project.

---

# Documentation

Detailed technical documentation is maintained separately:

```text
docs/
├── architecture.md
├── clustering.md
├── configuration.md
├── development.md
├── evaluation.md
└── troubleshooting.md
```

The README provides the high-level project description, architecture, 
installation, usage, configuration, and limitations.

The documentation files provide deeper technical information about the
individual parts of the system.

---

# Roadmap

Potential future improvements include:

* Human labeling interface
* Cross-event identity recognition
* Automated threshold calibration
* Improved occlusion handling
* Large-scale approximate nearest-neighbor retrieval
* Formal benchmark datasets
* Automated precision/recall evaluation
* Interactive cluster review tools
* More advanced temporal or event-context modeling

These are potential future directions and are not required by the
current core architecture.

---

# License

```text
MIT License
```

---

# Acknowledgements

PersonaCluster builds on open-source computer-vision and machine-learning
projects including:

* Ultralytics
* InsightFace
* TorchReID / Deep-Person-ReID
* PyTorch
* OpenCV
* NumPy
* scikit-learn

Please review and comply with the licenses and terms of the individual
dependencies before distributing the project.

---

# Project Philosophy

> **Do not ask clustering to compensate for bad observations.**

Reliable identity discovery starts with reliable observations.

```text
Reliable Detection
       ↓
Reliable Observations
       ↓
Reliable Persistence
       ↓
Constrained Identity Clustering
       ↓
Useful Image Selection
       ↓
Clear Final Output
```
