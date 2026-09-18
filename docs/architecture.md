# PersonaCluster Architecture

This document describes the internal architecture of PersonaCluster. The
main `README.md` provides the public overview and quick-start
instructions; this document focuses on how image processing, persistence, 
worker coordination, event-level clustering, and output generation
interact.

---

## 1. Architectural Overview

PersonaCluster separates image-level processing from event-level
identity discovery.

```text
Event Images
     │
     ▼
Recursive Image Discovery
     │
     ▼
SQLite Image Job Queue
     │
     ▼
Parallel Image Workers
     │
     ▼
Person Observations
     │
     ▼
SQLite Event Database
     │
     ▼
Event-Level Identity Clustering
     │
     ▼
Cluster Assignments
     │
     ▼
Event Output Manager
     │
     ├── All Images
     ├── Best Images
     └── Representative Face
```

The central architectural boundary is:

> Image processing happens independently per image; identity clustering
> happens only after the event observations are available.

---

## 2. Logical Layers

PersonaCluster can be viewed as five logical layers.

### Image Processing

Responsible for converting an image into person observations.

Components include:

* Person detection
* Face detection
* Face embedding
* Face/person association
* Face pose estimation
* Body/Re-ID embedding
* Quality calculation
* Embedding validation

### Observation and Quality

Responsible for describing the detected person occurrence and the
reliability of the available evidence.

### Persistence

Responsible for:

* Image jobs
* Observation persistence
* Job state
* Observation ID reservation
* Cluster assignments
* Event processing state

### Identity Discovery

Responsible for:

* Constrained identity clustering
* Identity anchors
* Pose-aware matching
* Face similarity
* Body appearance similarity
* Quality-aware matching
* Cluster formation

### Output

Responsible for:

* Organizing cluster images
* Selecting the strongest images for each cluster
* Creating the representative face image
* Writing the final event output
* Writing output metadata

---

## 3. Runtime Lifecycle

The event coordinator runs the following high-level phases:

```text
01. Prepare image jobs
02. Recover stale jobs
03. Run image workers
04. Verify/retry failed jobs
05. Load observations
06. Run constrained identity clustering
07. Generate final event output
```

`main.py` coordinates these phases rather than implementing the
individual vision models.

Best-image selection and representative-image generation are part of
the final output phase and are handled by `EventOutputManager` .

---

## 4. Image Processing Pipeline

`app/pipeline.py` contains `PersonPipeline` , which processes one image.

```text
Image
  │
  ▼
Person Detection
  │
  ▼
Face Detection + Face Embedding
  │
  ▼
Face/Person Association
  │
  ▼
PersonObservation
  │
  ├── Face Pose
  ├── Body Embedding
  ├── Face Quality
  ├── Body Quality
  └── Embedding Validation
```

The pipeline does not perform event-level identity matching or
clustering.

This makes the image-processing stage independent and suitable for
multiprocessing.

---

## 5. Observation Model

A `PersonObservation` represents one detected occurrence of something
that appears to be one person in one image.

It is not a confirmed identity.

An observation may contain:

* Observation ID
* Image ID/path
* Person bounding box
* Person detection confidence
* Face bounding box
* Face detection confidence
* Face pose
* Yaw, pitch, and roll
* Face/person association score
* Face embedding
* Body embedding
* Face quality
* Body quality
* Face embedding validity
* Body embedding validity
* Moment ID
* Cluster ID

Observations can represent:

```text
face_and_body
face_only
body_only
```

The observation layer deliberately separates detection, representation, 
quality, and identity assignment.

---

## 6. Image Job Queue

Every discovered source image becomes a database job.

The normal lifecycle is:

```text
PENDING
   │
   ▼
PROCESSING
   │
   ├──────────────► COMPLETED
   │
   └──────────────► FAILED
```

A worker:

01. Claims a pending job.
02. Reads the image.
03. Runs `PersonPipeline`.
04. Reserves observation IDs.
05. Saves the generated observations.
06. Marks the job completed.

Workers continue until no pending work remains.

---

## 7. SQLite Persistence

Each event has its own database:

```text
<event_path>/event.db
```

`app/storage/eventStore.py` is responsible for persistent event state.

The database stores information such as:

* Image processing jobs
* Job state
* Job errors
* Person observations
* Observation IDs
* Cluster assignments
* Event processing state

The complete event image collection is not kept in Python memory.
Observations are persisted and loaded when required by event-level
processing.

---

## 8. SQLite Concurrency

Each worker process owns its own `EventStore` and SQLite connection.

Workers do not share SQLite connection objects.

The SQLite configuration is designed for concurrent worker coordination
and includes:

```text
WAL mode
synchronous = NORMAL
busy_timeout
foreign_keys = ON
```

The exact values are maintained in the implementation rather than
duplicated throughout the codebase.

---

## 9. Atomic Observation Persistence

Worker processing saves observations and completes the corresponding
image job as one logical database transaction.

Conceptually:

```text
Save observations
       +
Mark job COMPLETED
       │
       ▼
Single transaction
```

This avoids leaving the database in an inconsistent state where
observations were persisted but the job still appears to be processing.

---

## 10. Observation ID Reservation

Observation IDs must be unique across worker processes.

Workers must not generate IDs using:

```text
MAX(observation_id) + 1
```

because concurrent workers could calculate the same value.

Instead, `EventStore` reserves blocks of IDs.

Example:

```text
Worker 1 → 1–10
Worker 2 → 11–18
Worker 3 → 19–27
```

Unused IDs after a failure are acceptable. Global uniqueness is the
important property.

---

## 11. Crash and Stale-Job Recovery

A worker can fail while a job is in `PROCESSING` .

The coordinator can identify jobs that have remained in that state
longer than the configured stale timeout and return them to `PENDING` .

This allows interrupted work to be retried without requiring the entire
event to restart.

The timeout is configured through:

```python
WORKER_STALE_TIMEOUT_SECONDS
```

---

## 12. Multiprocessing

The system uses processes rather than threads for image workers.

Each process owns its own ML stack:

```text
YOLO
InsightFace
FacePoseEstimator
OSNet / TorchReID
QualityCalculator
EmbeddingValidator
```

This avoids sharing model instances between processes.

The tradeoff is higher memory and VRAM usage because each worker may
load its own model instances.

---

## 13. Worker Scaling

The worker architecture allows image processing to run independently:

```text
Image A → Worker 1
Image B → Worker 2
Image C → Worker 3
```

All workers write observations to the same event database.

Identity clustering happens only after image processing completes:

```text
Workers
   │
   ▼
All Observations
   │
   ▼
One Event-Level Clustering Stage
```

Increasing worker count does not necessarily increase throughput. GPU
memory, CPU resources, RAM, model initialization cost, and contention
should be benchmarked.

For limited GPUs, start with:

```python
WORKER_COUNT = 1
```

and increase only when measurements justify it.

---

## 14. Why Clustering Is Outside Workers

Workers see only individual images.

Identity clustering requires the complete event:

```text
Observation A
Observation B
Observation C
...
Observation N
```

Running clustering independently inside workers would create competing
partial identity states that would later need synchronization and
reconciliation.

The architecture therefore keeps a clear separation:

```text
PARALLEL
Image → Observation

THEN

EVENT-LEVEL
All Observations → Identity Clusters
```

---

## 15. Image Discovery and Identity

Images are discovered recursively and may have identical filenames in
different directories.

For example:

```text
camera_A/IMG_001.jpg
camera_B/IMG_001.jpg
```

must remain distinguishable.

The event-store and output flow therefore uses canonical image paths
where appropriate and includes compatibility handling for legacy or
relative image identifiers.

This allows the same filename to exist in different source directories
without incorrectly treating the images as the same source image.

---

## 16. Memory Management

Workers process one image at a time.

```text
Image 1
   ↓
Process
   ↓
Persist
   ↓
Release

Image 2
   ↓
Process
   ↓
Persist
   ↓
Release
```

The event's source photographs are not loaded as one large image
collection.

Persistent observations are stored in SQLite and loaded during the
event-level stages.

---

## 17. Event-Level Processing

After all image jobs have been processed, the coordinator performs:

```text
All Observations
      │
      ▼
Constrained Identity Clustering
      │
      ▼
Cluster Assignments
      │
      ▼
Event Output Manager
      │
      ├── Select Best Images
      ├── Create Representative Face
      └── Write Final Output
```

The event-level stage operates on the complete observation set.

The clustering stage does not instantiate the image-processing workers
again.

---

## 18. Constrained Identity Clustering

`ConstrainedIdentityClustering` is responsible for discovering which
observations belong to the same real-world person.

The clustering process uses:

* Face embedding similarity as the primary identity signal
* Body appearance similarity as supporting evidence
* Observation quality as supporting evidence
* Trusted frontal observations as identity anchors
* Pose-aware similarity thresholds
* Same-source-image constraints
* Quality-aware cluster representatives
* Strongest-compatible-pair merging

The clustering process is event-level and operates after all image
observations have been persisted.

Conceptually:

```text
Observations
     │
     ▼
Validate usable observations
     │
     ▼
Build candidate identity clusters
     │
     ▼
Calculate pair compatibility
     │
     ▼
Select strongest compatible pair
     │
     ▼
Merge clusters when constraints allow
     │
     ▼
Repeat
     │
     ▼
Final Cluster Assignments
```

The clustering implementation uses cached embeddings, quality values, 
and pair scores to avoid unnecessarily repeating expensive calculations.

---

## 19. Identity Anchors and Pose

Strong frontal observations can act as trusted identity anchors.

An observation can qualify as a trusted anchor when it satisfies the
configured face-quality, detection-confidence, face-size, pose, and yaw
requirements.

Pose information is used during identity matching to distinguish normal
same-pose comparisons from cross-pose comparisons.

The current pose categories are:

```text
frontal
left
right
profile
```

Face pose is calculated from the face landmarks and is used by the
association and clustering stages.

---

## 20. Output Boundaries

`EventOutputManager` is responsible for the final event output.

It does not change identity assignments.

Its responsibilities include:

* Reading cluster assignments
* Grouping observations by cluster
* Selecting the strongest source images
* Avoiding duplicate source images in selected results
* Applying pose diversity when appropriate
* Creating the representative face image
* Copying images into the final cluster directories
* Writing output metadata

The final relationship is:

```text
Clustering
    ↓
determines identity assignments

EventOutputManager
    ↓
selects strong images
    ↓
creates representative face
    ↓
writes final output
```

There is no separate `BestImageSelector` component in the current
architecture.

There is also no separate cluster visualization stage in the current
execution pipeline.

---

## 21. Final Output Structure

The output manager first applies reference matching and exports only
clusters that belong to known people from the configured reference directory.
Multiple cluster IDs matched to the same person are combined into one person
directory.

```text
output/
├── Person Name/
│   ├── All Images/
│   ├── Best Images/
│   └── representative Image.jpg
│
└── final_results.xlsx
```

### `All Images`

Contains the source images associated with the matched person.

### `Best Images`

Contains the strongest selected images from the matched clusters.

### `representative Image.jpg`

Contains one representative face crop for the person. It is stored directly
inside the person's output directory.

### `final_results.xlsx`

The final Excel summary contains one row per exported person:

```text
Name of Person | Phone Number | Folder Shared Link
```

The phone number comes from the reference filename format
`Person Name - Phone Number.ext`. The folder link is populated after the
person folder has been uploaded to Google Drive.

---

## 22. Project Structure

The current active project structure is:

```text
app/
├── configuration.py
├── pipeline.py
│
├── models/
│   ├── detector.py
│   └── observation.py
│
├── detection/
│   ├── personDetector.py
│   ├── faceDetector.py
│   └── association.py
│
├── identity/
│   └── facePoseEstimator.py
│
├── embeddings/
│   └── bodyEncoder.py
│
├── quality/
│   └── qualityCalculator.py
│
├── validation/
│   └── embeddingValidator.py
│
├── storage/
│   └── eventStore.py
│
├── workers/
│   └── imageWorker.py
│
├── clustering/
│   └── constrainedClustering.py
│
└── output/
    └── eventOutputManager.py
```

The project intentionally does not contain separate modules for:

```text
identityProfile.py
sameImageDuplicateSuppressor.py
bestImageSelector.py
clusterVisualizer.py
```

because those components are not part of the current execution path.

---

## 23. Architectural Principles

The implementation follows these boundaries:

01. Detection produces observations; it does not establish identity.
02. Workers process images independently.
03. Observations are persisted before event-level clustering.
04. Face representation is the primary identity evidence.
05. Body appearance is supporting evidence.
06. Quality controls the reliability of evidence.
07. Identity clustering operates on the complete event observation set.
08. Trusted observations can act as identity anchors.
09. Pose-aware matching helps handle different face orientations.
10. Same-source-image observations are prevented from being merged into
    the same identity cluster.

11. Clustering determines identity assignments.
12. `EventOutputManager` selects strong images without changing identity
    assignments.

13. The representative face is generated as part of final output.
14. Output components do not modify clustering results.
15. Image-level processing remains independent from event-level identity
    discovery.
