# PersonaCluster Architecture

This document describes the current architecture of PersonaCluster, including multi-event discovery, event-local galleries and references, persistent event state, image workers, clustering, reference matching, output generation, and optional Google Drive upload.

---

# 1. Architectural Overview

```text
data/events/
     │
     ▼
Discover direct event folders
     │
     ├── Event A
     ├── Event B
     └── Event C
             │
             ▼
     For each event:
             │
             ├── Gallery discovery
             ├── Event-local references
             ├── EventStore
             ├── Image jobs
             ├── Workers
             ├── Clustering
             ├── Reference matching
             └── Output
                     │
                     ▼
             output/<event_name>/
```

The core boundary remains:

> Image processing happens independently per image; identity clustering happens only after the event observations are available.

The event boundary is above that pipeline: every event has independent gallery input, reference data, database state, clustering state, and output.

---

# 2. Event Boundary

An event is:

```text
data/events/<event_name>/
```

Inside it:

```text
Gallery/
reference/
event.db
```

Only `Gallery/` is an image-processing input.

Only `reference/` is used for known-person matching.

The database is persistent processing state and is not gallery input.

---

# 3. Runtime Layers

```text
┌────────────────────────────────────────────┐
│ Event Discovery                            │
│ data/events/*                              │
└──────────────────────┬─────────────────────┘
                       ▼
┌────────────────────────────────────────────┐
│ Gallery Discovery                          │
│ recursive under <event>/Gallery            │
└──────────────────────┬─────────────────────┘
                       ▼
┌────────────────────────────────────────────┐
│ Image Processing                           │
│ Detection → Association → Embeddings       │
└──────────────────────┬─────────────────────┘
                       ▼
┌────────────────────────────────────────────┐
│ Observation + Quality                      │
│ Pose + Quality + Validation                │
└──────────────────────┬─────────────────────┘
                       ▼
┌────────────────────────────────────────────┐
│ Persistence                                │
│ SQLite jobs + observations + state         │
└──────────────────────┬─────────────────────┘
                       ▼
┌────────────────────────────────────────────┐
│ Identity Discovery                         │
│ Constrained event-level clustering         │
└──────────────────────┬─────────────────────┘
                       ▼
┌────────────────────────────────────────────┐
│ Reference Matching                         │
│ <event>/reference                          │
└──────────────────────┬─────────────────────┘
                       ▼
┌────────────────────────────────────────────┐
│ Output                                     │
│ output/<event_name>/                       │
└────────────────────────────────────────────┘
```

---

# 4. Batch Event Coordinator

`main.py` is responsible for the event lifecycle.

Conceptually:

```text
discover events
     │
     ▼
for each event:
     │
     ├── open event database
     ├── discover Gallery recursively
     ├── prepare jobs
     ├── recover stale jobs
     ├── run workers
     ├── retry failures
     ├── verify processing
     ├── cluster observations
     ├── match event references
     ├── generate final output
     ├── optionally upload to Drive
     └── continue to next event
```

The event name is derived from the direct folder name.

The program must not treat a nested gallery folder as a separate event.

---

# 5. Image Processing Pipeline

`app/pipeline.py` contains the image-level processing pipeline.

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
Face Pose
  │
  ▼
Body/Re-ID Embedding
  │
  ▼
Quality Calculation
  │
  ▼
Embedding Validation
  │
  ▼
PersonObservation
```

The pipeline does not perform event-level clustering.

---

# 6. Worker Architecture

Each image worker owns its own process-local resources:

```text
EventStore
SQLite connection
PersonPipeline
YOLO detector
InsightFace
FacePoseEstimator
OSNet/TorchReID
QualityCalculator
EmbeddingValidator
```

Workers do not share SQLite connection objects or ML model instances.

They claim jobs, process images, persist observations, and complete jobs.

---

# 7. SQLite Persistence

Every event has its own database:

```text
data/events/<event_name>/event.db
```

The database tracks:

- Event metadata
- Runs
- Image registration
- Image hashes
- Image jobs
- Observations
- Observation/person comparison attempts
- Clusters
- Cluster memberships
- Known people
- Reference images
- Cluster/person matches
- Output artifacts
- Processing operations
- Model versions
- Configuration snapshots
- Image cluster state
- Event stage state

SQLite is configured for concurrent workers using WAL mode, `synchronous=NORMAL`, a busy timeout, and foreign-key enforcement.

---

# 8. Persistent Image Identity

Image identity must preserve nested paths and distinguish duplicate filenames.

A gallery may contain:

```text
Gallery/CameraA/Day1/IMG001.jpg
Gallery/CameraB/Day1/IMG001.jpg
```

These are two different source images.

The persistent identifier therefore uses a portable path representation rather than relying only on the basename.

Output and visualization lookup must use the same image identity.

---

# 9. Incremental Processing

The event database allows subsequent runs to reuse previous work.

```text
Image unchanged
    │
    └── completed state can be reused

Image new
    │
    └── queue for processing

Image content changed
    │
    └── invalidate dependent state
        and process again
```

This prevents every invocation from rebuilding the event from scratch.

---

# 10. Stale Job Recovery

A worker can terminate while an image remains `PROCESSING`.

The coordinator can detect jobs exceeding:

```python
WORKER_STALE_TIMEOUT_SECONDS
```

and return them to a retryable state.

This is event-local because the database is event-local.

---

# 11. Post-Cluster Recheck

The persistent architecture supports a second inference pass for eligible images that produced observations but did not contribute to a cluster.

The intended sequence is:

```text
First processing
      │
      ▼
Clustering
      │
      ▼
Find eligible unclustered images
      │
      ▼
Requeue/reprocess once
      │
      ▼
Re-cluster
```

The recheck is bounded so an event does not enter an endless processing loop.

---

# 12. Identity Clustering

Clustering operates over all valid observations for one event.

Evidence includes:

- Face similarity
- Body similarity
- Quality
- Pose
- Identity anchors
- Same-image constraints

Clustering does not mix observations across different events.

---

# 13. Reference Matching

Reference matching happens after event-level clustering.

Input:

```text
data/events/<event_name>/reference/
```

References are grouped by person name parsed from:

```text
Person Name - Phone Number.ext
```

A known person may own multiple discovered cluster IDs.

Those accepted clusters are aggregated during final output generation.

---

# 14. Output Architecture

The final local output is deliberately outside the event input tree:

```text
output/
└── <event_name>/
    ├── <person_name>/
    │   ├── All Images/
    │   ├── Best Images/
    │   └── representative Image.jpg
    └── final_results.xlsx
```

The output manager resolves source images from their persistent image IDs.

It should never use `reference/` as a source gallery.

---

# 15. Google Drive Boundary

Google Drive is an optional output integration.

The local output is generated first.

Then, if enabled:

```text
output/<event_name>/
       │
       ▼
Google Drive event folder
       │
       ├── Person folders
       └── Images
```

The Excel report remains local.

---

# 16. Failure Isolation

A failure in one image should not invalidate successfully persisted observations from other images.

A failure in one event should not silently turn another event into the same event.

The batch coordinator should report the failing event and continue where the configured batch behavior permits.

---

# 17. Architectural Invariants

The following invariants are important:

1. Direct children of `data/events/` are events.
2. Only `<event>/Gallery/` is gallery input.
3. `<event>/reference/` is reference input.
4. Nested gallery directories do not create events.
5. Event databases are event-local.
6. Clustering is event-local.
7. Reference matching is event-local.
8. Final output is `output/<event_name>/`.
9. Excel is local.
10. Google Drive is optional.
11. Image identity must distinguish duplicate filenames.
12. Workers do not perform global clustering.
