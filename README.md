# PersonaCluster

### Event-Based Person Detection, Recognition, Clustering, Reference Matching, and Output

PersonaCluster is an event-based computer-vision system for processing closed photographic events. It recursively discovers gallery images, detects people, extracts face and body representations, creates person observations, discovers anonymous identity clusters, matches those clusters against the event's reference people, and produces a structured final output.

It is designed for:

* Conferences
* Weddings
* Parties
* Sports events
* Graduations
* Corporate events
* Other closed event photo collections

The system processes **one event at a time**, while `main.py` can process all event folders found under `data/events/` .

---

# 1. Current Directory Structure

The current input layout is intentionally event-local:

```text
PersonaCluster/
│
├── data/
│   └── events/
│       ├── EEA/
│       │   ├── Gallery/
│       │   │   ├── Camera 1/
│       │   │   │   ├── Day 1/
│       │   │   │   │   ├── IMG001.jpg
│       │   │   │   │   └── IMG002.jpg
│       │   │   │   └── Day 2/
│       │   │   └── Camera 2/
│       │   │       └── ...
│       │   │
│       │   └── reference/
│       │       ├── Person 1 - 201xxxxxxxxx.jpg
│       │       ├── Person 2 - 201xxxxxxxxx.jpg
│       │       └── ...
│       │
│       ├── Event_2026/
│       │   ├── Gallery/
│       │   │   └── ...
│       │   └── reference/
│       │       └── ...
│       │
│       └── Wedding_September/
│           ├── Gallery/
│           │   └── ...
│           └── reference/
│               └── ...
│
├── output/
│   ├── EEA/
│   │   ├── Person 1/
│   │   │   ├── All Images/
│   │   │   ├── Best Images/
│   │   │   └── representative Image.jpg
│   │   └── Person 2/
│   │       └── ...
│   └── Event_2026/
│       └── ...
│
├── app/
├── main.py
└── ...
```

### Input rules

1. `data/events/` is the event container.
2. Every direct child of `data/events/` is one event.
3. Each event must contain a `Gallery/` directory.
4. The entire tree below `Gallery/` is recursively searched for images.
5. Each event has its own `reference/` directory.
6. Reference images are never treated as gallery images.
7. Different events may contain completely different reference people.
8. Event state is stored in that event's own `event.db`.

The old global `data/reference/` layout is no longer the event input model.

---

# 2. End-to-End Workflow

```text
data/events/
     │
     ├── EEA/
     ├── Event_2026/
     └── Wedding_September/
             │
             ▼
      Event Discovery
             │
             ▼
       Gallery Discovery
       (recursive only)
             │
             ▼
       SQLite Job Queue
             │
             ▼
       Parallel Workers
             │
             ├── Person Detection
             ├── Face Detection + Embedding
             ├── Face/Person Association
             ├── Face Pose
             ├── Body/Re-ID Embedding
             ├── Quality
             └── Embedding Validation
             │
             ▼
       Person Observations
             │
             ▼
        Event Database
             │
             ▼
   Constrained Identity Clustering
             │
             ▼
     Anonymous Clusters
             │
             ▼
   Event Reference Matching
             │
             ▼
      EventOutputManager
             │
             ├── All Images
             ├── Best Images
             └── Representative Image
             │
             ▼
       output/<event_name>/
             │
             ├── Local Excel report
             └── Optional Google Drive upload
```

`main.py` repeats this workflow for every event discovered under `data/events/` .

---

# 3. Event Processing

An event is defined by its direct folder:

```text
data/events/EEA/
```

The event name is:

```text
EEA
```

Its gallery is:

```text
data/events/EEA/Gallery/
```

Its references are:

```text
data/events/EEA/reference/
```

Its persistent database is:

```text
data/events/EEA/event.db
```

Its final output is:

```text
output/EEA/
```

The same rules apply independently to every other event.

---

# 4. Recursive Gallery Discovery

Only the `Gallery/` directory is searched.

For example:

```text
EEA/
├── Gallery/
│   ├── Camera A/
│   │   ├── Morning/
│   │   │   └── IMG001.jpg
│   │   └── Evening/
│   │       └── IMG002.jpg
│   └── Camera B/
│       └── Batch 01/
│           └── Original/
│               └── IMG003.jpg
└── reference/
    └── Person 1 - 201xxx.jpg
```

`IMG001.jpg` , `IMG002.jpg` , and `IMG003.jpg` belong to EEA.

The reference image does not.

Supported image formats are:

```text
.jpg
.jpeg
.png
.bmp
.webp
.tif
.tiff
```

Nested directory depth is unrestricted.

---

# 5. Image Processing

Each gallery image is processed independently.

```text
Image
  │
  ├── Person Detection
  ├── Face Detection + Embedding
  ├── Face/Person Association
  ├── Face Pose Estimation
  ├── Body/Re-ID Embedding
  ├── Quality Calculation
  └── Embedding Validation
          │
          ▼
   PersonObservation
```

Workers do not perform global identity clustering.

---

# 6. Persistent Event Memory

Each event owns its own SQLite database:

```text
data/events/<event_name>/event.db
```

The persistent state includes image registration, image hashes, jobs, observations, cluster state, reference information, cluster/person matches, output artifacts, configuration snapshots, and processing metadata.

The system is intended to avoid repeating work unnecessarily.

### Reuse rules

* New images are queued.
* Unchanged completed images are skipped.
* Changed images invalidate their previous image-level state and are processed again.
* Stale processing jobs can be recovered.
* Failed jobs can be retried.
* Reference comparisons can be cached.
* Clustering state can be reused when its inputs/configuration have not changed.

A post-cluster recheck can requeue eligible images whose observations did not contribute to a cluster, subject to the configured retry policy.

---

# 7. Identity Clustering

Clustering happens only after event image processing.

The clustering stage uses:

* Face similarity
* Body similarity
* Quality
* Pose context
* Identity anchors
* Same-image constraints

Face is the primary identity signal.

A same-image constraint prevents two observations from the same source image from being assigned to the same identity cluster.

The resulting clusters are event-local anonymous identities until reference matching is applied.

---

# 8. Event Reference Matching

References are loaded from:

```text
data/events/<event_name>/reference/
```

Expected filename pattern:

```text
Person Name - Phone Number.ext
```

The phone number is metadata. The person's name is used for the final output directory.

A person can have multiple reference images.

Reference matching happens after event-level clustering.

One known person may correspond to multiple discovered clusters; the output manager aggregates the accepted clusters into one person directory.

---

# 9. Final Output

For:

```text
data/events/EEA/
```

the final output is:

```text
output/EEA/
├── Person 1/
│   ├── All Images/
│   │   ├── ...
│   │   └── ...
│   ├── Best Images/
│   │   ├── ...
│   │   └── ...
│   └── representative Image.jpg
│
└── Person 2/
    ├── All Images/
    ├── Best Images/
    └── representative Image.jpg
```

The output is intentionally outside `data/events/<event>/` .

The Excel report is generated locally:

```text
output/<event_name>/final_results.xlsx
```

The Excel file is not uploaded to Google Drive.

---

# 10. Google Drive

When enabled, Google Drive receives the generated event/person folders and images.

The local output remains the source of truth for the final event result.

The Google Drive integration can create:

```text
Event Results/
└── <event_name>/
    ├── Person 1/
    └── Person 2/
```

The generated Excel report remains local.

---

# 11. Running the Project

Place events under:

```text
data/events/
```

For example:

```text
data/events/EEA/Gallery/
data/events/EEA/reference/
```

Then run:

```bash
python main.py
```

`main.py` discovers the event folders and processes them sequentially.

---

# 12. Important Separation of Responsibilities

```text
data/events/
    = input + persistent event state

data/events/<event>/Gallery/
    = gallery images only

data/events/<event>/reference/
    = reference images only

data/events/<event>/event.db
    = persistent event processing state

output/<event>/
    = final human-facing output
```

This separation is important because reference data, gallery data, database state, and generated output have different lifecycles.

---

# 13. Project Components

```text
app/
├── configuration.py
├── pipeline.py
├── models/
├── detection/
├── identity/
├── embeddings/
├── quality/
├── validation/
├── storage/
├── workers/
├── clustering/
├── output/
└── googleDriveUploader.py
```

`main.py` is the event-level coordinator.

`pipeline.py` processes one image.

`eventStore.py` manages persistent event state.

`constrainedClustering.py` performs event-level identity discovery.

`referenceMatcher.py` matches discovered clusters to that event's reference people.

`eventOutputManager.py` writes the final reference-matched output.

`googleDriveUploader.py` handles optional Drive upload.

---

# 14. GPU Runtime

The active face and body models may use CUDA when the environment supports it.

If ONNX Runtime reports:

```text
Available providers: CPUExecutionProvider
```

the application may fall back to CPU for InsightFace.

Worker count should be chosen based on available RAM and VRAM. Multiple workers can create multiple model instances.

---

# 15. Design Principle

The central design principle is:

> Gallery images belong to an event, references belong to that same event, processing state belongs to that event, and final output belongs to the event at the project-level `output/<event_name>/` directory.
