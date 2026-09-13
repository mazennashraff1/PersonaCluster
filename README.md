# **PersonaCluster**

## Intelligent person detection, recognition, and constrained clustering across event images.

An event-based computer-vision system for detecting people in photographs, extracting face and body appearance embeddings, storing observations in SQLite, and discovering which observations belong to the same real-world person.

The system is designed for **closed events** such as conferences, weddings, parties, sports events, and other photo collections covering a limited period of time.

It does **not** require a permanent person gallery and does not attempt to identify people by name automatically.

Instead:

```text
Event Photos
     │
     ▼
Image Job Queue
     │
     ▼
Parallel Image Workers
     │
     ├── Person Detection
     │
     ├── Face Detection
     │
     ├── Face ↔ Person Association
     │
     ├── Face Embedding
     │
     ├── Upper-Body Extraction
     │
     ├── Body/Re-ID Embedding
     │
     └── Quality Validation
     │
     ▼
Person Observations
     │
     ▼
SQLite Database
     │
     ▼
Constrained Identity Clustering
     │
     ▼
Final Person Clusters
     │
     ▼
Cluster Visualization
     │
     ▼
Human Labeling
```

---

# 1. Project Goal

Given a collection of event photographs:

```text
event_001/
├── IMG_0001.jpg
├── IMG_0002.jpg
├── IMG_0003.jpg
└── ...
```

the system discovers groups of observations that are likely to represent the same person.

The system starts without knowing anyone's identity.

For example:

```text
Cluster 1
Cluster 2
Cluster 3
Cluster 4
```

After clustering, a separate user-facing layer can assign names:

```text
Cluster 1 → Ahmed
Cluster 2 → Sara
Cluster 3 → Mazen
```

The distinction is important:

```text
Discovery
    ↓
Clustering
    ↓
Anonymous Person Cluster
    ↓
Human Labeling
    ↓
Named Identity
```

The clustering system itself does not need to know that Cluster 3 is "Mazen".

---

# 2. Design Principles

The project follows several important principles.

## 2.1 Observation-first architecture

The fundamental data object is a `PersonObservation` .

An observation represents:

> One detected occurrence of one person in one image.

Conceptually:

```python
{
    "observation_id": 123,
    "image_id": "IMG_0042",
    "person_bbox": [...],
    "face_bbox": [...],
    "face_embedding": [...],
    "body_embedding": [...],
    "face_quality": 0.91,
    "body_quality": 0.86,
    "moment_id": 12,
    "cluster_id": None
}
```

The observation does not initially have a person's name.

---

## 2.2 Face + body representation

A person is represented using two major sources of evidence:

```text
                Person
                   │
          ┌────────┴────────┐
          ▼                 ▼
        Face              Body
          │                 │
          ▼                 ▼
   Face Embedding     Body Embedding
```

Face embeddings provide identity-related information.

Body/Re-ID embeddings provide additional appearance information when:

* the face is unavailable
* the face is very small
* the person is looking away
* the face is partially occluded
* the face quality is insufficient

Body appearance is not treated as a permanent identity signal because clothing, pose, lighting, and context can change.

---

## 2.3 Quality-aware processing

Not every detection is equally reliable.

The system therefore tracks quality information such as:

* detection confidence
* face size
* face visibility
* image sharpness
* crop quality
* body quality
* embedding validity

Poor observations should not have the same influence as strong observations during clustering.

---

## 2.4 Closed-event processing

The system is designed around one complete event.

Example:

```text
Wedding
    │
    ├── 5,000 images
    ├── 200 detected people
    └── multiple photographs of the same people
```

The system processes that event and discovers the people appearing in it.

It does not require a permanent global identity database.

---

# 3. High-Level Architecture

The complete system is divided into two major stages.

```text
                 EVENT
                   │
                   ▼
        ┌──────────────────────┐
        │ Image Processing     │
        │ Workers              │
        └──────────┬───────────┘
                   │
                   ▼
            Person Observations
                   │
                   ▼
              SQLite DB
                   │
                   ▼
        ┌──────────────────────┐
        │ Event-Level          │
        │ Clustering           │
        └──────────┬───────────┘
                   │
                   ▼
           Final Clusters
```

The important architectural decision is:

> Image processing is parallelized; event-level clustering remains a separate final stage.

This prevents every worker from trying to modify the global clustering state simultaneously.

---

# 4. Complete Processing Pipeline

## Stage 1 — Image Discovery

`main.py` scans the event directory and discovers supported image files.

Supported formats:

```text
.jpg
.jpeg
.png
.bmp
.webp
.tif
.tiff
```

Images are inserted into the SQLite job queue.

---

## Stage 2 — SQLite Image Job Queue

Each image becomes a job.

The database stores:

```text
job_id
image_id
image_path
status
worker_id
attempts
created_at
started_at
completed_at
error_message
```

Possible states:

```text
PENDING
PROCESSING
COMPLETED
FAILED
```

Example:

```text
IMG_0001 → COMPLETED
IMG_0002 → PROCESSING
IMG_0003 → PENDING
IMG_0004 → FAILED
```

---

# 5. Worker Architecture

The project uses independent image workers.

Each worker owns:

```text
Worker
 │
 ├── EventStore
 │
 ├── SQLite connection
 │
 ├── PersonPipeline
 │
 ├── Person detector
 │
 ├── Face detector
 │
 ├── Face encoder
 │
 └── Body/Re-ID encoder
```

Workers do **not** share model objects or SQLite connections.

Conceptually:

```text
                SQLite
                   │
       ┌───────────┼───────────┐
       │           │           │
       ▼           ▼           ▼
   Worker 1    Worker 2    Worker 3
       │           │           │
       ▼           ▼           ▼
    Pipeline    Pipeline    Pipeline
```

This architecture is especially appropriate for multiprocessing because each process has its own Python runtime and model instances.

---

# 6. Worker Job Lifecycle

A worker repeatedly performs:

```text
01. Ask SQLite for a PENDING job
02. Atomically claim the job
03. Mark it PROCESSING
04. Read the image
05. Run PersonPipeline
06. Generate observations
07. Reserve observation IDs
08. Save observations
09. Mark job COMPLETED
10. Repeat
```

If processing fails:

```text
PROCESSING
     │
     ▼
   FAILED
```

The error is stored in the database.

---

# 7. Crash Recovery

The job queue is designed to survive worker failures.

A job may become:

```text
PROCESSING
```

and then the worker may crash.

Before starting the workers, the coordinator can recover jobs that have been stuck in `PROCESSING` longer than the configured stale timeout.

For example:

```text
WORKER_STALE_TIMEOUT_SECONDS = 600
```

means a job stuck for more than approximately 10 minutes can be recovered.

The recovery process changes:

```text
PROCESSING → PENDING
```

so another worker can process it.

Recovery is performed by the coordinator rather than continuously by every worker.

---

# 8. Transaction Safety

Saving an observation and completing its image job must be treated as one logical operation.

The project therefore uses:

```text
save_observations_and_complete_job()
```

to ensure that the following operations happen together:

```text
Save observations
       +
Mark image job COMPLETED
```

This avoids a dangerous state where:

```text
Observations were saved
but
Job is still PROCESSING
```

because that could cause the image to be processed again after recovery.

---

# 9. Observation IDs

Observation IDs must be unique across all workers.

Workers therefore do not independently generate global IDs.

Instead, SQLite maintains an observation sequence:

```text
observation_sequence
```

Workers reserve a block of IDs:

```text
Worker 1 → IDs 1–10
Worker 2 → IDs 11–18
Worker 3 → IDs 19–27
```

This prevents ID collisions.

IDs may have gaps after a worker failure.

For example:

```text
1
2
3
7
8
9
```

is acceptable as long as IDs remain unique.

---

# 10. SQLite Configuration

SQLite is configured for concurrent worker access.

The database uses:

```text
WAL mode
synchronous = NORMAL
busy_timeout
foreign_keys = ON
```

WAL allows multiple worker processes to read while SQLite coordinates writes more effectively.

Each worker must have its own `EventStore` and SQLite connection.

SQLite connections must not be shared between processes.

---

# 11. Image Processing Pipeline

The `PersonPipeline` processes one image.

The conceptual pipeline is:

```text
Image
  │
  ▼
Person Detection
  │
  ▼
Person Bounding Boxes
  │
  ▼
Face Detection
  │
  ▼
Face ↔ Person Association
  │
  ├───────────────┐
  ▼               ▼
Face Crop       Body Crop
  │               │
  ▼               ▼
Face Encoder    Body Encoder
  │               │
  └───────┬───────┘
          ▼
      Quality
          │
          ▼
    Validation
          │
          ▼
PersonObservation
```

---

# 12. Person Detection

The first vision stage detects visible people.

Output includes:

```text
Bounding box
Confidence
```

Example:

```text
IMG_001.jpg

Person 1
Person 2
Person 3
```

Detection does not identify people.

---

# 13. Face Detection

Each detected person is examined for a usable face.

A person can have:

```text
Good face
Partial face
Poor face
No usable face
```

A missing face does not automatically invalidate the person observation.

Body appearance can still provide useful evidence.

---

# 14. Face/Person Association

A detected face must be associated with the correct person bounding box.

The association uses spatial relationships such as:

* overlap
* containment
* face center position
* relative size

This is important because an image can contain multiple people and multiple faces.

---

# 15. Face Embeddings

A face encoder converts a usable face crop into a numerical vector.

The current architecture uses an InsightFace/ArcFace-style face embedding.

The conceptual flow is:

```text
Face Crop
    │
    ▼
InsightFace
    │
    ▼
Face Embedding
```

The project uses normalized face embeddings for similarity calculations.

---

# 16. Body/Re-ID Embeddings

The body encoder is separate from the face encoder.

The current dependency is:

```text
torchreid
```

with the Deep-Person-ReID implementation.

Conceptually:

```text
Person Crop
    │
    ▼
Upper-Body Crop
    │
    ▼
Person Re-ID Encoder
    │
    ▼
Body Embedding
```

Body embeddings are appearance evidence, not permanent identity evidence.

---

# 17. Observation Validation

Before an observation is stored, its embeddings and associated data are validated.

The system should avoid storing invalid numerical values such as:

```text
NaN
Inf
wrong embedding dimensions
empty vectors
```

This prevents corrupted observations from reaching the clustering stage.

---

# 18. Moment Information

Event photographs often contain temporal context.

Images captured close together may represent the same local activity or session.

The conceptual structure is:

```text
10:01:02 ─┐
10:01:05  ├── Moment 1
10:01:09  │
10:01:13 ─┘

10:24:01 ─┐
10:24:04  ├── Moment 2
10:24:08 ─┘
```

Moment information can help body appearance become more meaningful when photographs are temporally close.

---

# 19. Constrained Identity Clustering

After all image jobs are complete, clustering begins.

This is intentionally separate from the workers.

```text
All Workers
     │
     ▼
SQLite
     │
     ▼
All Observations
     │
     ▼
ConstrainedIdentityClustering
```

The clustering stage operates on the complete event dataset.

---

# 20. Same-Image Constraint

An important constraint in this project is:

> A photograph cannot contain the same person twice.

Therefore, observations from the same image cannot simply be merged together as if they were the same person.

For example:

```text
IMG_001

Person A
Person B
Person C
```

must remain three different identity candidates.

The clustering logic therefore explicitly rejects invalid same-image merges.

This prevents a purely embedding-based algorithm from incorrectly grouping two different people just because their embeddings are similar.

---

# 21. First-Pass Clustering

The first clustering stage is deliberately conservative.

The objective is:

```text
High precision
      >
Maximum recall
```

It is safer to temporarily split one real person into multiple clusters than to incorrectly merge two different people.

Example:

```text
Acceptable:

Cluster 1 → Mazen
Cluster 2 → Mazen
Cluster 3 → Ahmed
```

is preferable to:

```text
Bad:

Cluster 1 → Mazen + Ahmed
```

The second clustering stage can later merge fragmented clusters.

---

# 22. Evidence Used for Clustering

The clustering architecture considers multiple signals:

```text
Face similarity
Body similarity
Observation quality
Moment relationship
Same-image constraints
```

The exact weights and thresholds should be treated as project parameters and calibrated against real event data.

The Apple research is used as architectural inspiration, not as a source of fixed numerical thresholds.

---

# 23. Second-Pass Clustering

The first stage can produce fragmented clusters.

Example:

```text
Cluster 1 → Mazen
Cluster 2 → Mazen
Cluster 3 → Mazen
Cluster 4 → Ahmed
```

The second stage attempts to merge fragments belonging to the same person.

Face information is particularly useful here because body appearance can change due to:

* clothing
* lighting
* pose
* jackets
* event activity

The current architecture therefore uses face-based cluster refinement.

---

# 24. Final Clusters

The output is a set of anonymous discovered people.

Example:

```text
Cluster 001
    421 observations
    183 images

Cluster 002
    288 observations
    121 images

Cluster 003
    173 observations
    91 images
```

At this stage:

```text
Cluster 001
```

does not automatically mean:

```text
Ahmed
```

It is simply one discovered person cluster.

---

# 25. Cluster Visualization

After clustering, the system generates visual representations of the discovered clusters.

Visualization allows the developer to inspect:

* cluster membership
* representative observations
* image associations
* potential false merges
* fragmented identities

The visualization stage is a validation tool and should be used to evaluate clustering quality.

---

# 26. Project Structure

The current project should follow this structure:

```text
project_root/
│
├── app/
│   │
│   ├── configuration.py
│   ├── pipeline.py
│   │
│   ├── models/
│   │   └── observation.py
│   │
│   ├── storage/
│   │   └── eventStore.py
│   │
│   ├── workers/
│   │   └── imageWorker.py
│   │
│   └── clustering/
│       ├── constrainedClustering.py
│       └── clusterVisualizer.py
│
├── data/
│   └── events/
│       └── event_001/
│           ├── images/
│           └── output/
│
├── models/
│
├── main.py
├── requirements.txt
└── README.md
```

The exact directory names can be changed in `configuration.py` , but the separation of responsibilities should remain.

---

# 27. Responsibilities of the Main Files

## `main.py`

The application coordinator.

Responsible for:

```text
Discover images
      ↓
Create image jobs
      ↓
Recover stale jobs
      ↓
Start workers
      ↓
Wait for workers
      ↓
Verify job completion
      ↓
Load observations
      ↓
Run clustering
      ↓
Save cluster assignments
      ↓
Generate visualization
```

`main.py` should not contain the implementation of the image-processing pipeline.

---

## `app/pipeline.py`

Processes one image.

Responsible for:

```text
Detection
Face processing
Body processing
Quality
Validation
Observation creation
```

It should not control the event-level worker queue.

---

## `app/workers/imageWorker.py`

Responsible for parallel image processing.

Each worker:

```text
Claims a job
    ↓
Reads image
    ↓
Runs pipeline
    ↓
Saves observations
    ↓
Completes job
```

It does not perform event-level clustering.

---

## `app/storage/eventStore.py`

Responsible for persistence.

It manages:

```text
SQLite
Observations
Image jobs
Observation IDs
Cluster assignments
Transactions
Recovery
```

---

## `app/clustering/constrainedClustering.py`

Responsible for event-level identity clustering.

It operates after image processing is finished.

---

## `app/clustering/clusterVisualizer.py`

Responsible for visual validation of final clusters.

---

## `app/configuration.py`

Contains configurable paths, model settings, thresholds, and worker settings.

---

# 28. Environment Requirements

Recommended environment:

```text
Operating System:
Windows 10/11 or Linux

Python:
3.10 or 3.11

GPU:
NVIDIA GPU recommended

CUDA:
Compatible with the installed PyTorch build

RAM:
16 GB recommended

Storage:
SSD strongly recommended
```

The system can run without a GPU if the underlying models support CPU execution, but image processing will generally be substantially slower.

For large events such as thousands of images, an NVIDIA GPU is strongly recommended.

---

# 29. Creating the Python Environment

## Windows

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

Upgrade pip:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

---

## Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Then:

```bash
python -m pip install --upgrade pip setuptools wheel
```

---

# 30. Installing Dependencies

Install the requirements:

```bash
pip install -r requirements.txt
```
Then run:

```bash
python -m pip install --no-build-isolation git+https://github.com/KaiyangZhou/deep-person-reid.git
```
---

# 31. GPU Installation

PyTorch must be installed with a build compatible with the NVIDIA driver and CUDA runtime available on the machine.

Do not assume that:

```text
CUDA toolkit version
```

and:

```text
PyTorch CUDA build
```

must have exactly the same version number.

Verify the installation after installing PyTorch:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Expected GPU output should resemble:

```text
2.x.x
True
NVIDIA ...
```

If:

```text
torch.cuda.is_available()
```

returns:

```text
False
```

the system is not currently using the GPU through PyTorch.

---

# 32. Verify the Main Dependencies

Run:

```bash
python -c "import numpy, cv2, PIL, scipy, torch, torchvision, ultralytics, insightface, onnxruntime, sklearn, skimage, matplotlib, yaml; print('All core dependencies imported successfully')"
```

Then verify TorchReID:

```bash
python -c "import torchreid; print('TorchReID imported successfully')"
```

If both commands succeed, the Python environment is correctly configured at the package level.

---

# 33. Model Files

The project uses pretrained machine-learning models.

Depending on the implementation, model files may be:

```text
Downloaded automatically
```

or:

```text
Stored locally in the models/ directory
```

Do not commit large model weights to Git unless the project explicitly requires it.

Recommended structure:

```text
models/
├── person/
├── face/
└── body/
```

If a model is downloaded automatically, allow the first execution to complete before judging runtime performance.

---

# 34. Configuration

Worker settings should be defined in:

```text
app/configuration.py
```

Recommended starting configuration:

```python
WORKER_COUNT = 1

WORKER_STALE_TIMEOUT_SECONDS = 600

WORKER_USE_PROCESSES = True
```

---

# 35. Why Start With One Worker?

A GPU does not automatically become faster when more Python processes are launched.

Each process may create its own copy of:

```text
YOLO model
InsightFace model
Body/Re-ID model
```

Therefore:

```text
1 worker
```

is the safest starting point.

With a GPU containing limited VRAM, multiple workers may cause:

```text
GPU memory exhaustion
```

or excessive context switching.

Start with:

```python
WORKER_COUNT = 1
```

and benchmark before increasing it.

---

# 36. Running the System

After the environment and configuration are ready:

```bash
python main.py
```

The application performs:

```text
01. Discover images
02. Create image jobs
03. Recover stale jobs
04. Start image workers
05. Process images
06. Save observations
07. Verify job status
08. Run constrained clustering
09. Save cluster assignments
10. Generate cluster visualization
```

---

# 37. Expected Worker Output

You should see output similar to:

```text
============================================================
PREPARING IMAGE JOB QUEUE
============================================================

New jobs created: 23
PENDING:    23
PROCESSING: 0
COMPLETED:  0
FAILED:     0
```

Then:

```text
============================================================
STARTING IMAGE WORKERS
============================================================

Configured workers: 1
Worker mode: PROCESSES
Starting worker-1
```

Then:

```text
============================================================
WORKER STARTED: worker-1
============================================================

[worker-1] PID=12345 HOST=MY-PC

[worker-1] Processing: IMG_0001.jpg
[worker-1] COMPLETED: IMG_0001.jpg
[worker-1] Observations: 4
[worker-1] Pipeline time: 2.41s
[worker-1] Total image time: 2.42s
```

This continues until there are no remaining `PENDING` jobs.

---

# 38. Job Result Verification

After all workers finish:

```text
============================================================
IMAGE JOB RESULTS
============================================================

PENDING:    0
PROCESSING: 0
COMPLETED:  23
FAILED:     0
```

The clustering stage should only begin when:

```text
PENDING = 0
PROCESSING = 0
```

If processing jobs remain, the program stops rather than silently clustering incomplete data.

---

# 39. Clustering Output

The next stage loads all observations:

```text
============================================================
CONSTRAINED IDENTITY CLUSTERING
============================================================

Total observations in event: 87
```

Then it reports clustering statistics such as:

```text
Valid face observations: 71
Observations assigned to identities: 79
Unknown / unassigned observations: 8
Discovered identity clusters: 14
Rejected same-image merges: 23
Rejected low-similarity candidates: 41
```

These statistics are important when evaluating the system.

---

# 40. Visualization Output

After clustering:

```text
============================================================
VISUAL VALIDATION
============================================================
```

The program generates cluster visualizations in the configured event output directory.

These visualizations should be inspected to determine whether:

```text
Same people are grouped
Different people are separated
Poor observations are isolated
Clusters are overly fragmented
False merges exist
```

---

# 41. Existing Database Warning

This is extremely important when migrating from the old sequential version.

Suppose the database already contains observations for:

```text
IMG_0001.jpg
IMG_0002.jpg
IMG_0003.jpg
```

but the new worker job queue does not yet know that those images were processed.

Running the worker system may create jobs for those images and process them again.

That can result in duplicate observations.

For the first worker-based test, use either:

```text
A fresh event database
```

or:

```text
A fresh event directory
```

unless an explicit database migration/synchronization step has been performed.

Do not blindly run the new worker pipeline against an old partially processed database.

---

# 42. Reprocessing an Event

If you want to process an event from scratch, the safest development procedure is:

```text
01. Remove/reset the event database
02. Clear previous generated outputs
03. Keep the original input images
04. Run main.py again
```

The exact reset procedure depends on how `EVENTS_PATH` and database paths are configured.

Never delete the original photographs.

---

# 43. Failed Jobs

If an image fails:

```text
FAILED
```

the database stores the error message.

The error should be investigated before considering the event complete.

Typical causes include:

```text
Corrupt image
Unreadable file
Model error
Invalid crop
Out-of-memory
Unexpected embedding shape
Dependency problem
```

A run with:

```text
FAILED > 0
```

should be considered incomplete until the failures are understood.

---

# 44. Memory Management

The system intentionally processes images one at a time inside each worker.

The worker does not load the entire event image set into RAM.

Conceptually:

```text
Image 1
  ↓
Process
  ↓
Save
  ↓
Release memory

Image 2
  ↓
Process
  ↓
Save
  ↓
Release memory
```

The event database stores observations rather than requiring all source images to remain in memory.

This is important when scaling from:

```text
23 images
```

to:

```text
10,000+ images
```

---

# 45. GPU Memory Considerations

Each process may instantiate its own ML models.

Therefore:

```text
Worker count ↑
        ↓
Model instances ↑
        ↓
GPU memory usage ↑
```

Increasing the worker count is not automatically an optimization.

Benchmark:

```text
1 worker
```

first.

Then, if GPU memory allows:

```text
2 workers
```

can be tested.

Monitor GPU memory while processing.

---

# 46. CPU Threads vs Processes

The current architecture uses multiprocessing:

```python
WORKER_USE_PROCESSES = True
```

This is intentional.

The image-processing pipeline is ML-heavy and can involve:

* PyTorch
* ONNX Runtime
* OpenCV
* NumPy
* YOLO
* InsightFace
* Re-ID models

Each process gets isolated model state.

The current final configuration therefore prefers processes over Python threads.

---

# 47. Why Clustering Is Not Inside the Workers

The workers operate independently on individual images.

Clustering requires the complete event:

```text
Observation A
Observation B
Observation C
...
Observation N
```

Therefore clustering happens after all workers finish.

This avoids having:

```text
Worker 1
    ↓
partial clustering state

Worker 2
    ↓
different partial clustering state
```

and then attempting to synchronize global identity state between processes.

The architecture is:

```text
PARALLEL
────────

Image → Observation
Image → Observation
Image → Observation
Image → Observation

THEN

SEQUENTIAL EVENT STAGE
──────────────────────

All Observations
       ↓
   Clustering
       ↓
 Final Clusters
```

---

# 48. Scalability

The architecture is designed to scale image processing independently from clustering.

For example:

```text
23 images
    ↓
1 worker
```

can become:

```text
10,000 images
    ↓
multiple workers
    ↓
shared SQLite job queue
```

The important point is that workers do not need to know about each other.

They only need to:

```text
claim job
process job
save result
complete job
```

SQLite provides the coordination mechanism.

---

# 49. Performance Measurement

The worker reports:

```text
Pipeline time
Total image time
Worker time
Total execution time
```

This allows performance analysis.

For example:

```text
Pipeline time: 2.40s
Total image time: 2.43s
```

means the majority of the time is inside the ML pipeline.

If the pipeline dominates runtime, increasing workers may provide more benefit than optimizing file discovery.

---

# 50. Where the Processing Time Goes

The most expensive operations are generally expected to be ML inference stages:

```text
Person Detection
Face Detection
Face Embedding
Body/Re-ID Embedding
```

Clustering is a separate event-level stage.

The worker architecture therefore focuses parallelism on the image-processing stage.

---

# 51. Correctness Before Performance

The optimization order should be:

```text
Correctness
    ↓
Database reliability
    ↓
Worker reliability
    ↓
GPU utilization
    ↓
Parallelism
    ↓
Benchmarking
```

Do not increase worker count simply because more workers are available.

First confirm:

```text
No duplicate observations
No missing images
No invalid IDs
No unfinished jobs
No incorrect same-image merges
```

---

# 52. Reproducible Development Procedure

For a new machine:

```text
01. Clone/copy the project
02. Install Python
03. Create virtual environment
04. Install PyTorch
05. Install requirements.txt
06. Verify GPU
07. Verify model dependencies
08. Configure event path
09. Prepare event images
10. Run main.py
11. Inspect database/job statistics
12. Inspect cluster visualizations
```

---

# 53. First Test Dataset

Before processing thousands of images, use a small event.

Recommended progression:

```text
10–25 images
      ↓
50–100 images
      ↓
500 images
      ↓
1,000 images
      ↓
10,000+ images
```

At every stage measure:

```text
Runtime
GPU memory
RAM
Observation count
Failed jobs
Cluster count
Cluster purity
False merges
Fragmentation
```

---

# 54. Evaluation

The system should be evaluated at multiple levels.

## Detection

Are people correctly detected?

---

## Observation Quality

Measure:

```text
Valid faces
Missing faces
Poor crops
False detections
Invalid embeddings
```

---

## Cluster Purity

For each cluster:

> What percentage of observations belong to the dominant real person?

High purity means fewer identity mixtures.

---

## Fragmentation

For each real person:

> Into how many clusters was that person divided?

Lower fragmentation is better.

---

## False Merges

Measure how often two different real people are incorrectly placed in one cluster.

This is particularly important because false merges are usually more damaging than temporary fragmentation.

---

# 55. Desired Clustering Behavior

The preferred result is:

```text
High purity
+
Low false merges
+
Low fragmentation
```

rather than simply:

```text
Maximum number of merged observations
```

A clustering system that merges everyone into one cluster is technically "compact" but completely incorrect.

---

# 56. Apple-Inspired Architecture

The architecture is inspired by the ideas described in Apple's research around people recognition in Photos.

The project adopts concepts such as:

```text
Face + upper-body representation
Observation-level processing
Contextual information
Conservative clustering
Cluster refinement
Representative observations
Quality-aware processing
```

However, this project is not a reproduction of Apple's proprietary implementation.

Model choices, thresholds, distance functions, and weights must be calibrated against the project's own datasets.

---

# 57. Important Constraint

One image may contain multiple people.

For example:

```text
IMG_001.jpg

Person A
Person B
Person C
```

The clustering system must not conclude:

```text
Person A = Person B
```

merely because their embeddings happen to be close.

Same-image constraints are therefore a core part of identity clustering.

---

# 58. No Permanent Identity Gallery

The current system is event-based.

Its lifecycle is:

```text
Event
  ↓
Observation Extraction
  ↓
Clustering
  ↓
Cluster Visualization
  ↓
User Labeling
  ↓
Event Result
```

There is no requirement for:

```text
Permanent Ahmed Gallery
Permanent Sara Gallery
Permanent Mazen Gallery
```

A future version could add cross-event identity recognition, but that is outside the current architecture.

---

# 59. Why FAISS Is Not Required Yet

For a single closed event, the project can initially use:

```text
NumPy
scikit-learn
```

for similarity and clustering operations.

FAISS is not part of the core architecture at this stage.

It can be introduced later if the number of observations becomes large enough that approximate nearest-neighbor retrieval is required.

---

# 60. Development Roadmap

The implementation is conceptually organized as:

```text
01. Person Detection
02. Face Detection
03. Face ↔ Person Association
04. Face Embedding
05. Body Crop
06. Body Embedding
07. Quality Scoring
08. Observation Validation
09. Observation Storage
10. Image Job Queue
11. Worker Processing
12. Moment Information
13. Constrained Clustering
14. Cluster Visualization
15. Cluster Refinement
16. Representative Selection
17. Human Labeling
18. Event Results
```

The current architecture has reached the worker/database scalability stage.

---

# 61. Troubleshooting

## `torch.cuda.is_available()` is False

Check:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Then verify:

* NVIDIA driver
* PyTorch installation
* correct Python environment
* compatible PyTorch CUDA build

---

## CUDA out-of-memory

Reduce:

```python
WORKER_COUNT
```

Start with:

```python
WORKER_COUNT = 1
```

Multiple worker processes may each load their own models.

---

## `cv2.imread()` returns None

Check:

* file exists
* path is correct
* image is not corrupt
* OpenCV supports the format

---

## Worker jobs remain PROCESSING

Check the worker output and process exit codes.

If a worker crashed, stale jobs can be recovered on the next run using:

```python
WORKER_STALE_TIMEOUT_SECONDS
```

---

## Jobs are duplicated

Do not run the new worker pipeline against an old database containing observations without synchronizing the image job table.

For development, use a clean event database.

---

## TorchReID import fails

Verify:

```bash
python -c "import torchreid; print('OK')"
```

Then reinstall:

```bash
pip install -r requirements.txt
```

Make sure the virtual environment is activated.

---

# 62. Production Safety Checklist

Before processing a large event:

```text
[ ] Virtual environment active
[ ] Correct Python version
[ ] requirements installed
[ ] PyTorch verified
[ ] GPU verified
[ ] Models available
[ ] Event path correct
[ ] Database path correct
[ ] Worker count tested
[ ] Small dataset tested
[ ] No duplicate observations
[ ] No unfinished jobs
[ ] Cluster visualization verified
```

---

# 63. Recommended Initial Configuration

For the first GPU test:

```python
WORKER_COUNT = 1
WORKER_STALE_TIMEOUT_SECONDS = 600
WORKER_USE_PROCESSES = True
```

After a successful run, benchmark:

```text
1 worker
```

against:

```text
2 workers
```

only if GPU memory and system resources permit.

---

# 64. Complete Runtime Architecture

The final runtime architecture is:

```text
                         EVENT DIRECTORY
                               │
                               ▼
                         main.py
                               │
                               ▼
                     Image Job Creation
                               │
                               ▼
                        SQLite Database
                               │
                 ┌─────────────┼─────────────┐
                 │             │             │
                 ▼             ▼             ▼
             Worker 1      Worker 2      Worker N
                 │             │             │
                 ▼             ▼             ▼
             Pipeline      Pipeline      Pipeline
                 │             │             │
                 └─────────────┼─────────────┘
                               │
                               ▼
                      Person Observations
                               │
                               ▼
                         SQLite Database
                               │
                               ▼
                Job Completion Verification
                               │
                               ▼
                  Constrained Identity
                       Clustering
                               │
                               ▼
                     Cluster Assignments
                               │
                               ▼
                    Cluster Visualization
                               │
                               ▼
                       Human Labeling
                               │
                               ▼
                         Event Result
```

---

# 65. Core Architectural Rule

The most important rule of the entire system is:

> **Do not allow clustering to compensate for bad observations.**

The pipeline must first produce reliable:

```text
Person detections
Face associations
Face embeddings
Body embeddings
Quality scores
Validated observations
```

Only then should clustering be trusted.

The architecture therefore remains layered:

```text
Detection
    ↓
Representation
    ↓
Quality
    ↓
Persistence
    ↓
Parallel Processing
    ↓
Temporal Context
    ↓
Constrained Clustering
    ↓
Cluster Refinement
    ↓
Visualization
    ↓
Human Labeling
```

---

# 66. Final Usage

The normal workflow is simply:

```bash
# Activate environment

# Windows
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure event/model paths
# app/configuration.py

# Run the complete pipeline
python main.py
```

The system then performs the complete event workflow automatically:

```text
Images
  ↓
SQLite Jobs
  ↓
Workers
  ↓
Observations
  ↓
Database
  ↓
Constrained Clustering
  ↓
Visualization
  ↓
Event Results
```

---

# 67. Project Status

The system architecture is designed to support:

```text
✓ Event-based processing
✓ Person detection
✓ Face detection
✓ Face/person association
✓ Face embeddings
✓ Body/Re-ID embeddings
✓ Quality-aware observations
✓ SQLite persistence
✓ Persistent observation IDs
✓ SQLite image job queue
✓ Worker job claiming
✓ Worker failure handling
✓ Stale-job recovery
✓ Multiprocessing
✓ GPU-based ML inference
✓ Same-image clustering constraints
✓ Event-level constrained clustering
✓ Cluster assignment persistence
✓ Cluster visualization
✓ Large event scalability
```

The primary remaining work for a research/production-quality system is not adding more architectural layers, but **benchmarking and calibrating the existing implementation** against representative event datasets.

The most important measurements are:

```text
Runtime
GPU utilization
RAM usage
VRAM usage
Detection quality
Observation quality
Cluster purity
False merges
Cluster fragmentation
Failed image jobs
```

This turns the system from a working implementation into a measurable and reproducible computer-vision system.
