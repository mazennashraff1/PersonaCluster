# PersonaCluster Development Guide

This document describes the recommended development workflow for
PersonaCluster, including environment setup, clean test runs, worker
behavior, database considerations, reliability checks, performance
testing, and Git practices.

------------------------------------------------------------------------

## 1. Development Environment

Recommended development environment:

```text
Python 3.10 or 3.11
16 GB RAM or more
NVIDIA GPU recommended for large events
SSD recommended
```

Create a virtual environment before installing project dependencies.

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

Upgrade the packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

If TorchReID requires a separate installation in the development
environment:

```bash
python -m pip install --no-build-isolation git+https://github.com/KaiyangZhou/deep-person-reid.git
```

------------------------------------------------------------------------

## 2. Environment Verification

Before running an event, verify the machine and installed ML
environment.

### PyTorch and CUDA

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

This confirms:

* PyTorch is installed.
* CUDA availability can be detected.
* The active GPU can be identified when CUDA is available.

### Core Dependencies

```bash
python -c "import numpy, cv2, PIL, scipy, torch, torchvision, ultralytics, insightface, onnxruntime, sklearn, skimage, matplotlib, yaml; print('All core dependencies imported successfully')"
```

### TorchReID

```bash
python -c "import torchreid; print('TorchReID imported successfully')"
```

If one of these checks fails, resolve the environment problem before
evaluating the application.

------------------------------------------------------------------------

## 3. Configuration During Development

The project's tunable parameters are centralized in:

```text
app/configuration.py
```

Avoid scattering thresholds or model settings across individual
modules.

Configuration changes should be made deliberately and evaluated against
a consistent dataset.

Important configuration groups include:

```text
Detection
Association
Body encoding
Face quality
Body quality
Identity clustering
Face pose
Cross-pose matching
Identity anchors
Best-image selection
Representative-face selection
Workers
Event output
```

When changing an important threshold or weight:

01. Record the original value.
02. Change the value in `app/configuration.py`.
03. Run the same evaluation dataset.
04. Compare the resulting behavior.
05. Keep the change only if it provides a measurable improvement.

------------------------------------------------------------------------

## 4. Clean Development Run

For a completely fresh event test:

```text
01. Keep the original photographs unchanged.
02. Use a fresh event database or reset the existing development database.
03. Remove previously generated output.
04. Activate the virtual environment.
05. Verify dependencies and models.
06. Start with a conservative worker count.
07. Run main.py.
08. Review image-job statistics.
09. Review clustering statistics.
10. Inspect the generated cluster output.
```

Run the application with:

```bash
python main.py
```

The current processing flow is:

```text
Discover event images
        ↓
Create image jobs
        ↓
Process images with workers
        ↓
Persist observations
        ↓
Recover/retry failed or stale jobs
        ↓
Load event observations
        ↓
Constrained identity clustering
        ↓
EventOutputManager
        ↓
All Images
Best Images
Representative Face
```

------------------------------------------------------------------------

## 5. Database State

The event database is part of the application's processing state.

During development, avoid blindly reusing an old database after changing
the observation schema, worker architecture, or persistence behavior.

An old database can contain observations that no longer correspond
correctly to the current image-processing state.

For example:

```text
Existing observations
        +
Reprocessed images
        ↓
Potential duplicate observations
```

When making architectural or schema-level changes, prefer a fresh event
database unless a specific migration procedure is available.

A clean development run should therefore treat:

```text
Images
Database
Generated output
```

as a consistent experimental state.

------------------------------------------------------------------------

## 6. Worker Architecture

Image processing is performed by `ImageWorker` .

Each worker process owns its own processing resources, including:

```text
EventStore
SQLite connection
PersonPipeline
Person detector
Face detector
Face pose estimator
Body encoder
Quality/validation components
```

Workers should not share:

```text
SQLite connection objects
ML model instances
PersonPipeline instances
```

The process boundary is intentional because ML models and database
connections are process-local resources.

The workers are responsible for image-level processing only.

They do **not** perform event-level identity clustering.

------------------------------------------------------------------------

## 7. Worker Processing Lifecycle

Each image follows the job lifecycle:

```text
PENDING
   ↓
PROCESSING
   ↓
Read image
   ↓
Run PersonPipeline
   ↓
Reserve observation IDs
   ↓
Persist observations
   ↓
COMPLETED
```

If processing fails:

```text
PROCESSING
     ↓
FAILED
```

The failure remains associated with the corresponding image job so that
the problem can be diagnosed or the job can be retried.

------------------------------------------------------------------------

## 8. Worker Count and GPU Memory

Each worker may initialize multiple ML components.

Therefore:

```text
Worker count ↑
      ↓
Model instances ↑
      ↓
RAM / VRAM usage ↑
```

More workers do not automatically mean better performance.

For a GPU with limited VRAM, start with:

```python
WORKER_COUNT = 1
```

Then benchmark higher values if sufficient resources are available.

Compare:

```text
1 worker
2 workers
...
```

using the same event and configuration.

Measure both throughput and stability.

A configuration that is slightly faster but repeatedly causes CUDA
out-of-memory errors is not an effective production configuration.

------------------------------------------------------------------------

## 9. Image Processing Reliability

An individual image should be processed independently of other images.

The intended worker flow is:

```text
Claim image job
      ↓
Read image
      ↓
Run detection
      ↓
Associate face/person detections
      ↓
Extract embeddings
      ↓
Calculate quality
      ↓
Validate embeddings
      ↓
Create observations
      ↓
Reserve observation IDs
      ↓
Persist observations
      ↓
Complete image job
```

If processing fails, the corresponding image job should retain its
failure state and error information.

One problematic image should not silently invalidate observations
already persisted for other images.

------------------------------------------------------------------------

## 10. Failed Jobs

Typical image-processing failures can include:

```text
Corrupt image
Unreadable file
Invalid image dimensions
Invalid crop
CUDA out-of-memory
Model failure
Unexpected embedding shape
Missing model
Dependency problem
Unexpected runtime exception
```

When an event contains failed jobs, investigate them before considering
the event completely processed.

Useful information to inspect includes:

```text
Image path
Job state
Error message
Worker logs
Model/device configuration
```

A failed image is not equivalent to an image containing no person.

------------------------------------------------------------------------

## 11. Stale Jobs

A worker can terminate unexpectedly while an image remains in:

```text
PROCESSING
```

The event-processing logic can identify jobs that have exceeded:

```python
WORKER_STALE_TIMEOUT_SECONDS
```

and recover them so they can be processed again.

Conceptually:

```text
PROCESSING
    │
    │ timeout
    ▼
Stale job
    │
    ▼
PENDING
    │
    ▼
Retry
```

The current baseline timeout is:

```text
600 seconds
```

This mechanism helps recover work after unexpected worker termination.

------------------------------------------------------------------------

## 12. Event-Level Identity Processing

Image workers stop after observations have been persisted.

Identity discovery happens later at the event level:

```text
All persisted observations
          ↓
ConstrainedIdentityClustering
          ↓
Identity clusters
          ↓
EventOutputManager
          ↓
Final output
```

The clustering stage uses the complete available event observation set.

It considers:

```text
Face embeddings
Body embeddings
Quality
Face pose
Trusted anchors
Same-image constraints
```

The same-image constraint remains a clustering rule:

> Two observations originating from the same source image cannot belong
> to the same identity cluster.

There is no separate duplicate-suppression stage in the current
architecture.

------------------------------------------------------------------------

## 13. Final Output Processing

`EventOutputManager` handles the final local output after clustering and
reference matching.

Its responsibilities include:

```text
Reference-matched person directories
All-image copying
Best-image selection
Representative-face selection
Representative-face cropping
Output filename collision handling
```

The current output structure is:

```text
output/
├── Person Name/
│   ├── All Images/
│   ├── Best Images/
│   └── representative Image.jpg
└── final_results.xlsx
```

Only clusters that match people in `data/reference` are exported. Multiple
matched clusters for the same person are combined into the same person folder.

`final_results.xlsx` is generated after the local person folders are complete
and contains:

```text
Name of Person | Phone Number | Folder Shared Link
```

When Google Drive is enabled, the Drive uploader first uploads each person
folder and obtains its folder URL. Those URLs are then written to the Excel
file, and the completed Excel file is uploaded to the event folder.

There is no manifest file in the current output.

## 14. Performance Testing

Performance should be evaluated progressively.

A useful sequence is:

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

At each stage, record:

```text
Total runtime
Images per second
CPU usage
RAM usage
GPU utilization
VRAM usage
Observation count
Failed jobs
Cluster count
```

For identity-clustering evaluation, also record where ground-truth
identity labels are available:

```text
Cluster purity
False merge rate
Fragmentation
```

Performance measurements should be compared using the same dataset and
configuration whenever possible.

------------------------------------------------------------------------

## 15. Optimization Order

Use the following priority when improving the system:

```text
01. Correctness
02. Database reliability
03. Worker reliability
04. Observation quality
05. Identity-clustering quality
06. GPU utilization
07. Parallelism
08. Benchmarking
```

Do not increase worker count simply to obtain higher theoretical
throughput.

First establish that:

```text
Observations are correct
Jobs complete correctly
Embeddings are valid
Clusters are reasonable
Output is correct
```

Only then optimize execution speed.

------------------------------------------------------------------------

## 16. Image-Level vs Event-Level Work

The architecture deliberately separates image-level processing from
event-level identity discovery.

### Image-Level

These operations can run in parallel:

```text
Image A → Worker 1
Image B → Worker 2
Image C → Worker 3
```

Each worker produces independent observations.

### Event-Level

These operations require the event observation set:

```text
All observations
      ↓
Identity clustering
      ↓
Cluster assignments
      ↓
Best-image selection
      ↓
Representative-face selection
      ↓
Output
```

The current architecture therefore does **not** perform identity
clustering inside individual image workers.

This boundary should be preserved when extending the system.

------------------------------------------------------------------------

## 17. Adding a New Image-Level Component

When adding a new image-level model or calculation:

01. Determine whether it belongs to detection, representation, quality,
   association, or validation.
02. Keep it inside the per-image pipeline when possible.
03. Add its result to `PersonObservation` if it must be persisted.
04. Persist the observation through `EventStore`.
05. Keep event-level identity decisions inside the clustering layer.
06. Add tunable parameters to `app/configuration.py`.
07. Add evaluation measurements.
08. Verify that the new component is actually used by the active
   execution path.

Avoid adding infrastructure that is implemented but never called.

------------------------------------------------------------------------

## 18. Adding a New Clustering Rule

When changing identity clustering:

01. Define the rule clearly.
02. Determine whether it is a hard constraint or soft evidence.
03. Keep the rule inside `ConstrainedIdentityClustering`.
04. Avoid duplicating the same identity rule in image workers.
05. Update the relevant configuration values if applicable.
06. Add or update evaluation metrics.
07. Test the rule on synthetic cases.
08. Test it on representative event data.
09. Check for both false merges and unnecessary fragmentation.

Particular care should be taken with changes involving:

```text
Face similarity
Body similarity
Cross-pose matching
Trusted anchors
Same-image constraints
Cluster merging
```

------------------------------------------------------------------------

## 19. Testing Strategy

A useful development test sequence is:

```text
Unit-level checks
       ↓
Synthetic observations
       ↓
Small real/consented event
       ↓
Medium event
       ↓
Large event
```

At each stage verify:

```text
No unexpected duplicate observations
No invalid observation IDs
No invalid embeddings
No unexpected unfinished jobs
No prohibited same-image merges
Expected cluster behavior
Expected output structure
Representative face generated when valid face data exists
```

For clustering tests, synthetic observations are particularly useful
for testing:

```text
Strong same-person matches
Weak matches
Cross-pose matches
Different-person observations
Same-image constraints
Trusted-anchor behavior
Cluster merging
```

------------------------------------------------------------------------

## 20. Model Changes

When changing any of the following:

```text
YOLO model/version
InsightFace model
TorchReID model
PyTorch version
TorchVision version
ONNX Runtime
CUDA/PyTorch build
```

treat the result as a new experimental configuration.

Record:

```text
Model version
Configuration
Dataset
Runtime environment
Hardware
Clustering metrics
Output inspection
```

Do not assume that a newer model automatically produces better identity
clustering.

A model change can affect:

```text
Detection
Embedding quality
Pose estimation
Observation quality
Similarity distributions
Cluster formation
```

------------------------------------------------------------------------

## 21. Reproducibility

For meaningful comparisons, keep the following fixed whenever possible:

```text
Dataset
Configuration
Model versions
Hardware
Evaluation procedure
```

Change one major variable at a time.

For example:

```text
Experiment A
    Worker count = 1

Experiment B
    Worker count = 2
```

while keeping the dataset, models, and clustering configuration
unchanged.

This makes performance differences easier to attribute.

------------------------------------------------------------------------

## 22. Debugging the Pipeline

When an unexpected clustering result occurs, debug from the observation
level upward.

Use this order:

```text
Image
  ↓
Person detection
  ↓
Face detection
  ↓
Association
  ↓
Face/body embeddings
  ↓
Quality
  ↓
Pose
  ↓
Persisted observation
  ↓
Clustering
  ↓
Output
```

Do not immediately modify clustering thresholds when the underlying
problem may be an incorrect detection, association, embedding, or
quality score.

A useful debugging question is:

> Is the observation wrong, or is the identity decision wrong?

If the observation is wrong, fix the image-level pipeline first.

If the observation is correct but the identity assignment is wrong, 
investigate the clustering logic.

------------------------------------------------------------------------

## 23. Git Development Practices

Do not commit sensitive, private, or generated event data.

Avoid committing:

```text
Real event photographs
Real event databases
Face embeddings
Body embeddings
API keys
Passwords
Access tokens
Private configuration
Large generated outputs
Unlicensed model weights
```

Use:

```text
.gitignore
Environment variables
Local configuration
```

for development-only data and secrets.

The repository should contain the software and documentation required
to reproduce the project, not private event data.

------------------------------------------------------------------------

## 24. Development Principle

The most important development principle is:

> **Do not ask identity clustering to compensate for bad observations.**

If clustering behaves poorly, first determine whether the underlying
observations are reliable.

Investigate:

```text
Person detection
Face detection
Face/person association
Face embeddings
Body embeddings
Face pose
Quality scores
Embedding validation
```

before adding increasingly complicated identity-clustering rules.

The architecture should remain simple:

```text
Good observations
       ↓
Reliable persistence
       ↓
Constrained identity clustering
       ↓
Clear output
```

Each layer should solve its own responsibility rather than compensating
for failures in another layer.
