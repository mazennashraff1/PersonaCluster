# PersonaCluster Development Guide

This guide describes development, clean runs, event layout, persistent state, workers, debugging, and validation.

---

# 1. Development Environment

Recommended baseline:

```text
Python 3.10 or 3.11
16 GB RAM or more
NVIDIA GPU recommended
SSD recommended
```

Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If required by the environment:

```bash
python -m pip install --no-build-isolation git+https://github.com/KaiyangZhou/deep-person-reid.git
```

---

# 2. Environment Verification

Check PyTorch/CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Check core packages:

```bash
python -c "import numpy, cv2, PIL, scipy, torch, torchvision, ultralytics, insightface, onnxruntime, sklearn, skimage, matplotlib, yaml; print('All core dependencies imported successfully')"
```

Check TorchReID:

```bash
python -c "import torchreid; print('TorchReID imported successfully')"
```

---

# 3. Current Event Layout

Create events like:

```text
data/
└── events/
    ├── EEA/
    │   ├── Gallery/
    │   │   ├── Camera 1/
    │   │   └── Camera 2/
    │   └── reference/
    │       ├── Person 1 - 201xxx.jpg
    │       └── Person 2 - 201xxx.jpg
    │
    └── Event_2026/
        ├── Gallery/
        └── reference/
```

Do not put reference images inside `Gallery/` .

Do not create an event for each camera/day folder.

---

# 4. Running the Batch

Run:

```bash
python main.py
```

The coordinator discovers direct event folders under:

```text
data/events/
```

Then:

```text
Event 1
  ↓
complete event processing
  ↓
Event 2
  ↓
complete event processing
  ↓
...
```

---

# 5. Clean Development Run

For a completely fresh event:

01. Preserve the original gallery photographs.
02. Create the event folder.
03. Put all gallery folders under `Gallery/`.
04. Put event references under `reference/`.
05. Use a fresh `event.db` when testing schema/persistence changes.
06. Remove old generated output for the event if required.
07. Verify dependencies/models.
08. Start with a conservative worker count.
09. Run `python main.py`.
10. Review job statistics.
11. Review clustering statistics.
12. Review reference matches.
13. Inspect `output/<event_name>/`.

---

# 6. Persistent Database Rules

The event database is part of processing state:

```text
data/events/<event_name>/event.db
```

Do not blindly reuse an old database after changing:

* Observation schema
* Image identity rules
* Worker architecture
* Persistence schema
* Clustering state model

For those changes, use a fresh development event/database unless a migration exists.

---

# 7. Incremental Development Run

For normal repeated runs:

```text
Existing event
     │
     ▼
Discover Gallery
     │
     ▼
Compare image state
     │
     ├── unchanged → reuse
     ├── new       → process
     └── changed   → invalidate/reprocess
```

This is the intended persistent-memory behavior.

---

# 8. Worker Lifecycle

```text
PENDING
   ↓
PROCESSING
   ↓
Read image
   ↓
Run PersonPipeline
   ↓
Persist observations
   ↓
COMPLETED
```

On failure:

```text
PROCESSING
   ↓
FAILED
```

Failed images can be retried according to the event-processing logic.

---

# 9. Worker Count

Each process may initialize:

```text
YOLO
InsightFace
FacePoseEstimator
OSNet/TorchReID
Quality components
```

Therefore:

```text
Worker count ↑
      ↓
Model instances ↑
      ↓
RAM / VRAM usage ↑
```

For an RTX 3050 6 GB, start conservatively.

---

# 10. Stale Jobs

A worker can terminate while a job remains:

```text
PROCESSING
```

The configured stale timeout is:

```python
WORKER_STALE_TIMEOUT_SECONDS
```

The coordinator can recover stale jobs after the timeout.

---

# 11. Post-Cluster Recheck

The persistent workflow can perform a bounded recheck:

```text
Process images
      ↓
Cluster
      ↓
Find eligible images with no cluster contribution
      ↓
Retry/reprocess
      ↓
Cluster again
```

This exists to recover potentially useful observations without reprocessing the entire event indefinitely.

---

# 12. Reference Development

Reference files belong to the event:

```text
data/events/<event_name>/reference/
```

Filename:

```text
Person Name - Phone Number.ext
```

Multiple files may represent the same person.

Reference matching should be tested separately from anonymous clustering because a correct cluster can still fail reference matching.

---

# 13. Output Development

Expected local output:

```text
output/<event_name>/
├── Person A/
│   ├── All Images/
│   ├── Best Images/
│   └── representative Image.jpg
└── final_results.xlsx
```

If no clusters match references, the output can legitimately contain no person directories.

---

# 14. Source Image Identity

Do not reduce an image ID to:

```python
Path(image_path).name
```

when duplicate filenames are possible.

These must remain distinct:

```text
Gallery/A/IMG001.jpg
Gallery/B/IMG001.jpg
```

The persistent image identity must preserve the relative source path.

---

# 15. Code Validation

Before committing changes, run at minimum:

```bash
python -m py_compile main.py
python -m py_compile app/pipeline.py
python -m py_compile app/storage/eventStore.py
python -m py_compile app/workers/imageWorker.py
python -m py_compile app/clustering/constrainedClustering.py
python -m py_compile app/output/eventOutputManager.py
python -m py_compile app/identity/referenceMatcher.py
```

Also verify the critical imports.

---

# 16. Development Principle

Keep the following responsibilities separate:

```text
main.py
    event orchestration

pipeline.py
    one-image processing

imageWorker.py
    worker lifecycle

eventStore.py
    persistent state

constrainedClustering.py
    event identity discovery

referenceMatcher.py
    event reference matching

eventOutputManager.py
    final local output

googleDriveUploader.py
    optional external upload
```
