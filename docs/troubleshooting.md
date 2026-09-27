# PersonaCluster Troubleshooting

This guide covers common problems in the current multi-event, event-local-reference architecture.

---

# 1. No Events Found

Expected root:

```text
data/events/
```

An event must be a direct child:

```text
data/events/EEA/
```

If `data/events/` contains only:

```text
data/events/EEA/Gallery/...
```

then EEA is the event.

Do not place `EEA` inside another gallery folder.

---

# 2. Event Found but No Images

Check:

```text
data/events/<event>/Gallery/
```

The application searches recursively under `Gallery` .

Correct:

```text
EEA/
└── Gallery/
    └── Camera/
        └── Day1/
            └── IMG001.jpg
```

Incorrect for gallery processing:

```text
EEA/
└── reference/
    └── Person.jpg
```

Reference files are not gallery images.

---

# 3. References Are Not Found

Check:

```text
data/events/<event>/reference/
```

Do not use the old global:

```text
data/reference/
```

Reference filenames should follow:

```text
Person Name - Phone Number.ext
```

Example:

```text
Ahmed Hassan - 201012345678.jpg
```

The name is used as the output person name.

---

# 4. Reference Images Are Being Processed as Gallery Images

This indicates that image discovery is using the event root instead of:

```text
<event>/Gallery/
```

Verify that recursive discovery starts at `Gallery/` .

The following must never be scanned as gallery input:

```text
<event>/reference/
<event>/event.db
```

---

# 5. Output Is in the Wrong Directory

Correct:

```text
output/<event_name>/
```

Incorrect:

```text
data/events/<event_name>/output/
```

The event directory is the input/persistent-state location.

The project-level `output/` directory is the final-result location.

---

# 6. `EventOutputManager` Import Error

`main.py` must import:

```python
from app.output.eventOutputManager import EventOutputManager
```

If this fails, verify:

```text
app/output/eventOutputManager.py
```

exists.

Also verify that the project root is the directory containing `main.py` .

---

# 7. Cluster Visualizer Import Error

Expected module:

```text
app/clustering/clusterVisualizer.py
```

Expected import:

```python
from app.clustering.clusterVisualizer import visualizeClusters
```

If the module is missing, restore the visualizer file before running the application.

---

# 8. CUDA Is Unavailable

Run:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Check:

* NVIDIA driver
* Active Python environment
* PyTorch installation
* GPU visibility
* CUDA-compatible PyTorch build

InsightFace/ONNX Runtime may independently fall back to CPU depending on installed providers.

---

# 9. CUDA Out of Memory

Reduce worker count.

Start with:

```python
WORKER_COUNT = 1
```

Each worker can initialize:

```text
YOLO
InsightFace
OSNet/TorchReID
```

More workers can multiply memory consumption.

---

# 10. `cv2.imread()` Returns `None`

Check:

* File exists
* Path is correct
* File is readable
* File is not corrupt
* Extension is supported
* OpenCV can decode it

If the file is valid in another image viewer but OpenCV cannot decode it, test the decoder independently.

---

# 11. TorchReID Import Fails

Test:

```bash
python -c "import torchreid; print('OK')"
```

If necessary:

```bash
python -m pip install --no-build-isolation git+https://github.com/KaiyangZhou/deep-person-reid.git
```

Then restart the environment.

---

# 12. Jobs Remain `PROCESSING`

Inspect:

* Worker process termination
* CUDA errors
* Model initialization
* Image decoding
* Database errors
* Python exceptions

The stale-job timeout is:

```python
WORKER_STALE_TIMEOUT_SECONDS
```

The baseline documented value is 600 seconds.

---

# 13. Jobs Remain `FAILED`

Inspect the stored:

```text
image path
error message
attempt count
worker logs
```

Common causes:

```text
Corrupt image
Unreadable image
Invalid crop
Model error
CUDA out-of-memory
Embedding shape error
Dependency problem
```

A failed image is different from a successfully processed image that produced no useful observation.

---

# 14. Same Person Appears in Multiple Clusters

This is fragmentation.

Review:

```text
MIN_FACE_SIMILARITY
MERGE_THRESHOLD
CROSS_POSE_MIN_FACE_SIMILARITY
CROSS_POSE_MIN_BODY_SIMILARITY
CROSS_POSE_MERGE_THRESHOLD
```

Also inspect:

```text
Face quality
Face pose
Identity anchors
Body quality
Face/body embeddings
```

Do not change thresholds before checking observation quality.

---

# 15. Different People Are Merged

This is a false merge.

Review:

```text
MIN_FACE_SIMILARITY
MERGE_THRESHOLD
CROSS_POSE_MIN_FACE_SIMILARITY
CROSS_POSE_MIN_BODY_SIMILARITY
CROSS_POSE_MERGE_THRESHOLD
```

Also inspect:

* Face quality
* Pose
* Body similarity
* Identity anchors
* Same-image constraint
* Observation quality

---

# 16. Profile-Only Observations Do Not Form Identities

This can be expected when anchor requirements are strict.

Review:

```text
ANCHOR_MIN_FACE_QUALITY
ANCHOR_MIN_FACE_DETECTION_CONFIDENCE
ANCHOR_MIN_FACE_SIZE
ANCHOR_MAX_YAW_DEGREES
```

Weak side/profile observations may contribute to an already established identity without independently establishing one.

---

# 17. Duplicate Filenames

This is valid:

```text
Gallery/CameraA/IMG001.jpg
Gallery/CameraB/IMG001.jpg
```

They are different source images.

Do not identify images using only:

```python
Path(image_path).name
```

Persistent image identity must preserve the source path.

---

# 18. Duplicate Observations

First determine whether the apparent duplicates come from:

* Old database state
* Reprocessing
* Job recreation
* Architecture changes
* Reusing a database with incompatible code

For schema/persistence changes, use a fresh event database.

---

# 19. Empty Final Output

If processing succeeds but:

```text
Accepted reference matches: 0
```

then no cluster passed the configured reference matching rules.

Check:

```text
data/events/<event>/reference/
REFERENCE_MATCH_THRESHOLD
REFERENCE_MATCH_MIN_MARGIN
```

Also verify that the reference images contain readable faces.

An empty person output can be a valid result when no cluster matches the event references.

---

# 20. Google Drive Upload Issues

Check:

* Credentials path
* OAuth token
* Root folder configuration
* Network access
* Drive API permissions
* Public-link configuration
* Retry/timeout settings

The local output should be checked first:

```text
output/<event_name>/
```

Drive upload should not be used as the only verification of successful clustering.

---

# 21. Excel Is Missing

The Excel report is generated locally:

```text
output/<event_name>/final_results.xlsx
```

It is not intended to be uploaded to Google Drive.

If the file is missing, inspect the output-generation phase and verify that the event output directory exists.

---

# 22. Old Output Appears

The output cleanup setting controls whether previous final output is removed before rebuilding:

```python
EVENT_OUTPUT_CLEAN_BEFORE_RUN
```

Be careful when debugging incremental runs because deleting output does not necessarily delete the event database.

---

# 23. Database State Looks Wrong

Remember that:

```text
data/events/<event>/event.db
```

is persistent state.

If the image-processing architecture or database schema changes, an old database may no longer represent the current implementation.

For development, use a fresh event/database when appropriate.

---

# 24. Correct Diagnostic Checklist

For an event named `EEA` :

```text
[ ] data/events/EEA exists
[ ] data/events/EEA/Gallery exists
[ ] Gallery contains images or nested image folders
[ ] data/events/EEA/reference exists
[ ] reference filenames use Person - Phone.ext
[ ] event.db belongs to EEA
[ ] output/EEA is the final output root
[ ] EventOutputManager import is valid
[ ] clusterVisualizer exists if visualization is enabled
[ ] ML dependencies import correctly
[ ] CUDA/CPU provider is understood
[ ] worker count fits available VRAM/RAM
```
