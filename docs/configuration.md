# PersonaCluster Configuration

PersonaCluster centralizes tunable parameters in:

```text
app/configuration.py
```

This document describes the baseline configuration and the current event-local path model.

---

# 1. Event Paths

The event container is:

```text
data/events/
```

Each direct child is one event.

Within an event:

```text
data/events/<event_name>/Gallery/
data/events/<event_name>/reference/
data/events/<event_name>/event.db
```

The gallery is recursively searched.

The reference directory is used only for reference matching.

Final output is written outside the event input tree:

```text
output/<event_name>/
```

---

# 2. Person Detection

```python
PERSON_MODEL_NAME = "models/yolo11n.pt"
PERSON_DETECTION_THRESHOLD = 0.40
PERSON_DETECTION_DEVICE = None
PERSON_CLASS_ID = 0
```

`PERSON_CLASS_ID = 0` represents the person class in the configured detector.

---

# 3. Face Detection

```python
FACE_MODEL_NAME = "buffalo_l"
FACE_DETECTION_SIZE = (640, 640)
FACE_DETECTION_THRESHOLD = 0.40
FACE_CTX_ID = 0
```

`FACE_CTX_ID` controls the InsightFace execution context according to the installed runtime.

---

# 4. Face/Person Association

```python
ASSOCIATION_MIN_SCORE = 0.30
```

Association uses geometric relationships between face and person detections.

---

# 5. Body Encoder

```python
BODY_MODEL_NAME = "osnet_x1_0"
BODY_MODEL_PATH = None
BODY_DEVICE = None
BODY_UPPER_BODY_RATIO = 0.60
```

OSNet provides supporting body appearance evidence.

---

# 6. Face Quality

```python
FACE_QUALITY_DETECTION_WEIGHT = 0.40
FACE_QUALITY_SIZE_WEIGHT = 0.30
FACE_QUALITY_SHARPNESS_WEIGHT = 0.30

FACE_MIN_SIZE = 40
FACE_REFERENCE_SIZE = 120

FACE_SHARPNESS_MIN = 20.0
FACE_SHARPNESS_MAX = 150.0
```

The face quality score combines:

```text
Detection confidence → 40%
Face size            → 30%
Sharpness            → 30%
```

---

# 7. Body Quality

```python
BODY_QUALITY_DETECTION_WEIGHT = 0.40
BODY_QUALITY_SIZE_WEIGHT = 0.30
BODY_QUALITY_SHARPNESS_WEIGHT = 0.30

BODY_MIN_SIZE = 100
BODY_REFERENCE_SIZE = 400

BODY_SHARPNESS_MIN = 20.0
BODY_SHARPNESS_MAX = 150.0
```

---

# 8. Clustering Weights

```python
CLUSTER_FACE_WEIGHT = 0.65
CLUSTER_BODY_WEIGHT = 0.20
CLUSTER_QUALITY_WEIGHT = 0.15
```

Face is the primary identity signal.

For face-only observations:

```python
CLUSTER_FACE_ONLY_WEIGHT = 0.75
CLUSTER_FACE_ONLY_QUALITY_WEIGHT = 0.25
```

---

# 9. Identity Thresholds

```python
MIN_FACE_SIMILARITY = 0.55
MERGE_THRESHOLD = 0.78
```

These are project-specific values.

---

# 10. Cluster Representation

```python
MIN_CLUSTER_SIZE = 2
REPRESENTATIVE_COUNT = 3
```

`MIN_CLUSTER_SIZE` controls the minimum number of observations required for a discovered identity.

---

# 11. Face Pose

```python
FACE_POSE_FRONTAL_YAW_DEGREES = 20.0
FACE_POSE_PROFILE_YAW_DEGREES = 55.0
```

Pose categories are coarse contextual categories rather than precise 3D measurements.

---

# 12. Reference Matching

Reference files are event-local.

For event `EEA`:

```text
data/events/EEA/reference/
```

Expected filename:

```text
Person Name - Phone Number.ext
```

The reference matching threshold is:

```python
REFERENCE_MATCH_THRESHOLD = 0.55
```

The minimum best-vs-second-best margin is:

```python
REFERENCE_MATCH_MIN_MARGIN = 0.02
```

Reference matching is performed after clustering.

---

# 13. Output Configuration

The final output root is the project-level:

```text
output/
```

Each event gets:

```text
output/<event_name>/
```

The final mode is reference matched:

```python
EVENT_OUTPUT_MODE = "REFERENCE_MATCHED"
```

Output cleanup is controlled by:

```python
EVENT_OUTPUT_CLEAN_BEFORE_RUN = True
```

Best-image selection uses:

```python
BEST_IMAGES_PER_CLUSTER = 5
BEST_IMAGES_REQUIRE_POSE_DIVERSITY = True
```

The output manager aggregates multiple accepted clusters belonging to the same known person.

---

# 14. Worker Configuration

A worker loads multiple ML components, so worker count directly affects memory use.

The baseline should be conservative for a 6 GB GPU:

```python
WORKER_COUNT = 1
```

If the active implementation is configured with a higher value, verify VRAM/RAM usage before increasing it further.

Additional settings include:

```python
WORKER_STALE_TIMEOUT_SECONDS = 600
WORKER_USE_PROCESSES = True
```

---

# 15. Google Drive

Google Drive settings are controlled through configuration/environment values.

Important concepts:

- Authentication credentials
- OAuth token
- Root folder
- Event folder
- Public-link behavior
- Retry count
- HTTP timeout

Google Drive receives person folders and images, not the local Excel report.

---

# 16. Configuration Principle

Keep thresholds and paths centralized.

When changing a parameter:

1. Record the original value.
2. Change only the intended parameter.
3. Run the same evaluation event.
4. Compare false merges, fragmentation, unknown observations, output quality, and runtime.
5. Keep the change only when its effect is understood.
