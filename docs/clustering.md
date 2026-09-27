# PersonaCluster Identity Clustering

This document describes event-level identity discovery. Clustering is performed independently for each event after its Gallery images have been processed.

---

# 1. Identity Discovery Pipeline

```text
Completed Observations
        │
        ▼
Constrained Identity Clustering
        │
        ├── Face Evidence
        ├── Body Evidence
        ├── Quality
        ├── Pose
        ├── Identity Anchors
        └── Same-Image Constraints
        │
        ▼
Anonymous Identity Clusters
        │
        ▼
Event Reference Matching
        │
        ▼
Known-Person Output
```

A cluster is an event-level identity hypothesis. It is not automatically a person's real name.

---

# 2. Event Isolation

Clustering never mixes events.

For:

```text
data/events/EEA/
data/events/Event_2026/
```

there are two independent observation sets and two independent clustering runs.

References are also event-local:

```text
data/events/EEA/reference/
data/events/Event_2026/reference/
```

---

# 3. Observation vs Identity

```text
Image
  ↓
Observation
  ↓
Candidate identity
  ↓
Cluster
  ↓
Reference match
```

An observation represents an occurrence in one image. A cluster represents a collection of observations believed to belong to one event-level person.

---

# 4. Identity Evidence

The two main appearance modalities are:

```text
Face
 └── normalized face embedding

Body
 └── OSNet appearance embedding
```

Face is the primary identity signal.

Body is supporting evidence.

Quality indicates the reliability of the evidence.

---

# 5. Standard Matching Weights

The baseline configuration uses:

```python
CLUSTER_FACE_WEIGHT = 0.65
CLUSTER_BODY_WEIGHT = 0.20
CLUSTER_QUALITY_WEIGHT = 0.15
```

For observations without a valid body embedding:

```python
CLUSTER_FACE_ONLY_WEIGHT = 0.75
CLUSTER_FACE_ONLY_QUALITY_WEIGHT = 0.25
```

These are project-specific parameters and should be calibrated against representative datasets.

---

# 6. Face Similarity

The baseline face threshold is:

```python
MIN_FACE_SIMILARITY = 0.55
```

A comparison below the configured minimum cannot create an identity merge.

The threshold is not a universal biometric threshold.

---

# 7. Merge Threshold

The baseline combined-score threshold is:

```python
MERGE_THRESHOLD = 0.78
```

A candidate pair must satisfy the configured compatibility rules and combined score before merging.

---

# 8. Same-Image Constraint

Two observations from the same source image cannot be assigned to the same identity cluster.

For example:

```text
IMG001.jpg
 ├── Observation A
 └── Observation B
```

A and B must not be merged into one identity.

This is a correctness constraint.

---

# 9. Quality-Aware Matching

Quality is derived from signals such as:

- Detection confidence
- Face/body size
- Image sharpness

Quality is normalized to:

```text
0.0 → poor
1.0 → strong
```

Quality supports identity matching; it is not itself an identity signal.

---

# 10. Face Pose

Pose is estimated from facial landmarks using OpenCV `solvePnP`.

The system derives:

```text
Yaw
Pitch
Roll
```

and coarse categories such as:

```text
frontal
left
right
profile
unknown
```

Pose is contextual evidence.

---

# 11. Identity Anchors

Strong observations can act as identity anchors.

Anchor requirements include configurable limits for:

- Face quality
- Face detection confidence
- Face size
- Face pose/yaw

Conceptually:

```text
Clear + sufficiently large + approximately frontal
                    │
                    ▼
              Identity Anchor
```

A weak profile observation should not independently establish a strong identity when the anchor rules reject it.

It may still contribute to an existing identity when clustering rules allow it.

---

# 12. Cross-Pose Matching

Cross-pose comparisons use dedicated configuration values:

```text
CROSS_POSE_MIN_FACE_SIMILARITY
CROSS_POSE_MIN_BODY_SIMILARITY
CROSS_POSE_MERGE_THRESHOLD
```

The purpose is to allow an anchored identity to expand across viewpoints without treating weak cross-pose evidence as equivalent to strong same-pose evidence.

---

# 13. Cluster Merge Strategy

Conceptually:

```text
Candidate cluster pairs
        │
        ▼
Compatibility calculation
        │
        ▼
Priority ordering
        │
        ▼
Strongest compatible merge
        │
        ▼
Update cluster state
        │
        ▼
Continue
```

The main rules are:

1. Face similarity is primary.
2. Body similarity is supporting evidence.
3. Quality supports reliability.
4. Strong observations can represent an identity.
5. Same-image observations cannot share an identity cluster.

---

# 14. Cluster Persistence

Cluster assignments are persisted in the event database:

```text
data/events/<event_name>/event.db
```

This allows subsequent runs to reuse compatible clustering state when the relevant observations and configuration have not changed.

When an image changes, dependent state can be invalidated and rebuilt.

---

# 15. Reference Matching Is Separate

Clustering discovers anonymous identities.

Reference matching then maps accepted clusters to known people from:

```text
data/events/<event_name>/reference/
```

This separation is intentional:

```text
Event observations
       ↓
Anonymous clusters
       ↓
Reference matching
       ↓
Known-person output
```

A known person may own more than one cluster.

---

# 16. Output Implications

The output manager aggregates accepted clusters by person.

Example:

```text
Person A
 ├── Cluster 4
 ├── Cluster 9
 └── Cluster 17
```

becomes:

```text
output/EEA/Person A/
├── All Images/
├── Best Images/
└── representative Image.jpg
```

Clusters from another event are never included.

---

# 17. Tuning Guidance

Thresholds should be calibrated using labeled event data.

Evaluate:

- False merges
- Fragmentation
- Unknown observations
- Same-image violations
- Cross-pose behavior
- Anchor quality
- Best-image quality

Do not interpret a project-specific similarity threshold as a universal biometric standard.
