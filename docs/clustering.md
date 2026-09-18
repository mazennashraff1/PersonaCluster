# PersonaCluster Identity Clustering

This document describes the identity-discovery stage of PersonaCluster.

The clustering system is designed for **closed events**. It discovers
anonymous identity clusters from observations generated within the same
event rather than matching against a permanent named-person gallery.

---

## 1. Identity Discovery Pipeline

The current event-level identity stage is:

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
Event Output Manager
        │
        ├── Best Images
        └── Representative Face
```

An identity cluster represents observations believed to belong to the
same real-world person.

It does not automatically contain a person's real name.

---

## 2. Observation vs Identity

An observation is an image-level occurrence.

An identity is an event-level hypothesis.

```text
Image Detection
      ↓
Observation
      ↓
Candidate Identity
      ↓
Cluster
      ↓
Human Labeling
```

A detection should never be treated as a confirmed identity merely
because it has a face embedding.

---

## 3. Identity Evidence

PersonaCluster uses two appearance modalities:

```text
Face
  └── Face embedding

Body
  └── OSNet appearance embedding
```

The face embedding is the primary identity signal.

The body embedding provides supporting evidence, especially when
observations contain different viewpoints or imperfect facial evidence.

---

## 4. Face Similarity

Face similarity is the main identity comparison signal.

The implementation uses normalized face embeddings, allowing
cosine-style similarity to be calculated efficiently.

A face similarity threshold provides a lower boundary below which a face
comparison cannot establish an identity merge.

The configured value is project-specific and should be calibrated
against representative event datasets.

---

## 5. Body Similarity

Body similarity is calculated from normalized OSNet appearance
embeddings.

Body evidence is useful when:

* The face is partially informative.
* Viewpoints differ.
* Face quality is imperfect.
* Appearance provides additional supporting context.

Body evidence is intentionally weaker than face evidence because
clothing, pose, lighting, and appearance can change.

---

## 6. Quality-Aware Matching

Not all observations are equally reliable.

Matching therefore considers observation quality in addition to
appearance similarity.

Current quality signals include:

```text
Detection confidence
Bounding-box size
Image sharpness
```

Quality is normalized to the range:

```text
0.0 → poor
1.0 → strong
```

Quality supports matching decisions but does not independently determine
identity.

---

## 7. Standard Matching Weights

The standard combined identity score uses the configured weighting of:

```text
Face
Body
Quality
```

The face component has the strongest influence, while body appearance
and quality provide supporting evidence.

Conceptually:

```text
Combined Score
    =
    Face Evidence
    +
    Body Evidence
    +
    Quality Evidence
```

For observations without a valid body embedding, body evidence is not
used.

The exact weights and thresholds are configuration parameters and the
implementation remains authoritative.

---

## 8. Minimum Face Similarity

The clustering configuration defines a minimum face-similarity
requirement.

The important rule is:

> A face similarity below the configured minimum cannot create an
> identity merge, even when other evidence is favorable.

This provides a conservative guard against weak facial matches.

The configured value is a project parameter, not a universal biometric
threshold.

---

## 9. Merge Threshold

The clustering configuration defines a minimum combined-score threshold
for cluster merging.

A compatible candidate pair must satisfy the configured identity rules
and combined-score requirement before being merged.

Thresholds should be calibrated using labeled evaluation data.

---

## 10. Same-Image Constraint

Two different observations from the same source image cannot belong to
the same identity cluster.

Example:

```text
IMG_001.jpg

Observation A
Observation B
Observation C
```

The clustering stage must not merge A and B into the same identity
merely because their embeddings are similar.

This is a hard clustering constraint.

The purpose is to prevent the clustering algorithm from interpreting
multiple detected people in one photograph as the same event-level
identity.

---

## 11. Conservative Matching Philosophy

PersonaCluster prioritizes:

```text
High precision
    >
Maximum recall
```

A false merge can contaminate an entire identity cluster.

Therefore the system may intentionally leave uncertain observations
fragmented:

```text
Cluster A → Person X
Cluster B → Person X
```

rather than risk:

```text
Cluster A → Person X + Person Y
```

Fragmentation can later be reduced through threshold calibration and
stronger matching logic.

---

## 12. Face Pose

Face pose is estimated from InsightFace's five facial landmarks using
OpenCV `solvePnP`.

The system derives:

```text
Yaw
Pitch
Roll
```

and a coarse pose category:

```text
frontal
left
right
profile
```

Pose is contextual evidence.

It is not treated as an identity embedding.

---

## 13. Pose Thresholds

The coarse pose classification uses configured yaw thresholds.

Conceptually:

```text
Frontal
    |yaw| <= configured frontal threshold

Profile
    |yaw| >= configured profile threshold

Intermediate yaw
    → left or right according to yaw direction
```

The pose estimator is intended for coarse identity and representative
selection decisions, not precise 3D head-pose measurement.

---

## 14. Cross-Pose Matching

Different viewpoints can produce weaker face similarity than same-pose
comparisons.

PersonaCluster therefore supports a dedicated cross-pose matching path.

Cross-pose matching uses its own configured requirements for:

```text
Minimum face similarity
Minimum body similarity
Merge threshold
```

The purpose is to allow an anchored identity to expand across different
viewpoints while preventing weak cross-pose evidence from being treated
as a strong same-pose match.

The configured values are project-specific and should be calibrated
against representative event data.

---

## 15. Identity Anchors

A cluster can use strong frontal observations as trusted identity
anchors.

Anchor requirements are controlled through the clustering
configuration and include requirements related to:

```text
Face quality
Face detection confidence
Face size
Face pose
Yaw
```

Conceptually:

```text
Clear + sufficiently large + approximately frontal face
                         │
                         ▼
                   Identity Anchor
```

Profile-only or weak side-view observations should not independently
establish a strong identity when the configured anchor requirements are
not satisfied.

They can contribute to an existing identity when the clustering rules
allow the match.

---

## 16. Identity Expansion

The identity lifecycle is therefore:

```text
Strong frontal observation
          │
          ▼
     Identity Anchor
          │
          ├── Frontal observations
          ├── Left-view observations
          ├── Right-view observations
          └── Profile observations
```

This separates:

```text
Identity creation
```

from:

```text
Identity expansion
```

The creation step is deliberately stricter.

---

## 17. Cluster Merge Strategy

The clustering process maintains compatible candidate pairs and
prioritizes strong candidates.

Conceptually:

```text
Candidate Cluster Pairs
        │
        ▼
Calculate Pair Compatibility
        │
        ▼
Priority Queue
        │
        ▼
Strongest Compatible Pair
        │
        ▼
Merge
        │
        ▼
Update Affected Cluster State
        │
        ▼
Continue
        │
        ▼
Final Cluster Assignments
```

The merge rules remain authoritative:

1. Face similarity is primary.
2. Body similarity provides supporting evidence.
3. Quality supports reliability.
4. Strong observations can act as cluster representatives.
5. Same-image observations cannot share an identity cluster.
6. Cross-pose comparisons use their dedicated requirements.
7. Strong compatible pairs are considered first.
8. Configured thresholds remain authoritative.

---

## 18. Clustering Caches

The implementation can cache frequently used clustering information,
including:

```text
Face embeddings
Body embeddings
Quality values
Cluster representatives
Pair scores
```

These caches are performance optimizations.

They do not change the identity rules.

When clusters change, affected cached information is updated or
invalidated as required by the clustering implementation.

---

## 19. Cluster Statistics

Useful clustering statistics include:

```text
Total observations
Valid face observations
Clustered observations
Unknown observations
Cluster count
Rejected same-image merges
Rejected low-similarity merges
```

Cluster count alone is not sufficient to evaluate clustering quality.

A useful evaluation should also consider false merges and fragmentation.

---

## 20. Clustering vs Final Image Selection

Identity clustering and final image selection are separate
responsibilities.

```text
ConstrainedIdentityClustering
    ↓
Determines identity assignments

EventOutputManager
    ↓
Selects strong images
    ↓
Creates representative face
    ↓
Writes final cluster output
```

The output stage does not reassign observations or modify cluster
identity assignments.

There is no separate `BestImageSelector` component in the current
architecture.

Best-image selection is handled by `EventOutputManager` as part of final
event output generation.

---

## 21. Representative Face

The representative face is selected after identity clustering.

The purpose is to provide one useful face image that visually
represents the discovered identity.

Conceptually:

```text
Cluster
   │
   ▼
Candidate observations
   │
   ▼
Face quality / pose / availability
   │
   ▼
Best representative face
   │
   ▼
cluster01/face.jpg
```

The representative is a real image crop from the event.

It is not a generated identity image and does not introduce an external
person reference.

---

## 22. Event Output

After clustering, reference matching determines which discovered clusters
belong to known people. `EventOutputManager` then combines all matched
clusters belonging to the same person into one final person directory.

For example:

```text
output/
└── Person Name/
    ├── All Images/
    ├── Best Images/
    └── representative Image.jpg
```

After the local output is created, the final Excel summary is generated:

```text
final_results.xlsx
├── Name of Person
├── Phone Number
└── Folder Shared Link
```

The clustering stage determines cluster membership. Reference matching maps
eligible clusters to known people. The output manager only presents those
already-established matches and does not change identity assignments.

This separation is important:

```text
Clustering
    → Identity assignment

Output Manager
    → Image selection and presentation
```

---

## 23. Important Failure Modes

### False merge

Two real people are incorrectly placed in one cluster.

This is generally more damaging than fragmentation because it
contaminates the identity cluster.

### Fragmentation

Observations belonging to one person are divided across multiple
clusters.

Fragmentation is undesirable but often safer than a false merge under a
precision-first strategy.

### Weak anchor

An identity may fail to form when all available observations are
low-quality or do not satisfy the configured anchor requirements.

### Poor cross-pose evidence

A profile or side-view observation may not contain enough information to
safely connect to an anchored identity.

### Conflicting evidence

Face and body evidence may disagree.

The system should favor strong facial evidence while respecting the
configured similarity, quality, pose, and same-image constraints.

---

## 24. Calibration

Thresholds should not be treated as universal biometric values.

Calibration should use representative labeled event data.

A useful process is:

```text
Collect representative events
        ↓
Create ground truth
        ↓
Run baseline configuration
        ↓
Measure purity / false merges / fragmentation
        ↓
Tune thresholds
        ↓
Re-run
        ↓
Compare results
```

Threshold changes should be evaluated quantitatively and visually.

---

## 25. Current Responsibility Boundaries

The current implementation keeps identity discovery responsibilities
inside the clustering stage and final image organization inside
`EventOutputManager`.

```text
PersonPipeline
    → creates observations

EventStore
    → persists observations

ConstrainedIdentityClustering
    → discovers anonymous identity clusters

EventOutputManager
    → selects best images
    → creates representative face
    → writes final output
```

The following former components are **not separate stages in the current
architecture**:

```text
Same-Image Duplicate Suppression
IdentityProfile
BestImageSelector
```

Their former responsibilities have either been removed because they are
not part of the active execution path or incorporated into the current
clustering/output implementation.

---

## 26. Core Clustering Principles

1. Face is the primary identity signal.
2. Body is supporting evidence.
3. Quality determines evidence reliability.
4. Same-image identity merges are forbidden.
5. Cross-pose matching has dedicated requirements.
6. Trusted observations can establish or strengthen identity clusters.
7. Strong observations should represent clusters.
8. Conservative decisions are preferred when evidence is ambiguous.
9. Clustering operates on the complete event observation set.
10. Clustering produces anonymous identities, not names.
11. Best-image selection is performed after clustering.
12. `EventOutputManager` owns final image selection and representative
    generation.
13. Output generation does not modify identity assignments.
