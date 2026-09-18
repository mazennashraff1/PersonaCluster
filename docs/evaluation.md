# PersonaCluster Evaluation

PersonaCluster should be evaluated as a complete system rather than by
cluster count alone.

Evaluation should cover the complete active pipeline:

```text
Detection
    ↓
Observation generation
    ↓
Observation quality
    ↓
Identity clustering
    ↓
Best-image selection
    ↓
Representative-face generation
    ↓
Final output
```

Computational performance and reliability should also be evaluated.

------------------------------------------------------------------------

## 1. Evaluation Goals

The primary identity-discovery goals are:

```text
High cluster purity
+
Low false merge rate
+
Low fragmentation
```

The objective is not simply to maximize the number of merged
observations.

A strong configuration should balance identity coverage with conservative
identity separation.

------------------------------------------------------------------------

## 2. Ground Truth

Meaningful identity-clustering evaluation requires representative event
data with ground-truth identity labels.

A suitable evaluation dataset should contain:

* Multiple people
* Multiple images per person
* Different poses
* Different image qualities
* Realistic occlusions
* Multiple people in the same image
* Different distances from the camera
* Realistic event conditions

Use only data that you are legally and ethically permitted to process and
evaluate.

Ground-truth labels should identify the real person associated with each
observation whenever possible.

------------------------------------------------------------------------

## 3. Detection Evaluation

Person detection should be evaluated independently from identity
clustering.

Recommended metrics include:

### Precision

```text
Correct person detections
-------------------------
All person detections
```

### Recall

```text
Correct person detections
-------------------------
All ground-truth people
```

Also record:

```text
False positives
False negatives
Difficult scenes
Occlusion cases
Small detections
```

Poor person detection can propagate errors into every later stage.

------------------------------------------------------------------------

## 4. Face Detection Evaluation

Evaluate the face-detection stage independently.

Measure:

```text
Face detection success
Missed faces
False face detections
Face-size distribution
Detection-confidence distribution
```

Where possible, separate:

```text
Frontal faces
Non-frontal faces
Profile faces
Small faces
Occluded faces
```

This helps determine whether later clustering problems originate from
insufficient face evidence.

------------------------------------------------------------------------

## 5. Observation Evaluation

An observation should be evaluated independently of its eventual identity
cluster.

Measure:

```text
Face available
Body available
Face + body available
Face-only
Body-only
Invalid embeddings
Association failures
Poor crops
```

Useful rates include:

### Valid Embedding Rate

```text
Valid embeddings
----------------
Expected embeddings
```

### Face Availability Rate

```text
Observations with valid faces
----------------------------
All person observations
```

### Body Availability Rate

```text
Observations with valid body embeddings
---------------------------------------
All person observations
```

The purpose is to determine whether the image-level pipeline is
producing sufficiently reliable evidence for identity clustering.

------------------------------------------------------------------------

## 6. Quality Evaluation

Analyze the relationship between quality scores and actual observation
reliability.

Inspect:

```text
Face quality
Body quality
Detection confidence
Bounding-box size
Sharpness
```

Check whether stronger observations generally produce more reliable
identity matches.

Quality scores should be treated as evidence-reliability signals rather
than identity labels.

------------------------------------------------------------------------

## 7. Cluster Purity

Cluster purity measures how homogeneous an identity cluster is.

For a cluster:

```text
Purity =
Number of observations belonging to the dominant real identity
---------------------------------------------------------------
Total observations in the cluster
```

Higher is better.

Example:

```text
Cluster A
10 observations
9 belong to Person X
1 belongs to Person Y

Purity = 0.90
```

Purity should be evaluated across the complete set of predicted
clusters, not only the largest clusters.

------------------------------------------------------------------------

## 8. False Merge Rate

A false merge occurs when observations belonging to different real
people are assigned to the same identity cluster.

This is a critical PersonaCluster metric.

Track:

```text
Number of false identity merges
```

and, where useful:

```text
False merge rate =
Incorrect cross-identity assignments
------------------------------------
Relevant clustering decisions
```

The exact denominator should be defined consistently before comparing
experiments.

Because PersonaCluster is designed to be conservative, false merges
should receive particular attention during threshold calibration.

------------------------------------------------------------------------

## 9. Fragmentation

Fragmentation measures how many predicted clusters are created for one
real person.

Example:

```text
Ground-truth Person X
    ├── Cluster 2
    ├── Cluster 7
    └── Cluster 11
```

This indicates fragmentation.

A useful measurement is:

```text
Average predicted clusters per ground-truth identity
```

Lower is generally better, provided that false merges remain low.

Fragmentation should be analyzed separately for:

```text
Frontal observations
Cross-pose observations
Low-quality observations
Sparse identities
```

when sufficient ground-truth data is available.

------------------------------------------------------------------------

## 10. Unknown Observations

Track observations that do not become part of a discovered identity
cluster.

Useful statistics include:

```text
Total observations
Valid observations
Clustered observations
Unknown observations
```

High unknown rates can indicate:

* Poor face quality
* Excessively strict thresholds
* Insufficient identity anchors
* Difficult viewpoints
* Weak body evidence
* Detection failures

Unknown observations should therefore be investigated at the
observation level rather than automatically treated as clustering
failures.

------------------------------------------------------------------------

## 11. Same-Image Constraint Evaluation

The current clustering implementation contains a hard same-image
constraint.

Two observations originating from the same source image must not be
assigned to the same identity cluster.

This should be explicitly tested.

For every predicted cluster:

```text
Group observations by source image
```

A valid cluster should contain:

```text
At most one observation
per source image
```

This is a correctness requirement, not merely a quality metric.

The evaluation should report any violation as a clustering correctness
failure.

------------------------------------------------------------------------

## 12. Pose-Aware Evaluation

Identity matching should be evaluated separately across different pose
combinations.

Useful categories include:

```text
frontal → frontal
frontal → left
frontal → right
frontal → profile
left → right
```

This helps determine whether cross-pose matching improves identity
coverage without producing excessive false merges.

Compare standard and cross-pose behavior separately where possible.

------------------------------------------------------------------------

## 13. Identity Anchor Evaluation

Evaluate the trusted-anchor mechanism independently.

Measure:

```text
Anchors created
Anchors rejected
Valid identities without anchors
Weak observations incorrectly establishing identities
```

A useful qualitative check is:

> Does each discovered identity have at least one sufficiently strong, 
> approximately frontal observation when the available data allows it?

Anchor evaluation is particularly important because anchors can influence
subsequent identity expansion.

------------------------------------------------------------------------

## 14. Best-Image Evaluation

Best-image selection is evaluated separately from identity clustering.

The selected images should be inspected for:

```text
Strong face quality
Good face visibility
Good person detection
Strong face/person association
Useful pose diversity
No repeated source image within the selected set
```

The selector operates on already-established clusters.

Therefore:

> A best-image selection failure should not automatically be interpreted
> as an identity-clustering failure.

Conversely, a correct cluster with poor selected images indicates an
output-selection problem rather than an identity-discovery problem.

------------------------------------------------------------------------

## 15. Representative-Face Evaluation

The representative face is generated by `EventOutputManager` after
clustering.

Evaluate whether the generated `face.jpg` provides a useful visual
representation of the cluster.

Inspect:

```text
Face visibility
Face size
Image sharpness
Detection confidence
Face quality
Crop quality
Padding
```

A representative-face failure should be evaluated separately from the
identity assignment.

The expected location is:

```text
output/
└── cluster01/
    ├── allImages/
    ├── bestImages/
    └── face.jpg
```

------------------------------------------------------------------------

## 16. Final Output Evaluation

Evaluate the generated reference-matched person directories and final Excel
report as separate output artifacts.

Verify:

```text
Expected person directories
Only reference-matched people are exported
All Images contents are correct
Best Images contents are correct
Representative image exists when a valid face is available
No unexpected manifest or metadata files
No accidental image overwrites
final_results.xlsx exists
Excel contains the expected three columns
Each exported person has the expected phone number
Drive folder links are populated when Drive upload is enabled
```

The expected local structure is:

```text
output/
├── Person Name/
│   ├── All Images/
│   ├── Best Images/
│   └── representative Image.jpg
└── final_results.xlsx
```

For Google Drive runs, verify that each person link opens the corresponding
person folder with the intended read access.

The output layer should not modify identity assignments.

## 17. Runtime Evaluation

Record:

```text
Total runtime
Image processing time
Clustering time
Output generation time
Images per second
```

Where possible, measure image processing, clustering, and output
generation separately.

This helps identify whether a performance bottleneck originates from:

```text
ML inference
Database operations
Identity clustering
File operations
```

Compare measurements across worker counts and datasets.

------------------------------------------------------------------------

## 18. Resource Evaluation

Record:

```text
CPU utilization
RAM usage
GPU utilization
VRAM usage
Peak memory
Worker count
```

This is particularly important because each worker can initialize its
own ML model instances.

A configuration should therefore be evaluated for both speed and
resource stability.

------------------------------------------------------------------------

## 19. Scaling Evaluation

Test progressively larger datasets:

```text
10–25 images
50–100 images
500 images
1,000 images
10,000+ images
```

Measure both:

```text
Image-processing throughput
```

and:

```text
Identity-clustering runtime
```

Do not assume that image-processing throughput and clustering runtime
scale in the same way.

The image-processing stage is parallelized, while event-level clustering
operates on the accumulated observation set.

------------------------------------------------------------------------

## 20. Threshold Calibration

For important thresholds, use controlled experiments:

```text
Choose baseline
      ↓
Run evaluation
      ↓
Change one parameter
      ↓
Run evaluation
      ↓
Compare metrics
      ↓
Inspect resulting clusters and output
```

Important identity parameters include:

```text
MIN_FACE_SIMILARITY
MERGE_THRESHOLD
CROSS_POSE_MIN_FACE_SIMILARITY
CROSS_POSE_MIN_BODY_SIMILARITY
CROSS_POSE_MERGE_THRESHOLD
ANCHOR_*
```

Quality and detection thresholds should also be evaluated when the
problem originates earlier in the pipeline.

Avoid optimizing only one metric.

For example:

```text
Lower threshold
      ↓
More merges
      ↓
Lower fragmentation
      +
Potentially more false merges
```

A configuration that reduces fragmentation while creating unacceptable
false merges is not an improvement.

------------------------------------------------------------------------

## 21. Visual Evaluation

Numerical metrics should be supplemented by visual inspection.

Inspect:

```text
Cluster allImages
Cluster bestImages
Representative face
Unknown observations
```

Look for:

```text
Same person grouped together
Different people separated
False merges
Fragmented identities
Poor best images
Poor representative faces
Unexpected pose behavior
Unexpected same-image assignments
```

Visual inspection is especially important because embedding similarity
and numerical metrics cannot explain every failure mode.

------------------------------------------------------------------------

## 22. Recommended Evaluation Report

For each experiment, record:

```text
Experiment ID
Dataset
Model versions
Configuration
Worker count
Hardware

Total images
Total observations

Valid face observations
Valid body observations
Face-only observations
Body-only observations

Cluster count
Cluster purity
False merge rate
Fragmentation
Unknown observations

Same-image constraint violations
Best-image quality
Representative-face quality

Total runtime
Image processing time
Clustering time
Output generation time

Peak RAM
Peak VRAM
```

The report should contain measured values rather than estimates.

------------------------------------------------------------------------

## 23. Comparing Experiments

A useful comparison table is:

| Experiment | Purity | False Merges | Fragmentation | Unknown | Runtime |
|---|---:|---:|---:|---:|---:|
| Baseline | --- | --- | --- | --- | --- |
| Configuration A | --- | --- | --- | --- | --- |
| Configuration B | --- | --- | --- | --- | --- |

For more detailed experiments, additional columns can be added for:

```text
Same-image violations
Valid face rate
Valid body rate
Best-image quality
Representative-face quality
Peak RAM
Peak VRAM
```

Populate the table with measured results.

------------------------------------------------------------------------

## 24. Success Criteria

There is no single universal success threshold for PersonaCluster.

Project-specific target values should be defined before systematic
calibration.

A strong configuration should demonstrate:

```text
High cluster purity
Low false merge rate
Acceptable fragmentation
Acceptable unknown rate
No same-image constraint violations
Useful best-image selection
Useful representative faces
Stable runtime
Acceptable resource usage
```

Identity quality should take priority over simply increasing the number
of merged observations.

------------------------------------------------------------------------

## 25. Failure Diagnosis

When an evaluation result is poor, trace the failure backward through
the active pipeline:

```text
Final output
     ↓
Best-image / representative-face selection
     ↓
Identity cluster
     ↓
Clustering score and constraints
     ↓
Persisted observation
     ↓
Quality / validation
     ↓
Embeddings
     ↓
Association
     ↓
Face/person detection
     ↓
Source image
```

This helps distinguish between:

```text
Observation failure
```

and:

```text
Identity-clustering failure
```

For example, if the face embedding is unreliable because the detected
face is extremely small, changing the clustering threshold may not solve
the underlying problem.

------------------------------------------------------------------------

## 26. Evaluation Principle

The most important evaluation principle is:

> **A clustering configuration is only an improvement if it improves the
> overall identity-discovery objective without introducing unacceptable
> false merges.**

PersonaCluster should therefore be evaluated as a complete pipeline:

```text
Reliable detection
        ↓
Reliable observations
        ↓
Reliable identity clustering
        ↓
Useful image selection
        ↓
Clear final output
```

Each stage should be evaluated according to its own responsibility while
also measuring its effect on the complete system.
