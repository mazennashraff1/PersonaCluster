# PersonaCluster Evaluation

PersonaCluster should be evaluated as a complete event-processing system, not by cluster count alone.

---

# 1. Evaluation Pipeline

```text
Event discovery
    ↓
Gallery discovery
    ↓
Detection
    ↓
Observation generation
    ↓
Observation quality
    ↓
Identity clustering
    ↓
Reference matching
    ↓
Best-image selection
    ↓
Representative-image generation
    ↓
Final output
```

Reliability, incremental reuse, and runtime should also be evaluated.

---

# 2. Event-Level Evaluation

Use multiple representative events.

Each test event should have:

```text
data/events/<event_name>/Gallery/
data/events/<event_name>/reference/
```

The evaluation must verify that observations from one event never appear in another event's output.

---

# 3. Ground Truth

A useful evaluation event should contain:

- Multiple people
- Multiple images per person
- Different poses
- Different image qualities
- Occlusions
- Multiple people in the same image
- Different camera distances
- Realistic event conditions

Use only data you are permitted to process and evaluate.

---

# 4. Detection Evaluation

Measure:

- Person detection precision
- Person detection recall
- False positives
- False negatives
- Small detections
- Occlusion cases

Poor detection propagates errors into later stages.

---

# 5. Face Evaluation

Measure:

- Face detection success
- Missed faces
- False face detections
- Face-size distribution
- Detection confidence
- Pose distribution
- Occluded/non-frontal cases

---

# 6. Observation Evaluation

Measure:

```text
Face available
Body available
Face + body
Face-only
Body-only
Invalid embeddings
Association failures
Poor crops
```

Useful rates include valid embedding rate and face/body availability rate.

---

# 7. Clustering Metrics

Primary metrics include:

- Cluster purity
- False merge rate
- Fragmentation
- Unknown/unassigned observation rate
- Same-image constraint violations

A false merge occurs when observations from different real people share one identity cluster.

Fragmentation occurs when one real person is split across multiple predicted clusters.

---

# 8. Same-Image Correctness

For every predicted cluster:

```text
Group observations by source image
```

A valid cluster should contain at most one observation from each source image.

Any violation is a correctness failure.

---

# 9. Pose-Aware Evaluation

Compare identity behavior for:

```text
frontal → frontal
frontal → left
frontal → right
frontal → profile
left → right
```

This helps separate same-pose performance from cross-pose behavior.

---

# 10. Identity Anchor Evaluation

Measure:

```text
Anchors created
Anchors rejected
Identities without strong anchors
Weak observations establishing identities
```

Inspect whether discovered identities have sufficiently strong evidence under the configured anchor rules.

---

# 11. Reference Matching Evaluation

Reference matching should be evaluated separately from clustering.

Measure:

- Correct reference assignments
- Rejected low-similarity matches
- Ambiguous matches
- Wrong-person assignments
- Multiple clusters correctly aggregated to one person

References are event-local, so evaluate:

```text
EEA/reference/
```

against:

```text
output/EEA/
```

and separately evaluate another event.

---

# 12. Best-Image Evaluation

For each known person, inspect selected images for:

- Strong face quality
- Good face visibility
- Good person detection
- Strong association
- Useful pose diversity
- No unintended repeated source image

A best-image failure is an output-selection issue, not necessarily a clustering failure.

---

# 13. Representative Image Evaluation

Check that:

- The representative belongs to the correct person/cluster.
- The face crop is valid.
- The face is sufficiently visible.
- The representative does not come from the reference folder.
- The representative is generated from the event gallery.

---

# 14. Incremental Processing Evaluation

Run the same event twice.

First run:

```text
New images → processed
```

Second run:

```text
Unchanged images → reused/skipped
```

Then modify one gallery image and verify:

```text
Changed image → invalidated/reprocessed
Unchanged images → remain reusable
```

---

# 15. Post-Cluster Recheck Evaluation

Create a test event containing observations that initially remain unclustered.

Verify:

```text
Initial clustering
      ↓
Eligible image detected
      ↓
Bounded recheck
      ↓
Re-clustering
```

The event must not enter an endless retry loop.

---

# 16. Multi-Event Isolation Test

Create:

```text
data/events/Event_A/
data/events/Event_B/
```

Run the application and verify:

```text
output/Event_A/
output/Event_B/
```

No image from Event A should appear in Event B.

No reference from Event A should be used to match Event B.

Each event must have its own:

```text
event.db
```

---

# 17. Output Evaluation

For each event verify:

```text
output/<event_name>/
├── <person_name>/
│   ├── All Images/
│   ├── Best Images/
│   └── representative Image.jpg
└── final_results.xlsx
```

The Excel file is local only.

---

# 18. Performance Evaluation

Measure:

```text
Images processed / second
Total event runtime
Worker startup time
GPU memory
CPU memory
Database growth
Reference matching time
Output generation time
Drive upload time
```

Compare worker counts on the same event and configuration.

---

# 19. Failure Evaluation

Test:

- Corrupt image
- Missing image
- Unsupported image
- Model failure
- CUDA out-of-memory
- Worker termination
- Database lock
- Missing reference directory
- Empty Gallery
- Duplicate filenames in different folders

A single bad image should not silently invalidate successfully persisted work.

---

# 20. Evaluation Principle

The goal is not maximum cluster count.

The important properties are:

```text
Correct identity separation
+
Useful identity coverage
+
Stable incremental processing
+
Correct reference matching
+
Correct final output
+
Reliable event isolation
```
