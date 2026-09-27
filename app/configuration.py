"""
app/configuration.py

Central configuration for the Event Person Identity System.

IMPORTANT:
This is the SINGLE SOURCE OF TRUTH for all tunable
configuration parameters in the project.

Other modules should NOT define their own configurable
thresholds, weights, model names, paths, or visualization
settings.

If a parameter needs tuning, change it HERE.
"""

# ============================================================
# GENERAL
# ============================================================

# Enable live OpenCV visualization while processing images.
ENABLE_VISUALIZATION = False

# General quality range.
QUALITY_MIN = 0
QUALITY_MAX = 1


# ============================================================
# PERSON DETECTOR
# ============================================================

PERSON_MODEL_NAME = 'models/yolo11n.pt'

PERSON_DETECTION_THRESHOLD = 0.4

# None = automatically use the default Ultralytics device.
# Examples:
#   None
#   "cpu"
#   "cuda"
#   "cuda:0"
PERSON_DETECTION_DEVICE = None

# COCO class ID for person.
PERSON_CLASS_ID = 0


# ============================================================
# FACE DETECTOR
# ============================================================

FACE_MODEL_NAME = 'buffalo_l'

FACE_DETECTION_SIZE = [640, 640]

FACE_DETECTION_THRESHOLD = 0.4

# InsightFace execution context.
#
# 0  = GPU
# -1  = CPU
FACE_CTX_ID = 0


# ============================================================
# FACE / BODY ASSOCIATION
# ============================================================

ASSOCIATION_MIN_SCORE = 0.3


# ============================================================
# BODY ENCODER
# ============================================================

BODY_MODEL_NAME = 'osnet_x1_0'

# None = use Torchreid's normal pretrained model.
BODY_MODEL_PATH = None

# None = automatically use CUDA when available.
# Examples:
#   None
#   "cpu"
#   "cuda"
#   "cuda:0"
BODY_DEVICE = None

# Portion of the person bounding box used for
# appearance/body embedding.
BODY_UPPER_BODY_RATIO = 0.6


# ============================================================
# FACE QUALITY
# ============================================================

FACE_QUALITY_DETECTION_WEIGHT = 0.4

FACE_QUALITY_SIZE_WEIGHT = 0.3

FACE_QUALITY_SHARPNESS_WEIGHT = 0.3

# Minimum useful face dimension in pixels.
FACE_MIN_SIZE = 40

# Face dimension at which size quality reaches 1.0.
FACE_REFERENCE_SIZE = 120

# Laplacian variance calibration.
FACE_SHARPNESS_MIN = 20

FACE_SHARPNESS_MAX = 150


# ============================================================
# FACE POSE
# ============================================================

# Coarse yaw boundaries used for identity-profile grouping.
# These are intentionally configurable because the final values
# should be calibrated against the event's own images.
FACE_POSE_FRONTAL_YAW_DEGREES = 20
FACE_POSE_PROFILE_YAW_DEGREES = 55


# ============================================================
# BODY QUALITY
# ============================================================

BODY_QUALITY_DETECTION_WEIGHT = 0.4

BODY_QUALITY_SIZE_WEIGHT = 0.3

BODY_QUALITY_SHARPNESS_WEIGHT = 0.3

# Minimum useful body dimension in pixels.
BODY_MIN_SIZE = 100

# Body dimension at which size quality reaches 1.0.
BODY_REFERENCE_SIZE = 400

# Laplacian variance calibration.
BODY_SHARPNESS_MIN = 20

BODY_SHARPNESS_MAX = 150


# ============================================================
# IDENTITY CLUSTERING
# ============================================================

# ------------------------------------------------------------
# General clustering thresholds
# ------------------------------------------------------------

# Minimum face cosine similarity allowed for an identity merge.
# A face similarity below this value can NEVER create a merge.
MIN_FACE_SIMILARITY = 0.55

# Final combined similarity required for a normal identity merge.
MERGE_THRESHOLD = 0.78


# ------------------------------------------------------------
# Cross-pose clustering
# ------------------------------------------------------------

# Minimum face similarity required when comparing observations
# belonging to different face-pose groups.
CROSS_POSE_MIN_FACE_SIMILARITY = 0.6

# Minimum body similarity required when comparing observations
# belonging to different face-pose groups.
CROSS_POSE_MIN_BODY_SIMILARITY = 0.5

# Final combined similarity required for a cross-pose merge.
CROSS_POSE_MERGE_THRESHOLD = 0.75


# ------------------------------------------------------------
# Cluster representatives
# ------------------------------------------------------------

# Maximum number of representatives retained from each
# face-pose group.
MAX_REPRESENTATIVES_PER_POSE = 3

# Number of strongest observations used to represent a cluster.
REPRESENTATIVE_COUNT = 3


# ------------------------------------------------------------
# Anchor requirements
# ------------------------------------------------------------

# Minimum face quality required for an observation to be
# considered a reliable identity anchor.
ANCHOR_MIN_FACE_QUALITY = 0.5

# Minimum face detector confidence required for an observation
# to be considered a reliable identity anchor.
ANCHOR_MIN_FACE_DETECTION_CONFIDENCE = 0.5

# Minimum face dimension in pixels required for an observation
# to be used as an anchor.
ANCHOR_MIN_FACE_SIZE = 40

# Maximum absolute yaw allowed for a frontal anchor.
ANCHOR_MAX_YAW_DEGREES = 20


# ------------------------------------------------------------
# Cluster size
# ------------------------------------------------------------

# Minimum observations required for a discovered identity.
MIN_CLUSTER_SIZE = 2


# ------------------------------------------------------------
# Identity clustering weights
# ------------------------------------------------------------

# Combined identity score:
#
# Face      -> primary signal
# Body      -> supporting signal
# Quality   -> confidence/reliability signal
#
# These should normally add up to 1.0.
CLUSTER_FACE_WEIGHT = 0.65

CLUSTER_BODY_WEIGHT = 0.2

CLUSTER_QUALITY_WEIGHT = 0.15


# ============================================================
# FACE-ONLY IDENTITY CLUSTERING
# ============================================================

# Used when an observation has face information but
# does not have body information.

CLUSTER_FACE_ONLY_WEIGHT = 0.75

CLUSTER_FACE_ONLY_QUALITY_WEIGHT = 0.25


# ============================================================
# FACE-ONLY IDENTITY CLUSTERING
# ============================================================

# Used when an observation has face information but
# does not have body information.

CLUSTER_FACE_ONLY_WEIGHT = 0.75

CLUSTER_FACE_ONLY_QUALITY_WEIGHT = 0.25


# ============================================================
# IDENTITY MATCHING THRESHOLDS
# ============================================================

# A face similarity below this value can NEVER create
# an identity merge, regardless of body/quality score.
MIN_FACE_SIMILARITY = 0.55

# Final combined similarity required for merging.
MERGE_THRESHOLD = 0.78

# Minimum observations required for a cluster to be
# considered a discovered identity.
MIN_CLUSTER_SIZE = 2

# Number of strongest observations considered when
# representing a cluster.
REPRESENTATIVE_COUNT = 3


# ============================================================
# VISUALIZATION
# ============================================================

# Live processing window scale.
#
# 0.5 = 50% of original image size.
VISUALIZATION_SCALE = 0.5

# OpenCV waitKey delay in milliseconds.
#
# 1 = approximately real-time display.
VISUALIZATION_WAIT_KEY_MS = 1

# ESC key used to stop processing.
VISUALIZATION_STOP_KEY = 27


# ============================================================
# CLUSTER VISUALIZER
# ============================================================

# Padding around the face when creating the representative
# face image.
FACE_PADDING_RATIO = 0.25

# These are retained for visualizer compatibility.
# The current visualizer does NOT create contact sheets.
VISUALIZATION_COLUMNS = 5

VISUALIZATION_MAX_ITEMS_PER_CLUSTER = 100

# Weights used when selecting the best representative
# observation for a cluster.
REPRESENTATIVE_FACE_QUALITY_WEIGHT = 0.7

REPRESENTATIVE_FACE_DETECTION_WEIGHT = 0.3

# Output directory name.
CLUSTER_VISUALIZATION_DIR_NAME = 'cluster_visualization'


# ============================================================
# FILE PATHS
# ============================================================

# Root directory containing event folders/images.
EVENTS_PATH = 'data/events'

# SQLite database filename.
EVENT_DATABASE_FILENAME = 'event.db'

# ============================================================
# Worker configuration
# ============================================================

# Start with ONE worker on the RTX 3050 6GB.
#
# Each worker creates its own ML models, so increasing this
# number can multiply GPU VRAM usage.
WORKER_COUNT = 2

# If a worker has been PROCESSING an image longer than this,
# the job can be considered stale after the program restarts.
WORKER_STALE_TIMEOUT_SECONDS = 600

# Use multiprocessing.
#
# Recommended for the current ML pipeline.
WORKER_USE_PROCESSES = True


# ============================================================
# REFERENCE-BASED FINAL OUTPUT
# ============================================================

# Folder containing known/reference images. Each image must use:
#     Person Name - Phone Number.ext
# The phone number is metadata only; the person name becomes
# the final output folder name.
REFERENCES_PATH = 'reference'

# Minimum cosine similarity between a reference face embedding
# and an observation in a discovered cluster for that cluster
# to be assigned to a known person.
REFERENCE_MATCH_THRESHOLD = 0.55

# If two known people have nearly identical best scores for the
# same cluster, the cluster is considered ambiguous and is not
# exported. Set to 0.0 to disable the margin requirement.
REFERENCE_MATCH_MIN_MARGIN = 0.02

# Final output directory inside the event directory.
EVENT_OUTPUT_DIRECTORY_NAME = 'output'

# The reference-matched person output is the default final mode.
EVENT_OUTPUT_MODE = 'REFERENCE_MATCHED'

# Remove previous final output before rebuilding it.
EVENT_OUTPUT_CLEAN_BEFORE_RUN = True

# Number of best source images retained from EACH matched cluster
# when building the person's combined Best Images folder.
BEST_IMAGES_PER_CLUSTER = 5

# Keep pose diversity when selecting best images from a cluster.
BEST_IMAGES_REQUIRE_POSE_DIVERSITY = True


# ============================================================
# GOOGLE DRIVE SHARING
# ============================================================

# Permission applied ONLY to each generated person folder when
# GOOGLE_DRIVE_PUBLIC_LINK=True.
# "writer" means anyone with that person's folder link can edit
# that folder and its contents. The event folder is not public.
GOOGLE_DRIVE_PUBLIC_LINK_ROLE = 'writer'
