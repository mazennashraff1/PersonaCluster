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
QUALITY_MIN = 0.0
QUALITY_MAX = 1.0


# ============================================================
# PERSON DETECTOR
# ============================================================

PERSON_MODEL_NAME = "models/yolo11n.pt"

PERSON_DETECTION_THRESHOLD = 0.40

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

FACE_MODEL_NAME = "buffalo_l"

FACE_DETECTION_SIZE = (640, 640)

FACE_DETECTION_THRESHOLD = 0.40

# InsightFace execution context.
#
# 0  = GPU
# -1  = CPU
FACE_CTX_ID = 0


# ============================================================
# FACE / BODY ASSOCIATION
# ============================================================

ASSOCIATION_MIN_SCORE = 0.30


# ============================================================
# BODY ENCODER
# ============================================================

BODY_MODEL_NAME = "osnet_x1_0"

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
BODY_UPPER_BODY_RATIO = 0.60


# ============================================================
# FACE QUALITY
# ============================================================

FACE_QUALITY_DETECTION_WEIGHT = 0.40

FACE_QUALITY_SIZE_WEIGHT = 0.30

FACE_QUALITY_SHARPNESS_WEIGHT = 0.30

# Minimum useful face dimension in pixels.
FACE_MIN_SIZE = 40

# Face dimension at which size quality reaches 1.0.
FACE_REFERENCE_SIZE = 120

# Laplacian variance calibration.
FACE_SHARPNESS_MIN = 20.0

FACE_SHARPNESS_MAX = 150.0


# ============================================================
# BODY QUALITY
# ============================================================

BODY_QUALITY_DETECTION_WEIGHT = 0.40

BODY_QUALITY_SIZE_WEIGHT = 0.30

BODY_QUALITY_SHARPNESS_WEIGHT = 0.30

# Minimum useful body dimension in pixels.
BODY_MIN_SIZE = 100

# Body dimension at which size quality reaches 1.0.
BODY_REFERENCE_SIZE = 400

# Laplacian variance calibration.
BODY_SHARPNESS_MIN = 20.0

BODY_SHARPNESS_MAX = 150.0


# ============================================================
# IDENTITY CLUSTERING
# ============================================================

# Combined identity score:
#
# Face      -> primary signal
# Body      -> supporting signal
# Quality   -> confidence/reliability signal
#
# These should normally add up to 1.0.
CLUSTER_FACE_WEIGHT = 0.65

CLUSTER_BODY_WEIGHT = 0.20

CLUSTER_QUALITY_WEIGHT = 0.15


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
VISUALIZATION_SCALE = 0.50

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
REPRESENTATIVE_FACE_QUALITY_WEIGHT = 0.70

REPRESENTATIVE_FACE_DETECTION_WEIGHT = 0.30

# Output directory name.
CLUSTER_VISUALIZATION_DIR_NAME = "cluster_visualization"


# ============================================================
# FILE PATHS
# ============================================================

# Root directory containing event folders/images.
EVENTS_PATH = "data/events"

# Root directory containing known/reference people.
REFERENCES_PATH = "data/people"

# SQLite database filename.
EVENT_DATABASE_FILENAME = "event.db"

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
