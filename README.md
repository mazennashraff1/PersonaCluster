# PersonaCluster

### Event-Based Person Detection, Recognition, Clustering, Reference Matching, and Best-Image Selection

PersonaCluster is an event-based computer-vision system for processing collections of photographs from closed events.

It discovers people in event photographs, creates person observations using face and body appearance information, groups observations into anonymous identity clusters, matches those clusters against provided reference images, and produces organized results for each matched person.

Typical use cases include:

* Conferences
* Weddings
* Parties
* Graduations
* Sports events
* Corporate events
* Other closed event-based photo collections

PersonaCluster is designed for **event-level identity discovery**. It does not require a permanent named-person gallery or cross-event identity database.

---

## Project Flow

```text
Event Photos
     ↓
Recursive Image Discovery
     ↓
Image Processing
     ├── Person Detection
     ├── Face Detection + Embedding
     ├── Face/Person Association
     ├── Face Pose Estimation
     ├── Body/Re-ID Embedding
     └── Quality Assessment
     ↓
Person Observations
     ↓
SQLite Event Database
     ↓
Constrained Identity Clustering
     ↓
Anonymous Identity Clusters
     ↓
Reference Matching
     ↓
Event Output
     ├── All Images
     ├── Best Images
     └── Representative Face
```

The core design separates **image processing** from **identity discovery**:

> Images are processed independently, then the complete event observation set is used for identity clustering.

For the detailed architecture, see [ `architecture.md` ](docs/architecture.md).

---

# Key Features
* Recursive event image discovery
* Nested gallery directories
* YOLO-based person detection
* InsightFace face detection and embeddings
* Face/person association
* Face pose estimation
* TorchReID/OSNet body appearance embeddings
* Face and body quality assessment
* Embedding validation
* Face, body, and face+body observations
* SQLite-backed processing state
* Parallel image workers
* Crash and stale-job recovery
* Persistent event-level processing
* Face-primary identity clustering
* Quality-aware identity matching
* Cross-pose matching
* Identity anchors
* Same-image identity constraints
* Best-image selection
* Representative-face generation
* Reference-based final output
* Excel result generation
* Optional Google Drive upload
* PersonaCluster Studio web interface

---

# Requirements

Recommended environment:

| Component | Recommendation         |
| --------- | ---------------------- |
| OS        | Windows 10/11 or Linux |
| Python    | 3.10 or 3.11           |
| RAM       | 16 GB or more          |
| GPU       | NVIDIA GPU recommended |
| Storage   | SSD recommended        |

A GPU is strongly recommended for larger event collections.

For GPUs with limited VRAM, start with a single worker.

---

# Environment Setup

Create a virtual environment from the project root.

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

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

If TorchReID is not installed by the requirements:

```bash
python -m pip install --no-build-isolation git+https://github.com/KaiyangZhou/deep-person-reid.git
```

---

# Verify the Environment

Check Python:

```bash
python --version
```

Check PyTorch and CUDA:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Check the main dependencies:

```bash
python -c "import numpy, cv2, PIL, scipy, torch, torchvision, ultralytics, insightface, onnxruntime, sklearn, skimage, matplotlib, yaml; print('All core dependencies imported successfully')"
```

Check TorchReID:

```bash
python -c "import torchreid; print('TorchReID imported successfully')"
```

---

# Project Structure

The main project structure is:

```text
PersonaCluster/
│
├── app/
│   ├── configuration.py
│   ├── pipeline.py
│   ├── models/
│   ├── detection/
│   ├── identity/
│   ├── embeddings/
│   ├── quality/
│   ├── validation/
│   ├── storage/
│   ├── workers/
│   ├── clustering/
│   └── output/
│
├── data/
│   └── events/
│
├── models/
│   └── yolo11n.pt
│
├── credentials/
│
├── docs/
│
├── studio/
│
├── main.py
├── requirements.txt
├── .env
└── README.md
```

The detailed responsibilities of these components are documented in [ `architecture.md` ](docs/architecture.md).

---

# Models and Technologies

| Component           | Technology                    |
| ------------------- | ----------------------------- |
| Person detection    | Ultralytics YOLO11n           |
| Face detection      | InsightFace                   |
| Face embedding      | InsightFace `buffalo_l` |
| Body / Re-ID        | TorchReID `osnet_x1_0` |
| Face pose           | OpenCV `solvePnP` |
| Image processing    | OpenCV / NumPy                |
| Persistence         | SQLite                        |
| Identity clustering | Custom constrained clustering |
| Backend Studio      | FastAPI                       |
| Frontend Studio     | React / Vite                  |

---

# Prepare an Event

Events are stored under:

```text
data/events/
```

Each direct child represents an event.

Recommended structure:

```text
data/
└── events/
    └── EEA/
        ├── Gallery/
        │   ├── Camera 1/
        │   ├── Camera 2/
        │   └── ...
        │
        └── reference/
            ├── Person 1 - 201xxxxxxxxx.jpg
            └── Person 2 - 201xxxxxxxxx.jpg
```

### Gallery

Event photographs can be placed anywhere inside:

```text
data/events/<event_name>/Gallery/
```

Nested directories are supported.

Supported image formats include:

```text
.jpg
.jpeg
.png
.bmp
.webp
.tif
.tiff
```

### Reference Images

Known-person reference images are placed in:

```text
data/events/<event_name>/reference/
```

Use:

```text
Person Name - Phone Number.ext
```

as the filename format.

Reference images should **not** be placed inside `Gallery` .

---

# Run PersonaCluster

From the project root:

```bash
python main.py
```

The application discovers the available events and processes them.

Processing state is maintained per event, allowing the system to work with persistent event databases and continue processing without unnecessarily repeating completed image work.

For development, incremental processing, worker behavior, and recovery, see [ `development.md` ](docs/development.md).

---

# Results

Results are created under:

```text
output/<event_name>/
```

The active final-output workflow is reference matched, meaning only discovered clusters that can be matched to the supplied reference people are exported.

Example:

```text
output/
└── EEA/
    ├── Person 1/
    │   ├── All Images/
    │   ├── Best Images/
    │   └── representative Image.jpg
    │
    ├── Person 2/
    │   ├── All Images/
    │   ├── Best Images/
    │   └── representative Image.jpg
    │
    └── final_results.xlsx
```

The Excel report contains:

```text
Name of Person
Phone Number
Folder Shared Link
```

The local output is the primary result of the application.

Google Drive upload is optional.

---

# Google Drive

When enabled, PersonaCluster can upload the generated person folders and Excel report to Google Drive.

The Drive output follows the event/person structure:

```text
Event Results/
└── Event Name/
    ├── Person Name/
    │   ├── All Images/
    │   ├── Best Images/
    │   └── representative Image.jpg
    │
    ├── Another Person/
    │   ├── All Images/
    │   ├── Best Images/
    │   └── representative Image.jpg
    │
    └── final_results.xlsx
```

When public sharing is enabled, the generated person-folder links are written to the Excel report.

Drive configuration is documented in [ `configuration.md` ](docs/configuration.md).

---

# PersonaCluster Studio

PersonaCluster includes **Studio**, a local web interface for controlling the project.

Studio provides:

* Dashboard
* Event creation
* Processing controls
* Run monitoring
* Results viewing
* Configuration editing

Studio is located in:

```text
studio/
```

Install the backend dependencies:

```powershell
.venv\Scripts\activate
pip install -r studio\backend\requirements.txt
```

Install the frontend dependencies:

```powershell
cd studio\frontend
npm install
cd ..
```

Start Studio:

```powershell
python run_studio.py
```

Then open:

```text
http://127.0.0.1:5173
```

Studio-specific instructions are available in [ `studio/README.md` ](studio/README.md).

---

# Configuration

Project configuration is centralized in:

```text
app/configuration.py
```

Environment-specific and secret values can be provided through:

```text
.env
```

Configuration includes areas such as:

* Detection
* Face processing
* Body/Re-ID
* Quality
* Pose
* Identity clustering
* Reference matching
* Output
* Workers
* Google Drive

Do not use the README as the source of truth for individual thresholds or tuning values.

See [ `configuration.md` ](docs/configuration.md) for the current configuration reference.

---

# Documentation

The root README intentionally provides only the information needed to understand, install, and run the project.

Use the dedicated documentation for deeper information:

| Document                                        | Purpose                                                     |
| ----------------------------------------------- | ----------------------------------------------------------- |
| [ `architecture.md` ](docs/architecture.md)       | System architecture and component responsibilities          |
| [ `clustering.md` ](docs/clustering.md)           | Identity clustering and matching logic                      |
| [ `configuration.md` ](docs/configuration.md)     | Configuration and tunable parameters                        |
| [ `development.md` ](docs/development.md)         | Development, persistence, workers, recovery, and validation |
| [ `evaluation.md` ](docs/evaluation.md)           | Evaluation methodology and metrics                          |
| [ `troubleshooting.md` ](docs/troubleshooting.md) | Common problems and diagnostic steps                        |
| [ `studio/README.md` ](studio/README.md)          | PersonaCluster Studio                                       |

---

# Limitations

PersonaCluster is intentionally an **event-level identity discovery system**.

Current limitations include:

* Identity clusters are anonymous until matched or labeled.
* Cross-event identity recognition is not part of the core architecture.
* Face visibility and image quality affect matching quality.
* Body appearance can change between photographs.
* Clustering thresholds require calibration for different datasets.
* Very large-scale approximate nearest-neighbor retrieval is not required by the current core implementation.
* The system should not be treated as a universal biometric identification system.

---

# Privacy and Security

PersonaCluster may process photographs containing identifiable people and generate face and body appearance embeddings.

Do not commit the following to a public repository:

```text
Real event photographs
Real event databases
Face embeddings
Body embeddings
Generated private output
API keys
Passwords
Access tokens
Private configuration
```

Use synthetic, consented, or appropriately licensed sample data when demonstrating the project.

Review the licenses and terms of the underlying models and dependencies before distributing the project.

---

# License

```text
MIT License
```

---

# Acknowledgements

PersonaCluster builds on open-source projects including:

* Ultralytics
* InsightFace
* TorchReID / Deep-Person-ReID
* PyTorch
* OpenCV
* NumPy
* scikit-learn

Please review the licenses and terms of the individual dependencies before distribution.

---

# Project Philosophy

> **Do not ask clustering to compensate for bad observations.**

The system follows this general principle:

```text
Reliable Detection
       ↓
Reliable Observations
       ↓
Reliable Persistence
       ↓
Constrained Identity Clustering
       ↓
Useful Image Selection
       ↓
Clear Final Output
```

The detailed implementation decisions behind this philosophy are documented in the supporting documentation.

---

# Quick Start

```powershell
# Create environment
python -m venv .venv

# Activate
.venv\Scripts\activate

# Install dependencies
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Prepare:
# data/events/<event>/Gallery/
# data/events/<event>/reference/

# Run
python main.py
```

Then check:

```text
output/<event_name>/
```

---

## Project Entry Point

If you are new to PersonaCluster:

1. Set up the Python environment.
2. Prepare an event under `data/events/`.
3. Add the gallery images.
4. Add reference images if reference matching is required.
5. Run `python main.py`.
6. Check `output/<event_name>/`.
7. Use the supporting documentation when you need to understand or modify a specific part of the system.
