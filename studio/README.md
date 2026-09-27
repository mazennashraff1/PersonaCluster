# PersonaCluster Studio

A local React + FastAPI control panel for the existing PersonaCluster engine.

## Important architecture

Studio is **not** a replacement for PersonaCluster. It controls the existing project and calls the production `main.process_event()` path. Detection, embeddings, observation persistence, clustering, reference matching, output generation, Google Drive upload and Excel generation remain owned by the existing project.

The expected placement is:

```text
PersonaCluster/
├── main.py
├── app/
├── data/
├── output/
├── models/
├── credentials/
├── .env
└── studio/
    ├── run_studio.py
    ├── backend/
    │   ├── __init__.py
    │   ├── server.py
    │   └── requirements.txt
    ├── frontend/
    │   ├── package.json
    │   ├── vite.config.js
    │   ├── index.html
    │   ├── public/
    │   └── src/
    │       ├── main.jsx
    │       └── styles.css
    └── README.md
```

Keep `.env`, `main.py`, `app/`, `data/`, `output/`, `models/` and `credentials/` at the project root. Do not move them into `studio/`.

## Install

From the PersonaCluster root:

```powershell
.\.venv\Scripts\activate
pip install -r studio\backend\requirements.txt
cd studio\frontend
npm install
cd ..
python run_studio.py
```

Studio opens at `http://127.0.0.1:5173` and the API runs at `http://127.0.0.1:8765`.

## What the pages do

- **Dashboard** — events, image/reference counts, hardware profile and event selection.
- **New Event** — select a Gallery folder and optional reference folder. Nested Gallery paths are preserved.
- **Run Monitor** — starts/stops the selected event and shows the real `main.py` process log and SQLite job counts.
- **Results** — reads the project-level `output/<event>/` result created by `EventOutputManager` and provides the local Excel report.
- **Configuration** — edits top-level values in `app/configuration.py` and Google Drive values in the root `.env`. A `.bak` is created before the first edit.

## Output location

The existing project writes final output to:

```text
output/<event_name>/
```

Studio intentionally reads this location. It does not create an `output` folder inside `data/events/<event>`.

## Hardware

The configuration page can apply a conservative hardware profile. This is important because each worker owns its own ML model instances. Start with one worker on a 6 GB GPU.

## Optional project root override

Normally Studio automatically uses its parent directory as the PersonaCluster root. If Studio is launched from somewhere else, set:

```powershell
$env:PERSONACLUSTER_ROOT="D:\path\to\PersonaCluster"
python studio\run_studio.py
```
