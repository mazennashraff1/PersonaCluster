from __future__ import annotations

import ast
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import psutil
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# studio/backend/server.py -> PersonaCluster/
STUDIO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROJECT_ROOT = STUDIO_ROOT.parent
PROJECT_ROOT = Path(os.getenv("PERSONACLUSTER_ROOT", DEFAULT_PROJECT_ROOT)).resolve()
FRONTEND_DIR = STUDIO_ROOT / "frontend"
EVENTS_ROOT = PROJECT_ROOT / "data" / "events"
CONFIG_PATH = PROJECT_ROOT / "app" / "configuration.py"
ENV_PATH = PROJECT_ROOT / ".env"
OUTPUT_ROOT = PROJECT_ROOT / "output"
RUNTIME = STUDIO_ROOT / "runtime"
RUNTIME.mkdir(parents=True, exist_ok=True)
EVENTS_ROOT.mkdir(parents=True, exist_ok=True)
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="PersonaCluster Studio API", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5173", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

state: dict[str, Any] = {
    "running": False,
    "event": None,
    "started_at": None,
    "process": None,
    "log_path": None,
    "returncode": None,
    "error": None,
}
lock = threading.Lock()
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".avif"}


def project_check() -> None:
    missing = []
    if not (PROJECT_ROOT / "main.py").exists():
        missing.append("main.py")
    if not CONFIG_PATH.exists():
        missing.append("app/configuration.py")
    if missing:
        raise RuntimeError(
            f"PersonaCluster project root is invalid: {PROJECT_ROOT}. Missing: {', '.join(missing)}"
        )


project_check()


def safe_relative(name: str) -> Path:
    p = Path(name.replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts:
        raise HTTPException(400, f"Unsafe upload path: {name}")
    return p


def read_constants(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: dict[str, Any] = {}
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                pass
    return values


def update_constants(path: Path, updates: dict[str, Any]) -> list[str]:
    text = path.read_text(encoding="utf-8")
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    replacements = []
    found = set()
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
            if name not in updates:
                continue
            value = updates[name]
            if isinstance(value, str):
                literal = repr(value)
            elif isinstance(value, bool):
                literal = "True" if value else "False"
            elif value is None:
                literal = "None"
            elif isinstance(value, (int, float)):
                literal = repr(value)
            elif isinstance(value, (list, tuple, dict)):
                literal = repr(value)
            else:
                continue
            replacements.append(
                (
                    node.value.lineno,
                    node.value.col_offset,
                    node.value.end_lineno,
                    node.value.end_col_offset,
                    literal,
                )
            )
            found.add(name)

    new_text = text
    for sl, sc, el, ec, literal in sorted(replacements, reverse=True):
        a = offsets[sl - 1] + sc
        b = offsets[el - 1] + ec
        new_text = new_text[:a] + literal + new_text[b:]

    if not path.with_suffix(path.suffix + ".bak").exists():
        shutil.copy2(path, path.with_suffix(path.suffix + ".bak"))
    path.write_text(new_text, encoding="utf-8")
    return sorted(found)


def parse_env() -> dict[str, str]:
    if not ENV_PATH.exists():
        return {}
    out: dict[str, str] = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def update_env(updates: dict[str, str]) -> None:
    original = ENV_PATH.read_text(encoding="utf-8") if ENV_PATH.exists() else ""
    lines = original.splitlines()
    updated = set()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            lines[i] = f"{key}={updates[key]}"
            updated.add(key)
    for key, value in updates.items():
        if key not in updated:
            lines.append(f"{key}={value}")
    if ENV_PATH.exists() and not ENV_PATH.with_suffix(".env.bak").exists():
        shutil.copy2(ENV_PATH, ENV_PATH.with_suffix(".env.bak"))
    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def hardware() -> dict[str, Any]:
    ram = psutil.virtual_memory()
    gpus = []
    try:
        import torch

        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                gpus.append(
                    {
                        "index": i,
                        "name": p.name,
                        "vram_gb": round(p.total_memory / 1024**3, 1),
                    }
                )
    except Exception:
        pass

    gpu = gpus[0] if gpus else None
    if gpu and gpu["vram_gb"] <= 6.5:
        profile = {
            "name": "6 GB GPU / safe",
            "workers": 1,
            "person_device": "cuda:0",
            "face_ctx": 0,
            "body_device": "cuda:0",
            "note": "One worker is recommended because every worker creates its own YOLO, InsightFace and OSNet model instances.",
        }
    elif gpu:
        profile = {
            "name": "GPU / balanced",
            "workers": 1,
            "person_device": "cuda:0",
            "face_ctx": 0,
            "body_device": "cuda:0",
            "note": "Start with one worker and increase only after benchmarking VRAM usage.",
        }
    else:
        profile = {
            "name": "CPU / conservative",
            "workers": 1,
            "person_device": "cpu",
            "face_ctx": -1,
            "body_device": "cpu",
            "note": "CPU inference is supported but will be substantially slower.",
        }
    return {
        "os": platform.platform(),
        "cpu_cores": psutil.cpu_count(logical=False) or 1,
        "logical_cpus": psutil.cpu_count() or 1,
        "ram_gb": round(ram.total / 1024**3, 1),
        "ram_available_gb": round(ram.available / 1024**3, 1),
        "gpus": gpus,
        "profile": profile,
    }


def event_output(event: Path) -> Path:
    config = read_constants(CONFIG_PATH)
    output_name = str(config.get("EVENT_OUTPUT_DIRECTORY_NAME", "output"))
    # The production EventOutputManager writes PROJECT_ROOT/output/<event>.
    return PROJECT_ROOT / output_name / event.name


def event_info(event: Path) -> dict[str, Any]:
    gallery = event / "Gallery"
    refs = event / "reference"
    out = event_output(event)
    db = event / "event.db"
    images = (
        sum(
            1
            for p in gallery.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if gallery.exists()
        else 0
    )
    refs_n = (
        sum(
            1 for p in refs.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if refs.exists()
        else 0
    )
    people = sum(1 for p in out.iterdir() if p.is_dir()) if out.exists() else 0
    return {
        "name": event.name,
        "path": str(event),
        "gallery_images": images,
        "references": refs_n,
        "people": people,
        "has_db": db.exists(),
        "output": str(out),
    }


def db_status(event: Path | None) -> dict[str, Any]:
    if not event:
        return {
            "pending": 0,
            "processing": 0,
            "completed": 0,
            "failed": 0,
            "observations": 0,
        }
    try:
        import sqlite3

        db = event / "event.db"
        if not db.exists():
            return {
                "pending": 0,
                "processing": 0,
                "completed": 0,
                "failed": 0,
                "observations": 0,
            }
        con = sqlite3.connect(db)
        cur = con.cursor()
        counts = {
            r[0]: r[1]
            for r in cur.execute(
                "SELECT status, COUNT(*) FROM image_jobs GROUP BY status"
            ).fetchall()
        }
        obs = cur.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
        con.close()
        return {
            "pending": counts.get("PENDING", 0),
            "processing": counts.get("PROCESSING", 0),
            "completed": counts.get("COMPLETED", 0),
            "failed": counts.get("FAILED", 0),
            "observations": obs,
        }
    except Exception as exc:
        return {"error": str(exc)}


def runner(event: Path, log_path: Path) -> None:
    # This calls the same process_event() used by main.py's normal event loop.
    code = (
        "import multiprocessing; multiprocessing.freeze_support(); "
        "from main import process_event; process_event(r'''%s''')" % str(event)
    )
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"[PersonaCluster Studio] Project: {PROJECT_ROOT}\n")
        log.write(f"[PersonaCluster Studio] Event: {event}\n\n")
        log.flush()
        try:
            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(PROJECT_ROOT) + (
                os.pathsep + existing_pythonpath if existing_pythonpath else ""
            )
            proc = subprocess.Popen(
                [sys.executable, "-u", "-c", code],
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
            with lock:
                state["process"] = proc
            rc = proc.wait()
            with lock:
                state["returncode"] = rc
                state["running"] = False
                state["process"] = None
            log.write(
                f"\n[PersonaCluster Studio] Process finished with exit code {rc}\n"
            )
        except Exception as exc:
            with lock:
                state["error"] = str(exc)
                state["returncode"] = -1
                state["running"] = False
                state["process"] = None
            log.write(f"\n[PersonaCluster Studio] Runner error: {exc}\n")


def terminate_process_tree(proc: subprocess.Popen) -> None:
    try:
        parent = psutil.Process(proc.pid)
        children = parent.children(recursive=True)
        for child in children:
            try:
                child.terminate()
            except psutil.Error:
                pass
        try:
            parent.terminate()
        except psutil.Error:
            pass
        _, alive = psutil.wait_procs(children + [parent], timeout=3)
        for p in alive:
            try:
                p.kill()
            except psutil.Error:
                pass
    except psutil.Error:
        try:
            proc.kill()
        except Exception:
            pass


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "project_root": str(PROJECT_ROOT),
        "main_exists": (PROJECT_ROOT / "main.py").exists(),
        "configuration_exists": CONFIG_PATH.exists(),
        "events_root": str(EVENTS_ROOT),
        "output_root": str(OUTPUT_ROOT),
    }


@app.get("/api/hardware")
def get_hardware():
    return hardware()


@app.get("/api/config")
def get_config():
    return {
        "path": str(CONFIG_PATH),
        "values": read_constants(CONFIG_PATH),
        "env": parse_env(),
    }


@app.put("/api/config")
def put_config(payload: dict):
    updated = (
        update_constants(CONFIG_PATH, payload.get("configuration", {}))
        if payload.get("configuration")
        else []
    )
    if payload.get("env"):
        update_env({str(k): str(v) for k, v in payload["env"].items()})
    # Return the same shape as GET /api/config.
    # The frontend stores this response as its configuration state and
    # expects configuration values under `values`.
    return {
        "updated": updated,
        "values": read_constants(CONFIG_PATH),
        "env": parse_env(),
    }


@app.get("/api/events")
def events():
    return [
        event_info(p)
        for p in sorted(EVENTS_ROOT.iterdir(), key=lambda p: p.name.casefold())
        if p.is_dir()
    ]


@app.post("/api/events")
async def create_event(
    event_name: str = Form(...),
    gallery: list[UploadFile] = File(...),
    references: list[UploadFile] = File(default=[]),
):
    name = re.sub(r"[^A-Za-z0-9 _.-]+", "_", event_name).strip() or "Event"
    event = EVENTS_ROOT / name
    if event.exists():
        raise HTTPException(409, "An event with this name already exists.")
    (event / "Gallery").mkdir(parents=True)
    (event / "reference").mkdir(parents=True)

    for upload in gallery:
        rel = safe_relative(upload.filename or "")
        parts = list(rel.parts)
        if parts and parts[0].lower() in {"gallery", "images", "photos"}:
            parts = parts[1:]
        if not parts:
            continue
        dest = event / "Gallery" / Path(*parts)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as out:
            while chunk := await upload.read(1024 * 1024):
                out.write(chunk)

    for upload in references:
        rel = safe_relative(upload.filename or "")
        dest = event / "reference" / rel.name
        with dest.open("wb") as out:
            while chunk := await upload.read(1024 * 1024):
                out.write(chunk)

    return event_info(event)


@app.post("/api/events/{name}/run")
def run_event(name: str):
    if Path(name).name != name or name in {"", ".", ".."}:
        raise HTTPException(400, "Invalid event name")
    event = EVENTS_ROOT / name
    if not event.is_dir():
        raise HTTPException(404, "Event not found")
    if not (event / "Gallery").is_dir():
        raise HTTPException(400, "Event is missing its Gallery folder")

    with lock:
        if state["running"]:
            raise HTTPException(409, "Another event is already running.")
        state.update(
            {
                "running": True,
                "event": name,
                "started_at": time.time(),
                "returncode": None,
                "error": None,
            }
        )
        log_path = (
            RUNTIME / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', name)}_{int(time.time())}.log"
        )
        state["log_path"] = str(log_path)
        threading.Thread(target=runner, args=(event, log_path), daemon=True).start()
    return {"started": True, "event": name, "log_path": state["log_path"]}


@app.post("/api/events/{name}/stop")
def stop_event(name: str):
    with lock:
        if state.get("event") != name or not state.get("running"):
            return {"stopped": False}
        proc = state.get("process")
    if proc and proc.poll() is None:
        terminate_process_tree(proc)
        with lock:
            state["running"] = False
            state["process"] = None
        return {"stopped": True}
    return {"stopped": False}


@app.get("/api/diagnostics")
def diagnostics():
    checks = {
        "project_root": str(PROJECT_ROOT),
        "project_root_exists": PROJECT_ROOT.is_dir(),
        "main_py": (PROJECT_ROOT / "main.py").is_file(),
        "configuration_py": CONFIG_PATH.is_file(),
        "events_root": str(EVENTS_ROOT),
        "events_root_exists": EVENTS_ROOT.is_dir(),
        "output_root": str(OUTPUT_ROOT),
        "python": sys.executable,
    }
    try:
        import main as project_main

        checks["main_import"] = "ok"
        checks["process_event"] = callable(getattr(project_main, "process_event", None))
    except Exception as exc:
        checks["main_import"] = f"error: {type(exc).__name__}: {exc}"
        checks["process_event"] = False
    return checks


@app.get("/api/status")
def status():
    with lock:
        snapshot = {k: v for k, v in state.items() if k != "process"}

    event = EVENTS_ROOT / snapshot["event"] if snapshot.get("event") else None
    snapshot["jobs"] = db_status(event)
    jobs = snapshot["jobs"] or {}
    total = sum(
        jobs.get(k, 0) for k in ("pending", "processing", "completed", "failed")
    )
    terminal = jobs.get("completed", 0) + jobs.get("failed", 0)
    image_progress = round(terminal / total * 100, 1) if total else 0

    # The image queue has an exact percentage because it is persisted in SQLite.
    # Clustering/reference matching/output are later phases and the clustering
    # implementation does not expose an iteration count, so those phases are
    # represented explicitly instead of incorrectly reporting 100% when the
    # image queue is finished.
    phase = "idle"
    phase_progress = image_progress
    if snapshot.get("running") and event:
        log_path = snapshot.get("log_path")
        log_text = ""
        if log_path and Path(log_path).is_file():
            try:
                log_text = Path(log_path).read_text(encoding="utf-8", errors="replace")[
                    -12000:
                ]
            except OSError:
                pass
        upper = log_text.upper()
        if "PROCESSING COMPLETE" in upper:
            phase, phase_progress = "complete", 100
        elif "GENERATING FINAL OUTPUT" in upper or "[OUTPUT]" in upper:
            phase, phase_progress = "output", 95
        elif (
            "EVENT REFERENCE MATCHING" in upper
            or "REFERENCE MATCH" in upper
            or "ACCEPTED REFERENCE MATCHES" in upper
        ):
            phase, phase_progress = "reference_matching", 88
        elif "CONSTRAINED IDENTITY CLUSTERING" in upper:
            phase, phase_progress = "clustering", 70
        elif total and terminal >= total:
            phase, phase_progress = "clustering", 70
        else:
            phase, phase_progress = "image_processing", round(image_progress * 0.70, 1)
    elif snapshot.get("returncode") == 0:
        phase, phase_progress = "complete", 100
    elif total:
        phase, phase_progress = "image_processing", round(image_progress * 0.70, 1)

    snapshot["phase"] = phase
    snapshot["image_progress"] = image_progress
    snapshot["progress"] = phase_progress
    snapshot["progress_indeterminate"] = phase in {
        "clustering",
        "reference_matching",
        "output",
    }

    if event:
        out = event_output(event)
        output_files = (
            [p for p in out.rglob("*") if p.is_file()] if out.is_dir() else []
        )
        snapshot["output"] = {
            "exists": out.is_dir(),
            "files": len(output_files),
            "images": sum(1 for p in output_files if p.suffix.lower() in IMAGE_EXTS),
        }
    else:
        snapshot["output"] = {"exists": False, "files": 0, "images": 0}
    return snapshot


@app.get("/api/log")
def log():
    # Path("") resolves to the current working directory ("."), which
    # exists but is a directory.  The frontend polls this endpoint even
    # before a run has started, so explicitly require a real log file.
    raw_path = state.get("log_path")
    if not raw_path:
        return {"text": ""}

    path = Path(str(raw_path))
    if not path.is_file():
        return {"text": ""}

    try:
        return {"text": path.read_text(encoding="utf-8", errors="replace")[-40000:]}
    except OSError as exc:
        # A log can briefly be unavailable while the runner is creating
        # or replacing it. Do not turn that transient condition into a
        # 500 response for the UI.
        return {"text": f"[Studio] Unable to read log: {exc}"}


@app.get("/api/events/{name}/results")
def results(name: str):
    event = EVENTS_ROOT / name
    if not event.is_dir():
        raise HTTPException(404, "Event not found")
    out = event_output(event)
    people = []
    if out.exists():
        for person in sorted(
            (p for p in out.iterdir() if p.is_dir()), key=lambda p: p.name.casefold()
        ):
            all_dir = person / "All Images"
            best_dir = person / "Best Images"
            all_count = (
                sum(1 for x in all_dir.rglob("*") if x.is_file())
                if all_dir.exists()
                else 0
            )
            best_count = (
                sum(1 for x in best_dir.rglob("*") if x.is_file())
                if best_dir.exists()
                else 0
            )
            # Prefer the generated representative face. If it is unavailable,
            # use a real image from Best Images, then All Images.
            image_exts = IMAGE_EXTS
            rep = next(
                (
                    x
                    for x in person.iterdir()
                    if x.is_file()
                    and x.suffix.lower() in image_exts
                    and x.stem.casefold() in {"representative image", "representative"}
                ),
                None,
            )
            representative_is_fallback = False

            if rep is None and best_dir.exists():
                rep = next(
                    (
                        x
                        for x in sorted(best_dir.rglob("*"))
                        if x.is_file() and x.suffix.lower() in image_exts
                    ),
                    None,
                )
                representative_is_fallback = rep is not None

            if rep is None and all_dir.exists():
                rep = next(
                    (
                        x
                        for x in sorted(all_dir.rglob("*"))
                        if x.is_file() and x.suffix.lower() in image_exts
                    ),
                    None,
                )
                representative_is_fallback = rep is not None

            representative_url = None
            representative_file = None

            if rep:
                from urllib.parse import quote

                relative_file = rep.relative_to(person).as_posix()
                representative_file = rep.name
                representative_url = (
                    f"/api/events/{quote(name, safe='')}/output-image"
                    f"?person={quote(person.name, safe='')}"
                    f"&file={quote(relative_file, safe='')}"
                )

            people.append(
                {
                    "name": person.name,
                    "all_images": all_count,
                    "best_images": best_count,
                    "representative": representative_url,
                    "representative_file": representative_file,
                    "representative_is_fallback": representative_is_fallback,
                }
            )
    excel = out / "final_results.xlsx"
    output_files = [p for p in out.rglob("*") if p.is_file()] if out.is_dir() else []
    output_images = sum(1 for p in output_files if p.suffix.lower() in IMAGE_EXTS)
    return {
        "people": people,
        "excel": f"/api/events/{name}/download-excel" if excel.exists() else None,
        "output": str(out),
        "output_exists": out.is_dir(),
        "output_files": len(output_files),
        "output_images": output_images,
    }


@app.get("/api/events/{name}/download-excel")
def download_excel(name: str):
    event = EVENTS_ROOT / name
    path = event_output(event) / "final_results.xlsx"
    if not path.is_file():
        raise HTTPException(404, "Excel report not found")
    return FileResponse(path, filename="final_results.xlsx")


@app.get("/api/events/{name}/output-image")
def output_image(name: str, person: str, file: str):
    """Serve a generated person image safely, including Best/All Images fallbacks."""
    event = EVENTS_ROOT / name
    if not event.is_dir():
        raise HTTPException(404, "Event not found")

    base = event_output(event).resolve()
    person_dir = (base / person).resolve()

    if person_dir.parent != base or not person_dir.is_dir():
        raise HTTPException(404, "Output person folder not found")

    path = (person_dir / file).resolve()

    # Permit the representative or files in Best Images / All Images,
    # but never anything outside this person's output folder.
    if person_dir not in path.parents or not path.is_file():
        raise HTTPException(404, "Output image not found")

    if path.suffix.lower() not in IMAGE_EXTS:
        raise HTTPException(400, "Requested output file is not an image")

    return FileResponse(path, media_type=None, filename=path.name)


@app.get("/api/events/{name}/output-file/{person}/{file_name}")
def output_file(name: str, person: str, file_name: str):
    event = EVENTS_ROOT / name
    base = event_output(event).resolve()
    path = (base / person / file_name).resolve()
    if base not in path.parents or not path.is_file():
        raise HTTPException(404, "Output file not found")
    return FileResponse(path)
