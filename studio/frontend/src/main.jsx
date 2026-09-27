import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity,
  AlertCircle,
  CheckCircle2,
  ChevronRight,
  Cloud,
  Cpu,
  Database,
  FileSpreadsheet,
  FolderOpen,
  Gauge,
  HardDrive,
  Image as ImageIcon,
  LayoutDashboard,
  Play,
  RefreshCw,
  Save,
  Settings,
  ShieldCheck,
  Square,
  UploadCloud,
  Users,
  Zap,
} from "lucide-react";
import "./styles.css";

const API = "http://127.0.0.1:8765/api";
const NAV = [
  { id: "dashboard", label: "Dashboard", icon: LayoutDashboard },
  { id: "new", label: "New Event", icon: UploadCloud },
  { id: "run", label: "Run Monitor", icon: Activity },
  { id: "results", label: "Results", icon: Users },
  { id: "config", label: "Configuration", icon: Settings },
];

// Keep this list aligned with app/configuration.py. The backend still remains
// the source of truth: fields are rendered only when they exist in the file.
const CONFIG_GROUPS = {
  "Runtime & workers": [
    "WORKER_COUNT",
    "WORKER_USE_PROCESSES",
    "WORKER_STALE_TIMEOUT_SECONDS",
    "ENABLE_VISUALIZATION",
  ],
  "Person detection": [
    "PERSON_MODEL_NAME",
    "PERSON_DETECTION_THRESHOLD",
    "PERSON_DETECTION_DEVICE",
    "PERSON_CLASS_ID",
  ],
  "Face detection": [
    "FACE_MODEL_NAME",
    "FACE_DETECTION_SIZE",
    "FACE_DETECTION_THRESHOLD",
    "FACE_CTX_ID",
  ],
  Association: ["ASSOCIATION_MIN_SCORE"],
  "Body / Re-ID": [
    "BODY_MODEL_NAME",
    "BODY_MODEL_PATH",
    "BODY_DEVICE",
    "BODY_UPPER_BODY_RATIO",
  ],
  "Face quality": [
    "FACE_QUALITY_DETECTION_WEIGHT",
    "FACE_QUALITY_SIZE_WEIGHT",
    "FACE_QUALITY_SHARPNESS_WEIGHT",
    "FACE_MIN_SIZE",
    "FACE_REFERENCE_SIZE",
    "FACE_SHARPNESS_MIN",
    "FACE_SHARPNESS_MAX",
  ],
  "Face pose": [
    "FACE_POSE_FRONTAL_YAW_DEGREES",
    "FACE_POSE_PROFILE_YAW_DEGREES",
  ],
  Clustering: [
    "MIN_FACE_SIMILARITY",
    "MERGE_THRESHOLD",
    "CROSS_POSE_MIN_FACE_SIMILARITY",
    "CROSS_POSE_MIN_BODY_SIMILARITY",
    "CROSS_POSE_MERGE_THRESHOLD",
    "MAX_REPRESENTATIVES_PER_POSE",
    "REPRESENTATIVE_COUNT",
    "ANCHOR_MIN_FACE_QUALITY",
    "ANCHOR_MIN_FACE_DETECTION_CONFIDENCE",
    "ANCHOR_MIN_FACE_SIZE",
    "ANCHOR_MAX_YAW_DEGREES",
    "MIN_CLUSTER_SIZE",
    "CLUSTER_FACE_WEIGHT",
    "CLUSTER_BODY_WEIGHT",
    "CLUSTER_QUALITY_WEIGHT",
    "CLUSTER_FACE_ONLY_WEIGHT",
    "CLUSTER_FACE_ONLY_QUALITY_WEIGHT",
  ],
  "Reference matching": [
    "REFERENCES_PATH",
    "REFERENCE_MATCH_THRESHOLD",
    "REFERENCE_MATCH_MIN_MARGIN",
  ],
  Output: [
    "EVENT_OUTPUT_DIRECTORY_NAME",
    "EVENT_OUTPUT_MODE",
    "EVENT_OUTPUT_CLEAN_BEFORE_RUN",
    "BEST_IMAGES_PER_CLUSTER",
    "BEST_IMAGES_REQUIRE_POSE_DIVERSITY",
  ],
  "Visualization / advanced": [
    "VISUALIZATION_SCALE",
    "VISUALIZATION_WAIT_KEY_MS",
    "VISUALIZATION_STOP_KEY",
    "FACE_PADDING_RATIO",
    "VISUALIZATION_COLUMNS",
    "VISUALIZATION_MAX_ITEMS_PER_CLUSTER",
    "REPRESENTATIVE_FACE_QUALITY_WEIGHT",
    "REPRESENTATIVE_FACE_DETECTION_WEIGHT",
    "CLUSTER_VISUALIZATION_DIR_NAME",
    "EVENTS_PATH",
    "EVENT_DATABASE_FILENAME",
    "QUALITY_MIN",
    "QUALITY_MAX",
    "PERSON_MODEL_NAME",
    "BODY_QUALITY_DETECTION_WEIGHT",
    "BODY_QUALITY_SIZE_WEIGHT",
    "BODY_QUALITY_SHARPNESS_WEIGHT",
    "BODY_MIN_SIZE",
    "BODY_REFERENCE_SIZE",
    "BODY_SHARPNESS_MIN",
    "BODY_SHARPNESS_MAX",
    "GOOGLE_DRIVE_PUBLIC_LINK_ROLE",
  ],
};
const ENV_KEYS = [
  "GOOGLE_DRIVE_ENABLED",
  "GOOGLE_DRIVE_CREDENTIALS_PATH",
  "GOOGLE_DRIVE_TOKEN_PATH",
  "GOOGLE_DRIVE_ROOT_FOLDER_ID",
  "GOOGLE_DRIVE_FOLDER_NAME",
  "GOOGLE_DRIVE_PUBLIC_LINK",
  "GOOGLE_DRIVE_RETRY_COUNT",
  "GOOGLE_DRIVE_HTTP_TIMEOUT_SECONDS",
];

function App() {
  const [page, setPage] = useState("dashboard");
  const [events, setEvents] = useState([]);
  const [selected, setSelected] = useState("");
  const [hardware, setHardware] = useState(null);
  const [config, setConfig] = useState({ values: {}, env: {} });
  const [status, setStatus] = useState(null);
  const [log, setLog] = useState("");
  const [toast, setToast] = useState(null);

  const refresh = async () => {
    try {
      const [e, h, s, l] = await Promise.all([
        fetch(API + "/events").then((r) => r.json()),
        fetch(API + "/hardware").then((r) => r.json()),
        fetch(API + "/status").then((r) => r.json()),
        fetch(API + "/log").then((r) => r.json()),
      ]);
      setEvents(e);
      setHardware(h);
      setStatus(s);
      setLog(l.text || "");
      if (!selected && e[0]) setSelected(e[0].name);
    } catch (err) {
      setToast({
        type: "error",
        text: "Studio backend is not reachable. Start run_studio.py.",
      });
    }
  };

  const loadConfig = async () => {
    try {
      setConfig(await fetch(API + "/config").then((r) => r.json()));
    } catch (err) {
      setToast({ type: "error", text: "Could not read configuration." });
    }
  };

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 1500);
    return () => clearInterval(t);
  }, []);
  useEffect(() => {
    if (page === "config") loadConfig();
  }, [page]);
  useEffect(() => {
    if (toast) {
      const t = setTimeout(() => setToast(null), 3500);
      return () => clearTimeout(t);
    }
  }, [toast]);

  const selectedEvent = events.find((e) => e.name === selected);
  const goto = (p) => {
    if ((p === "run" || p === "results") && !selected && events[0])
      setSelected(events[0].name);
    setPage(p);
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            <Zap size={20} />
          </div>
          <div>
            <b>PersonaCluster</b>
            <span>Studio</span>
          </div>
        </div>
        <div className="project-chip">
          <span className="dot" /> LOCAL ENGINE{" "}
          <span className="ok">READY</span>
        </div>
        <nav>
          {NAV.map((n) => {
            const Icon = n.icon;
            return (
              <button
                className={page === n.id ? "nav active" : "nav"}
                onClick={() => goto(n.id)}
                key={n.id}
              >
                <Icon size={18} />
                <span>{n.label}</span>
              </button>
            );
          })}
        </nav>
        <div className="sidebar-bottom">
          <div className="engine-card">
            <div className="engine-icon">
              <Cpu size={17} />
            </div>
            <div>
              <b>{hardware?.profile?.name || "Detecting hardware"}</b>
              <span>{hardware?.profile?.workers || 1} worker recommended</span>
            </div>
          </div>
          <div className="version">PersonaCluster Studio 1.0</div>
        </div>
      </aside>
      <main className="main">
        <header className="topbar">
          <div>
            <span className="eyebrow">
              PERSONACLUSTER / {page.replace("-", " ").toUpperCase()}
            </span>
            <h1>{NAV.find((x) => x.id === page)?.label || "Studio"}</h1>
          </div>
          <div className="top-actions">
            <div className="system">
              <span className="dot" /> Engine connected
            </div>
            <button className="icon-btn" onClick={refresh} title="Refresh">
              <RefreshCw size={18} />
            </button>
          </div>
        </header>
        {page === "dashboard" && (
          <Dashboard
            events={events}
            hardware={hardware}
            selected={selected}
            setSelected={setSelected}
            setPage={goto}
          />
        )}
        {page === "new" && (
          <NewEvent
            refresh={refresh}
            setSelected={setSelected}
            setPage={goto}
            toast={setToast}
          />
        )}
        {page === "run" && (
          <Run
            selected={selected}
            status={status}
            log={log}
            refresh={refresh}
            toast={setToast}
            stop={async () => {
              try {
                const r = await fetch(
                  API + `/events/${encodeURIComponent(selected)}/stop`,
                  { method: "POST" },
                );
                const body = await r.json().catch(() => ({}));
                if (!r.ok)
                  throw new Error(
                    body.detail || `Backend returned ${r.status}`,
                  );
                refresh();
              } catch (err) {
                setToast({ type: "error", text: err.message });
              }
            }}
          />
        )}
        {page === "results" && <Results selected={selected} status={status} />}
        {page === "config" && (
          <Config
            config={config}
            setConfig={setConfig}
            hardware={hardware}
            toast={setToast}
          />
        )}
      </main>
      {toast && (
        <div className={"toast " + toast.type}>
          {toast.type === "error" ? (
            <AlertCircle size={17} />
          ) : (
            <CheckCircle2 size={17} />
          )}
          <span>{toast.text}</span>
        </div>
      )}
    </div>
  );
}

function Dashboard({ events, hardware, selected, setSelected, setPage }) {
  const total = events.reduce((a, e) => a + e.gallery_images, 0),
    refs = events.reduce((a, e) => a + e.references, 0),
    processed = events.filter((e) => e.has_db).length;
  return (
    <div className="page">
      <section className="hero">
        <div>
          <span className="pill">
            <span className="dot" /> LOCAL PROCESSING
          </span>
          <h2>Your event photo pipeline, in one place.</h2>
          <p>
            Upload an event, tune the engine for your hardware, run the existing
            PersonaCluster pipeline, and inspect the final reference-matched
            output.
          </p>
          <div className="hero-actions">
            <button className="primary" onClick={() => setPage("new")}>
              <UploadCloud size={17} /> New event
            </button>
            {selected && (
              <button className="secondary" onClick={() => setPage("run")}>
                <Play size={17} /> Run {selected}
              </button>
            )}
          </div>
        </div>
        <div className="hero-visual">
          <div className="orb">
            <Users size={42} />
          </div>
          <div className="orbit-card">
            <Gauge size={15} />
            <span>{hardware?.profile?.name || "Hardware profile"}</span>
          </div>
        </div>
      </section>
      <div className="stats">
        <Stat icon={FolderOpen} label="Events" value={events.length} />
        <Stat
          icon={ImageIcon}
          label="Gallery images"
          value={total.toLocaleString()}
        />
        <Stat icon={Users} label="References" value={refs.toLocaleString()} />
        <Stat icon={Database} label="Processed events" value={processed} />
      </div>
      <section className="panel">
        <div className="panel-head">
          <div>
            <h3>Events</h3>
            <p>
              Each event owns its Gallery, references and SQLite processing
              state. Final output is written to{" "}
              <code>output/&lt;event&gt;</code>.
            </p>
          </div>
          <button className="text-btn" onClick={() => setPage("new")}>
            Create event <ChevronRight size={16} />
          </button>
        </div>
        {events.length ? (
          <div className="event-table">
            {events.map((e) => (
              <button
                key={e.name}
                className={
                  "event-row " + (selected === e.name ? "selected" : "")
                }
                onClick={() => setSelected(e.name)}
              >
                <div className="event-icon">
                  <FolderOpen size={18} />
                </div>
                <div className="event-main">
                  <strong>{e.name}</strong>
                  <span>
                    {e.gallery_images.toLocaleString()} gallery images ·{" "}
                    {e.references} references
                  </span>
                </div>
                <div className="event-meta">
                  {e.has_db ? (
                    <span className="badge success">DATABASE READY</span>
                  ) : (
                    <span className="badge">NEW</span>
                  )}
                  <ChevronRight size={17} />
                </div>
              </button>
            ))}
          </div>
        ) : (
          <Empty text="No events yet. Create your first event to start processing." />
        )}
      </section>
    </div>
  );
}
function Stat({ icon: Icon, label, value }) {
  return (
    <div className="stat">
      <div className="stat-icon">
        <Icon size={18} />
      </div>
      <div>
        <span>{label}</span>
        <strong>{value}</strong>
      </div>
    </div>
  );
}

function uploadWithProgress(fd, onProgress) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", API + "/events");
    xhr.upload.onprogress = (e) => {
      if (e.lengthComputable)
        onProgress(Math.round((e.loaded / e.total) * 100));
    };
    xhr.onload = () => {
      let body = {};
      try {
        body = JSON.parse(xhr.responseText || "{}");
      } catch {}
      if (xhr.status >= 200 && xhr.status < 300) resolve(body);
      else reject(new Error(body.detail || `Upload failed (${xhr.status})`));
    };
    xhr.onerror = () =>
      reject(new Error("Upload failed. Check the backend connection."));
    xhr.send(fd);
  });
}

function NewEvent({ refresh, setSelected, setPage, toast }) {
  const [name, setName] = useState(""),
    [gallery, setGallery] = useState([]),
    [refs, setRefs] = useState([]),
    [busy, setBusy] = useState(false),
    [uploadProgress, setUploadProgress] = useState(0);
  const gRef = useRef(),
    rRef = useRef();
  const submit = async () => {
    if (!name.trim() || !gallery.length) {
      toast({
        type: "error",
        text: "Event name and a Gallery folder are required.",
      });
      return;
    }
    setBusy(true);
    setUploadProgress(0);
    const fd = new FormData();
    fd.append("event_name", name);
    gallery.forEach((f) =>
      fd.append("gallery", f, f.webkitRelativePath || f.name),
    );
    refs.forEach((f) =>
      fd.append("references", f, f.webkitRelativePath || f.name),
    );
    try {
      const e = await uploadWithProgress(fd, setUploadProgress);
      await refresh();
      setSelected(e.name);
      setPage("run");
      toast({
        type: "success",
        text: `${e.name} uploaded and is ready to run.`,
      });
    } catch (err) {
      toast({ type: "error", text: err.message });
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="page narrow">
      <div className="steps">
        <div className="step active">
          <b>01</b>
          <span>Event input</span>
        </div>
        <div className="step">
          <b>02</b>
          <span>References</span>
        </div>
        <div className="step">
          <b>03</b>
          <span>Process</span>
        </div>
      </div>
      <section className="panel form-panel">
        <div className="section-title">
          <div>
            <span className="eyebrow">NEW EVENT</span>
            <h2>Prepare an event</h2>
            <p>
              Gallery files are uploaded with their relative folder paths,
              preserving nested Camera/Day folders. References are stored in the
              event-local reference directory.
            </p>
          </div>
          <ShieldCheck size={27} />
        </div>
        <label className="field big">
          <span>Event name</span>
          <input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. EEA 2026"
          />
        </label>
        <UploadBox
          title="Gallery folder"
          subtitle="Select the root Gallery folder. Nested camera/day folders are preserved."
          files={gallery}
          setFiles={setGallery}
          inputRef={gRef}
          folder
        />
        <UploadBox
          title="Reference folder"
          subtitle="Use: Person Name - Phone Number.ext"
          files={refs}
          setFiles={setRefs}
          inputRef={rRef}
          folder
        />
        <div className="notice">
          <ShieldCheck size={17} />
          <div>
            <b>Local by default</b>
            <span>
              Files are copied into{" "}
              <code>data/events/{name || "<event>"}/Gallery</code> and{" "}
              <code>reference</code>.
            </span>
          </div>
        </div>
        {busy && (
          <div className="upload-progress">
            <div>
              <span>Uploading event</span>
              <strong>{uploadProgress}%</strong>
            </div>
            <div className="progress-track">
              <i style={{ width: `${uploadProgress}%` }} />
            </div>
          </div>
        )}
        <button className="primary wide" disabled={busy} onClick={submit}>
          {busy ? (
            <>
              <RefreshCw size={17} className="spin" /> Uploading{" "}
              {uploadProgress}%…
            </>
          ) : (
            <>
              <UploadCloud size={17} /> Create event & continue
            </>
          )}
        </button>
      </section>
    </div>
  );
}
function UploadBox({ title, subtitle, files, setFiles, inputRef, folder }) {
  return (
    <div
      className={"upload-box " + (files.length ? "has-files" : "")}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        multiple
        webkitdirectory={folder ? "true" : undefined}
        directory={folder ? "true" : undefined}
        onChange={(e) => setFiles(Array.from(e.target.files || []))}
      />
      <div className="upload-icon">
        <UploadCloud size={21} />
      </div>
      <div className="upload-copy">
        <strong>{title}</strong>
        <span>{subtitle}</span>
        {files.length > 0 && (
          <small>
            <CheckCircle2 size={14} /> {files.length.toLocaleString()} file(s)
            selected
          </small>
        )}
      </div>
      <ChevronRight size={18} />
    </div>
  );
}

function phaseLabel(phase) {
  return (
    {
      idle: "READY",
      image_processing: "READING & PROCESSING IMAGES",
      clustering: "CONSTRAINED IDENTITY CLUSTERING",
      reference_matching: "REFERENCE MATCHING",
      output: "GENERATING OUTPUT",
      complete: "COMPLETE",
    }[phase] ||
    String(phase || "PROCESSING")
      .replaceAll("_", " ")
      .toUpperCase()
  );
}
function Run({ selected, status, log, refresh, stop, toast }) {
  const jobs = status?.jobs || {};
  const total =
    (jobs.pending || 0) +
    (jobs.processing || 0) +
    (jobs.completed || 0) +
    (jobs.failed || 0);
  const running = Boolean(status?.running);
  const phase = status?.phase || (running ? "image_processing" : "idle");
  const imageProgress = Number(status?.image_progress || 0);
  const observations = Number(jobs.observations || 0);
  const terminalJobs = Number(jobs.completed || 0) + Number(jobs.failed || 0);
  const stages = [
    {
      id: "image_processing",
      label: "Read & process images",
      short: "Images",
      icon: ImageIcon,
    },
    {
      id: "clustering",
      label: "Find people & clusters",
      short: "Clustering",
      icon: Users,
    },
    {
      id: "reference_matching",
      label: "Match references",
      short: "References",
      icon: ShieldCheck,
    },
    {
      id: "output",
      label: "Build final output",
      short: "Output",
      icon: FolderOpen,
    },
  ];
  const phaseIndex = Math.max(
    0,
    stages.findIndex((s) => s.id === phase),
  );
  const stageNumber =
    phase === "complete" ? 4 : phase === "idle" ? 0 : phaseIndex + 1;
  const phaseText = phaseLabel(phase);
  const imageDone =
    imageProgress >= 100 ||
    ["clustering", "reference_matching", "output", "complete"].includes(phase);

  const stageState = (index) => {
    if (phase === "complete") return "done";
    if (index < phaseIndex) return "done";
    if (index === phaseIndex && running) return "active";
    return "upcoming";
  };

  const start = async () => {
    try {
      const r = await fetch(
        API + `/events/${encodeURIComponent(selected)}/run`,
        { method: "POST" },
      );
      const body = await r.json().catch(() => ({}));
      if (!r.ok) throw new Error(body.detail || `Backend returned ${r.status}`);
      refresh();
    } catch (err) {
      toast({ type: "error", text: err.message });
    }
  };

  return (
    <div className="page run-monitor-page">
      <div className="run-monitor-header">
        <div>
          <div className="run-event-label">
            <Activity size={15} /> RUN MONITOR
          </div>
          <div className="run-title-row">
            <div>
              <h2>{selected || "Select an event"}</h2>
              <p>
                {running
                  ? "PersonaCluster is working through the event. You can follow each stage below."
                  : phase === "complete"
                    ? "Processing is complete. The generated output is ready to review."
                    : "Everything is ready. Start processing when you are ready."}
              </p>
            </div>
            <div className="run-actions">
              {selected && !running && phase !== "complete" && (
                <button className="primary" onClick={start}>
                  <Play size={17} /> Start processing
                </button>
              )}
              {running && (
                <button className="danger" onClick={stop}>
                  <Square size={16} /> Stop safely
                </button>
              )}
              <button className="secondary" onClick={refresh}>
                <RefreshCw size={16} /> Refresh
              </button>
            </div>
          </div>
        </div>
      </div>

      <section
        className={"current-stage-card " + (running ? "is-running" : "")}
      >
        <div className="current-stage-main">
          <div className="stage-kicker">CURRENT STAGE</div>
          <div className="current-stage-title">
            <span className="current-stage-icon">
              <Zap size={20} />
            </span>
            <div>
              <h3>{phaseText}</h3>
              <p>
                {phase === "image_processing"
                  ? "Reading the Gallery and creating person observations."
                  : phase === "clustering"
                    ? "Grouping observations into event-level identities. This may take time after all images finish."
                    : phase === "reference_matching"
                      ? "Comparing discovered identities with the event reference images."
                      : phase === "output"
                        ? "Creating the final person folders and selected images."
                        : phase === "complete"
                          ? "All processing stages have finished successfully."
                          : "Waiting for the event to start."}
              </p>
            </div>
          </div>
        </div>
        <div className="stage-progress-summary">
          {phase === "image_processing" ? (
            <>
              <strong>{imageProgress}%</strong>
              <span>
                {terminalJobs.toLocaleString()} of {total.toLocaleString()}{" "}
                image jobs finished
              </span>
            </>
          ) : phase === "complete" ? (
            <>
              <strong>100%</strong>
              <span>All stages completed</span>
            </>
          ) : running ? (
            <>
              <strong>Working…</strong>
              <span>Stage {stageNumber} of 4</span>
            </>
          ) : (
            <>
              <strong>Ready</strong>
              <span>4 processing stages</span>
            </>
          )}
        </div>
        <div className="big-progress-track">
          {phase === "image_processing" ? (
            <i
              style={{ width: `${Math.max(0, Math.min(100, imageProgress))}%` }}
            />
          ) : phase === "complete" ? (
            <i style={{ width: "100%" }} />
          ) : running ? (
            <i className="indeterminate-fill" />
          ) : (
            <i style={{ width: "0%" }} />
          )}
        </div>
      </section>

      <section className="pipeline-card">
        <div className="section-heading">
          <div>
            <h3>Processing pipeline</h3>
            <p>
              Each stage has its own state, so finishing image processing does
              not mean the whole event is finished.
            </p>
          </div>
          {running && (
            <span className="live-badge">
              <i /> LIVE
            </span>
          )}
        </div>
        <div className="pipeline-stages">
          {stages.map((stage, index) => {
            const state = stageState(index);
            const Icon = stage.icon;
            const detail =
              stage.id === "image_processing"
                ? imageDone
                  ? "Completed"
                  : `${imageProgress}% complete`
                : state === "done"
                  ? "Completed"
                  : state === "active"
                    ? "Working now"
                    : "Waiting";
            return (
              <div className={"pipeline-stage " + state} key={stage.id}>
                <div className="stage-node">
                  {state === "done" ? (
                    <CheckCircle2 size={19} />
                  ) : (
                    <Icon size={19} />
                  )}
                </div>
                <div className="stage-content">
                  <strong>{stage.label}</strong>
                  <span>{detail}</span>
                  {stage.id === "image_processing" && state === "active" && (
                    <small>
                      {terminalJobs.toLocaleString()} / {total.toLocaleString()}{" "}
                      jobs
                    </small>
                  )}
                  {stage.id === "clustering" && state === "active" && (
                    <small>
                      {observations.toLocaleString()} observations to cluster
                    </small>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </section>

      <section className="run-metrics-grid">
        <MetricCard
          label="Pending images"
          value={jobs.pending}
          hint="Waiting to be processed"
        />
        <MetricCard
          label="Processing now"
          value={jobs.processing}
          hint="Currently being handled"
          live={running && jobs.processing > 0}
        />
        <MetricCard
          label="Completed images"
          value={jobs.completed}
          hint="Successfully processed"
        />
        <MetricCard
          label="Failed images"
          value={jobs.failed}
          hint="Can be inspected or retried"
          danger={jobs.failed > 0}
        />
        <MetricCard
          label="Observations"
          value={observations}
          hint="Person observations stored"
          wide
        />
      </section>

      <section className="pipeline-details-grid">
        <div className="panel pipeline-explanation">
          <div className="section-heading">
            <div>
              <h3>What is happening?</h3>
              <p>A simple explanation of the current work.</p>
            </div>
          </div>
          <div className="explanation-list">
            <div
              className={
                phase === "image_processing"
                  ? "explanation-item active"
                  : "explanation-item"
              }
            >
              <span>1</span>
              <div>
                <strong>Read images</strong>
                <p>
                  Gallery images are discovered recursively and sent to the
                  workers.
                </p>
              </div>
            </div>
            <div
              className={
                phase === "clustering"
                  ? "explanation-item active"
                  : "explanation-item"
              }
            >
              <span>2</span>
              <div>
                <strong>Discover identities</strong>
                <p>
                  The stored observations are compared and grouped into
                  anonymous clusters.
                </p>
              </div>
            </div>
            <div
              className={
                phase === "reference_matching"
                  ? "explanation-item active"
                  : "explanation-item"
              }
            >
              <span>3</span>
              <div>
                <strong>Match known people</strong>
                <p>Clusters are compared with the event's reference images.</p>
              </div>
            </div>
            <div
              className={
                phase === "output"
                  ? "explanation-item active"
                  : "explanation-item"
              }
            >
              <span>4</span>
              <div>
                <strong>Create results</strong>
                <p>
                  All Images, Best Images and representative images are written
                  to the output folder.
                </p>
              </div>
            </div>
          </div>
        </div>

        <div className="panel run-log-panel">
          <div className="section-heading">
            <div>
              <h3>Live activity</h3>
              <p>Latest messages from the real PersonaCluster process.</p>
            </div>
            <span className="log-status">
              <i /> LIVE
            </span>
          </div>
          <pre className="friendly-log">
            {log ||
              "No processing activity yet. Start an event to see live messages here."}
          </pre>
        </div>
      </section>
    </div>
  );
}

function MetricCard({
  label,
  value = 0,
  hint,
  danger = false,
  live = false,
  wide = false,
}) {
  return (
    <div
      className={
        "run-metric-card " + (danger ? "danger" : "") + (wide ? "wide" : "")
      }
    >
      <div className="metric-top">
        <span>{label}</span>
        {live && <i className="metric-live" />}
      </div>
      <strong>{Number(value || 0).toLocaleString()}</strong>
      <small>{hint}</small>
    </div>
  );
}

function Results({ selected, status }) {
  const [data, setData] = useState(null);
  const load = async () => {
    if (!selected) return;
    try {
      setData(
        await fetch(
          API +
            `/events/${encodeURIComponent(selected)}/results?ts=${Date.now()}`,
        ).then((r) => r.json()),
      );
    } catch {}
  };
  useEffect(() => {
    load();
    const t = setInterval(load, 1500);
    return () => clearInterval(t);
  }, [selected]);
  if (!selected)
    return (
      <div className="page">
        <Empty text="Select an event to view its final output." />
      </div>
    );
  const hasOutput = Boolean(data?.people?.length || data?.output_images);
  return (
    <div className="page">
      <div className="run-header">
        <div>
          <span className="pill">LIVE OUTPUT DIRECTORY</span>
          <h2>{selected}</h2>
          <p>
            Results are scanned directly from the configured project output
            folder on every refresh. Nothing is hardcoded to a specific event or
            person.
          </p>
        </div>
        {data?.excel && (
          <a className="secondary link" href={API + data.excel}>
            <FileSpreadsheet size={17} /> Download Excel
          </a>
        )}
      </div>
      <div className="stats">
        <Stat
          icon={Users}
          label="Matched people"
          value={data?.people?.length || 0}
        />
        <Stat
          icon={ImageIcon}
          label="Output images"
          value={data?.output_images || 0}
        />
        <Stat
          icon={HardDrive}
          label="Output folder"
          value={data?.output_exists ? "Detected" : "Not created"}
        />
      </div>
      <section className="panel">
        <div className="panel-head">
          <div>
            <h3>Generated output</h3>
            <p>
              {status?.running
                ? `Pipeline stage: ${phaseLabel(status?.phase)}. This page updates while files appear.`
                : "Showing whatever currently exists in the real output directory."}
            </p>
          </div>
          <button className="icon-btn" onClick={load}>
            <RefreshCw size={17} />
          </button>
        </div>
        {hasOutput ? (
          <div className="people-grid">
            {data.people.map((p) => (
              <div className="person-card" key={p.name}>
                <div className="person-avatar">
                  {p.representative ? (
                    <img
                      className="avatar-image"
                      src={
                        API +
                        p.representative +
                        (p.representative.includes("?") ? "&" : "?") +
                        "v=" +
                        encodeURIComponent(p.representative_file || "image")
                      }
                      alt={`${p.name} representative`}
                      loading="lazy"
                      onLoad={(e) => {
                        e.currentTarget.style.display = "block";
                        e.currentTarget.nextElementSibling?.classList.add(
                          "hidden",
                        );
                      }}
                      onError={(e) => {
                        e.currentTarget.style.display = "none";
                        e.currentTarget.nextElementSibling?.classList.remove(
                          "hidden",
                        );
                      }}
                    />
                  ) : null}
                  <div
                    className={"avatar" + (p.representative ? " hidden" : "")}
                  >
                    {p.name.slice(0, 1).toUpperCase()}
                  </div>
                </div>

                <div className="person-copy">
                  <strong>{p.name}</strong>
                  <span>{p.all_images.toLocaleString()} all images</span>
                  <span>{p.best_images.toLocaleString()} best images</span>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <Empty
            text={
              status?.running
                ? "No output images have appeared yet. Keep this page open; it refreshes automatically."
                : "The output directory is empty for this event."
            }
          />
        )}
      </section>
    </div>
  );
}

function Config({ config, setConfig, hardware, toast }) {
  const [draft, setDraft] = useState(config),
    [dirty, setDirty] = useState(false),
    [saving, setSaving] = useState(false);
  useEffect(() => {
    if (!dirty) setDraft(config);
  }, [config, dirty]);
  const setVal = (k, v) => {
    setDirty(true);
    setDraft((d) => ({
      ...d,
      values: { ...d.values, [k]: coerce(v, d.values[k]) },
    }));
  };
  const setEnv = (k, v) => {
    setDirty(true);
    setDraft((d) => ({ ...d, env: { ...d.env, [k]: v } }));
  };
  const save = async () => {
    setSaving(true);
    try {
      const r = await fetch(API + "/config", {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ configuration: draft.values, env: draft.env }),
      });
      const body = await r.json().catch(() => ({}));
      if (!r.ok)
        throw new Error(body.detail || "Could not save configuration.");
      // Normalize the saved response so the editor always receives the
      // same shape as GET /api/config, even if an older backend is running.
      const savedConfig = {
        values: body.values ?? body.configuration ?? {},
        env: body.env ?? {},
      };
      setConfig(savedConfig);
      setDraft(savedConfig);
      setDirty(false);
      toast({
        type: "success",
        text: `Saved ${body.updated?.length || 0} configuration value(s). Changes apply to the next run.`,
      });
    } catch (err) {
      toast({ type: "error", text: err.message });
    } finally {
      setSaving(false);
    }
  };
  const reload = async () => {
    await fetch(API + "/config")
      .then((r) => r.json())
      .then((c) => {
        setConfig(c);
        setDraft(c);
        setDirty(false);
      });
  };
  const apply = () => {
    setDirty(true);
    setDraft((d) => ({
      ...d,
      values: {
        ...d.values,
        WORKER_COUNT: hardware?.profile?.workers ?? d.values.WORKER_COUNT,
        PERSON_DETECTION_DEVICE:
          hardware?.profile?.person_device ?? d.values.PERSON_DETECTION_DEVICE,
        FACE_CTX_ID: hardware?.profile?.face_ctx ?? d.values.FACE_CTX_ID,
        BODY_DEVICE: hardware?.profile?.body_device ?? d.values.BODY_DEVICE,
      },
    }));
  };
  return (
    <div className="page">
      <section className="config-hero">
        <div>
          <span className="pill">SINGLE SOURCE OF TRUTH</span>
          <h2>Configure the real engine.</h2>
          <p>
            Studio edits <code>app/configuration.py</code> and the project root{" "}
            <code>.env</code>. The draft stays local to this page until you
            explicitly save it.
          </p>
        </div>
        <div className="resource">
          <Cpu size={19} />
          <div>
            <b>{hardware?.profile?.name || "Detecting hardware…"}</b>
            <span>{hardware?.profile?.note || ""}</span>
          </div>
        </div>
      </section>
      <div className="config-toolbar">
        <span className={dirty ? "badge warning" : "badge success"}>
          {dirty ? "UNSAVED CHANGES" : "SAVED"}
        </span>
        <button className="secondary" onClick={reload}>
          <RefreshCw size={16} /> Reload from file
        </button>
        <button className="secondary" onClick={apply}>
          <Cpu size={16} /> Apply safe hardware profile
        </button>
        <button className="primary" disabled={!dirty || saving} onClick={save}>
          {saving ? (
            <>
              <RefreshCw size={16} className="spin" /> Saving…
            </>
          ) : (
            <>
              <Save size={16} /> Save changes
            </>
          )}
        </button>
      </div>
      {Object.entries(CONFIG_GROUPS).map(([title, keys]) => (
        <section className="panel config-panel" key={title}>
          <div className="panel-head">
            <div>
              <h3>{title}</h3>
              <p>
                Editable values are read from the current configuration file.
              </p>
            </div>
          </div>
          <div className="config-grid">
            {keys
              .filter((k, i, a) => a.indexOf(k) === i)
              .map(
                (k) =>
                  draft.values[k] !== undefined && (
                    <Field
                      key={k}
                      name={k}
                      value={draft.values[k]}
                      onChange={(v) => setVal(k, v)}
                    />
                  ),
              )}
          </div>
        </section>
      ))}
      <section className="panel config-panel">
        <div className="panel-head">
          <div>
            <h3>Google Drive / .env</h3>
            <p>
              Credentials stay in your local project. Drive receives generated
              person folders/images only.
            </p>
          </div>
          <Cloud size={18} />
        </div>
        <div className="config-grid">
          {ENV_KEYS.map((k) => (
            <Field
              key={k}
              name={k}
              value={draft.env[k] ?? ""}
              onChange={(v) => setEnv(k, String(v))}
              env
            />
          ))}
        </div>
      </section>
    </div>
  );
}
function Field({ name, value, onChange }) {
  const bool =
    typeof value === "boolean" ||
    ["true", "false"].includes(String(value).toLowerCase());
  const num = typeof value === "number";
  const array = Array.isArray(value);
  return (
    <label className="field">
      <span>{pretty(name)}</span>
      {bool ? (
        <select
          value={String(value)}
          onChange={(e) => onChange(e.target.value === "true")}
        >
          <option value="true">True</option>
          <option value="false">False</option>
        </select>
      ) : array ? (
        <input
          value={JSON.stringify(value)}
          onChange={(e) => {
            try {
              onChange(JSON.parse(e.target.value));
            } catch {}
          }}
        />
      ) : num ? (
        <input
          type="number"
          step="any"
          value={value}
          onChange={(e) => onChange(e.target.value)}
        />
      ) : (
        <input value={value ?? ""} onChange={(e) => onChange(e.target.value)} />
      )}
    </label>
  );
}
function pretty(s) {
  return s
    .replaceAll("_", " ")
    .toLowerCase()
    .replace(/\b\w/g, (c) => c.toUpperCase());
}
function coerce(v, old) {
  if (typeof old === "number") return Number(v);
  if (typeof old === "boolean") return v === true || v === "true";
  return v;
}
function Empty({ text }) {
  return (
    <div className="empty">
      <FolderOpen size={25} />
      <span>{text}</span>
    </div>
  );
}
createRoot(document.getElementById("root")).render(<App />);
