# Roadwatch: Real-Time Vehicle Analytics Prototype

This repository is a **prototype** for the Elansol Technologies vehicle analytics assignment. It demonstrates the application path from crossing events through PostgreSQL to a live React dashboard. Its default event source is a simulator. It does **not** currently ingest video, detect vehicles, classify real vehicle colors, track objects, or run NVIDIA DeepStream.

The assignment brief says its contents and the resulting work are for recruitment use and must not be shared or disclosed. Keep this repository private and do not publish the brief or this submission publicly.

## What works in this prototype

- Python/FastAPI endpoints for health, metrics, recent events, and Server-Sent Events (SSE).
- PostgreSQL persistence with an idempotency key, event timestamps, camera/session/vehicle identifiers, class, color, direction, line, and source.
- A demo event producer that creates clearly labeled `simulated` crossing events every three seconds.
- A responsive React dashboard with totals, direction counts, color breakdown, recent events, and live updates.
- Docker Compose definitions for PostgreSQL and MediaMTX.

## Architecture

```text
Current prototype:
Demo event producer -> FastAPI -> PostgreSQL -> REST/SSE -> React dashboard

Assignment target (later):
MP4/RTSP camera -> MediaMTX -> DeepStream detector -> tracker -> color classifier
                                                   -> line crossing -> event writer
                                                                      -> PostgreSQL
                                                           FastAPI REST/SSE -> React
```

The API and database are separated from the event source. A future DeepStream probe can publish the same event fields the simulator currently creates. Database writes use an idempotency key so retries do not create duplicate events. Inference callbacks should hand events to a queue/writer rather than wait on database I/O in the video-processing thread.

### Main folders

```text
backend/app/main.py       API, demo event source, event persistence
backend/app/schema.sql    PostgreSQL table and index definitions
frontend/src/main.jsx     React dashboard and API/SSE client
frontend/src/style.css    Responsive dashboard styling
infra/docker-compose.yml  API, PostgreSQL, and MediaMTX services
```

## Requirements and setup

Installed tools observed on the development laptop include Python 3.10, Node.js 24, npm 11, Git, PostgreSQL client tools, and Docker CLI. The Docker Linux engine was **not running** when checked, and FFmpeg was not found. The NVIDIA card detected by `nvidia-smi` was a GeForce GTX 1650 with 4 GB VRAM. Windows hardware details and WSL state could not be fully checked in this session.

### Start the API, PostgreSQL, and MediaMTX

Start Docker Desktop and wait until its Linux engine is ready. From the repository root:

```powershell
docker compose -f infra/docker-compose.yml up -d --build
docker compose -f infra/docker-compose.yml ps
```

PostgreSQL is exposed on `localhost:5433` (container port 5432); MediaMTX RTSP is exposed on `rtsp://localhost:8554`; the API is exposed on `http://localhost:8000`. Using host port 5433 avoids colliding with a PostgreSQL server already installed on Windows. Inside Compose, the API connects to the database by service name. The Compose database credentials are for local development only. Do not reuse them in a deployed environment. The API creates the prototype schema on startup. Visit `http://localhost:8000/api/health`, `http://localhost:8000/api/metrics`, and `http://localhost:8000/docs` to inspect it.

For direct host execution instead of the API container, create a Python virtual environment, install `backend\requirements.txt`, set `$env:DATABASE_URL='postgresql://vehicle:vehicle_dev@localhost:5433/vehicle_analytics'`, and run `python -m uvicorn app.main:app --app-dir backend --reload --port 8000`.

### Start the React dashboard

In another PowerShell terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL shown in the terminal (normally `http://localhost:5173`). The dashboard connects to `http://localhost:8000` by default. Set `VITE_API_URL` before starting Vite if the API is hosted elsewhere.

### Run the prototype without Docker

Docker is optional for the current simulated-event prototype. You need a native PostgreSQL **server** installed and running on Windows; having only the `psql` client is not enough. In pgAdmin, create a login role named `vehicle` with password `vehicle_dev`, then create database `vehicle_analytics` owned by `vehicle`. If the role/database already exist, update/check their password and owner rather than creating duplicates. The API's native default uses `localhost:5433`; if your PostgreSQL server listens on the usual port `5432`, set the URL as shown below.

In a PowerShell terminal at the repository root:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --default-timeout=120 --retries 5 -r backend\requirements.txt
$env:DATABASE_URL = 'postgresql://vehicle:vehicle_dev@localhost:5432/vehicle_analytics'
python -m uvicorn app.main:app --app-dir backend --reload --port 8000
```

Keep that terminal running. In another terminal, start the dashboard using the React instructions above. On API startup, the schema is created and simulated crossing events begin arriving. This route skips Docker and MediaMTX; it still demonstrates PostgreSQL-to-API-to-dashboard flow, but does not process camera video.

To stop services:

```powershell
docker compose -f infra/docker-compose.yml down
```

That keeps the local PostgreSQL volume. To remove local database data as well, use `docker compose -f infra/docker-compose.yml down -v` only when you intentionally want to reset the prototype.

## Event/API details

- `GET /api/health` reports API/database readiness and current mode.
- `GET /api/metrics` returns total events and counts by color and direction.
- `GET /api/events?limit=50` returns recent persisted events (limit 1–200).
- `GET /api/events/stream` is an SSE stream for newly persisted demo events.
- Demo mode is on by default. Set `DEMO_MODE=false` to stop simulated inserts while developing a real event producer.
- `source` is `simulated` in demo mode. The schema reserves `deepstream` for future real inference output.

## What is still needed for assignment compliance

1. Verify a supported Ubuntu/Jetson DeepStream host, its NVIDIA driver, and compatible SDK/runtime. The local GTX 1650 and Docker CLI alone do not prove this environment is ready.
2. Publish a traffic MP4/camera stream into MediaMTX with FFmpeg, then verify RTSP playback in VLC. FFmpeg and a sample clip were not present in the checked workspace.
3. Implement and validate a primary vehicle detector, `nvtracker`, a secondary color classifier on vehicle crops, and `nvdsanalytics` line crossing (or equivalent DeepStream probe logic).
4. Connect the DeepStream event probe to a bounded asynchronous writer that calls the event persistence layer, carrying a unique session ID because tracker IDs can reset after restarts.
5. Add optional live video playback in the dashboard. This is a bonus in the assignment and is not implemented here.
6. Validate one count per tracked vehicle crossing, color accuracy, timestamp behavior, and dashboard updates with real video.

The color data in demo mode is random. Do not present those values as computer-vision predictions. The demo proves database/API/dashboard wiring only.

## Walkthrough explanation

### Short presentation outline

1. **Introduction:** “This is a prototype for a real-time vehicle analytics system. I separated event ingestion, persistence, and visualization so the dashboard can be developed before the DeepStream host is ready.”
2. **Architecture:** “The assignment target sends RTSP through MediaMTX to DeepStream. DeepStream detects vehicles, tracks them across frames, classifies each vehicle crop, and emits one event when a tracked object crosses the configured line. The event is written to PostgreSQL, and the React dashboard reads metrics and receives live event updates from the API.”
3. **Current implementation:** “For this prototype, a simulator produces sample crossing events. FastAPI validates the app flow, saves each event to PostgreSQL, and publishes new records via SSE. React displays counts and recent rows. The source label makes simulated data visible.”
4. **Demonstration:** Start PostgreSQL/MediaMTX, API, then the React app. Show `/api/health`, the metrics endpoint, and the dashboard changing as demo events arrive. Point out that MediaMTX is running but not yet connected to a real video source in this prototype.
5. **DeepStream upgrade:** Show where a future probe would send the event payload to the writer. Explain that `nvtracker` IDs are scoped to a pipeline session and the event key prevents retry duplicates.
6. **Limitations:** State plainly that this walkthrough demonstrates the application integration, not real detection, classification, tracking, or RTSP processing. A separate supported Linux/Jetson setup and validated models are required for that milestone.

The assignment requests keeping the camera on throughout its walkthrough recording. Follow that instruction if/when recording the submission.

## Git and GitHub

The folder has been initialized as a local Git repository. After creating a **private** GitHub repository, connect it from this folder with the URL GitHub gives you:

```powershell
git add .
git commit -m "Create vehicle analytics prototype"
git branch -M main
git remote add origin <your-private-repository-url>
git push -u origin main
```

Do not commit `.env` files, model weights you are not permitted to redistribute, credentials, or the assignment PDF. `.gitignore` already excludes `.env` and common build artifacts.
