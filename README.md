# Roadwatch: Real-Time Vehicle Analytics Prototype

This repository is a **prototype** for the Elansol Technologies vehicle analytics assignment. It has two selectable event sources: the original simulator and a Windows OpenCV/ONNX path that processes the local MP4, detects and tracks vehicles, estimates crop colors, and posts line-crossing events to PostgreSQL. It does **not** run NVIDIA DeepStream, use an NVIDIA DeepStream tracker, or use a trained secondary color classifier. Treat it as a constrained-device prototype rather than a completed assignment submission.

The assignment brief says its contents and the resulting work are for recruitment use and must not be shared or disclosed. Keep this repository private and do not publish the brief or this submission publicly.

## What works in this prototype

- Python/FastAPI endpoints for health, metrics, recent events, and Server-Sent Events (SSE).
- PostgreSQL persistence with an idempotency key, event timestamps, camera/session/vehicle identifiers, class, color, direction, line, and source.
- A demo event producer that creates clearly labeled `simulated` crossing events every three seconds.
- An optional Windows-only OpenCV + ONNX Runtime DirectML worker for local MP4 inference, simple IoU tracking, heuristic crop-color estimates, line crossings, annotated output, and asynchronous API event submission.
- A responsive React dashboard with totals, direction counts, color breakdown, recent events, and live updates.
- Docker Compose definitions for PostgreSQL and MediaMTX.

## Architecture

```text
Current prototype:
Demo event producer -> FastAPI -> PostgreSQL -> REST/SSE -> React dashboard

Optional Windows video prototype:
MP4 -> OpenCV/YOLOX ONNX -> IoU tracker -> line crossing -> FastAPI -> PostgreSQL
                                                            -> REST/SSE -> React

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
vision/worker.py          Optional Windows MP4 detector/tracker/counting worker
vision/requirements.txt   Windows vision-only dependencies
```

## Requirements and setup

The observed laptop has Windows with WSL2 Ubuntu 26.04, Python 3.10, Node.js 24, npm 11, Git, PostgreSQL tools, Docker Desktop, 8 GB system RAM, and an NVIDIA GeForce GTX 1650 with 4 GB VRAM. Docker engine availability can vary; the instructions below use the already-working native PostgreSQL setup if Docker is unavailable. The OpenCV worker attempts DirectML adapter 1 (the reported GTX 1650) and falls back to CPU if DirectML cannot initialize. This is not a guarantee of real-time speed.

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
- Demo mode is on by default. For video inference, set `DEMO_MODE=false` and `ACTIVE_SOURCE=opencv` before starting the API. The dashboard then filters to OpenCV events and marks the color method as heuristic.
- Valid event sources are `simulated`, `opencv`, and the reserved `deepstream` value. `deepstream` is only a schema/API source label; no DeepStream pipeline is implemented.

## Run the MP4 vision prototype on Windows

This uses the supplied local clip and does not need a camera, MediaMTX, Docker, or Linux. PostgreSQL, the API, and React must be running. Run all commands from PowerShell.

1. Start PostgreSQL using your existing Windows service/pgAdmin setup. Confirm the API's `DATABASE_URL` points to the correct host port and database. If PostgreSQL is on port 5432:

   ```powershell
   $env:DATABASE_URL = 'postgresql://vehicle:vehicle_dev@localhost:5432/vehicle_analytics'
   $env:DEMO_MODE = 'false'
   $env:ACTIVE_SOURCE = 'opencv'
   .\.venv\Scripts\python.exe -m uvicorn app.main:app --app-dir backend --reload --port 8000
   ```

   Keep this terminal open. Verify `http://localhost:8000/api/health` reports `mode: opencv` and the database is connected. Start the dashboard in a second terminal with `cd frontend; npm run dev` and use the Vite URL shown (typically `http://localhost:5173`).

2. Install the Windows-only vision dependencies into the existing `.venv`. This is a separate requirements file so DirectML is not installed into the Linux API container:

   ```powershell
   .\.venv\Scripts\python.exe -m pip install --default-timeout=120 --retries 5 -r vision\requirements.txt
   ```

3. Download the OpenCV Zoo YOLOX INT8 model (OpenCV Zoo model terms apply; see the model README linked below):

   ```powershell
   New-Item -ItemType Directory -Force models | Out-Null
   Invoke-WebRequest 'https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/object_detection_yolox/object_detection_yolox_2022nov_int8bq.onnx' -OutFile 'models\object_detection_yolox_2022nov_int8bq.onnx'
   ```

4. In a third terminal, run the worker:

   ```powershell
   .\.venv\Scripts\python.exe vision\worker.py --video .\mixkit-cars-driving-by-on-road-2022-hd-ready.mp4 --device-id 1 --line-orientation vertical --line-ratio 0.5
   ```

   It draws boxes, track IDs, estimated colors, and the count line, writes `output\opencv_annotated.mp4`, and posts each first crossing per track to `POST /api/events`. Use `--line-orientation horizontal` or change `--line-ratio` to place the line for the clip. The `in`/`out` labels follow the chosen line direction and are not a road-specific traffic direction until calibrated.

5. Watch the dashboard's event table and counts update. The worker timestamps events from their frame offset relative to processing start because this MP4 has no live camera capture clock. Press Ctrl+C after processing. A useful live processing demonstration can use `--max-frames 1500` to stop after a subset; omit it to process the whole file.

Limitations: the COCO detector only detects vehicle categories learned by its pretrained model; the basic IoU tracker can lose IDs through occlusion; HSV crop color is a rough heuristic, not a secondary learned classifier; one crossing is emitted per track; this is offline MP4 processing rather than RTSP. The GTX 1650 DirectML path may fall back to CPU or be slower than video playback. Model source and usage notes: [OpenCV Zoo YOLOX model README](https://github.com/opencv/opencv_zoo/blob/main/models/object_detection_yolox/README.md).

## What is still needed for assignment compliance

1. Verify a supported Ubuntu/Jetson DeepStream host, its NVIDIA driver, and compatible SDK/runtime. The local GTX 1650 and Docker CLI alone do not prove this environment is ready.
2. Publish the available traffic MP4 into MediaMTX with FFmpeg, then verify RTSP playback in VLC. The clip is present locally, but FFmpeg/RTSP publication and playback have not been configured.
3. Implement and validate a primary vehicle detector, `nvtracker`, a secondary color classifier on vehicle crops, and `nvdsanalytics` line crossing (or equivalent DeepStream probe logic).
4. Connect the DeepStream event probe to a bounded asynchronous writer that calls the event persistence layer, carrying a unique session ID because tracker IDs can reset after restarts.
5. Add optional live video playback in the dashboard. This is a bonus in the assignment and is not implemented here.
6. Validate one count per tracked vehicle crossing, color accuracy, timestamp behavior, and dashboard updates with real video.

The color data in demo mode is random. Do not present those values as computer-vision predictions. The demo proves database/API/dashboard wiring only.

## Walkthrough explanation

### Short presentation outline

1. **Introduction:** “This is a prototype for a real-time vehicle analytics system. I separated event ingestion, persistence, and visualization so the dashboard can be developed before the DeepStream host is ready.”
2. **Architecture:** “The assignment target sends RTSP through MediaMTX to DeepStream. DeepStream detects vehicles, tracks them across frames, classifies each vehicle crop, and emits one event when a tracked object crosses the configured line. The event is written to PostgreSQL, and the React dashboard reads metrics and receives live event updates from the API.”
3. **Current implementation:** “The simulator remains useful for checking the API and UI. In the optional Windows video mode, OpenCV runs a pretrained YOLOX ONNX vehicle detector, a basic IoU tracker, crop-based HSV color estimates, and a configurable line crossing. Events are saved through FastAPI and PostgreSQL and delivered to React over SSE.”
4. **Demonstration:** Start PostgreSQL/API and React, then run the worker on the local MP4. Show the annotated output, `/api/health`, and the dashboard updating from events labeled `opencv`.
5. **DeepStream upgrade:** Explain the intended assignment path: MediaMTX RTSP → DeepStream primary detector → `nvtracker` → secondary learned color classifier → line crossing → PostgreSQL → React. Explain that the prototype worker uses a different detector/tracker/classification approach, so it is a feasibility prototype, not DeepStream compliance.
6. **Limitations:** State plainly that DirectML can fall back to CPU, the tracker is simple, color estimates are heuristic, and the current clip is processed offline. A supported DeepStream environment and validated models are still needed for the required pipeline.

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
