"""Prototype API for vehicle crossing analytics.

Demo mode deliberately emits simulated crossing events. It does not claim to
perform computer-vision detection; a future DeepStream adapter can publish the
same event shape through ``record_event``.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

import psycopg
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from psycopg.rows import dict_row

DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://vehicle:vehicle_dev@localhost:5432/vehicle_analytics"
)
DEMO_MODE = os.getenv("DEMO_MODE", "true").lower() == "true"
DEMO_SESSION = str(uuid.uuid5(uuid.NAMESPACE_DNS, "vehicle-analytics-prototype-demo"))
COLORS = ["white", "black", "silver", "gray", "blue", "red"]
VEHICLE_CLASSES = ["car", "truck", "bus", "motorcycle"]

SCHEMA = """
CREATE TABLE IF NOT EXISTS crossing_events (
    id BIGSERIAL PRIMARY KEY,
    event_key TEXT NOT NULL UNIQUE,
    camera_id TEXT NOT NULL DEFAULT 'camera-01',
    session_id UUID NOT NULL,
    vehicle_id TEXT NOT NULL,
    vehicle_class TEXT NOT NULL,
    color TEXT NOT NULL,
    direction TEXT NOT NULL CHECK (direction IN ('in', 'out')),
    line_id TEXT NOT NULL DEFAULT 'gate-01',
    occurred_at TIMESTAMPTZ NOT NULL,
    source TEXT NOT NULL CHECK (source IN ('simulated', 'deepstream'))
);
CREATE INDEX IF NOT EXISTS crossing_events_time_idx ON crossing_events (occurred_at DESC);
CREATE INDEX IF NOT EXISTS crossing_events_color_time_idx ON crossing_events (color, occurred_at DESC);
"""


def connect() -> psycopg.Connection:
    return psycopg.connect(DATABASE_URL, row_factory=dict_row, connect_timeout=3)


def initialize_database() -> None:
    with connect() as conn:
        conn.execute(SCHEMA)


def record_event(event: dict[str, Any]) -> dict[str, Any] | None:
    """Persist one crossing idempotently; returns None for a duplicate event."""
    with connect() as conn:
        row = conn.execute(
            """
            INSERT INTO crossing_events
                (event_key, camera_id, session_id, vehicle_id, vehicle_class,
                 color, direction, line_id, occurred_at, source)
            VALUES (%(event_key)s, %(camera_id)s, %(session_id)s, %(vehicle_id)s,
                    %(vehicle_class)s, %(color)s, %(direction)s, %(line_id)s,
                    %(occurred_at)s, %(source)s)
            ON CONFLICT (event_key) DO NOTHING
            RETURNING id, event_key, camera_id, session_id, vehicle_id,
                      vehicle_class, color, direction, line_id, occurred_at, source
            """,
            event,
        ).fetchone()
        return dict(row) if row else None


async def simulate_events(app: FastAPI) -> None:
    """Demo-only event source so API, PostgreSQL, and UI can be shown together."""
    rng = random.Random()
    while True:
        await asyncio.sleep(3)
        event = {
            "event_key": str(uuid.uuid4()),
            "camera_id": "camera-01",
            "session_id": DEMO_SESSION,
            "vehicle_id": f"demo-{uuid.uuid4().hex[:8]}",
            "vehicle_class": rng.choice(VEHICLE_CLASSES),
            "color": rng.choice(COLORS),
            "direction": rng.choice(["in", "out"]),
            "line_id": "gate-01",
            "occurred_at": datetime.now(timezone.utc),
            "source": "simulated",
        }
        try:
            saved = await asyncio.to_thread(record_event, event)
            if saved:
                await app.state.new_events.put(saved)
        except Exception as exc:
            app.state.last_db_error = str(exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.new_events = asyncio.Queue(maxsize=100)
    app.state.last_db_error = None
    try:
        await asyncio.to_thread(initialize_database)
        app.state.db_ready = True
    except Exception as exc:
        app.state.db_ready = False
        app.state.last_db_error = str(exc)
    task = asyncio.create_task(simulate_events(app)) if DEMO_MODE else None
    try:
        yield
    finally:
        if task:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


app = FastAPI(title="Vehicle Analytics Prototype API", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "http://localhost:5173").split(","),
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, Any]:
    try:
        with connect() as conn:
            conn.execute("SELECT 1")
        return {"status": "ok", "database": "connected", "mode": "demo" if DEMO_MODE else "inference"}
    except Exception as exc:
        raise HTTPException(503, detail={"status": "degraded", "database": str(exc)}) from exc


@app.get("/api/metrics")
def metrics() -> dict[str, Any]:
    with connect() as conn:
        total = conn.execute("SELECT count(*) AS total FROM crossing_events").fetchone()["total"]
        colors = conn.execute(
            "SELECT color, count(*) AS count FROM crossing_events GROUP BY color ORDER BY color"
        ).fetchall()
        directions = conn.execute(
            "SELECT direction, count(*) AS count FROM crossing_events GROUP BY direction"
        ).fetchall()
    return {"total": total, "by_color": colors, "by_direction": directions,
            "mode": "demo" if DEMO_MODE else "inference"}


@app.get("/api/events")
def events(limit: int = Query(default=50, ge=1, le=200)) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM crossing_events ORDER BY occurred_at DESC, id DESC LIMIT %s", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]


@app.get("/api/events/stream")
async def event_stream():
    async def stream():
        queue: asyncio.Queue = app.state.new_events
        heartbeat_at = time.monotonic()
        while True:
            try:
                event = await asyncio.wait_for(queue.get(), timeout=1)
                yield f"data: {json.dumps(event, default=str)}\n\n"
            except asyncio.TimeoutError:
                if time.monotonic() - heartbeat_at >= 15:
                    yield ": keep-alive\n\n"
                    heartbeat_at = time.monotonic()

    return StreamingResponse(stream(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
