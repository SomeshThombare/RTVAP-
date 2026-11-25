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

CREATE INDEX IF NOT EXISTS crossing_events_time_idx
    ON crossing_events (occurred_at DESC);

CREATE INDEX IF NOT EXISTS crossing_events_color_time_idx
    ON crossing_events (color, occurred_at DESC);
