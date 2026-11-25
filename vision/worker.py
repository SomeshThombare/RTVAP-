"""Windows-friendly real-video vehicle analytics prototype.

Uses an OpenCV Zoo YOLOX ONNX detector, ONNX Runtime DirectML when available,
a lightweight IoU tracker, and a heuristic crop-color estimate. This is an
alternative prototype and is not NVIDIA DeepStream.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import queue
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
COCO_VEHICLES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
COCO_CLASSES = 80
INPUT_SIZE = 640
STRIDES = (8, 16, 32)
LOG = logging.getLogger("vehicle-vision")


@dataclass
class Detection:
    box: tuple[int, int, int, int]
    confidence: float
    vehicle_class: str
    color: str


@dataclass
class Track:
    track_id: int
    box: tuple[int, int, int, int]
    vehicle_class: str
    color: str
    confidence: float
    age: int = 0
    previous_point: tuple[float, float] | None = None
    crossed: bool = False


def box_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    x1, y1 = max(ax1, bx1), max(ay1, by1)
    x2, y2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return intersection / max(1, area_a + area_b - intersection)


def estimate_color(frame: np.ndarray, box: tuple[int, int, int, int]) -> str:
    """Estimate a coarse body color from the center of the vehicle crop."""
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    if width < 3 or height < 3:
        return "unknown"
    # Avoid crop edges and road pixels below the vehicle body.
    crop = frame[y1 + int(height * 0.15): y1 + int(height * 0.75),
                 x1 + int(width * 0.18): x1 + int(width * 0.82)]
    if crop.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)
    saturation = pixels[:, 1]
    value = pixels[:, 2]
    chromatic = (saturation >= 55) & (value >= 35) & (value <= 245)
    if int(chromatic.sum()) >= max(8, int(len(pixels) * 0.08)):
        hue = pixels[chromatic, 0]
        bins = np.array([0, 10, 22, 35, 78, 100, 130, 160, 180])
        labels = ["red", "orange", "yellow", "green", "green", "blue", "purple", "red"]
        counts, _ = np.histogram(hue, bins=bins)
        return labels[int(np.argmax(counts))]
    bright = value[value > 15]
    if bright.size == 0:
        return "black"
    median_value = float(np.median(bright))
    if median_value < 58:
        return "black"
    if median_value > 205:
        return "white"
    if median_value > 145:
        return "silver"
    return "gray"


class VehicleDetector:
    def __init__(self, model_path: Path, device_id: int, confidence: float = 0.32):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found: {model_path}. Download the OpenCV Zoo YOLOX model first."
            )
        self.model_path = model_path
        self.confidence = confidence
        self.scale = 1.0
        self.mode = "CPU"
        providers = ort.get_available_providers()
        if "DmlExecutionProvider" in providers:
            try:
                options = ort.SessionOptions()
                options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                options.enable_mem_pattern = False
                self.session = ort.InferenceSession(
                    str(model_path), options,
                    providers=[("DmlExecutionProvider", {"device_id": str(device_id)}),
                               "CPUExecutionProvider"],
                )
                self.mode = f"DirectML adapter {device_id}"
            except Exception as exc:
                LOG.warning("DirectML adapter %s unavailable (%s); using CPU", device_id, exc)
                self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        else:
            LOG.warning("DirectML provider unavailable; using CPU")
            self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.grids, self.expanded_strides = self._make_grids()
        LOG.info("Detector ready: %s (%s)", model_path.name, self.mode)

    @staticmethod
    def _make_grids() -> tuple[np.ndarray, np.ndarray]:
        grids, expanded = [], []
        for stride in STRIDES:
            side = INPUT_SIZE // stride
            xv, yv = np.meshgrid(np.arange(side), np.arange(side))
            grids.append(np.stack((xv, yv), axis=2).reshape(1, -1, 2))
            expanded.append(np.full((1, side * side, 1), stride, dtype=np.float32))
        return np.concatenate(grids, axis=1).astype(np.float32), np.concatenate(expanded, axis=1)

    def infer(self, frame: np.ndarray) -> list[Detection]:
        height, width = frame.shape[:2]
        self.scale = min(INPUT_SIZE / height, INPUT_SIZE / width)
        resized = cv2.resize(frame, (int(width * self.scale), int(height * self.scale)))
        padded = np.full((INPUT_SIZE, INPUT_SIZE, 3), 114, dtype=np.uint8)
        padded[:resized.shape[0], :resized.shape[1]] = resized
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32)
        blob = np.transpose(rgb, (2, 0, 1))[None, ...]
        raw = self.session.run([self.output_name], {self.input_name: blob})[0][0].copy()
        raw[:, :2] = (raw[:, :2] + self.grids[0]) * self.expanded_strides[0]
        raw[:, 2:4] = np.exp(np.clip(raw[:, 2:4], -10, 10)) * self.expanded_strides[0]
        class_scores = raw[:, 4:5] * raw[:, 5:]
        class_ids = class_scores.argmax(axis=1)
        scores = class_scores[np.arange(class_scores.shape[0]), class_ids]
        keep = (scores >= self.confidence) & np.isin(class_ids, list(COCO_VEHICLES))
        indices = np.flatnonzero(keep)
        if not len(indices):
            return []
        boxes: list[tuple[int, int, int, int]] = []
        confidences: list[float] = []
        classes: list[int] = []
        for idx in indices:
            cx, cy, bw, bh = raw[idx, :4]
            x1 = int((cx - bw / 2) / self.scale)
            y1 = int((cy - bh / 2) / self.scale)
            x2 = int((cx + bw / 2) / self.scale)
            y2 = int((cy + bh / 2) / self.scale)
            boxes.append((max(0, x1), max(0, y1), min(width, x2), min(height, y2)))
            confidences.append(float(scores[idx]))
            classes.append(int(class_ids[idx]))
        selected: list[Detection] = []
        for class_id in sorted(set(classes)):
            members = [i for i, cid in enumerate(classes) if cid == class_id]
            order = sorted(members, key=lambda i: confidences[i], reverse=True)
            while order:
                best = order.pop(0)
                x1, y1, x2, y2 = boxes[best]
                if x2 > x1 and y2 > y1:
                    selected.append(Detection(boxes[best], confidences[best], COCO_VEHICLES[class_id], "unknown"))
                order = [i for i in order if box_iou(boxes[best], boxes[i]) < 0.45]
        return selected


class IoUTracker:
    """Small greedy IoU tracker for a single offline video stream."""
    def __init__(self, max_age: int = 12, match_iou: float = 0.18):
        self.max_age = max_age
        self.match_iou = match_iou
        self.next_id = 1
        self.tracks: dict[int, Track] = {}

    def update(self, detections: list[Detection]) -> list[Track]:
        for track in self.tracks.values():
            track.age += 1
        pairs = sorted(
            ((box_iou(track.box, det.box), track_id, idx)
             for track_id, track in self.tracks.items()
             for idx, det in enumerate(detections)
             if track.vehicle_class == det.vehicle_class),
            reverse=True,
        )
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        for overlap, track_id, idx in pairs:
            if overlap < self.match_iou or track_id in matched_tracks or idx in matched_dets:
                continue
            track = self.tracks[track_id]
            det = detections[idx]
            track.box = det.box
            track.confidence = det.confidence
            track.color = det.color
            track.age = 0
            matched_tracks.add(track_id)
            matched_dets.add(idx)
        for idx, det in enumerate(detections):
            if idx in matched_dets:
                continue
            track_id = self.next_id
            self.next_id += 1
            self.tracks[track_id] = Track(track_id, det.box, det.vehicle_class, det.color, det.confidence)
            matched_tracks.add(track_id)
        self.tracks = {tid: track for tid, track in self.tracks.items() if track.age <= self.max_age}
        return [self.tracks[tid] for tid in matched_tracks if tid in self.tracks]


class EventPoster:
    def __init__(self, api_url: str):
        self.api_url = api_url.rstrip("/") + "/api/events"
        self.events: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self.thread = threading.Thread(target=self._run, name="event-writer", daemon=True)
        self.thread.start()

    def submit(self, event: dict[str, Any]) -> None:
        self.events.put(event)

    def _run(self) -> None:
        while True:
            event = self.events.get()
            try:
                if event is None:
                    return
                body = json.dumps(event).encode("utf-8")
                for attempt in range(3):
                    request = urllib.request.Request(
                        self.api_url, body, {"Content-Type": "application/json"}, method="POST"
                    )
                    try:
                        with urllib.request.urlopen(request, timeout=5) as response:
                            if response.status < 300:
                                LOG.info("Crossing saved: vehicle=%s direction=%s color=%s",
                                         event["vehicle_id"], event["direction"], event["color"])
                            break
                    except (urllib.error.URLError, TimeoutError) as exc:
                        if attempt == 2:
                            LOG.error("Could not submit crossing event: %s", exc)
                        else:
                            time.sleep(1 + attempt)
            finally:
                self.events.task_done()

    def close(self) -> None:
        self.events.join()
        self.events.put(None)
        self.thread.join(timeout=5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect, track, and count vehicles in an MP4.")
    parser.add_argument("--video", type=Path, default=ROOT / "mixkit-cars-driving-by-on-road-2022-hd-ready.mp4")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "object_detection_yolox_2022nov_int8bq.onnx")
    parser.add_argument("--api", default=os.getenv("API_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--camera-id", default="camera-01")
    parser.add_argument("--line-orientation", choices=("vertical", "horizontal"), default="vertical")
    parser.add_argument("--line-ratio", type=float, default=0.5)
    parser.add_argument("--device-id", type=int, default=1, help="DirectML adapter index; GPU 1 is the GTX 1650 on this laptop")
    parser.add_argument("--confidence", type=float, default=0.32)
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap; 0 processes the whole clip")
    parser.add_argument("--output", type=Path, default=ROOT / "output" / "opencv_annotated.mp4")
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if not 0.05 <= args.line_ratio <= 0.95:
        raise SystemExit("--line-ratio must be between 0.05 and 0.95")
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise SystemExit(f"Cannot open video: {args.video}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width < 1 or height < 1:
        raise SystemExit("Video has invalid dimensions")
    detector = VehicleDetector(args.model, args.device_id, args.confidence)
    tracker = IoUTracker()
    session_id = str(uuid.uuid4())
    video_start = datetime.now(timezone.utc)
    line = int((width if args.line_orientation == "vertical" else height) * args.line_ratio)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit(f"Cannot create annotated video: {args.output}")
    poster = EventPoster(args.api)
    frame_index = 0
    crossings = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            detections = detector.infer(frame)
            for det in detections:
                det.color = estimate_color(frame, det.box)
            tracks = tracker.update(detections)
            for track in tracks:
                x1, y1, x2, y2 = track.box
                point = ((x1 + x2) / 2, (y1 + y2) / 2)
                coordinate = point[0] if args.line_orientation == "vertical" else point[1]
                old_coordinate = (track.previous_point[0] if args.line_orientation == "vertical"
                                  else track.previous_point[1]) if track.previous_point else None
                if old_coordinate is not None and not track.crossed:
                    crossed_forward = old_coordinate < line <= coordinate
                    crossed_backward = coordinate < line <= old_coordinate
                    if crossed_forward or crossed_backward:
                        direction = "in" if crossed_forward else "out"
                        event_number = 1
                        event_key = str(uuid.uuid5(uuid.UUID(session_id), f"{track.track_id}:{event_number}"))
                        event_time = video_start + timedelta(seconds=frame_index / fps)
                        poster.submit({
                            "event_key": event_key,
                            "camera_id": args.camera_id,
                            "session_id": session_id,
                            "vehicle_id": f"{session_id[:8]}-{track.track_id}",
                            "vehicle_class": track.vehicle_class,
                            "color": track.color,
                            "direction": direction,
                            "line_id": f"{args.line_orientation}-{args.line_ratio:.2f}",
                            "occurred_at": event_time.isoformat(),
                            "source": "opencv",
                        })
                        track.crossed = True
                        crossings += 1
                track.previous_point = point
                cv2.rectangle(frame, (x1, y1), (x2, y2), (72, 240, 130), 2)
                label = f"ID {track.track_id} {track.vehicle_class} {track.color} {track.confidence:.2f}"
                cv2.putText(frame, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                            0.48, (72, 240, 130), 1, cv2.LINE_AA)
            if args.line_orientation == "vertical":
                cv2.line(frame, (line, 0), (line, height), (20, 190, 255), 2)
            else:
                cv2.line(frame, (0, line), (width, line), (20, 190, 255), 2)
            cv2.putText(frame, f"OpenCV prototype | crossings: {crossings}", (14, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
            frame_index += 1
            if frame_index % 50 == 0:
                LOG.info("Processed %s/%s frames; crossings=%s", frame_index, total_frames or "?", crossings)
            if args.max_frames and frame_index >= args.max_frames:
                break
    finally:
        capture.release()
        writer.release()
        poster.close()
    LOG.info("Finished %s frames; %s crossing events submitted; annotated video: %s",
             frame_index, crossings, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
