"""Face landmarks (MediaPipe Face Mesh), F8 blink rate, head pose, iris offset and a
face-crop stream other analysers reuse (physio, face V-A, gaze).

Head pose and gaze are geometric proxies from landmarks so their signs are well defined:
yaw > 0  = speaker looks to *their* left (image right in the raw, non-mirrored video),
pitch > 0 = speaker looks up.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from .base import Analyzer, AnalyzerSkipped, SessionContext
from .ingest import iter_frames
from .settings import get_settings
from .utils import nanmean, nanstd, rnd

LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
NOSE_TIP, CHIN = 1, 152
MOUTH_L, MOUTH_R, LIP_TOP, LIP_BOTTOM = 61, 291, 13, 14
BROW_R, BROW_L, LID_TOP_R, LID_TOP_L = 105, 334, 159, 386
LID_BOT_R, LID_BOT_L = 145, 374
IRIS_R, IRIS_L = 468, 473
# Nose tip sits ~0.30 inter-ocular distances in front of the eye plane (generic head model).
NOSE_DEPTH_RATIO = 0.30
EYE_YAW_GAIN_DEG = 120.0
EYE_PITCH_GAIN_DEG = 80.0

COLUMNS = ("t", "detected", "bbox", "ear", "yaw", "pitch", "roll", "iris_dx", "iris_dy",
           "gaze_yaw", "gaze_pitch", "smile", "corner_lift", "mouth_open", "brow")


def eye_aspect_ratio(pts: np.ndarray) -> float:
    """EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|) for six eye landmarks."""
    p1, p2, p3, p4, p5, p6 = pts
    horizontal = np.linalg.norm(p1 - p4)
    if horizontal < 1e-6:
        return float("nan")
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * horizontal))


def detect_blinks(t: np.ndarray, ear: np.ndarray, ratio: float = 0.75, min_gap_s: float = 0.25,
                  max_closed_s: float = 0.5, baseline_s: float = 2.0) -> list[float]:
    """Blink onsets: EAR dips below `ratio` x its local (2 s rolling median) open-eye level and recovers
    within 0.5 s. Looking down also lowers EAR, but slowly and for longer, so it is not counted."""
    valid = ~np.isnan(ear)
    if valid.sum() < 10:
        return []
    baseline = np.array([np.nanmedian(ear[(t >= ti - baseline_s / 2) & (t <= ti + baseline_s / 2)]) for ti in t])
    blinks: list[float] = []
    onset: float | None = None
    for ti, e, b in zip(t, ear, baseline):
        if np.isnan(e) or np.isnan(b):
            continue
        if e < b * ratio:
            onset = float(ti) if onset is None else onset
        elif onset is not None:
            if ti - onset <= max_closed_s and (not blinks or onset - blinks[-1] >= min_gap_s):
                blinks.append(onset)
            onset = None
    return blinks


def landmark_metrics(lm: np.ndarray) -> dict[str, float]:
    """Pose/expression proxies from 478 pixel-space landmarks (x, y)."""
    eye_r_outer, eye_l_outer = lm[33], lm[263]
    iod = float(np.linalg.norm(eye_l_outer - eye_r_outer)) or 1.0
    mid_eye = (eye_r_outer + eye_l_outer) / 2
    nose = lm[NOSE_TIP]
    yaw = math.degrees(math.asin(np.clip((nose[0] - mid_eye[0]) / iod / NOSE_DEPTH_RATIO, -1, 1)))
    # Raw vertical nose offset; gaze.py calibrates it against the session median.
    pitch = -math.degrees(math.asin(np.clip((nose[1] - mid_eye[1]) / iod / 1.5 - 0.25, -1, 1)))
    roll = math.degrees(math.atan2(eye_l_outer[1] - eye_r_outer[1], eye_l_outer[0] - eye_r_outer[0]))

    metrics = {
        "ear": (eye_aspect_ratio(lm[LEFT_EYE]) + eye_aspect_ratio(lm[RIGHT_EYE])) / 2,
        "yaw": yaw, "pitch": pitch, "roll": roll,
        "smile": float(np.linalg.norm(lm[MOUTH_R] - lm[MOUTH_L]) / iod),
        "corner_lift": float(((lm[LIP_TOP][1] + lm[LIP_BOTTOM][1]) / 2 - (lm[MOUTH_L][1] + lm[MOUTH_R][1]) / 2) / iod),
        "mouth_open": float(np.linalg.norm(lm[LIP_BOTTOM] - lm[LIP_TOP]) / iod),
        "brow": float(((lm[LID_TOP_R][1] - lm[BROW_R][1]) + (lm[LID_TOP_L][1] - lm[BROW_L][1])) / 2 / iod),
        "iris_dx": float("nan"), "iris_dy": float("nan"),
    }
    if lm.shape[0] > IRIS_L:
        dx, dy = [], []
        for iris, a, b, top, bot in ((IRIS_R, 33, 133, LID_TOP_R, LID_BOT_R), (IRIS_L, 362, 263, LID_TOP_L, LID_BOT_L)):
            x0, x1 = sorted((lm[a][0], lm[b][0]))
            if x1 - x0 > 1e-6:
                dx.append((lm[iris][0] - x0) / (x1 - x0) - 0.5)
            if lm[bot][1] - lm[top][1] > 1e-6:
                dy.append(0.5 - (lm[iris][1] - lm[top][1]) / (lm[bot][1] - lm[top][1]))
        metrics["iris_dx"] = float(np.mean(dx)) if dx else float("nan")
        metrics["iris_dy"] = float(np.mean(dy)) if dy else float("nan")
    return metrics


class _MeshBackend:
    name = "mediapipe-facemesh"

    def __init__(self) -> None:
        import mediapipe as mp

        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def landmarks(self, frame_bgr: np.ndarray) -> np.ndarray | None:
        import cv2

        h, w = frame_bgr.shape[:2]
        res = self._mesh.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        if not res.multi_face_landmarks:
            return None
        pts = res.multi_face_landmarks[0].landmark
        return np.array([[p.x * w, p.y * h] for p in pts], dtype=float)

    def close(self) -> None:
        self._mesh.close()


class _HaarBackend:
    """Bounding boxes only; used when MediaPipe is not installed."""
    name = "opencv-haar"

    def __init__(self) -> None:
        import cv2

        self._det = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    def bbox(self, frame_bgr: np.ndarray) -> list[float] | None:
        import cv2

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._det.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return [float(x), float(y), float(w), float(h)]

    def close(self) -> None:
        pass


def make_backend():
    try:
        return _MeshBackend()
    except (ImportError, AttributeError):
        return _HaarBackend()


def _empty_row(t: float) -> dict[str, Any]:
    row = {c: None for c in COLUMNS}
    row.update(t=round(t, 3), detected=False)
    return row


def analyze_frame(backend, frame: np.ndarray, t: float) -> dict[str, Any]:
    row = _empty_row(t)
    if isinstance(backend, _HaarBackend):
        box = backend.bbox(frame)
        if box:
            row.update(detected=True, bbox=[round(v, 1) for v in box])
        return row
    lm = backend.landmarks(frame)
    if lm is None:
        return row
    x0, y0 = lm.min(axis=0)
    x1, y1 = lm.max(axis=0)
    m = landmark_metrics(lm)
    row.update(detected=True, bbox=[round(float(x0), 1), round(float(y0), 1),
                                    round(float(x1 - x0), 1), round(float(y1 - y0), 1)])
    row.update({k: rnd(v, 4) for k, v in m.items()})
    if row["iris_dx"] is not None:
        row["gaze_yaw"] = rnd(m["yaw"] + EYE_YAW_GAIN_DEG * m["iris_dx"], 2)
        row["gaze_pitch"] = rnd(m["pitch"] + EYE_PITCH_GAIN_DEG * m["iris_dy"], 2)
    else:
        row["gaze_yaw"], row["gaze_pitch"] = rnd(m["yaw"], 2), rnd(m["pitch"], 2)
    return row


# ---------- face crop stream (shared with physio / affect / gaze) ----------

@dataclass
class FaceTrack:
    t: np.ndarray
    boxes: np.ndarray  # (n, 4) x, y, w, h with NaN where missing
    max_gap_s: float = 1.0

    @classmethod
    def from_face(cls, face: dict[str, Any]) -> "FaceTrack":
        t = np.asarray(face["frames"]["t"], dtype=float)
        boxes = np.array([b if b else [np.nan] * 4 for b in face["frames"]["bbox"]], dtype=float)
        return cls(t=t, boxes=boxes)

    def at(self, t: float) -> np.ndarray | None:
        valid = ~np.isnan(self.boxes[:, 0])
        if not valid.any():
            return None
        tv = self.t[valid]
        if np.min(np.abs(tv - t)) > self.max_gap_s:
            return None
        return np.array([np.interp(t, tv, self.boxes[valid, k]) for k in range(4)])


def crop_face(frame_bgr: np.ndarray, box: np.ndarray, size: int, scale: float = 1.2,
              rgb: bool = True) -> np.ndarray:
    import cv2

    h, w = frame_bgr.shape[:2]
    cx, cy = box[0] + box[2] / 2, box[1] + box[3] / 2
    half = max(box[2], box[3]) * scale / 2
    x0, y0 = int(max(cx - half, 0)), int(max(cy - half, 0))
    x1, y1 = int(min(cx + half, w)), int(min(cy + half, h))
    crop = frame_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        crop = frame_bgr
    crop = cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(crop, cv2.COLOR_BGR2RGB) if rgb else crop


def iter_face_crops(ctx: SessionContext, face: dict[str, Any], fps: float | None = None,
                    size: int = 72, scale: float = 1.2) -> Iterator[tuple[float, np.ndarray | None]]:
    """Yield (t, RGB crop or None) for every sampled frame of the normalised video."""
    track = FaceTrack.from_face(face)
    for _, t, frame in iter_frames(ctx.video_path, fps):
        box = track.at(t)
        yield t, (crop_face(frame, box, size, scale) if box is not None else None)


class FaceAnalyzer(Analyzer):
    name = "face"
    requires = ("ingest",)

    def analyze(self, ctx: SessionContext) -> dict[str, Any]:
        ingest = self.need(ctx, "ingest")
        duration = float(ingest["duration_s"])
        backend = make_backend()
        rows: list[dict[str, Any]] = []
        try:
            for _, t, frame in iter_frames(ctx.video_path, get_settings().analysis_fps):
                rows.append(analyze_frame(backend, frame, t))
        finally:
            backend.close()
        detected = sum(r["detected"] for r in rows)
        if not rows or detected == 0:
            raise AnalyzerSkipped("no face detected in the video")

        frames = {c: [r[c] for r in rows] for c in COLUMNS}
        t = np.asarray(frames["t"], dtype=float)
        ear = np.asarray([np.nan if e is None else e for e in frames["ear"]], dtype=float)
        blinks = detect_blinks(t, ear)
        yaw = [v for v in frames["yaw"] if v is not None]
        expr = [np.nanstd([v for v in frames[k] if v is not None] or [np.nan])
                for k in ("smile", "corner_lift", "brow", "mouth_open")]
        summary = {
            "detection_rate": rnd(detected / len(rows), 3),
            "blink_count": len(blinks),
            "blink_rate_per_min": rnd(len(blinks) / max(duration / 60, 1e-6), 2) if backend.name != "opencv-haar" else None,
            "head_motion_deg": rnd(nanstd(np.diff(yaw).tolist()) if len(yaw) > 2 else None, 3),
            "yaw_std_deg": rnd(nanstd(yaw), 2),
            "landmark_expressiveness": rnd(nanmean(expr), 4),
        }
        return {"summary": summary, "blinks": [rnd(b) for b in blinks], "frames": frames,
                "fps": get_settings().analysis_fps, "models": {"face": backend.name}}
