"""E2 / F7 / F18: spatio-temporal gaze vs a 3x3 virtual audience grid.

Per-frame gaze angles come from L2CS-Net (models/gaze/, Gaze360 weights) when present,
otherwise from Face Mesh head pose + iris offset. Angles are calibrated against the first
seconds of the recording (the recorder asks the speaker to look at the camera), mapped to a
grid cell, smoothed (trained GRU if models/gaze/gaze_gru.pt exists, else a mode filter), and
summarised as dwell time, coverage and neglected zones.

Cells are indexed row-major from the speaker's viewpoint: 0 = back-left ... 8 = front-right.
Column "left" means the speaker's left. -1 = off-grid (looking away / at notes).
"""
from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .base import Analyzer, AnalyzerSkipped, SessionContext
from .face import FaceTrack, crop_face
from .ingest import iter_frames
from .settings import get_settings
from .utils import configure_torch, resolve_device, rnd, torch_available

CELL_NAMES = ["back-left", "back-centre", "back-right", "mid-left", "centre", "mid-right",
              "front-left", "front-centre", "front-right"]
CENTER = 4
YAW_EDGE_DEG, PITCH_EDGE_DEG = 8.0, 6.0
OFF_YAW_DEG, OFF_PITCH_DEG = 45.0, 35.0
NEGLECT_DWELL = 0.02
COVERED_DWELL = 0.04
DEFAULT_CALIBRATION_S = 3.0
MODE_WINDOW = 5


def angles_to_cell(yaw: float, pitch: float) -> int:
    if np.isnan(yaw) or np.isnan(pitch) or abs(yaw) > OFF_YAW_DEG or abs(pitch) > OFF_PITCH_DEG:
        return -1
    col = 0 if yaw > YAW_EDGE_DEG else 2 if yaw < -YAW_EDGE_DEG else 1
    row = 0 if pitch > PITCH_EDGE_DEG else 2 if pitch < -PITCH_EDGE_DEG else 1
    return row * 3 + col


def calibrate(t: np.ndarray, yaw: np.ndarray, pitch: np.ndarray, calibration_s: float | None,
              yaw_absolute: bool) -> tuple[float, float, str]:
    """Offsets so that looking at the camera maps to the centre cell."""
    if calibration_s:
        mask = (t <= calibration_s) & ~np.isnan(yaw) & ~np.isnan(pitch)
        if mask.sum() >= 5:
            return float(np.median(yaw[mask])), float(np.median(pitch[mask])), "camera-window"
    yaw0 = 0.0 if yaw_absolute else float(np.nanmedian(yaw))
    return yaw0, float(np.nanmedian(pitch)), "session-median"


def mode_filter(cells: list[int], k: int = MODE_WINDOW) -> list[int]:
    half = k // 2
    out = []
    for i in range(len(cells)):
        window = [c for c in cells[max(0, i - half):i + half + 1]]
        out.append(Counter(window).most_common(1)[0][0])
    return out


def fixations(t: list[float], cells: list[int], min_s: float = 0.3) -> list[dict[str, Any]]:
    runs, start = [], 0
    for i in range(1, len(cells) + 1):
        if i == len(cells) or cells[i] != cells[start]:
            dur = (t[i - 1] - t[start]) + (t[1] - t[0] if len(t) > 1 else 0.1)
            if cells[start] >= 0 and dur >= min_s:
                runs.append({"cell": cells[start], "start": rnd(t[start], 2), "duration": rnd(dur, 2)})
            start = i
    return runs


def grid_summary(t: list[float], cells: list[int], duration: float) -> dict[str, Any]:
    on = [c for c in cells if c >= 0]
    n = max(len(cells), 1)
    dwell = [on.count(i) / n for i in range(9)]
    probs = np.array([d for d in dwell if d > 0])
    entropy = float(-(probs / probs.sum() * np.log(probs / probs.sum())).sum() / np.log(9)) if probs.size else 0.0
    neglected = [i for i in range(9) if dwell[i] < NEGLECT_DWELL]
    fix = fixations(t, cells)
    return {
        "dwell": [rnd(d, 4) for d in dwell],
        "dwell_seconds": [rnd(d * duration, 1) for d in dwell],
        "coverage": rnd(sum(d >= COVERED_DWELL for d in dwell) / 9, 3),
        "eye_contact_ratio": rnd(dwell[CENTER], 3),
        "on_grid_ratio": rnd(len(on) / n, 3),
        "off_grid_ratio": rnd(1 - len(on) / n, 3),
        "entropy": rnd(entropy, 3),
        "neglected_cells": neglected,
        "fixation_count": len(fix),
        "mean_fixation_s": rnd(np.mean([f["duration"] for f in fix]), 2) if fix else None,
    }


def neglect_alerts(t: list[float], cells: list[int], neglected: list[int], duration: float) -> list[dict[str, Any]]:
    alerts = []
    for cell in neglected:
        visits = [ti for ti, c in zip(t, cells) if c == cell]
        alerts.append({
            "cell": cell, "name": CELL_NAMES[cell],
            "message": f"You almost never looked at the {CELL_NAMES[cell]} of the room"
                       + (f" (last glance at {visits[-1]:.0f}s)." if visits else f" in {duration:.0f}s."),
        })
    return alerts


def per_second_cells(t: list[float], cells: list[int], duration: float) -> list[int | None]:
    out: list[int | None] = []
    for sec in range(int(np.ceil(duration))):
        window = [c for ti, c in zip(t, cells) if sec <= ti < sec + 1]
        out.append(Counter(window).most_common(1)[0][0] if window else None)
    return out


# --------------------------------------------------------------------------- learned parts
def _model_path(name: str) -> Path:
    return get_settings().models_dir / "gaze" / name


# py-feat's safetensors re-pack of the Gaze360 weights (unchanged values) or the original pickle.
L2CS_FILES = ("l2cs_gaze360_resnet50.safetensors", "L2CSNet_gaze360.pkl")
L2CS_INPUT = 224


def l2cs_path() -> Path | None:
    return next((p for p in map(_model_path, L2CS_FILES) if p.exists()), None)


def _l2cs_state(path: Path) -> dict[str, Any]:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        import torch

        state = torch.load(str(path), map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
    prefixes = ("module.", "model.", "backbone.")
    clean = {}
    for key, value in state.items():
        for prefix in prefixes:
            key = key.removeprefix(prefix)
        clean[key] = value
    return clean


@lru_cache(maxsize=1)
def load_l2cs(path: str, device: str):
    from .nets import L2CS

    model = L2CS()
    missing, _ = model.load_state_dict(_l2cs_state(Path(path)), strict=False)
    if missing:
        raise RuntimeError(f"L2CS checkpoint is missing {len(missing)} weights, e.g. {missing[:3]}")
    return model.to(device).eval()


L2CS_FPS = 3.0
L2CS_BATCH = 16


def l2cs_angles(ctx: SessionContext, face: dict[str, Any], t_target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Batched L2CS on face crops at 5 fps, interpolated onto the face timeline (NaN across >0.5 s gaps)."""
    import torch

    configure_torch()
    device = resolve_device()
    model = load_l2cs(str(l2cs_path()), device)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    track = FaceTrack.from_face(face)
    ts: list[float] = []
    yaws: list[float] = []
    pitches: list[float] = []
    batch_t: list[float] = []
    batch: list[np.ndarray] = []

    def flush() -> None:
        x = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).to(device)
        with torch.no_grad():
            yaw_l, pitch_l = model(x)
        ts.extend(batch_t)
        yaws.extend(model.to_degrees(yaw_l).cpu().tolist())
        pitches.extend(model.to_degrees(pitch_l).cpu().tolist())
        batch.clear()
        batch_t.clear()

    for _, t, frame in iter_frames(ctx.video_path, L2CS_FPS):
        box = track.at(t)
        if box is None:
            continue
        batch.append((crop_face(frame, box, L2CS_INPUT, 1.0).astype(np.float32) / 255 - mean) / std)
        batch_t.append(t)
        if len(batch) == L2CS_BATCH:
            flush()
    if batch:
        flush()
    if len(ts) < 2:
        raise RuntimeError("too few face crops for L2CS")
    src = np.asarray(ts)
    near = np.min(np.abs(t_target[:, None] - src[None, :]), axis=1) <= 0.5
    out_yaw = np.where(near, np.interp(t_target, src, yaws), np.nan)
    out_pitch = np.where(near, np.interp(t_target, src, pitches), np.nan)
    return out_yaw, out_pitch


def align_sign(raw: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """L2CS sign conventions differ from ours; flip if it anti-correlates with head pose."""
    mask = ~np.isnan(raw) & ~np.isnan(reference)
    if mask.sum() > 10 and np.corrcoef(raw[mask], reference[mask])[0, 1] < 0:
        return -raw
    return raw


@lru_cache(maxsize=1)
def load_smoother(path: str):
    import torch

    from .nets import GazeSmoother

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = GazeSmoother(**ckpt.get("config", {}))
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()


def smoother_features(yaw_rel: np.ndarray, pitch_rel: np.ndarray, head_yaw: np.ndarray,
                      head_pitch: np.ndarray) -> np.ndarray:
    feats = np.stack([yaw_rel, pitch_rel, head_yaw, head_pitch], axis=1) / 30.0
    return np.nan_to_num(feats.astype(np.float32))


def gru_cells(features: np.ndarray, raw_cells: list[int]) -> list[int]:
    import torch

    model = load_smoother(str(_model_path("gaze_gru.pt")))
    with torch.no_grad():
        logits = model(torch.from_numpy(features)[None])[0]
    cells = logits.argmax(dim=-1).tolist()
    return [-1 if raw < 0 else c for raw, c in zip(raw_cells, cells)]


def _arr(frames: dict[str, list], key: str) -> np.ndarray:
    return np.asarray([np.nan if v is None else v for v in frames[key]], dtype=float)


class GazeAnalyzer(Analyzer):
    name = "gaze"
    requires = ("ingest", "face")

    def analyze(self, ctx: SessionContext) -> dict[str, Any]:
        ingest = self.need(ctx, "ingest")
        face = self.need(ctx, "face")
        frames = face["frames"]
        duration = float(ingest["duration_s"])
        t = _arr(frames, "t")
        head_yaw, head_pitch = _arr(frames, "yaw"), _arr(frames, "pitch")
        if np.isnan(head_yaw).all():
            raise AnalyzerSkipped("face backend gave no landmarks; gaze needs Face Mesh")

        backend = "facemesh-headpose+iris"
        yaw, pitch = _arr(frames, "gaze_yaw"), _arr(frames, "gaze_pitch")
        if l2cs_path() is not None and torch_available():
            try:
                l_yaw, l_pitch = l2cs_angles(ctx, face, t)
                yaw, pitch = align_sign(l_yaw, head_yaw), align_sign(l_pitch, head_pitch)
                backend = "l2cs-gaze360"
            except Exception:  # noqa: BLE001 - fall back to landmark gaze
                pass

        meta = _read_meta(ctx)
        yaw0, pitch0, calib = calibrate(t, yaw, pitch, meta.get("calibration_s", None),
                                        yaw_absolute=backend.startswith("facemesh"))
        yaw_rel, pitch_rel = yaw - yaw0, pitch - pitch0
        raw_cells = [angles_to_cell(y, p) for y, p in zip(yaw_rel, pitch_rel)]

        smoother = "mode-filter"
        cells = mode_filter(raw_cells)
        if _model_path("gaze_gru.pt").exists() and torch_available():
            try:
                cells = gru_cells(smoother_features(yaw_rel, pitch_rel, head_yaw - yaw0, head_pitch - pitch0), raw_cells)
                smoother = "gru"
            except Exception:  # noqa: BLE001
                pass

        t_list = t.tolist()
        summary = grid_summary(t_list, cells, duration)
        return {
            "summary": summary,
            "alerts": neglect_alerts(t_list, cells, summary["neglected_cells"], duration),
            "cells": cells,
            "raw_cells": raw_cells,
            "t": [rnd(v, 2) for v in t_list],
            "yaw_rel": [rnd(v, 2) for v in yaw_rel],
            "pitch_rel": [rnd(v, 2) for v in pitch_rel],
            "per_second": per_second_cells(t_list, cells, duration),
            "calibration": {"method": calib, "yaw0": rnd(yaw0, 2), "pitch0": rnd(pitch0, 2)},
            "cell_names": CELL_NAMES,
            "models": {"gaze": backend, "gaze_smoother": smoother},
        }


def _read_meta(ctx: SessionContext) -> dict[str, Any]:
    import json

    path = ctx.path("meta.json")
    meta = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    meta.setdefault("calibration_s", DEFAULT_CALIBRATION_S if meta.get("source") == "webcam" else None)
    return meta
