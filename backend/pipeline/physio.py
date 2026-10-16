"""E1 / F17: webcam heart rate + HRV (rPPG) and a physiological-stress curve.

Primary model: pretrained PhysNet from rPPG-Toolbox (models/rppg/*.pth).
Fallback and evaluation baseline: the classical POS algorithm (Wang et al. 2017), plus CHROM.
Windows with a low signal-quality index (SQI) are marked unreliable instead of reported.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

from .base import Analyzer, AnalyzerSkipped, SessionContext
from .face import FaceTrack, crop_face
from .ingest import iter_frames
from .settings import get_settings
from .utils import configure_torch, nanmean, resolve_device, rnd, torch_available, window_starts

HR_BAND = (0.7, 3.0)  # 42-180 BPM
WINDOW_S, HOP_S = 10.0, 5.0
SQI_MIN = 0.35
AGREE_BPM = 8.0          # methods must agree within this to call a window reliable
EDGE_MARGIN_BPM = 5.0    # spectral peaks hugging the band edge are motion artefacts
RMSSD_PLAUSIBLE = (20.0, 150.0)
BASELINE_S = 20.0
PHYSNET_CHUNK = 128
DISCLAIMER = "Indicative only: webcam rPPG is not a medical measurement."


# --------------------------------------------------------------------------- signal extraction
def skin_rois(box: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Forehead + both cheeks inside a landmark bounding box (x, y, w, h)."""
    x, y, w, h = box
    rel = [(0.30, 0.06, 0.40, 0.14), (0.12, 0.48, 0.22, 0.16), (0.66, 0.48, 0.22, 0.16)]
    return [(int(x + rx * w), int(y + ry * h), max(int(rw * w), 2), max(int(rh * h), 2)) for rx, ry, rw, rh in rel]


def roi_mean_rgb(frame_bgr: np.ndarray, box: np.ndarray) -> np.ndarray:
    pixels = []
    H, W = frame_bgr.shape[:2]
    for x, y, w, h in skin_rois(box):
        patch = frame_bgr[max(y, 0):min(y + h, H), max(x, 0):min(x + w, W)]
        if patch.size:
            pixels.append(patch.reshape(-1, 3))
    if not pixels:
        return np.full(3, np.nan)
    return np.concatenate(pixels).mean(axis=0)[::-1]  # BGR -> RGB


def extract_trace(video_path: Path, track: FaceTrack, keep_crops: bool = False
                  ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray | None]:
    """Per-frame skin RGB means at native fps (+ optional 72x72 crops for PhysNet)."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    times, rgb, crops = [], [], []
    for _, t, frame in iter_frames(video_path):
        box = track.at(t)
        times.append(t)
        rgb.append(roi_mean_rgb(frame, box) if box is not None else np.full(3, np.nan))
        if keep_crops:
            crops.append(crop_face(frame, box, 72, 1.0) if box is not None else np.zeros((72, 72, 3), np.uint8))
    return np.asarray(times), np.asarray(rgb, dtype=float), float(fps), (np.stack(crops) if keep_crops and crops else None)


def fill_gaps(rgb: np.ndarray) -> np.ndarray:
    out = rgb.copy()
    idx = np.arange(len(out))
    for c in range(out.shape[1]):
        good = ~np.isnan(out[:, c])
        if good.sum() >= 2:
            out[~good, c] = np.interp(idx[~good], idx[good], out[good, c])
    return out


def bandpass(x: np.ndarray, fps: float, band: tuple[float, float] = HR_BAND) -> np.ndarray:
    nyq = fps / 2
    b, a = signal.butter(3, [band[0] / nyq, min(band[1] / nyq, 0.99)], btype="band")
    if len(x) <= 3 * max(len(a), len(b)):
        return x - x.mean()
    return signal.filtfilt(b, a, signal.detrend(x))


def pos(rgb: np.ndarray, fps: float) -> np.ndarray:
    """Plane-Orthogonal-to-Skin projection with overlap-add over 1.6 s windows."""
    n = len(rgb)
    win = max(int(1.6 * fps), 2)
    proj = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]])
    h = np.zeros(n)
    for start in range(0, n - win + 1):
        c = rgb[start:start + win].T
        mean = c.mean(axis=1, keepdims=True)
        if np.any(mean <= 0):
            continue
        s = proj @ (c / mean)
        std1 = s[1].std()
        p = s[0] + (s[0].std() / std1 if std1 > 1e-9 else 0.0) * s[1]
        h[start:start + win] += p - p.mean()
    return bandpass(h, fps)


def chrom(rgb: np.ndarray, fps: float) -> np.ndarray:
    """CHROM (de Haan & Jeanne 2013), used as an evaluation baseline."""
    norm = rgb / np.clip(rgb.mean(axis=0, keepdims=True), 1e-9, None)
    xs = 3 * norm[:, 0] - 2 * norm[:, 1]
    ys = 1.5 * norm[:, 0] + norm[:, 1] - 1.5 * norm[:, 2]
    xf, yf = bandpass(xs, fps), bandpass(ys, fps)
    alpha = xf.std() / max(yf.std(), 1e-9)
    return xf - alpha * yf


# --------------------------------------------------------------------------- PhysNet
def physnet_weights() -> Path | None:
    folder = get_settings().models_dir / "rppg"
    found = sorted(folder.glob("*PhysNet*.pth")) if folder.exists() else []
    return found[0] if found else None


@lru_cache(maxsize=1)
def load_physnet(path: str, device: str):
    import torch

    from .nets import PhysNet

    model = PhysNet(frames=PHYSNET_CHUNK)
    state = torch.load(path, map_location=device, weights_only=True)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def diff_normalize(frames: np.ndarray) -> np.ndarray:
    """rPPG-Toolbox 'DiffNormalized' input: frame differences over sums, unit std."""
    f = frames.astype(np.float32)
    d = (f[1:] - f[:-1]) / (f[1:] + f[:-1] + 1e-7)
    d = d / (d.std() + 1e-7)
    d = np.concatenate([d, np.zeros_like(d[:1])], axis=0)
    return np.nan_to_num(d)


def physnet_bvp(crops: np.ndarray, fps: float, weights: Path) -> np.ndarray:
    import torch

    configure_torch()
    device = resolve_device()
    model = load_physnet(str(weights), device)
    x = diff_normalize(crops)
    out = np.zeros(len(x))
    with torch.no_grad():
        for start in range(0, len(x) - PHYSNET_CHUNK + 1, PHYSNET_CHUNK):
            chunk = torch.from_numpy(x[start:start + PHYSNET_CHUNK]).permute(3, 0, 1, 2)[None].to(device)
            out[start:start + PHYSNET_CHUNK] = model(chunk)[0].cpu().numpy()
    # DiffNormalized labels are the BVP derivative: integrate before peak-based HRV.
    return bandpass(np.cumsum(out), fps)


# --------------------------------------------------------------------------- HR / HRV
def heart_rate(bvp: np.ndarray, fps: float) -> tuple[float | None, float]:
    """(BPM at the spectral peak, SQI = share of band power near peak + 1st harmonic)."""
    if len(bvp) < fps * 4 or np.allclose(bvp, 0):
        return None, 0.0
    nfft = max(4096, 1 << int(np.ceil(np.log2(len(bvp)))))
    freqs, power = signal.periodogram(bvp, fs=fps, nfft=nfft, window="hann")
    band = (freqs >= HR_BAND[0]) & (freqs <= HR_BAND[1])
    if not band.any() or power[band].sum() <= 0:
        return None, 0.0
    f_peak = freqs[band][np.argmax(power[band])]
    near = (np.abs(freqs - f_peak) <= 0.12) | (np.abs(freqs - 2 * f_peak) <= 0.12)
    total = power[(freqs >= HR_BAND[0]) & (freqs <= 2 * HR_BAND[1])].sum()
    sqi = float(power[near].sum() / total) if total > 0 else 0.0
    return float(f_peak * 60.0), min(sqi, 1.0)


def rmssd_ms(bvp: np.ndarray, fps: float, bpm_hint: float | None) -> float | None:
    """RMSSD from peak-to-peak intervals of a 256 Hz-resampled BVP window."""
    if len(bvp) < fps * 5:
        return None
    up = 256.0
    t = np.arange(len(bvp)) / fps
    tu = np.arange(0, t[-1], 1 / up)
    sig = np.interp(tu, t, bvp)
    min_dist = up * 60.0 / (bpm_hint * 1.4 if bpm_hint else 180.0)
    peaks, _ = signal.find_peaks(sig, distance=max(int(min_dist), 1), prominence=np.std(sig) * 0.3)
    if len(peaks) < 4:
        return None
    ibi = np.diff(peaks) / up * 1000.0
    med = np.median(ibi)
    ibi = ibi[(ibi > 333) & (ibi < 1500) & (np.abs(ibi - med) < 0.3 * med)]
    if len(ibi) < 3:
        return None
    return float(np.sqrt(np.mean(np.diff(ibi) ** 2)))


def windowed_estimates(bvp: np.ndarray, fps: float, t0: float = 0.0) -> list[dict[str, Any]]:
    duration = len(bvp) / fps
    out = []
    for s in window_starts(duration, WINDOW_S, HOP_S):
        seg = bvp[int(s * fps):int((s + WINDOW_S) * fps)]
        bpm, sqi = heart_rate(seg, fps)
        out.append({"start": rnd(t0 + s, 2), "end": rnd(t0 + min(s + WINDOW_S, duration), 2),
                    "bpm": rnd(bpm, 1), "rmssd_ms": rnd(rmssd_ms(seg, fps, bpm), 1),
                    "sqi": rnd(sqi, 3), "reliable": bool(bpm is not None and sqi >= SQI_MIN)})
    return out


def stress_series(windows: list[dict[str, Any]], duration: float) -> tuple[list, list, dict[str, Any]]:
    """Per-second BPM and a 0-1 stress index: BPM rise over the opening baseline, blended with RMSSD drop."""
    good = [w for w in windows if w["reliable"]]
    base_w = [w for w in good if w["start"] < BASELINE_S] or good[:2]
    bpm_base = float(np.median([w["bpm"] for w in base_w])) if base_w else None
    rm_vals = [w["rmssd_ms"] for w in base_w if w["rmssd_ms"]]
    rm_base = float(np.median(rm_vals)) if rm_vals else None
    bpm_s, stress_s = [], []
    for sec in range(int(np.ceil(duration))):
        cover = [w for w in good if w["start"] <= sec < w["end"]]
        if not cover or bpm_base is None:
            bpm_s.append(None)
            stress_s.append(None)
            continue
        bpm = float(np.mean([w["bpm"] for w in cover]))
        parts = [float(np.clip((bpm - bpm_base) / 20.0, 0, 1))]
        rms = [w["rmssd_ms"] for w in cover if w["rmssd_ms"]]
        if rm_base and rms:
            parts.append(float(np.clip((rm_base - np.mean(rms)) / rm_base, 0, 1)))
        bpm_s.append(rnd(bpm, 1))
        stress_s.append(rnd(float(np.mean(parts)), 3))
    bpms = [w["bpm"] for w in good]
    summary = {
        "bpm_mean": rnd(nanmean(bpms), 1), "bpm_baseline": rnd(bpm_base, 1),
        "bpm_max": rnd(max(bpms), 1) if bpms else None,
        "bpm_rise": rnd(max(bpms) - bpm_base, 1) if bpms and bpm_base else None,
        "rmssd_mean_ms": rnd(nanmean([w["rmssd_ms"] for w in good]), 1),
        "reliable_ratio": rnd(len(good) / len(windows), 3) if windows else 0.0,
        "stress_mean": rnd(nanmean(stress_s), 3),
    }
    return bpm_s, stress_s, summary


def estimate_bvp(times: np.ndarray, rgb: np.ndarray, fps: float, crops: np.ndarray | None,
                 method: str) -> np.ndarray:
    rgb = fill_gaps(rgb)
    if method == "physnet" and crops is not None:
        weights = physnet_weights()
        if weights is None:
            raise FileNotFoundError("no PhysNet weights in models/rppg")
        return physnet_bvp(crops, fps, weights)
    return chrom(rgb, fps) if method == "chrom" else pos(rgb, fps)


def consensus_windows(per_method: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """Combine per-method windows: reliable only if >= 2 methods agree within AGREE_BPM, away from the
    band edges, with adequate SQI. BPM = mean of the agreeing estimates; RMSSD kept only if plausible."""
    names = list(per_method)
    lo, hi = HR_BAND[0] * 60 + EDGE_MARGIN_BPM, HR_BAND[1] * 60 - EDGE_MARGIN_BPM
    out = []
    for i, base in enumerate(per_method[names[0]]):
        cands = [(n, per_method[n][i]) for n in names
                 if per_method[n][i]["bpm"] is not None and lo <= per_method[n][i]["bpm"] <= hi
                 and per_method[n][i]["sqi"] >= SQI_MIN]
        best: list[tuple[str, dict[str, Any]]] = []
        for _, anchor in cands:
            group = [(n, w) for n, w in cands if abs(w["bpm"] - anchor["bpm"]) <= AGREE_BPM]
            best = group if len(group) > len(best) else best
        agreeing = best if len(best) >= min(2, len(names)) else []
        rmssd = [w["rmssd_ms"] for _, w in agreeing if w["rmssd_ms"] and RMSSD_PLAUSIBLE[0] <= w["rmssd_ms"] <= RMSSD_PLAUSIBLE[1]]
        out.append({
            "start": base["start"], "end": base["end"],
            "bpm": rnd(float(np.mean([w["bpm"] for _, w in agreeing])), 1) if agreeing else None,
            "rmssd_ms": rnd(float(np.median(rmssd)), 1) if rmssd else None,
            "sqi": rnd(max((w["sqi"] for _, w in agreeing), default=0.0), 3),
            "reliable": bool(agreeing), "agree": [n for n, _ in agreeing],
            "per_method": {n: per_method[n][i]["bpm"] for n in names},
        })
    return out


class PhysioAnalyzer(Analyzer):
    name = "physio"
    requires = ("ingest", "face")

    def analyze(self, ctx: SessionContext) -> dict[str, Any]:
        ingest = self.need(ctx, "ingest")
        face = self.need(ctx, "face")
        duration = float(ingest["duration_s"])
        use_physnet = physnet_weights() is not None and torch_available()
        times, rgb, fps, crops = extract_trace(ctx.video_path, FaceTrack.from_face(face), keep_crops=use_physnet)
        if np.isnan(rgb).all() or len(rgb) < fps * WINDOW_S:
            raise AnalyzerSkipped("not enough face video for rPPG (need 10 s)")
        per_method: dict[str, list[dict[str, Any]]] = {}
        for method in (["physnet"] if use_physnet else []) + ["pos", "chrom"]:
            try:
                per_method[method] = windowed_estimates(estimate_bvp(times, rgb, fps, crops, method), fps)
            except Exception:  # noqa: BLE001 - e.g. PhysNet weights mismatch: consensus of the rest
                continue
        windows = consensus_windows(per_method)
        bpm_s, stress_s, summary = stress_series(windows, duration)
        return {"summary": summary, "windows": windows, "series": {"bpm": bpm_s, "stress": stress_s},
                "fps": rnd(fps, 2), "disclaimer": DISCLAIMER,
                "models": {"rppg": "consensus(" + ",".join(per_method) + ")"}}
