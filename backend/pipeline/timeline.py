"""FR5 segmentation and F13 timeline: per-second channels + events (confidence dips,
mismatches, heart-rate spikes, neglected zones, filler clusters, long pauses)."""
from __future__ import annotations

from typing import Any

import numpy as np

from .base import Analyzer, SessionContext
from .text import SENTENCE_END
from .utils import nanmean, nanstd, rnd

SNAP_S = 1.5
MIN_SEGMENT_S = 2.0
DIP_DROP = 12.0
DIP_FLOOR = 60.0
BPM_SPIKE = 12.0
FILLER_CLUSTER = (3, 10.0)  # >= 3 fillers within 10 s
LONG_PAUSE_S = 2.0


def build_segments(duration: float, words: list[dict[str, Any]], size: float = 5.0) -> list[dict[str, Any]]:
    """~`size`-second windows whose edges snap to a sentence end within +-1.5 s."""
    ends = [w["end"] for w in words if w["w"].rstrip('"\'').endswith(SENTENCE_END)]
    segments, start = [], 0.0
    while start < duration - 0.5:
        target = start + size
        cands = [e for e in ends if target - SNAP_S <= e <= target + SNAP_S and e > start + MIN_SEGMENT_S]
        end = min(cands, key=lambda e: abs(e - target)) if cands else min(target, duration)
        if duration - end < MIN_SEGMENT_S:
            end = duration
        seg_words = [w for w in words if start <= (w["start"] + w["end"]) / 2 < end]
        segments.append({"id": len(segments), "start": rnd(start, 2), "end": rnd(end, 2),
                         "text": " ".join(w["w"] for w in seg_words)})
        start = end
    return segments


def confidence_per_second(segments: list[dict[str, Any]], duration: float) -> list[float | None]:
    out: list[float | None] = []
    for sec in range(int(np.ceil(duration))):
        seg = next((s for s in segments if s["start"] <= sec + 0.5 < s["end"]), None)
        out.append(seg["traits"].get("confidence") if seg else None)
    return out


def find_dips(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scores = [s["traits"].get("confidence") for s in segments]
    mean, std = nanmean(scores), nanstd(scores) or 0.0
    events = []
    for i, seg in enumerate(segments):
        c = scores[i]
        if c is None or mean is None:
            continue
        prev = next((scores[j] for j in range(i - 1, -1, -1) if scores[j] is not None), None)
        below = c < mean - max(std, 5.0) and c < DIP_FLOOR + 10
        drop = prev is not None and prev - c >= DIP_DROP and c < DIP_FLOOR
        if below or drop:
            worst = seg.get("weakest") or "several channels"
            events.append(_event("confidence_dip", seg["start"], seg["end"], (mean - c) / 30,
                                 f"Confidence dip ({c:.0f})", f"Driven mainly by {worst}."))
    return events


def _event(kind: str, start: float, end: float, severity: float, label: str, why: str) -> dict[str, Any]:
    return {"type": kind, "start": rnd(start, 2), "end": rnd(end, 2),
            "severity": rnd(float(np.clip(severity, 0.05, 1.0)), 2), "label": label, "why": why}


def bpm_spikes(physio: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not physio:
        return []
    base = physio["summary"].get("bpm_baseline")
    if base is None:
        return []
    events = []
    for w in physio["windows"]:
        if w["reliable"] and w["bpm"] and w["bpm"] - base >= BPM_SPIKE:
            events.append(_event("bpm_spike", w["start"], w["end"], (w["bpm"] - base) / 30,
                                 f"Heart rate up {w['bpm'] - base:.0f} BPM",
                                 f"{w['bpm']:.0f} BPM vs. {base:.0f} at the start (indicative)."))
    return _merge(events)


def filler_clusters(fillers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    n, span = FILLER_CLUSTER
    events, i = [], 0
    times = [f["start"] for f in fillers]
    while i < len(times):
        j = i
        while j + 1 < len(times) and times[j + 1] - times[i] <= span:
            j += 1
        if j - i + 1 >= n:
            words = ", ".join(f["text"] for f in fillers[i:j + 1])
            events.append(_event("filler_cluster", times[i], fillers[j]["end"], (j - i + 1) / 8,
                                 f"{j - i + 1} fillers in {span:.0f}s", f"Heard: {words}."))
            i = j + 1
        else:
            i += 1
    return events


def long_pauses(pauses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_event("long_pause", p["start"], p["end"], p["duration"] / 6, f"{p['duration']:.1f}s pause",
                   "A long silence; fine if deliberate, otherwise a lost thread.")
            for p in pauses if p["duration"] >= LONG_PAUSE_S]


def _merge(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for e in sorted(events, key=lambda e: e["start"]):
        if merged and e["start"] <= merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], e["end"])
            merged[-1]["severity"] = max(merged[-1]["severity"], e["severity"])
        else:
            merged.append(dict(e))
    return merged


class TimelineAnalyzer(Analyzer):
    name = "timeline"
    requires = ("ingest", "fusion")

    def analyze(self, ctx: SessionContext) -> dict[str, Any]:
        duration = float(self.need(ctx, "ingest")["duration_s"])
        fusion = self.need(ctx, "fusion")
        voice, physio, gaze = ctx.read("voice"), ctx.read("physio"), ctx.read("gaze")
        face_va, audio_va = ctx.read("face_va"), ctx.read("audio_va")
        segments = fusion["segments"]

        events = find_dips(segments)
        events += [_event("mismatch", m["start"], m["end"], m.get("distance", 1.0) / 2, "Voice/face mismatch", m["reason"])
                   for m in fusion["mismatches"]]
        events += bpm_spikes(physio)
        if voice:
            events += filler_clusters(voice["fillers"]) + long_pauses(voice["pauses"])
        if gaze:
            events += [{"type": "neglected_zone", "start": 0.0, "end": rnd(duration, 2), "severity": 0.5,
                        "label": f"Neglected: {a['name']}", "why": a["message"], "cell": a["cell"]}
                       for a in gaze["alerts"]]
        events.sort(key=lambda e: (e["start"], e["type"]))

        n = int(np.ceil(duration))
        pad = lambda xs: (list(xs or []) + [None] * n)[:n]  # noqa: E731
        channels = {
            "t": list(range(n)),
            "confidence": confidence_per_second(segments, duration),
            "bpm": pad(physio["series"]["bpm"]) if physio else [None] * n,
            "stress": pad(physio["series"]["stress"]) if physio else [None] * n,
            "face_valence": pad(face_va["per_second"]["valence"]) if face_va else [None] * n,
            "face_arousal": pad(face_va["per_second"]["arousal"]) if face_va else [None] * n,
            "audio_valence": pad(audio_va["per_second"]["valence"]) if audio_va else [None] * n,
            "audio_arousal": pad(audio_va["per_second"]["arousal"]) if audio_va else [None] * n,
            "gaze_cell": pad(gaze["per_second"]) if gaze else [None] * n,
            "energy_db": pad(voice["series"]["energy_db"]) if voice else [None] * n,
            "pitch_hz": pad(voice["series"]["pitch_hz"]) if voice else [None] * n,
        }
        return {"channels": channels, "events": events,
                "counts": {k: sum(e["type"] == k for e in events) for k in
                           ("confidence_dip", "mismatch", "bpm_spike", "filler_cluster", "long_pause", "neglected_zone")}}
