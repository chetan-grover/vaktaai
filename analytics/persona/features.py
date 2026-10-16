"""E5: the 24-number session feature vector the persona model is trained on.

Missing modalities are imputed with the reference norm, so a session without (say) rPPG
still lands on the map. NORMS are rough population priors (mean, std) used for
standardisation until a trained model provides its own statistics.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

NORMS: dict[str, tuple[float, float]] = {
    "wpm": (145.0, 25.0),
    "wpm_std": (20.0, 10.0),
    "articulation_wpm": (170.0, 25.0),
    "pauses_per_min": (10.0, 5.0),
    "mean_pause_s": (0.8, 0.4),
    "long_pause_count": (1.0, 1.5),
    "filler_rate_per_min": (3.0, 2.5),
    "pitch_std_st": (2.5, 1.0),
    "pitch_range_st": (8.0, 3.0),
    "energy_std_db": (6.0, 2.0),
    "eye_contact_ratio": (0.35, 0.2),
    "gaze_coverage": (0.45, 0.2),
    "gaze_entropy": (0.5, 0.2),
    "blink_rate_per_min": (17.0, 8.0),
    "head_motion": (4.0, 2.5),
    "face_valence": (0.1, 0.25),
    "face_arousal": (0.0, 0.25),
    "expressiveness": (0.15, 0.08),
    "audio_arousal": (0.0, 0.3),
    "congruence": (80.0, 8.0),
    "bpm_rise": (6.0, 6.0),
    "rmssd_ms": (40.0, 20.0),
    "appeal_balance": (0.85, 0.1),
    "transition_quality": (0.45, 0.2),
}
FEATURE_NAMES: list[str] = list(NORMS)


def _get(d: dict[str, Any] | None, *path: str) -> Any:
    for key in path:
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d


def _appeal_balance(report: dict[str, Any]) -> float | None:
    appeals = _get(report, "content", "rhetoric", "appeals")
    if not appeals:
        return None
    p = np.array([appeals["ethos"], appeals["pathos"], appeals["logos"]], dtype=float)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log(p)).sum() / math.log(3))


def session_vector(report: dict[str, Any]) -> dict[str, float | None]:
    """Raw (unstandardised) features from a report.json; None where unavailable."""
    v, g, ph = _get(report, "voice", "summary") or {}, _get(report, "gaze", "summary") or {}, _get(report, "physio", "summary") or {}
    head = [s["features"].get("head_motion") for s in report.get("segments") or [] if s.get("features")]
    head = [h for h in head if h is not None]
    return {
        "wpm": v.get("wpm"), "wpm_std": v.get("wpm_std"), "articulation_wpm": v.get("articulation_wpm"),
        "pauses_per_min": v.get("pauses_per_min"), "mean_pause_s": v.get("mean_pause_s"),
        "long_pause_count": v.get("long_pause_count"), "filler_rate_per_min": v.get("filler_rate_per_min"),
        "pitch_std_st": v.get("pitch_std_st"), "pitch_range_st": v.get("pitch_range_st"),
        "energy_std_db": v.get("energy_std_db"),
        "eye_contact_ratio": g.get("eye_contact_ratio"), "gaze_coverage": g.get("coverage"), "gaze_entropy": g.get("entropy"),
        "blink_rate_per_min": _get(report, "face", "summary", "blink_rate_per_min"),
        "head_motion": float(np.mean(head)) if head else None,
        "face_valence": _get(report, "affect", "face", "valence_mean"),
        "face_arousal": _get(report, "affect", "face", "arousal_mean"),
        "expressiveness": _get(report, "affect", "face", "expressiveness"),
        "audio_arousal": _get(report, "affect", "audio", "arousal_mean"),
        "congruence": _get(report, "congruence", "score"),
        "bpm_rise": ph.get("bpm_rise"), "rmssd_ms": ph.get("rmssd_mean_ms"),
        "appeal_balance": _appeal_balance(report),
        "transition_quality": _get(report, "content", "rhetoric", "transition_quality"),
    }


def to_array(vector: dict[str, float | None], norms: dict[str, tuple[float, float]] | None = None) -> np.ndarray:
    """Standardised (z-score) array in FEATURE_NAMES order; missing -> 0 (= the norm)."""
    norms = norms or NORMS
    out = []
    for name in FEATURE_NAMES:
        mean, std = norms[name]
        value = vector.get(name)
        out.append(0.0 if value is None or (isinstance(value, float) and math.isnan(value)) else (value - mean) / std)
    return np.clip(np.asarray(out, dtype=np.float32), -4, 4)
