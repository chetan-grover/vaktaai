"""Persona names from a z-scored feature profile (cluster centroid or single session)."""
from __future__ import annotations

from typing import Any

from .features import FEATURE_NAMES

# (name, description, score function over z-profile). Highest score wins; > 0.7 required.
RULES: list[tuple[str, str, Any]] = [
    ("Fast Talker", "Rushes through content with little pausing.",
     lambda z: z["wpm"] - 0.5 * z["mean_pause_s"]),
    ("Deliberate Pacer", "Slow, measured delivery; could use more drive.",
     lambda z: -z["wpm"]),
    ("Monotone Presenter", "Flat pitch and volume; content carries the talk alone.",
     lambda z: -(z["pitch_std_st"] + z["energy_std_db"]) / 2),
    ("Filler-Heavy Speaker", "Ums, likes and 'basically' break up the flow.",
     lambda z: z["filler_rate_per_min"]),
    ("Note Reader", "Gaze stays down or off to one side; the room is half-ignored.",
     lambda z: -(z["eye_contact_ratio"] + z["gaze_coverage"]) / 2),
    ("Nervous Starter", "Physiological stress climbs; delivery tightens under pressure.",
     lambda z: (z["bpm_rise"] + z["blink_rate_per_min"]) / 2),
    ("Energetic Storyteller", "Expressive voice and face; can be over the top.",
     lambda z: (z["expressiveness"] + z["pitch_std_st"] + z["audio_arousal"]) / 3),
    ("Polished Communicator", "Balanced pace, clean speech, steady gaze across the room.",
     lambda z: (z["gaze_coverage"] + z["transition_quality"] - z["filler_rate_per_min"] + z["congruence"]) / 4),
]
FALLBACK = ("Developing Speaker", "No single habit dominates yet; keep practising the assigned drills.")


def name_profile(z: dict[str, float], taken: set[str] | None = None) -> tuple[str, str]:
    taken = taken or set()
    full = {n: float(z.get(n, 0.0)) for n in FEATURE_NAMES}
    scored = sorted(((fn(full), name, desc) for name, desc, fn in RULES if name not in taken), reverse=True)
    if scored and scored[0][0] > 0.7:
        return scored[0][1], scored[0][2]
    return FALLBACK
