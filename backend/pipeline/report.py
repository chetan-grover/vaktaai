"""Assemble report.json (the one document the frontend reads) from the module JSON files."""
from __future__ import annotations

import platform
from datetime import datetime, timezone
from importlib import metadata
from typing import Any

from .base import SessionContext
from .settings import get_settings

DISCLAIMER = ("Heart-rate, stress and emotion outputs are indicative only: they are not medical "
              "measurements and must not be used for lie detection.")
MODULES = ("ingest", "asr", "voice", "content", "audio_va", "face", "gaze", "face_va", "physio",
           "fusion", "timeline", "drills")
PACKAGES = ("faster-whisper", "mediapipe", "torch", "transformers", "hsemotion-onnx", "numpy", "opencv-contrib-python")


def package_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
    return out


def _pick(data: dict[str, Any] | None, *keys: str) -> dict[str, Any] | None:
    return {k: data.get(k) for k in keys} if data else None


def build_report(ctx: SessionContext) -> dict[str, Any]:
    envelopes = {name: ctx.read_envelope(name) for name in MODULES}
    data = {name: (env or {}).get("data") if (env or {}).get("status") == "ok" else None
            for name, env in envelopes.items()}
    models: dict[str, str] = {}
    for d in data.values():
        if d and isinstance(d.get("models"), dict):
            models.update(d["models"])

    fusion, ingest = data["fusion"] or {}, data["ingest"] or {}
    face_va, audio_va = data["face_va"], data["audio_va"]
    report: dict[str, Any] = {
        "session_id": ctx.session_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "duration_s": ingest.get("duration_s"),
        "disclaimer": DISCLAIMER,
        "modules": {name: {"status": env.get("status"), "error": env.get("error"), "elapsed_s": env.get("elapsed_s")}
                    if env else {"status": "missing", "error": None, "elapsed_s": None}
                    for name, env in envelopes.items()},
        "reproducibility": {"models": models, "packages": package_versions(), "seed": get_settings().seed},
        "traits": fusion.get("traits"),
        "congruence": fusion.get("congruence"),
        "segments": [{k: s.get(k) for k in ("id", "start", "end", "text", "traits", "mismatch", "mismatch_rule",
                                             "congruence_distance", "features", "learned")}
                     for s in fusion.get("segments", [])],
        "mismatches": fusion.get("mismatches", []),
        "timeline": data["timeline"],
        "transcript": _pick(data["asr"], "text", "words", "language"),
        "voice": _pick(data["voice"], "summary", "fillers", "pauses", "wpm_windows"),
        "face": _pick(data["face"], "summary", "blinks"),
        "gaze": _pick(data["gaze"], "summary", "alerts", "calibration", "cell_names"),
        "physio": _pick(data["physio"], "summary", "windows", "disclaimer"),
        "affect": {
            "face": {**face_va["summary"], "proxy": face_va["proxy"], "t": face_va["t"], "valence": face_va["valence"],
                     "arousal": face_va["arousal"]} if face_va else None,
            "audio": {**audio_va["summary"], "proxy": audio_va["proxy"], "t": audio_va["t"],
                      "valence": audio_va["valence"], "arousal": audio_va["arousal"]} if audio_va else None,
        },
        "content": _pick(data["content"], "structure", "rhetoric", "note"),
        "drills": _pick(data["drills"], "weaknesses", "plan"),
    }
    try:
        from analytics.persona.features import session_vector

        report["persona_features"] = session_vector(report)
    except Exception:  # noqa: BLE001 - persona is optional
        report["persona_features"] = None
    ctx.write_json("report.json", report)
    return report
