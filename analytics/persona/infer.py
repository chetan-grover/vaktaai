"""E5 / F21 inference: place sessions on the persona map and name the persona.

Uses the trained VAE + GMM (models/persona/) when present; otherwise a transparent
rule map whose two axes are averages of standardised features:
  x = delivery energy (pace, pitch/volume variation, expressiveness, vocal arousal)
  y = polish (few fillers, eye contact, room coverage, transitions, congruence, calm heart rate)
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .features import FEATURE_NAMES, NORMS, to_array
from .naming import FALLBACK, RULES, name_profile

ENERGY = {"wpm": 1, "pitch_std_st": 1, "energy_std_db": 1, "expressiveness": 1, "audio_arousal": 1}
POLISH = {"filler_rate_per_min": -1, "eye_contact_ratio": 1, "gaze_coverage": 1, "transition_quality": 1,
          "congruence": 1, "bpm_rise": -1}
PROTOTYPES: dict[str, dict[str, float]] = {
    "Fast Talker": {"wpm": 1.8, "mean_pause_s": -1.0, "filler_rate_per_min": 0.5},
    "Deliberate Pacer": {"wpm": -1.6, "audio_arousal": -0.6, "pitch_std_st": -0.4},
    "Monotone Presenter": {"pitch_std_st": -1.5, "energy_std_db": -1.3, "expressiveness": -0.8},
    "Filler-Heavy Speaker": {"filler_rate_per_min": 2.0, "transition_quality": -0.8},
    "Note Reader": {"eye_contact_ratio": -1.6, "gaze_coverage": -1.4},
    "Nervous Starter": {"bpm_rise": 1.8, "blink_rate_per_min": 1.2, "filler_rate_per_min": 0.6},
    "Energetic Storyteller": {"expressiveness": 1.5, "pitch_std_st": 1.4, "audio_arousal": 1.2, "wpm": 0.5},
    "Polished Communicator": {"gaze_coverage": 1.2, "transition_quality": 1.2, "filler_rate_per_min": -1.2, "congruence": 1.0},
}


def _axis(z: dict[str, float], weights: dict[str, int]) -> float:
    return float(np.mean([sign * z.get(name, 0.0) for name, sign in weights.items()]))


class RuleMap:
    kind = "rule-axes"
    axes = ("Delivery energy", "Polish")

    def project(self, vectors: list[dict[str, Any]]) -> np.ndarray:
        rows = [dict(zip(FEATURE_NAMES, to_array(v).tolist())) for v in vectors]
        return np.array([[_axis(z, ENERGY), _axis(z, POLISH)] for z in rows]).reshape(-1, 2)

    def assign(self, vector: dict[str, Any]) -> dict[str, Any]:
        name, desc = name_profile(dict(zip(FEATURE_NAMES, to_array(vector).tolist())))
        return {"name": name, "description": desc, "component": None}

    def regions(self) -> list[dict[str, Any]]:
        out = []
        for name, desc, _ in RULES:
            z = PROTOTYPES[name]
            out.append({"name": name, "description": desc, "x": round(_axis(z, ENERGY), 3), "y": round(_axis(z, POLISH), 3)})
        out.append({"name": FALLBACK[0], "description": FALLBACK[1], "x": 0.0, "y": 0.0})
        return out


class VaeMap:
    kind = "vae+gmm"
    axes = ("Latent 1", "Latent 2")

    def __init__(self, folder: Path) -> None:
        import joblib
        import torch

        from .model import SessionVAE

        self.meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        self.norms = {k: tuple(v) for k, v in self.meta["norms"].items()}
        ckpt = torch.load(folder / "vae.pt", map_location="cpu", weights_only=True)
        self.model = SessionVAE(**ckpt["config"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.gmm = joblib.load(folder / "gmm.joblib")
        self.by_component = {p["component"]: p for p in self.meta["personas"]}

    def _latent(self, vectors: list[dict[str, Any]]) -> np.ndarray:
        import torch

        x = torch.tensor(np.stack([to_array(v, self.norms) for v in vectors]), dtype=torch.float32)
        with torch.no_grad():
            mu, _ = self.model.encode(x)
        return mu.numpy()

    def project(self, vectors: list[dict[str, Any]]) -> np.ndarray:
        return self._latent(vectors)[:, :2]

    def assign(self, vector: dict[str, Any]) -> dict[str, Any]:
        comp = int(self.gmm.predict(self._latent([vector]))[0])
        p = self.by_component.get(comp, {"name": FALLBACK[0], "description": FALLBACK[1]})
        return {"name": p["name"], "description": p["description"], "component": comp}

    def regions(self) -> list[dict[str, Any]]:
        return [{"name": p["name"], "description": p["description"], "x": p["centroid"][0],
                 "y": p["centroid"][1] if len(p["centroid"]) > 1 else 0.0} for p in self.meta["personas"]]


@lru_cache(maxsize=1)
def load_map(models_dir: str) -> RuleMap | VaeMap:
    folder = Path(models_dir) / "persona"
    if (folder / "meta.json").exists():
        try:
            return VaeMap(folder)
        except Exception:  # noqa: BLE001 - torch missing or stale files -> rules
            pass
    return RuleMap()


def persona_trajectory(sessions: list[dict[str, Any]], models_dir: str) -> dict[str, Any]:
    """sessions: [{"session_id", "created_at", "title", "features": vector}] oldest first."""
    pmap = load_map(models_dir)
    usable = [s for s in sessions if s.get("features")]
    coords = pmap.project([s["features"] for s in usable]) if usable else np.zeros((0, 2))
    points = []
    for s, (x, y) in zip(usable, coords):
        points.append({"session_id": s["session_id"], "created_at": s.get("created_at"), "title": s.get("title"),
                       "x": round(float(x), 3), "y": round(float(y), 3), "persona": pmap.assign(s["features"])})
    return {"kind": pmap.kind, "axes": list(pmap.axes), "regions": pmap.regions(), "points": points,
            "current": points[-1]["persona"] if points else None}


def kmeans_raw(vectors: list[dict[str, Any]], k: int, seed: int = 13) -> np.ndarray:
    """Baseline for eval: k-means on standardised raw features (no VAE)."""
    from sklearn.cluster import KMeans

    x = np.stack([to_array(v, NORMS) for v in vectors])
    return KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(x)
