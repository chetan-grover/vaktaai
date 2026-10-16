"""F10-F12 / E3: multimodal fusion -> per-segment trait scores, congruence distance, mismatches.

Two layers run side by side so they can be compared in eval/:
- rule baseline: YAML bands + weights (config/traits.yaml) and hand-written mismatch rules;
- learned E3 head (models/fusion/e3_head.pt): cross-attention over the audio and face
  embedding streams saved by affect.py; gives fused V-A, per-modality V-A and a mismatch probability.
The learned head is used when its weights and both embedding files exist; otherwise the
congruence distance is computed from the per-modality V-A streams directly.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .base import Analyzer, SessionContext
from .gaze import CENTER
from .settings import get_settings
from .text import lexicon_sentiment
from .timeline import build_segments
from .utils import nanmean, resolve_device, rnd, torch_available, trapezoid

CONFIG_DIR = Path(__file__).parent / "config"
MAX_VA_DISTANCE = float(np.sqrt(8))


@lru_cache
def load_trait_config() -> dict[str, Any]:
    return yaml.safe_load((CONFIG_DIR / "traits.yaml").read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- segment features
def _in(t: float, seg: dict[str, Any]) -> bool:
    return seg["start"] <= t < seg["end"]


def _series_slice(series: list | None, seg: dict[str, Any]) -> list[float]:
    if not series:
        return []
    lo, hi = int(seg["start"]), int(np.ceil(seg["end"]))
    return [v for v in series[lo:hi] if v is not None]


def _timed_slice(t: list | None, values: list | None, seg: dict[str, Any]) -> list[float]:
    if not t or not values:
        return []
    return [v for ti, v in zip(t, values) if ti is not None and v is not None and _in(ti, seg)]


def segment_features(seg: dict[str, Any], m: dict[str, Any], ctx_stats: dict[str, Any]) -> dict[str, Any]:
    length = max(seg["end"] - seg["start"], 1e-6)
    f: dict[str, Any] = {}
    asr, voice, face, gaze = m.get("asr"), m.get("voice"), m.get("face"), m.get("gaze")
    if asr:
        n_words = sum(_in((w["start"] + w["end"]) / 2, seg) for w in asr["words"])
        f["wpm"] = n_words * 60 / length if n_words else None
        f["text_sentiment"] = lexicon_sentiment(seg["text"])
    if voice and asr and asr.get("words"):
        f["filler_rate"] = sum(_in(x["start"], seg) for x in voice["fillers"]) * 60 / length
        paused = sum(max(0.0, min(p["end"], seg["end"]) - max(p["start"], seg["start"])) for p in voice["pauses"])
        f["pause_ratio"] = paused / length
    if voice:
        pitch = _series_slice(voice["series"]["pitch_hz"], seg)
        if len(pitch) >= 2 and ctx_stats.get("pitch_median"):
            f["pitch_std_st"] = float(np.std(12 * np.log2(np.asarray(pitch) / ctx_stats["pitch_median"])))
        energy = _series_slice(voice["series"]["energy_db"], seg)
        if energy and ctx_stats.get("energy_median") is not None:
            f["energy_rel_db"] = float(np.mean(energy)) - ctx_stats["energy_median"]
    if face:
        fr = face["frames"]
        yaw = _timed_slice(fr["t"], fr["yaw"], seg)
        f["head_motion"] = head_jitter(yaw)
        if face["summary"].get("blink_rate_per_min") is not None:
            f["blink_rate"] = sum(_in(b, seg) for b in face["blinks"]) * 60 / length
    if gaze:
        cells = _timed_slice(gaze["t"], gaze["cells"], seg)
        if cells:
            f["on_grid"] = sum(c >= 0 for c in cells) / len(cells)
            f["eye_contact"] = sum(c == CENTER for c in cells) / len(cells)
            f["gaze_cells"] = len({c for c in cells if c >= 0})
    for key, mod in (("face", m.get("face_va")), ("audio", m.get("audio_va"))):
        if mod:
            v = _timed_slice(mod["t"], mod["valence"], seg)
            a = _timed_slice(mod["t"], mod["arousal"], seg)
            f[f"{key}_valence"] = float(np.mean(v)) if v else None
            f[f"{key}_arousal"] = float(np.mean(a)) if a else None
            if key == "face" and len(v) > 1:
                f["face_va_std"] = float(np.mean([np.std(v), np.std(a)]))
    physio = m.get("physio")
    if physio:
        bpm = _series_slice(physio["series"]["bpm"], seg)
        base = physio["summary"].get("bpm_baseline")
        f["bpm"] = float(np.mean(bpm)) if bpm else None
        f["bpm_rise"] = float(np.mean(bpm)) - base if bpm and base else None
    return {k: (None if v is None else float(v)) for k, v in f.items()}


def head_jitter(yaw: list[float], window: int = 10) -> float | None:
    """Std of yaw around its 1 s moving average: fidgeting, not deliberate turns toward the room."""
    if len(yaw) <= window:
        return None
    arr = np.asarray(yaw, dtype=float)
    trend = np.convolve(arr, np.ones(window) / window, mode="valid")
    return float(np.std(arr[window // 2: window // 2 + len(trend)] - trend))


def goodness(features: dict[str, Any], cfg: dict[str, Any], exclude: set[str] = frozenset()) -> dict[str, float]:
    out = {}
    for name, spec in cfg["features"].items():
        if spec["metric"] in exclude:
            continue
        g = trapezoid(features.get(spec["metric"]), *spec["band"])
        if g is not None:
            out[name] = round(float(g), 3)
    return out


def trait_scores(good: dict[str, float], cfg: dict[str, Any]) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for trait, spec in cfg["traits"].items():
        pairs = [(w, good[f]) for f, w in spec["weights"].items() if f in good]
        total = sum(w for w, _ in pairs)
        scores[trait] = round(100 * sum(w * g for w, g in pairs) / total, 1) if total >= 0.3 else None
    return scores


def explain(trait: str, mean_good: dict[str, float], mean_raw: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    """One-line 'why' (NFR6): the weakest and strongest weighted feature."""
    weights = cfg["traits"][trait]["weights"]
    present = [f for f in weights if f in mean_good]
    if not present:
        return {"why": "Not enough signal to score this trait.", "weakest": None, "strongest": None}
    weakest = min(present, key=lambda f: mean_good[f])
    strongest = max(present, key=lambda f: mean_good[f] * weights[f])

    def describe(f: str) -> str:
        spec = cfg["features"][f]
        raw = mean_raw.get(spec["metric"])
        if raw is None:
            return spec["label"].lower()
        value = f"{raw * 100:.0f}% {spec['unit']}" if spec["unit"] == "of time" else f"{raw:.3g} {spec['unit']}"
        return f"{spec['label'].lower()} ({value})"

    why = f"{describe(weakest).capitalize()} held this back; {describe(strongest)} helped."
    if mean_good[weakest] > 0.8:
        why = f"Strong across the board; best on {describe(strongest)}."
    return {"why": why, "weakest": weakest, "strongest": strongest}


# --------------------------------------------------------------------------- mismatch
def rule_mismatch(f: dict[str, Any], cfg: dict[str, Any]) -> list[str]:
    reasons = []
    for rule in cfg["mismatch"]["rules"].values():
        fv, fa, aa, ts = f.get("face_valence"), f.get("face_arousal"), f.get("audio_arousal"), f.get("text_sentiment")
        checks = []
        if "face_valence_min" in rule:
            checks.append(fv is not None and fv >= rule["face_valence_min"])
        if "face_valence_abs_max" in rule:
            checks.append(fv is not None and abs(fv) <= rule["face_valence_abs_max"])
        if "face_arousal_max" in rule:
            checks.append(fa is not None and fa <= rule["face_arousal_max"])
        if "audio_arousal_max" in rule:
            checks.append(aa is not None and aa <= rule["audio_arousal_max"])
        if "audio_arousal_min" in rule:
            checks.append(aa is not None and aa >= rule["audio_arousal_min"])
        if "text_sentiment_max" in rule:
            checks.append(ts is not None and ts <= rule["text_sentiment_max"])
        if checks and all(checks):
            reasons.append(rule["message"])
    return reasons


def va_distance(f: dict[str, Any]) -> float | None:
    keys = ("audio_valence", "audio_arousal", "face_valence", "face_arousal")
    if any(f.get(k) is None for k in keys):
        return None
    return float(np.hypot(f["audio_valence"] - f["face_valence"], f["audio_arousal"] - f["face_arousal"]))


def head_path() -> Path:
    return get_settings().models_dir / "fusion" / "e3_head.pt"


@lru_cache(maxsize=1)
def load_head(path: str, device: str):
    import torch

    from .nets import CrossAttentionFusionHead

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = CrossAttentionFusionHead(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt


def learned_segments(ctx: SessionContext, segments: list[dict[str, Any]]) -> list[dict[str, Any] | None]:
    """Run the E3 head per segment on the saved audio/face embedding streams."""
    import torch

    device = resolve_device()
    model, ckpt = load_head(str(head_path()), device)
    audio, face = np.load(ctx.path("audio_emb.npz")), np.load(ctx.path("face_emb.npz"))
    norm = ckpt.get("norm", {})
    max_len = model.config["max_len"]
    threshold = float(ckpt.get("mismatch_threshold", 0.5))
    out: list[dict[str, Any] | None] = []
    for seg in segments:
        ia = (audio["t"] >= seg["start"]) & (audio["t"] < seg["end"])
        jf = (face["t"] >= seg["start"]) & (face["t"] < seg["end"])
        if ia.sum() == 0 or jf.sum() == 0:
            out.append(None)
            continue
        a = (audio["emb"][ia][:max_len] - norm.get("audio_mean", 0)) / norm.get("audio_std", 1)
        f = (face["emb"][jf][:max_len] - norm.get("face_mean", 0)) / norm.get("face_std", 1)
        with torch.no_grad():
            res = model(torch.tensor(a, dtype=torch.float32)[None].to(device),
                        torch.tensor(f, dtype=torch.float32)[None].to(device))
        a_va, f_va = res["audio_va"][0].cpu().numpy(), res["face_va"][0].cpu().numpy()
        prob = float(torch.sigmoid(res["mismatch_logit"])[0])
        out.append({"va": res["va"][0].cpu().numpy().round(3).tolist(), "audio_va": a_va.round(3).tolist(),
                    "face_va": f_va.round(3).tolist(), "distance": float(np.linalg.norm(a_va - f_va)),
                    "mismatch_prob": round(prob, 3), "mismatch": prob >= threshold,
                    "emotion": ckpt.get("emotions", [])[int(res["emotion_logits"][0].argmax())] if ckpt.get("emotions") else None})
    return out


# --------------------------------------------------------------------------- analyzer
def session_stats(m: dict[str, Any]) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    voice = m.get("voice")
    if voice:
        pitch = [p for p in voice["series"]["pitch_hz"] if p]
        energy = [e for e in voice["series"]["energy_db"] if e is not None]
        stats["pitch_median"] = float(np.median(pitch)) if pitch else None
        stats["energy_median"] = float(np.median(energy)) if energy else None
    return stats


def _mean_dicts(dicts: list[dict[str, Any]], weights: list[float]) -> dict[str, float]:
    keys = {k for d in dicts for k in d}
    out = {}
    for k in keys:
        pairs = [(w, d[k]) for d, w in zip(dicts, weights) if d.get(k) is not None]
        if pairs:
            out[k] = sum(w * v for w, v in pairs) / sum(w for w, _ in pairs)
    return out


class FusionAnalyzer(Analyzer):
    name = "fusion"
    requires = ("ingest",)
    MODULES = ("asr", "voice", "face", "gaze", "physio", "face_va", "audio_va", "content")

    def __init__(self, force_rule: bool = False) -> None:
        self.force_rule = force_rule  # eval/fusion_eval.py compares learned vs rule on the same session

    def analyze(self, ctx: SessionContext) -> dict[str, Any]:
        cfg = load_trait_config()
        duration = float(self.need(ctx, "ingest")["duration_s"])
        m = {name: ctx.read(name) for name in self.MODULES}
        words = (m["asr"] or {}).get("words", [])
        segments = build_segments(duration, words, get_settings().segment_seconds)
        stats = session_stats(m)

        learned: list[dict[str, Any] | None] = [None] * len(segments)
        method = "rule"
        if (not self.force_rule and head_path().exists() and torch_available() and ctx.path("audio_emb.npz").exists()
                and ctx.path("face_emb.npz").exists()):
            try:
                learned = learned_segments(ctx, segments)
                method = "learned+rule"
            except Exception:  # noqa: BLE001 - keep the rule baseline live
                method = "rule"
        threshold = float(cfg["mismatch"]["distance_threshold"])

        # Proxy V-A is z-scored within the session: fine for face-vs-voice congruence, meaningless as an absolute score.
        exclude = set()
        if (m["audio_va"] or {}).get("proxy"):
            exclude |= {"audio_valence", "audio_arousal"}
        if (m["face_va"] or {}).get("proxy"):
            exclude |= {"face_valence", "face_arousal", "face_va_std"}

        mismatches = []
        for seg, lrn in zip(segments, learned):
            feats = segment_features(seg, m, stats)
            good = goodness(feats, cfg, exclude)
            seg["features"] = {k: rnd(v, 3) for k, v in feats.items()}
            seg["goodness"] = good
            seg["traits"] = trait_scores(good, cfg)
            seg["weakest"] = (cfg["features"][min(good, key=good.get)]["label"].lower() if good else None)
            dist = lrn["distance"] if lrn else va_distance(feats)
            rules = rule_mismatch(feats, cfg)
            flagged_learned = bool(lrn and lrn["mismatch"])
            flagged_distance = dist is not None and dist > threshold
            seg["congruence_distance"] = rnd(dist, 3)
            seg["mismatch_rule"] = rules
            seg["learned"] = lrn
            seg["mismatch"] = flagged_learned if lrn else (flagged_distance or bool(rules))
            if seg["mismatch"]:
                reason = rules[0] if rules else ("Voice and face convey different emotions"
                                                 + (f" (p={lrn['mismatch_prob']:.2f})" if lrn else f" (distance {dist:.2f})"))
                mismatches.append({"segment": seg["id"], "start": seg["start"], "end": seg["end"],
                                   "distance": rnd(dist, 3), "reason": reason, "text": seg["text"]})

        weights = [s["end"] - s["start"] for s in segments]
        mean_good = _mean_dicts([s["goodness"] for s in segments], weights)
        mean_raw = _mean_dicts([s["features"] for s in segments], weights)
        traits = {}
        for trait, spec in cfg["traits"].items():
            vals = [(w, s["traits"][trait]) for s, w in zip(segments, weights) if s["traits"].get(trait) is not None]
            score = sum(w * v for w, v in vals) / sum(w for w, _ in vals) if vals else None
            traits[trait] = {"label": spec["label"], "score": rnd(score, 1), **explain(trait, mean_good, mean_raw, cfg)}

        dists = [s["congruence_distance"] for s in segments if s["congruence_distance"] is not None]
        proxy = any((m[k] or {}).get("proxy") for k in ("face_va", "audio_va"))
        congruence = {
            "score": rnd(100 * (1 - np.mean(dists) / MAX_VA_DISTANCE), 1) if dists else None,
            "mean_distance": rnd(nanmean(dists), 3), "threshold": threshold, "method": method,
            "mismatch_count": len(mismatches), "proxy_inputs": proxy,
        }
        return {"segments": segments, "traits": traits, "congruence": congruence, "mismatches": mismatches,
                "feature_means": {k: rnd(v, 3) for k, v in mean_raw.items()},
                "goodness_means": {k: rnd(v, 3) for k, v in mean_good.items()},
                "method": method, "models": {"fusion": method}}
