"""F14 trait-to-drill engine: weaknesses from every module (incl. E1, E2, E3, E4) -> 1-3 drills
each, with the evidence segment attached (FR8). Rules and the drill library live in config/drills.yaml."""
from __future__ import annotations

import operator
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from .base import Analyzer, SessionContext
from .utils import rnd

CONFIG = Path(__file__).parent / "config" / "drills.yaml"
OPS = {">": operator.gt, ">=": operator.ge, "<": operator.lt, "<=": operator.le}
MAX_WEAKNESSES = 6


@lru_cache
def load_drill_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def collect_metrics(ctx: SessionContext) -> dict[str, float]:
    """Flatten module summaries into 'module.metric' keys used by the weakness rules."""
    metrics: dict[str, Any] = {}
    for name in ("voice", "face", "face_va", "physio"):
        data = ctx.read(name)
        if data:
            metrics.update({f"{name}.{k}": v for k, v in data.get("summary", {}).items()})
    gaze = ctx.read("gaze")
    if gaze:
        metrics.update({f"gaze.{k}": v for k, v in gaze["summary"].items() if not isinstance(v, list)})
        metrics["gaze.neglected_count"] = len(gaze["summary"]["neglected_cells"])
    content = ctx.read("content")
    if content:
        rh, st = content["rhetoric"], content["structure"]
        metrics.update({
            "content.opener_score": st["opener"]["score"], "content.closer_score": st["closer"]["score"],
            "content.transition_quality": rh["transition_quality"], "content.device_count": len(rh["devices"]),
            **{f"content.{k}": v for k, v in rh["appeals"].items()},
        })
    fusion = ctx.read("fusion")
    if fusion:
        metrics.update({f"fusion.{k}": v for k, v in fusion["feature_means"].items()})
        metrics["fusion.mismatch_count"] = len(fusion["mismatches"])
        metrics["fusion.congruence_score"] = fusion["congruence"]["score"]
        metrics.update({f"traits.{k}": v["score"] for k, v in fusion["traits"].items()})
    return {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _segment_value(seg: dict[str, Any], feature: str) -> float | None:
    for source in (seg.get("features", {}), seg.get("traits", {}), seg):
        value = source.get(feature)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def find_evidence(spec: dict[str, Any] | None, segments: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not spec or not segments:
        return None
    direction = spec["direction"]
    if direction in ("first", "last"):
        seg, value = segments[0 if direction == "first" else -1], None
    else:
        scored = [(v, s) for s in segments if (v := _segment_value(s, spec["feature"])) is not None]
        if not scored:
            return None
        value, seg = (max if direction == "max" else min)(scored, key=lambda p: p[0])
    return {"segment": seg["id"], "start": seg["start"], "end": seg["end"], "value": rnd(value, 2),
            "text": seg.get("text", "")[:240]}


def assign_drills(metrics: dict[str, float], segments: list[dict[str, Any]],
                  cfg: dict[str, Any]) -> list[dict[str, Any]]:
    library = cfg["drills"]
    found = []
    for rule in cfg["weaknesses"]:
        value = metrics.get(rule["metric"])
        if value is None or not OPS[rule["op"]](value, rule["value"]):
            continue
        severity = min(abs(value - rule["value"]) / rule["scale"], 1.0)
        drills = [{"id": d, **library[d]} for d in rule["drills"][:3] if d in library]
        found.append({"id": rule["id"], "label": rule["label"], "metric": rule["metric"],
                      "value": rnd(value, 3), "threshold": rule["value"], "severity": rnd(max(severity, 0.05), 2),
                      "evidence": find_evidence(rule.get("evidence"), segments), "drills": drills})
    found.sort(key=lambda w: w["severity"], reverse=True)
    return found[:MAX_WEAKNESSES]


class DrillsAnalyzer(Analyzer):
    name = "drills"
    requires = ("fusion",)

    def analyze(self, ctx: SessionContext) -> dict[str, Any]:
        cfg = load_drill_config()
        metrics = collect_metrics(ctx)
        segments = (ctx.read("fusion") or {}).get("segments", [])
        weaknesses = assign_drills(metrics, segments, cfg)
        seen, plan = set(), []
        for w in weaknesses:
            for d in w["drills"]:
                if d["id"] not in seen:
                    seen.add(d["id"])
                    plan.append({**d, "weakness": w["id"], "evidence": w["evidence"]})
        return {"weaknesses": weaknesses, "plan": plan, "metrics": metrics}
