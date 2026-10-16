"""Whole system: before/after-drills study (5-8 participants, 2 sessions each; get consent first).
For every user with >= 2 finished sessions, compares the first and the latest session.

    python eval/user_study.py
"""
from __future__ import annotations

import json

import numpy as np
from common import save_plot, write_csv

from app.db import SpeechSession, get_engine
from pipeline.settings import get_settings
from sqlmodel import Session, select

TRAITS = ("confidence", "authoritativeness", "engagement", "congruence")
METRICS = {"wpm": ("voice", "summary", "wpm"), "fillers_per_min": ("voice", "summary", "filler_rate_per_min"),
           "eye_contact": ("gaze", "summary", "eye_contact_ratio"), "gaze_coverage": ("gaze", "summary", "coverage")}


def dig(d, path):
    for k in path:
        d = d.get(k) if isinstance(d, dict) else None
    return d


def report_of(session_id: str) -> dict:
    path = get_settings().sessions_dir / session_id / "report.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> None:
    with Session(get_engine()) as db:
        rows = db.exec(select(SpeechSession).where(SpeechSession.status == "done").order_by(SpeechSession.created_at)).all()
    by_user: dict[int, list[SpeechSession]] = {}
    for r in rows:
        by_user.setdefault(r.user_id, []).append(r)
    out = []
    for user, sessions in by_user.items():
        if len(sessions) < 2:
            continue
        first, last = sessions[0], sessions[-1]
        rf, rl = report_of(first.id), report_of(last.id)
        row = {"user_id": user, "sessions": len(sessions), "persona_before": first.persona, "persona_after": last.persona}
        for t in TRAITS:
            a, b = getattr(first, t), getattr(last, t)
            row[f"{t}_delta"] = None if a is None or b is None else b - a
        for name, path in METRICS.items():
            a, b = dig(rf, path), dig(rl, path)
            row[f"{name}_delta"] = None if a is None or b is None else b - a
        out.append(row)
    if not out:
        print("no user has two finished sessions yet")
        return
    write_csv("user_study.csv", out)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    deltas = [[r[f"{t}_delta"] for r in out if r[f"{t}_delta"] is not None] for t in TRAITS]
    ax.bar(TRAITS, [np.mean(d) if d else 0 for d in deltas], yerr=[np.std(d) if d else 0 for d in deltas], capsize=4)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("score change (after - before)")
    ax.set_title(f"Before/after drills, n={len(out)} (descriptive)")
    save_plot(fig, "user_study.png")


if __name__ == "__main__":
    main()
