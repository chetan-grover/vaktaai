"""Shared setup for eval/ and training/ scripts: import paths, relaxed duration limits, CSV/plot output."""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "backend", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
# Benchmark clips are often shorter/longer than the app's 30 s - 5 min upload window.
os.environ.setdefault("VAKTA_MIN_DURATION_S", "1")
os.environ.setdefault("VAKTA_MAX_DURATION_S", "7200")

RESULTS = ROOT / "eval" / "results"


def write_csv(name: str, rows: Iterable[dict[str, Any]]) -> Path:
    rows = list(rows)
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path.relative_to(ROOT)} ({len(rows)} rows)")
    return path


def save_plot(fig, name: str) -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def videos_in(folder: Path, patterns: tuple[str, ...] = ("*.mp4", "*.webm", "*.mov", "*.avi", "*.mkv")) -> list[Path]:
    return sorted(p for pat in patterns for p in folder.glob(pat))
