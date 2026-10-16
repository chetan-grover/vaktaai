"""NFR1: end-to-end analysis time per clip and per stage (target: 3-min speech in <= 3 min on a T4).

    python eval/latency.py --clips tests/fixtures
"""
from __future__ import annotations

import argparse
import shutil
import time
import uuid
from pathlib import Path

from common import videos_in, write_csv

from pipeline.base import SessionContext
from pipeline.runner import STAGE_NAMES, run_pipeline
from pipeline.utils import resolve_device


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=Path, required=True)
    args = ap.parse_args()
    rows = []
    for clip in videos_in(args.clips):
        sid = f"latency-{uuid.uuid4().hex[:6]}"
        ctx = SessionContext.for_session(sid)
        shutil.copy(clip, ctx.session_dir / f"input{clip.suffix.lower()}")
        start = time.perf_counter()
        report = run_pipeline(sid)
        total = time.perf_counter() - start
        row = {"clip": clip.name, "device": resolve_device(), "duration_s": report["duration_s"], "total_s": round(total, 1),
               "realtime_factor": round(total / max(report["duration_s"] or 1, 1e-6), 2)}
        row.update({f"{name}_s": (report["modules"].get(name) or {}).get("elapsed_s") for name in STAGE_NAMES[:-1]})
        rows.append(row)
        print(row)
    write_csv("latency.csv", rows)


if __name__ == "__main__":
    main()
