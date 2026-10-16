"""E2: 9-cell gaze accuracy on own labelled clips (protocol from training/e2_gaze_smoother.py).
Rows: head-pose only (Face Mesh) | raw gaze (L2CS or mesh+iris) | + mode filter | + trained GRU.

    python eval/gaze_eval.py --clips data/gaze_clips_test --hold 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from common import videos_in, write_csv

from pipeline.gaze import angles_to_cell, mode_filter
from pipeline.settings import get_settings
from training.e2_gaze_smoother import accuracy, clip_sequence, predict


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--hold", type=float, default=3.0)
    args = ap.parse_args()
    gru = None
    gru_path = get_settings().models_dir / "gaze" / "gaze_gru.pt"
    if gru_path.exists():
        from pipeline.gaze import load_smoother

        gru = load_smoother(str(gru_path))
    rows = []
    for clip in videos_in(args.clips):
        feats, raw, _, labels = clip_sequence(clip, args.hold, None)
        head_cells = np.array([angles_to_cell(y * 30, p * 30) for y, p in feats[:, 2:4]])
        row = {"clip": clip.name, "headpose_only": accuracy(head_cells, labels), "raw_gaze": accuracy(raw, labels),
               "mode_filter": accuracy(np.array(mode_filter(raw.tolist())), labels)}
        if gru is not None:
            row["gru"] = accuracy(predict(gru, feats), labels)
        rows.append(row)
    keys = [k for k in rows[0] if k != "clip"] if rows else []
    rows.append({"clip": "MEAN", **{k: float(np.nanmean([r[k] for r in rows])) for k in keys}})
    write_csv("e2_gaze_eval.csv", rows)


if __name__ == "__main__":
    main()
