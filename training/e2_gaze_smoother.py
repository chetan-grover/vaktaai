"""E2: train the GRU gaze smoother on your own labelled clips.

Recording protocol (labels come from timing, so no manual annotation is needed):
  1. Stick 9 targets on the wall in a 3x3 grid around the camera (camera = centre target).
  2. Start recording, look at the camera for 3 s (calibration).
  3. Look at each target for --hold seconds in row-major order from YOUR viewpoint:
     back-left, back-centre, back-right, mid-left, centre, mid-right, front-left, front-centre, front-right.
  Record 20-30 clips (different lighting, glasses on/off, head-only vs eyes-only movement).

Alternatively pass --labels labels.csv with columns clip,start,end,cell for free-form clips.

    python training/e2_gaze_smoother.py --clips data/gaze_clips --hold 3
Writes models/gaze/gaze_gru.pt and eval/results/e2_gaze.csv (raw vs mode filter vs GRU, held-out clips).
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval.common import videos_in, write_csv  # noqa: E402

CALIBRATION_S = 3.0
TRANSITION_S = 0.6
IGNORE = -100


def protocol_labels(t: np.ndarray, hold: float) -> np.ndarray:
    labels = np.full(len(t), IGNORE)
    for cell in range(9):
        start = CALIBRATION_S + cell * hold
        mask = (t >= start + TRANSITION_S) & (t < start + hold)
        labels[mask] = cell
    labels[(t >= 0.5) & (t < CALIBRATION_S)] = 4
    return labels


def csv_labels(t: np.ndarray, rows: list[dict[str, str]]) -> np.ndarray:
    labels = np.full(len(t), IGNORE)
    for r in rows:
        labels[(t >= float(r["start"])) & (t < float(r["end"]))] = int(r["cell"])
    return labels


def clip_sequence(path: Path, hold: float, label_rows: list[dict[str, str]] | None):
    from pipeline.gaze import angles_to_cell, smoother_features
    from pipeline.runner import analyse_file

    ctx = analyse_file(path, ["face"])
    ctx.write_json("meta.json", {"source": "webcam", "calibration_s": CALIBRATION_S})
    analyse_file(path, ["gaze"], session_id=ctx.session_id)
    gaze, face = ctx.read("gaze"), ctx.read("face")
    if gaze is None:
        raise RuntimeError(f"gaze failed on {path.name}")
    t = np.asarray(gaze["t"], dtype=float)
    arr = lambda xs: np.asarray([np.nan if v is None else v for v in xs], dtype=float)  # noqa: E731
    yaw_rel, pitch_rel = arr(gaze["yaw_rel"]), arr(gaze["pitch_rel"])
    calib = gaze["calibration"]
    head_yaw = arr(face["frames"]["yaw"]) - calib["yaw0"]
    head_pitch = arr(face["frames"]["pitch"]) - calib["pitch0"]
    feats = smoother_features(yaw_rel, pitch_rel, head_yaw, head_pitch)
    raw = np.array([angles_to_cell(y, p) for y, p in zip(yaw_rel, pitch_rel)])
    labels = csv_labels(t, label_rows) if label_rows else protocol_labels(t, hold)
    return feats, raw, np.asarray(gaze["cells"]), labels


def accuracy(pred: np.ndarray, labels: np.ndarray) -> float:
    mask = labels != IGNORE
    return float((pred[mask] == labels[mask]).mean()) if mask.any() else float("nan")


def train(sequences, epochs: int = 300, seed: int = 13):
    import torch

    from pipeline.nets import GazeSmoother

    torch.manual_seed(seed)
    model = GazeSmoother()
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=IGNORE)
    for _ in range(epochs):
        model.train()
        for feats, _, _, labels in random.sample(sequences, len(sequences)):
            x = torch.from_numpy(feats)[None]
            y = torch.from_numpy(labels.astype(np.int64))[None]
            # random crop for robustness to clip length
            if x.shape[1] > 80:
                s = random.randint(0, x.shape[1] - 80)
                x, y = x[:, s:s + 80], y[:, s:s + 80]
            loss = loss_fn(model(x).reshape(-1, 9), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model.eval()


def predict(model, feats: np.ndarray) -> np.ndarray:
    import torch

    with torch.no_grad():
        return model(torch.from_numpy(feats)[None])[0].argmax(-1).numpy()


def main() -> None:
    import torch

    from pipeline.settings import get_settings

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--hold", type=float, default=3.0)
    ap.add_argument("--labels", type=Path, default=None)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--test-frac", type=float, default=0.25)
    args = ap.parse_args()

    by_clip: dict[str, list[dict[str, str]]] = {}
    if args.labels:
        with args.labels.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                by_clip.setdefault(row["clip"], []).append(row)
    clips = videos_in(args.clips)
    if len(clips) < 4:
        sys.exit("need at least 4 labelled clips")
    sequences = [clip_sequence(c, args.hold, by_clip.get(c.name)) for c in clips]
    random.seed(get_settings().seed)
    order = random.sample(range(len(clips)), len(clips))
    n_test = max(1, int(len(clips) * args.test_frac))
    test_idx, train_idx = order[:n_test], order[n_test:]
    model = train([sequences[i] for i in train_idx], args.epochs, get_settings().seed)

    rows = []
    for i in test_idx:
        feats, raw, mode, labels = sequences[i]
        rows.append({"clip": clips[i].name, "raw_l2cs_or_mesh": accuracy(raw, labels),
                     "mode_filter": accuracy(mode, labels), "gru": accuracy(predict(model, feats), labels)})
    rows.append({"clip": "MEAN", **{k: float(np.nanmean([r[k] for r in rows])) for k in ("raw_l2cs_or_mesh", "mode_filter", "gru")}})
    write_csv("e2_gaze.csv", rows)
    out = get_settings().models_dir / "gaze" / "gaze_gru.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": {"in_features": 4, "hidden": 32, "cells": 9}, "state_dict": model.state_dict()}, out)
    print(f"saved {out}; held-out 9-cell accuracy: {rows[-1]}")


if __name__ == "__main__":
    main()
