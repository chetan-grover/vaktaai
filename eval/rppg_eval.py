"""E1: heart-rate MAE / RMSE on UBFC-rPPG (DATASET_2 layout: subjectN/vid.avi + ground_truth.txt).
Compares PhysNet (if models/rppg/*PhysNet*.pth exists) with the classical POS and CHROM baselines.

    python eval/rppg_eval.py --ubfc data/UBFC/DATASET_2 --methods physnet pos chrom
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from common import save_plot, write_csv

from pipeline.face import FaceTrack
from pipeline.physio import consensus_windows, estimate_bvp, extract_trace, physnet_weights, windowed_estimates
from pipeline.runner import analyse_file


def ground_truth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """UBFC DATASET_2 ground_truth.txt: line 1 PPG, line 2 HR (BPM), line 3 time (s)."""
    lines = [np.array(l.split(), dtype=float) for l in path.read_text().strip().splitlines()]
    return lines[2], lines[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ubfc", type=Path, required=True)
    ap.add_argument("--methods", nargs="+", default=["physnet", "pos", "chrom"])
    ap.add_argument("--reliable-only", action="store_true", help="score only windows with SQI above threshold")
    args = ap.parse_args()
    methods = [m for m in args.methods if m != "physnet" or physnet_weights() is not None]
    rows, pairs = [], {m: [] for m in [*methods, "consensus"]}
    for subject in sorted(p for p in args.ubfc.iterdir() if (p / "vid.avi").exists()):
        gt_t, gt_hr = ground_truth(subject / "ground_truth.txt")
        ctx = analyse_file(subject / "vid.avi", ["face"], session_id=f"ubfc-{subject.name}")
        face = ctx.read("face")
        if face is None:
            print("no face in", subject.name)
            continue
        times, rgb, fps, crops = extract_trace(ctx.video_path, FaceTrack.from_face(face), keep_crops="physnet" in methods)
        per_method = {m: windowed_estimates(estimate_bvp(times, rgb, fps, crops, m), fps) for m in methods}
        per_method["consensus"] = consensus_windows({m: per_method[m] for m in methods})
        for method in [*methods, "consensus"]:
            windows = per_method[method]
            errs = []
            for w in windows:
                mask = (gt_t >= w["start"]) & (gt_t < w["end"])
                if w["bpm"] is None or not mask.any() or (args.reliable_only and not w["reliable"]):
                    continue
                truth = float(gt_hr[mask].mean())
                errs.append(w["bpm"] - truth)
                pairs[method].append((truth, w["bpm"]))
            if errs:
                e = np.abs(errs)
                rows.append({"subject": subject.name, "method": method, "mae_bpm": e.mean(),
                             "rmse_bpm": float(np.sqrt(np.mean(np.square(errs)))), "windows": len(errs)})
    for method in [*methods, "consensus"]:
        if pairs[method]:
            t, p = np.array(pairs[method]).T
            rows.append({"subject": "ALL", "method": method, "mae_bpm": float(np.abs(p - t).mean()),
                         "rmse_bpm": float(np.sqrt(np.mean((p - t) ** 2))), "pearson_r": float(np.corrcoef(t, p)[0, 1]),
                         "windows": len(t)})
    write_csv("e1_rppg.csv", rows)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 5))
    for method in methods:
        if pairs[method]:
            t, p = np.array(pairs[method]).T
            ax.scatter(t, p, s=10, label=method)
    ax.plot([40, 140], [40, 140], "k--", lw=1)
    ax.set_xlabel("ground-truth BPM")
    ax.set_ylabel("estimated BPM")
    ax.legend()
    save_plot(fig, "e1_rppg_scatter.png")


if __name__ == "__main__":
    main()
