"""E3: mismatch detection on own "acted mismatch" clips, learned head vs rule-based fusion.

labels.csv columns: clip,mismatch   (1 = e.g. smiling while giving bad news, 0 = congruent delivery)
RAVDESS held-out emotion accuracy / CCC come from training/e3_fusion.ipynb (e3_results.csv).

    python eval/fusion_eval.py --clips data/eval/mismatch --labels data/eval/mismatch/labels.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import write_csv

from pipeline.fusion import FusionAnalyzer
from pipeline.runner import analyse_file

STAGES = ["asr", "voice", "audio_va", "face", "face_va"]


def prf(pred: list[int], gold: list[int]) -> dict[str, float]:
    tp = sum(p and g for p, g in zip(pred, gold))
    fp = sum(p and not g for p, g in zip(pred, gold))
    fn = sum(g and not p for p, g in zip(pred, gold))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    acc = sum(p == g for p, g in zip(pred, gold)) / len(gold) if gold else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": acc}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    args = ap.parse_args()
    with args.labels.open(encoding="utf-8") as fh:
        labels = {r["clip"]: int(r["mismatch"]) for r in csv.DictReader(fh)}
    rows, gold, pred_rule, pred_learned = [], [], [], []
    learned_available = False
    for clip, label in labels.items():
        ctx = analyse_file(args.clips / clip, STAGES)
        rule = FusionAnalyzer(force_rule=True).run(ctx)["data"]
        learned = FusionAnalyzer().run(ctx)["data"]
        learned_available |= learned["method"] != "rule"
        gold.append(label)
        pred_rule.append(int(bool(rule["mismatches"])))
        pred_learned.append(int(bool(learned["mismatches"])))
        rows.append({"clip": clip, "gold": label, "rule": pred_rule[-1], "learned": pred_learned[-1],
                     "rule_congruence": rule["congruence"]["score"], "learned_congruence": learned["congruence"]["score"]})
    write_csv("e3_mismatch_clips.csv", rows)
    summary = [{"system": "rule-based fusion", **prf(pred_rule, gold)}]
    if learned_available:
        summary.append({"system": "learned cross-attention", **prf(pred_learned, gold)})
    else:
        print("models/fusion/e3_head.pt not found: only the rule baseline was scored")
    write_csv("e3_mismatch_summary.csv", summary)


if __name__ == "__main__":
    main()
