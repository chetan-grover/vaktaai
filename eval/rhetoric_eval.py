"""E4: agreement with the 100 hand labels (gold.jsonl from training/e4_qlora.ipynb):
F1 per device type (segment-level presence), MAE on ethos/pathos/logos, JSON validity.
Systems: any OpenAI-compatible endpoints (fine-tuned vs base prompting) + the heuristic fallback.

    python eval/rhetoric_eval.py --gold data/eval/gold.jsonl \
        --endpoint qlora=http://127.0.0.1:8001 --endpoint prompting=http://127.0.0.1:8002 --heuristic
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

from common import write_csv

from pipeline.content import (DEVICE_TYPES, RhetoricReport, heuristic_rhetoric, llm_rhetoric, structure_checks)
from pipeline.text import split_sentences


def score(predict: Callable[[str], RhetoricReport], gold: list[dict]) -> dict[str, float]:
    valid, maes = 0, []
    f1: dict[str, list[float]] = {t: [] for t in DEVICE_TYPES}
    for row in gold:
        try:
            pred = predict(row["text"])
            valid += 1
        except Exception:  # noqa: BLE001 - counts as invalid output
            continue
        ref = RhetoricReport.model_validate(row["label"])
        for t in DEVICE_TYPES:
            p, g = any(d.type == t for d in pred.devices), any(d.type == t for d in ref.devices)
            if p or g:
                f1[t].append(2 * (p and g) / (p + g))
        maes.append(sum(abs(getattr(pred.appeals, k) - getattr(ref.appeals, k)) for k in ("ethos", "pathos", "logos")) / 3)
    out = {"json_valid": valid / len(gold), "appeal_mae": sum(maes) / max(len(maes), 1)}
    out.update({f"f1_{t}": sum(v) / len(v) for t, v in f1.items() if v})
    return out


def heuristic(text: str) -> RhetoricReport:
    sentences = split_sentences(text)
    return heuristic_rhetoric(sentences, structure_checks(sentences))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gold", type=Path, required=True)
    ap.add_argument("--endpoint", action="append", default=[], help="name=url")
    ap.add_argument("--model", default="vakta-rhetoric")
    ap.add_argument("--heuristic", action="store_true")
    args = ap.parse_args()
    gold = [json.loads(line) for line in args.gold.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for spec in args.endpoint:
        name, url = spec.split("=", 1)
        rows.append({"system": name, **score(lambda t, u=url: llm_rhetoric(t, u, args.model, retries=1), gold)})
    if args.heuristic:
        rows.append({"system": "heuristic", **score(heuristic, gold)})
    write_csv("e4_rhetoric.csv", rows)


if __name__ == "__main__":
    main()
