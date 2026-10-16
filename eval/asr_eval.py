"""ASR + fillers: WER and filler precision/recall on own hand-labelled clips, Whisper small vs large-v3.

Each clip <name>.mp4 needs <name>.txt with the verbatim reference, fillers included
("um so, like, the plan ..."; soft fillers such as 'like,' marked with a comma).

    python eval/asr_eval.py --clips data/eval/asr --models small large-v3
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

from common import videos_in, write_csv

from pipeline.asr import transcribe
from pipeline.runner import analyse_file
from pipeline.voice import detect_fillers


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s']", " ", text.lower())).strip()


def fillers_of_text(text: str) -> Counter:
    words = [{"w": w, "start": i * 0.3, "end": i * 0.3 + 0.28} for i, w in enumerate(text.split())]
    return Counter(f["text"] for f in detect_fillers(words))


def filler_pr(pred: Counter, ref: Counter) -> tuple[float, float]:
    tp = sum(min(pred[k], ref[k]) for k in pred)
    return (tp / sum(pred.values()) if pred else 1.0), (tp / sum(ref.values()) if ref else 1.0)


def main() -> None:
    import jiwer

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=["small", "large-v3"])
    ap.add_argument("--language", default=None)
    args = ap.parse_args()
    rows = []
    for clip in videos_in(args.clips, ("*.mp4", "*.webm", "*.mov", "*.wav")):
        ref_path = clip.with_suffix(".txt")
        if not ref_path.exists():
            print("no reference for", clip.name)
            continue
        ref = ref_path.read_text(encoding="utf-8")
        ctx = analyse_file(clip, [])
        for size in args.models:
            hyp = transcribe(str(ctx.audio_path), size, args.language)
            p, r = filler_pr(Counter(f["text"] for f in detect_fillers(hyp["words"])), fillers_of_text(ref))
            rows.append({"clip": clip.name, "model": size, "wer": jiwer.wer(normalise(ref), normalise(hyp["text"])),
                         "filler_precision": p, "filler_recall": r})
    for size in args.models:
        sub = [r for r in rows if r["model"] == size]
        if sub:
            rows.append({"clip": "MEAN", "model": size, **{k: sum(r[k] for r in sub) / len(sub)
                                                         for k in ("wer", "filler_precision", "filler_recall")}})
    write_csv("asr.csv", rows)


if __name__ == "__main__":
    main()
