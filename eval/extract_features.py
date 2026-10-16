"""E5 data: run the full pipeline on a folder of public speech clips and write their 24-d persona
vectors to data/persona/public_features.csv (input for analytics/persona/train.py).

    python eval/extract_features.py --clips data/public_clips
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from common import ROOT, videos_in

from analytics.persona.features import FEATURE_NAMES, session_vector
from pipeline.report import build_report
from pipeline.runner import analyse_file

ALL_STAGES = ["asr", "voice", "audio_va", "content", "face", "gaze", "face_va", "physio", "fusion", "timeline", "drills"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "persona" / "public_features.csv")
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    clips = videos_in(args.clips)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["clip", *FEATURE_NAMES])
        writer.writeheader()
        for i, clip in enumerate(clips, 1):
            try:
                ctx = analyse_file(clip, ALL_STAGES)
                writer.writerow({"clip": clip.name, **session_vector(build_report(ctx))})
                fh.flush()
            except Exception as exc:  # noqa: BLE001 - one bad clip should not stop the batch
                print("skip", clip.name, exc)
            print(f"{i}/{len(clips)} {clip.name}")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
