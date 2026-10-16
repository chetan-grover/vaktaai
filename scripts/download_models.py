"""Download pretrained weights into models/ (nothing is fetched unless you name it).

    python scripts/download_models.py --list
    python scripts/download_models.py --get whisper-small hsemotion audio-va physnet

Trained-by-you artefacts (not downloadable): models/fusion/e3_head.pt (E3 notebook),
models/gaze/gaze_gru.pt (training/e2_gaze_smoother.py), models/persona/* (E5),
models/rhetoric/merged (E4 notebook, served by training/serve_rhetoric.py).
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from pipeline.settings import get_settings  # noqa: E402

MODELS = {
    "whisper-small": ("~480 MB", "MIT", "F3 ASR (default size)"),
    "whisper-large-v3": ("~3 GB", "MIT", "F3 ASR on GPU"),
    "audio-va": ("~1.2 GB", "CC BY-NC-SA 4.0", "E3 audeering wav2vec2 dimensional emotion (MSP-Podcast)"),
    "hsemotion": ("~16 MB", "Apache-2.0", "E3 face valence-arousal (AffectNet)"),
    "physnet": ("~3 MB", "rPPG-Toolbox licence (research)", "E1 PhysNet trained on PURE"),
    "l2cs": ("~96 MB", "MIT code; Gaze360 non-commercial", "E2 L2CS-Net Gaze360 (py-feat safetensors re-pack)"),
}
PHYSNET_URL = "https://github.com/ubicomplab/rPPG-Toolbox/raw/main/final_model_release/PURE_PhysNet_DiffNormalized.pth"
L2CS_REPO, L2CS_FILE = "py-feat/l2cs", "l2cs_gaze360_resnet50.safetensors"  # original Drive link is dead


def get(name: str, models_dir: Path) -> None:
    if name.startswith("whisper-"):
        from faster_whisper import WhisperModel

        WhisperModel(name.split("-", 1)[1], device="cpu", compute_type="int8", download_root=str(models_dir / "whisper"))
    elif name == "audio-va":
        import shutil

        from huggingface_hub import snapshot_download

        from pipeline.affect import AUDIO_MODEL_ID, audio_model_dir

        # Plain folder, no symlinks (Windows without Developer Mode cannot create them).
        snapshot_download(AUDIO_MODEL_ID, local_dir=str(audio_model_dir()))
        shutil.rmtree(models_dir / "hf" / f"models--{AUDIO_MODEL_ID.replace('/', '--')}", ignore_errors=True)
    elif name == "hsemotion":
        from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

        from pipeline.affect import FACE_MODEL_ID

        HSEmotionRecognizer(model_name=FACE_MODEL_ID)
    elif name == "physnet":
        target = models_dir / "rppg" / "PURE_PhysNet_DiffNormalized.pth"
        target.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(PHYSNET_URL, target)
    elif name == "l2cs":
        from huggingface_hub import hf_hub_download

        hf_hub_download(L2CS_REPO, L2CS_FILE, local_dir=str(models_dir / "gaze"))
    else:
        raise SystemExit(f"unknown model '{name}'")
    print("ok:", name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--get", nargs="+", default=[])
    args = ap.parse_args()
    models_dir = get_settings().models_dir
    if args.list or not args.get:
        for name, (size, licence, use) in MODELS.items():
            print(f"{name:<18} {size:<9} {licence:<34} {use}")
        print("\nMediaPipe Face Mesh ships inside the mediapipe package; no download needed.")
    for name in args.get:
        get(name, models_dir)


if __name__ == "__main__":
    main()
