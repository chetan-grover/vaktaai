"""E3 / F6 / F19 inputs: per-modality valence-arousal (V-A) streams in [-1, 1].

audio_va: audeering wav2vec2 dimensional model (MSP-Podcast fine-tuned), 2 s windows, 0.5 s hop.
          Fallback: prosody proxy from voice.json (energy + pitch), labelled as such.
face_va:  HSEmotion enet_b0_8_va_mtl (AffectNet-trained), on face crops at 2 fps.
          Fallback: landmark proxy (smile / corner lift -> valence, eyes + brows + mouth -> arousal).

When the learned models run, their embeddings are also saved (audio_emb.npz, face_emb.npz)
so fusion.py can feed the E3 cross-attention head.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from .base import Analyzer, AnalyzerSkipped, SessionContext
from .face import iter_face_crops
from .ingest import load_audio
from .settings import get_settings
from .utils import configure_torch, nanstd, per_second, resolve_device, rnd

AUDIO_MODEL_ID = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
FACE_MODEL_ID = "enet_b0_8_va_mtl"
STEP_S = 1.0
AUDIO_WIN_S = 2.0
AUDIO_CHUNK_S = 10.0
FACE_FPS = 2.0


def _zscore(x: np.ndarray) -> np.ndarray:
    m, s = np.nanmean(x), np.nanstd(x)
    return (x - m) / s if s > 1e-9 else np.zeros_like(x)


def va_summary(valence: list, arousal: list) -> dict[str, Any]:
    v = [x for x in valence if x is not None]
    a = [x for x in arousal if x is not None]
    return {"valence_mean": rnd(np.mean(v), 3) if v else None, "arousal_mean": rnd(np.mean(a), 3) if a else None,
            "valence_std": rnd(nanstd(v), 3), "arousal_std": rnd(nanstd(a), 3)}


# --------------------------------------------------------------------------- audio
def audio_model_dir() -> Path:
    return get_settings().models_dir / "audio_va"


@lru_cache(maxsize=1)
def load_audio_model(device: str):
    import torch
    from torch import nn
    from transformers import Wav2Vec2Model, Wav2Vec2PreTrainedModel, Wav2Vec2Processor

    class RegressionHead(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
            self.dropout = nn.Dropout(config.final_dropout)
            self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        def forward(self, x):
            return self.out_proj(self.dropout(torch.tanh(self.dense(self.dropout(x)))))

    class EmotionModel(Wav2Vec2PreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.wav2vec2 = Wav2Vec2Model(config)
            self.classifier = RegressionHead(config)
            self.post_init()

        def forward(self, input_values):
            hidden = self.wav2vec2(input_values)[0].mean(dim=1)
            return hidden, self.classifier(hidden)

    # Prefer the plain-folder copy from download_models.py (the HF cache needs symlinks, often blocked on Windows).
    folder = audio_model_dir()
    if (folder / "config.json").exists():
        source, kwargs = str(folder), {}
    else:
        source = AUDIO_MODEL_ID
        kwargs = {"cache_dir": str(get_settings().models_dir / "hf"), "local_files_only": not get_settings().allow_downloads}
    processor = Wav2Vec2Processor.from_pretrained(source, **kwargs)
    model = EmotionModel.from_pretrained(source, **kwargs)
    model = model.to(device).eval()
    if device == "cpu":
        import torch

        model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    return processor, model


def _chunk_bounds(n: int, sr: int) -> list[tuple[int, int]]:
    step = int(AUDIO_CHUNK_S * sr)
    bounds = [(i, min(i + step, n)) for i in range(0, n, step)]
    if len(bounds) > 1 and bounds[-1][1] - bounds[-1][0] < sr:  # fold a <1 s tail into the previous chunk
        bounds[-2:] = [(bounds[-2][0], n)]
    return bounds


def audio_model_va(samples: np.ndarray, sr: int, duration: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns t (n,), va (n, 2) and embeddings (n, 1027 = hidden 1024 + arousal/dominance/valence).

    wav2vec2 runs once per 10 s chunk; each 2 s window (1 s hop) mean-pools the frame-level hidden
    states it covers and goes through the regression head, mirroring the model's own pooling.
    """
    import torch

    configure_torch()
    device = resolve_device()
    processor, model = load_audio_model(device)
    frames, frame_t = [], []
    with torch.no_grad():
        for a, b in _chunk_bounds(len(samples), sr):
            inputs = processor(samples[a:b], sampling_rate=sr, return_tensors="pt").input_values.to(device)
            hidden = model.wav2vec2(inputs)[0][0].cpu()
            frames.append(hidden)
            frame_t.append(a / sr + (np.arange(len(hidden)) + 0.5) * (b - a) / sr / len(hidden))
        hidden_all, t_all = torch.cat(frames), np.concatenate(frame_t)
        ts, vas, embs = [], [], []
        for start in np.arange(0.0, max(duration - AUDIO_WIN_S, 0) + 1e-9, STEP_S):
            mask = torch.from_numpy((t_all >= start) & (t_all < start + AUDIO_WIN_S))
            if mask.sum() < 10:
                continue
            pooled = hidden_all[mask].mean(dim=0)
            arousal, dominance, valence = model.classifier(pooled.to(device)[None])[0].cpu().numpy().tolist()
            ts.append(start + AUDIO_WIN_S / 2)
            vas.append([2 * valence - 1, 2 * arousal - 1])
            embs.append(np.concatenate([pooled.numpy(), [arousal, dominance, valence]]))
    return np.asarray(ts), np.clip(np.asarray(vas), -1, 1), np.asarray(embs, dtype=np.float32)


def prosody_proxy_va(voice: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Arousal ~ loudness + pitch height; valence ~ pitch height + variation (weak proxy)."""
    energy = np.asarray([np.nan if v is None else v for v in voice["series"]["energy_db"]], dtype=float)
    pitch = np.asarray([np.nan if v is None else v for v in voice["series"]["pitch_hz"]], dtype=float)
    n = min(len(energy), len(pitch))
    energy, pitch = energy[:n], pitch[:n]
    pitch_var = np.array([np.nanstd(pitch[max(0, i - 2):i + 3]) if np.isfinite(pitch[max(0, i - 2):i + 3]).sum() > 1
                          else np.nan for i in range(n)])
    arousal = np.tanh(0.6 * _zscore(energy) + 0.4 * _zscore(pitch))
    valence = np.tanh(0.25 * _zscore(pitch) + 0.35 * _zscore(pitch_var))
    t = np.arange(n) + 0.5
    return t, np.stack([valence, arousal], axis=1)


class AudioAffectAnalyzer(Analyzer):
    name = "audio_va"
    requires = ("ingest", "voice")

    def analyze(self, ctx: SessionContext) -> dict[str, Any]:
        ingest = self.need(ctx, "ingest")
        if not ingest.get("has_audio"):
            raise AnalyzerSkipped("recording has no audio track")
        duration = float(ingest["duration_s"])
        backend, note = AUDIO_MODEL_ID, None
        try:
            samples, sr = load_audio(ctx.audio_path)
            t, va, emb = audio_model_va(samples, sr, duration)
            np.savez_compressed(ctx.path("audio_emb.npz"), t=t, emb=emb, va=va)
        except Exception as exc:  # noqa: BLE001 - model missing/offline -> proxy
            voice = self.need(ctx, "voice")
            t, va = prosody_proxy_va(voice)
            backend, note = "prosody-proxy", f"learned audio V-A unavailable ({type(exc).__name__})"
        valence = [rnd(v, 3) for v in va[:, 0]] if len(va) else []
        arousal = [rnd(a, 3) for a in va[:, 1]] if len(va) else []
        return {
            "t": [rnd(x, 2) for x in t], "valence": valence, "arousal": arousal,
            "per_second": {"valence": per_second(t, valence, duration, np.nanmean),
                           "arousal": per_second(t, arousal, duration, np.nanmean)},
            "summary": va_summary(valence, arousal),
            "proxy": backend == "prosody-proxy", "note": note,
            "models": {"audio_va": backend},
        }


# --------------------------------------------------------------------------- face
@lru_cache(maxsize=1)
def load_face_model():
    from hsemotion_onnx.facial_emotions import HSEmotionRecognizer

    return HSEmotionRecognizer(model_name=FACE_MODEL_ID)


def face_model_va(ctx: SessionContext, face: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cached = Path.home() / ".hsemotion" / f"{FACE_MODEL_ID}.onnx"
    if not get_settings().allow_downloads and not cached.exists():
        raise RuntimeError("HSEmotion weights not cached and downloads disabled")
    fer = load_face_model()
    ts, vas, embs = [], [], []
    for t, crop in iter_face_crops(ctx, face, fps=FACE_FPS, size=224, scale=1.1):
        if crop is None:
            continue
        _, scores = fer.predict_emotions(crop, logits=True)
        scores = np.asarray(scores, dtype=np.float32).ravel()
        ts.append(t)
        vas.append([scores[-2], scores[-1]])
        feats = getattr(fer, "extract_features", None)
        embs.append(np.concatenate([np.asarray(feats(crop), np.float32).ravel(), scores[-2:]]) if feats else scores)
    if not ts:
        raise RuntimeError("no face crops")
    return np.asarray(ts), np.clip(np.asarray(vas), -1, 1), np.asarray(embs, dtype=np.float32)


def landmark_proxy_va(face: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    f = face["frames"]
    arr = {k: np.asarray([np.nan if v is None else v for v in f[k]], dtype=float)
           for k in ("t", "smile", "corner_lift", "brow", "mouth_open", "ear")}
    valence = np.tanh(0.7 * _zscore(arr["smile"]) + 0.6 * _zscore(arr["corner_lift"]))
    arousal = np.tanh(0.4 * _zscore(arr["ear"]) + 0.4 * _zscore(arr["brow"]) + 0.4 * _zscore(arr["mouth_open"]))
    return arr["t"], np.stack([valence, arousal], axis=1)


class FaceAffectAnalyzer(Analyzer):
    name = "face_va"
    requires = ("ingest", "face")

    def analyze(self, ctx: SessionContext) -> dict[str, Any]:
        ingest = self.need(ctx, "ingest")
        face = self.need(ctx, "face")
        duration = float(ingest["duration_s"])
        backend, note = f"hsemotion/{FACE_MODEL_ID}", None
        try:
            t, va, emb = face_model_va(ctx, face)
            np.savez_compressed(ctx.path("face_emb.npz"), t=t, emb=emb, va=va)
        except Exception as exc:  # noqa: BLE001
            if face["frames"]["smile"].count(None) == len(face["frames"]["smile"]):
                raise AnalyzerSkipped("no landmarks for the face V-A proxy") from exc
            t, va = landmark_proxy_va(face)
            backend, note = "landmark-proxy", f"learned face V-A unavailable ({type(exc).__name__})"
        valence = [rnd(v, 3) for v in va[:, 0]]
        arousal = [rnd(a, 3) for a in va[:, 1]]
        summary = va_summary(valence, arousal)
        summary["expressiveness"] = rnd(np.nanmean([summary["valence_std"] or np.nan, summary["arousal_std"] or np.nan]), 3)
        return {
            "t": [rnd(x, 2) for x in t], "valence": valence, "arousal": arousal,
            "per_second": {"valence": per_second(t, valence, duration, np.nanmean),
                           "arousal": per_second(t, arousal, duration, np.nanmean)},
            "summary": summary, "proxy": backend == "landmark-proxy", "note": note,
            "models": {"face_va": backend},
        }
