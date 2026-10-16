"""F3 transcription with word-level timestamps (faster-whisper)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from .base import Analyzer, AnalyzerSkipped, SessionContext
from .ingest import load_audio
from .settings import get_settings
from .utils import resolve_device, rnd

# Whisper tends to drop disfluencies; a disfluent prompt makes it keep them (needed for F5).
FILLER_PROMPT = "Umm, let me think like, hmm... Okay, so, uh, here's what I'm, you know, thinking. Matlab, basically."


@lru_cache(maxsize=2)
def load_whisper(size: str, device: str):
    from faster_whisper import WhisperModel

    settings = get_settings()
    compute_type = "float16" if device == "cuda" else "int8"
    return WhisperModel(
        size,
        device=device,
        compute_type=compute_type,
        download_root=str(settings.models_dir / "whisper"),
        local_files_only=not settings.allow_downloads,
        cpu_threads=max(1, (os.cpu_count() or 4) // 2),
    )


def transcribe(audio_path: str, size: str, language: str | None) -> dict[str, Any]:
    try:
        model = load_whisper(size, resolve_device())
    except Exception as exc:  # noqa: BLE001 - usually weights not downloaded yet
        raise RuntimeError(f"Whisper '{size}' could not be loaded ({type(exc).__name__}). "
                           f"Run: python scripts/download_models.py --get whisper-{size}") from exc
    samples, _ = load_audio(Path(audio_path))  # ingest already wrote 16 kHz mono
    segments, info = model.transcribe(
        samples,
        language=language,
        word_timestamps=True,
        initial_prompt=FILLER_PROMPT,
        condition_on_previous_text=False,
        vad_filter=False,
        beam_size=5 if resolve_device() == "cuda" else 1,
    )
    out_segments: list[dict[str, Any]] = []
    words: list[dict[str, Any]] = []
    for seg in segments:
        out_segments.append({"start": rnd(seg.start), "end": rnd(seg.end), "text": seg.text.strip()})
        for w in seg.words or []:
            token = w.word.strip()
            if token:
                words.append({"w": token, "start": rnd(w.start), "end": rnd(w.end), "p": rnd(w.probability)})
    return {
        "language": info.language,
        "language_probability": rnd(info.language_probability),
        "text": " ".join(s["text"] for s in out_segments).strip(),
        "segments": out_segments,
        "words": words,
    }


class AsrAnalyzer(Analyzer):
    name = "asr"
    requires = ("ingest",)

    def analyze(self, ctx: SessionContext) -> dict[str, Any]:
        ingest = self.need(ctx, "ingest")
        if not ingest.get("has_audio"):
            raise AnalyzerSkipped("recording has no audio track")
        settings = get_settings()
        result = transcribe(str(ctx.audio_path), settings.whisper_model, settings.whisper_language)
        result["models"] = {"asr": f"faster-whisper/{settings.whisper_model}"}
        return result
