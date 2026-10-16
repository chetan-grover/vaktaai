"""Shared fixtures. Every test runs against a temp data/models dir with downloads disabled,
so no learned model is ever fetched and each module exercises its fallback path."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
PULSE_HZ = 1.2  # 72 BPM in the synthetic "skin" patch
SKIN_BOX = (100, 60, 120, 140)  # x, y, w, h of the pulsing patch in the 320x240 synthetic video


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    monkeypatch.setenv("VAKTA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("VAKTA_MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("VAKTA_ALLOW_DOWNLOADS", "0")
    monkeypatch.setenv("VAKTA_MIN_DURATION_S", "1")
    monkeypatch.delenv("VAKTA_LLM_URL", raising=False)
    _clear_caches()
    yield
    _clear_caches()


def _clear_caches() -> None:
    from pipeline.settings import get_settings

    get_settings.cache_clear()
    try:
        from app.db import get_engine

        get_engine.cache_clear()
    except ImportError:
        pass
    from analytics.persona.infer import load_map

    load_map.cache_clear()


def speech_like_audio(duration: float, sr: int = 16000, seed: int = 0) -> np.ndarray:
    """Voiced harmonic signal with a 4 Hz syllable envelope, gliding pitch and two silences."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(duration * sr)) / sr
    f0 = 170 + 30 * np.sin(2 * np.pi * 0.3 * t)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    voiced = sum(np.sin(k * phase) / k for k in range(1, 6))
    envelope = np.clip(np.sin(2 * np.pi * 4 * t), 0, None) ** 0.5
    envelope[(t > duration * 0.4) & (t < duration * 0.4 + 0.8)] = 0
    envelope[(t > duration * 0.75) & (t < duration * 0.75 + 0.6)] = 0
    audio = 0.25 * voiced * envelope + 0.003 * rng.standard_normal(len(t))
    return audio.astype(np.float32)


def make_synthetic_video(path: Path, duration: float = 12.0, fps: int = 30) -> Path:
    """320x240 video: grey background, a skin-coloured patch whose green channel pulses at 72 BPM."""
    import cv2
    import soundfile as sf

    from pipeline.utils import ffmpeg_exe

    tmp_video, tmp_audio = path.with_suffix(".raw.avi"), path.with_suffix(".wav")
    writer = cv2.VideoWriter(str(tmp_video), cv2.VideoWriter_fourcc(*"MJPG"), fps, (320, 240))
    x, y, w, h = SKIN_BOX
    for i in range(int(duration * fps)):
        frame = np.full((240, 320, 3), 90, dtype=np.uint8)
        pulse = 6 * np.sin(2 * np.pi * PULSE_HZ * i / fps)
        frame[y:y + h, x:x + w] = np.clip([120, 150 + pulse, 205 + 0.5 * pulse], 0, 255).astype(np.uint8)  # BGR
        writer.write(frame)
    writer.release()
    sf.write(str(tmp_audio), speech_like_audio(duration), 16000)
    subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(tmp_video), "-i", str(tmp_audio),
                    "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path)],
                   check=True)
    tmp_video.unlink()
    tmp_audio.unlink()
    return path


@pytest.fixture(scope="session")
def synthetic_video(tmp_path_factory) -> Path:
    return make_synthetic_video(tmp_path_factory.mktemp("fixtures") / "synthetic.mp4")


@pytest.fixture
def real_fixture() -> Path:
    """User-recorded fixture (tests/fixtures/short.mp4); tests using it skip when absent."""
    path = FIXTURES / "short.mp4"
    if not path.exists():
        pytest.skip("record tests/fixtures/short.mp4 to run real-face tests")
    return path


@pytest.fixture
def make_ctx(tmp_path):
    """Create a session folder with an input video copied in."""
    from pipeline.base import SessionContext

    def _make(video: Path | None = None, session_id: str = "s1") -> SessionContext:
        ctx = SessionContext.for_session(session_id)
        if video is not None:
            (ctx.session_dir / f"input{video.suffix}").write_bytes(video.read_bytes())
        return ctx

    return _make


def write_module(ctx, name: str, data: dict, status: str = "ok") -> None:
    ctx.write_json(f"{name}.json", {"module": name, "status": status, "data": data, "error": None})


def fake_words(text: str, start: float = 0.5, rate: float = 0.35, gaps: dict[int, float] | None = None) -> list[dict]:
    """Timestamped words; gaps={index: extra_seconds_before_word}."""
    gaps = gaps or {}
    words, t = [], start
    for i, token in enumerate(text.split()):
        t += gaps.get(i, 0.0)
        words.append({"w": token, "start": round(t, 3), "end": round(t + rate * 0.8, 3), "p": 0.9})
        t += rate
    return words


@pytest.fixture
def helpers():
    class H:
        write_module = staticmethod(write_module)
        fake_words = staticmethod(fake_words)
        json = json
        os = os
    return H
